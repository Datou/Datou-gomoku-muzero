# main.py (FINAL, CLEANED VERSION)

# --- Standard Imports ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import signal
import logging
import torch.multiprocessing as mp
from queue import Empty
import time  # <--- [LINUS'S FIX] ADD THIS. IT IS NOT OPTIONAL.

# --- Project-specific Imports ---
from config import config
from network import GomokuNetEZ
from db_manager import DatabaseManager
from logger_config import setup_worker_logging, logger_process

# Import refactored workers and loss function
from workers import universal_worker, inference_server_worker, data_loader_worker, training_worker, orchestrator_worker, visualize_and_log_worker, display_manager
from loss import calculate_loss

# Import refactored IPC messages and helper classes/functions
from ipc_messages import InitialModelRequest, ModelWeightsUpdate, HeatmapTask
from utils import _convert_to_json_serializable # Note: Only used in a function that is now in workers.py, but kept for context if needed elsewhere.

def log_and_display_config(logger):
    """Logs the key configuration parameters at startup."""
    header = "="*30
    config_details = f"\n{header} Key Configuration {header}\n"
    config_details += f"[System & Environment]\n  - Device: {config.DEVICE}\n  - Universal Workers: {config.NUM_WORKERS}\n\n"
    config_details += f"[MCTS Configuration]\n  - Simulations per move: {config.NUM_SIMULATIONS}\n  - Gumbel Top-K Actions: {config.NUM_TOP_ACTIONS}\n\n"
    config_details += f"[Buffer Configuration]\n  - Prefill Requirement: {config.REPLAY_BUFFER_PREFILL}\n  - Total Buffer Size: {config.TRAIN_BUFFER_SIZE}\n\n"
    config_details += f"[Network & Training]\n  - ResNet Blocks: {config.NUM_RES_BLOCKS}\n  - Learning Rate: {config.LEARNING_RATE}\n  - Physical Batch Size: {config.PHYSICAL_BATCH_SIZE}\n  - Unroll Steps: {config.NUM_UNROLL_STEPS}\n"
    config_details += header + "========================" + header
    logger.info(config_details)
    print(config_details)

# =====================================================================
#                      MAIN EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 1. --- Centralized Logging Setup ---
    log_queue = mp.Queue()
    log_file = "outputs/training.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging_proc = mp.Process(target=logger_process, args=(log_queue, log_file), name="Logger")
    logging_proc.start()
    
    setup_worker_logging(log_queue)
    logger = logging.getLogger("Main")
    log_and_display_config(logger)
    
    # 2. --- Inter-Process Communication Objects ---
    # (This part is correct, no changes needed)
    log_status_queue = mp.Queue()
    ui_queue = mp.Queue()
    trainer_event_queue = mp.Queue()
    resume_info_queue = mp.Queue(maxsize=1)
    data_queue = mp.Queue(maxsize=50000)
    vis_task_queue = mp.Queue()
    inference_request_queue = mp.Queue()
    model_update_queue = mp.Queue(maxsize=1)
    replay_data_queue = mp.Queue(maxsize=20)
    batch_queue = mp.Queue(maxsize=config.PREPARED_BATCH_QUEUE_SIZE)
    priority_update_queue = mp.Queue()
    initial_model_requests_queue = mp.Queue(maxsize=1)
    inference_result_queues = {i: mp.Queue() for i in range(config.NUM_WORKERS)}
    buffer_status_queue = mp.Queue()
    shutdown_event = mp.Event()
    server_ready_event = mp.Event()
    pause_event = mp.Event() 
    WORKER_MODE = mp.Value('i', 0)
    LATEST_MODEL_STEP = mp.Value('i', 0)
    
    # 3. --- Initial State Check ---
    db_manager = DatabaseManager()
    is_first_ever_run = not db_manager.load_trainer_state()
    if is_first_ever_run:
        logger.info("First ever run detected. Database appears empty.")
    else:
        logger.info("Resuming from existing state found in database.")
        
    # 4. --- Process Definitions ---
    processes = []
    process_definitions = {
        "Orchestrator": (orchestrator_worker, (WORKER_MODE, shutdown_event, log_queue, ui_queue, pause_event, LATEST_MODEL_STEP)),
        "InferenceServer": (inference_server_worker, (inference_request_queue, inference_result_queues, model_update_queue, initial_model_requests_queue, shutdown_event, server_ready_event, log_queue)),
        # ======== [LINUS'S FIX] ADD replay_data_queue HERE ========
        "DataLoader": (data_loader_worker, (data_queue, batch_queue, priority_update_queue, log_status_queue, ui_queue, shutdown_event, resume_info_queue, buffer_status_queue, replay_data_queue, log_queue)),
        "Visualizer": (visualize_and_log_worker, (vis_task_queue, log_status_queue, shutdown_event, replay_data_queue, log_queue)),
        "DisplayManager": (display_manager, (ui_queue, shutdown_event, server_ready_event, WORKER_MODE, log_queue)),
        "Trainer": (training_worker, (batch_queue, priority_update_queue, log_status_queue, ui_queue, vis_task_queue, shutdown_event, model_update_queue, initial_model_requests_queue, resume_info_queue, trainer_event_queue, WORKER_MODE, buffer_status_queue, LATEST_MODEL_STEP, log_queue, pause_event)),
    }
    for i in range(config.NUM_WORKERS):
        process_definitions[f"Worker-{i}"] = (universal_worker, (i, WORKER_MODE, data_queue, log_status_queue, ui_queue, shutdown_event, inference_request_queue, inference_result_queues[i], replay_data_queue, trainer_event_queue, LATEST_MODEL_STEP, log_queue, pause_event))

    for name, (target, args) in process_definitions.items():
        processes.append(mp.Process(target=target, args=args, name=name))

    # 5. --- Process Execution and Shutdown ---
    logger.info(f"Starting {len(processes)} processes...")
    for p in processes:
        p.start()
        
    if is_first_ever_run:
        logger.info("Queueing initial heatmap generation for step 0.")
        cpu_weights = {k: v.cpu() for k, v in GomokuNetEZ(config).state_dict().items()}
        vis_task_queue.put(HeatmapTask(step=0, model_state_dict=cpu_weights))
        
    def sigint_handler(signum, frame):
        if not shutdown_event.is_set():
            logger.info("Ctrl+C detected. Initiating graceful shutdown...")
            shutdown_event.set()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
    
    # =================== [LINUS'S FIX START] ===================
    # This is the correct way to manage the main process.
    # It waits for the shutdown signal, not for a single child process to die.
    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    finally:
        logger.info("Shutdown signal received in main process. Finalizing shutdown of all workers...")
        
        # Give processes a moment to finish cleanly
        for p in processes:
            if p.is_alive():
                p.join(timeout=5)
                
        # Now, terminate any stubborn processes
        for p in processes:
            if p.is_alive():
                logger.warning(f"Process {p.name} (PID: {p.pid}) did not exit gracefully. Terminating.")
                p.terminate()

        # Cleanly shut down the logger
        log_queue.put(None)
        logging_proc.join(timeout=5)
        if logging_proc.is_alive():
            logging_proc.terminate()
        
        logger.info("All workers shut down.")
    # =================== [LINUS'S FIX END] ===================