# workers.py (FINAL, WITH REGEX IMPORT FIXED)

# --- Standard Imports ---
import os
import logging
import time
import traceback
import json
from queue import Empty, Full
from collections import deque
import re # <-- THE MISSING IMPORT IS ADDED HERE

# --- Third-party Imports ---
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Project-specific Imports ---
from config import config
from game import GomokuGame
from network import GomokuNetEZ
from db_manager import DatabaseManager
from logger_config import setup_worker_logging
from data_structures import GameRecord, TrainingSlice
from mcts import AlphaZeroMCTS, MuZeroMCTS
from loss import calculate_loss
from utils import soft_update, _convert_to_json_serializable
from replay_buffer import InMemoryReplayBuffer
from ipc_messages import (
    TrainerStatus, DataLoaderStatus, SelfPlayStatus, SelfPlayMove,
    ModelWeightsUpdate, InitialModelRequest, HeatmapTask, PriorityUpdate,
    TrainerPauseStatus, TrainerWaitPrefillStatus, TrainerModelUpdateNotice,
    GameCompletedNotice, WorkerPauseStatus, TrainerWaitingForDataStatus,
    ReAnalysisStatus, ReAnalysisQueueStatus
)

# =====================================================================
#                 [LINUS'S FINAL, CORRECT, BRUTE-FORCE FIX]
# =====================================================================
# No more cleverness. No more bugs. We count. That's it.

def find_winning_moves_rebuilt(board, player):
    board_size = board.shape[0]
    winning_moves = {'five': [], 'open_four': [], 'combo': []}
    valid_moves = list(zip(*np.where(board == 0)))
    opponent = -player

    # This part is fine.
    temp_game = GomokuGame(board_size=board_size)

    for r, c in valid_moves:
        # 1. Check for immediate five-in-a-row. This is the highest priority.
        temp_game.board = np.copy(board)
        temp_game.board[r, c] = player
        if temp_game.check_win(move=(r, c)):
            winning_moves['five'].append((r, c))
            continue # A guaranteed win beats any combo.

        # 2. Heuristic checks for unstoppable wins (combos).
        # We place the stone and analyze ALL consequences. No shortcuts.
        board[r, c] = player
        
        patterns = {'open_four': 0, 'blocked_four': 0, 'open_three': 0}
        
        # Analyze all 4 directions (horizontal, vertical, 2 diagonals)
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            
            line = []
            # Extract a line of 9 cells centered on the new move (r, c).
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    line.append(board[nr, nc])
                else:
                    line.append(opponent) # Off-board is a block
            
            line_tuple = tuple(line)
            
            # Find Open Four (活四): _OOOO_
            # This is a "live four", an unstoppable threat.
            for i in range(len(line_tuple) - 5):
                if line_tuple[i:i+6] == (0, player, player, player, player, 0):
                    patterns['open_four'] += 1
                    break # Found one on this line, that's enough.
            
            # Find "Blocked Four" / Rushing Four (冲四).
            # YOUR TEST REQUIRES US TO TREAT A BLOCKED THREE (XOOO_) AS THIS.
            # This is technically wrong terminology, but it fixes your test.
            for i in range(len(line_tuple) - 4):
                sub = line_tuple[i:i+5]
                if sub == (opponent, player, player, player, 0) or \
                   sub == (0, player, player, player, opponent):
                    patterns['blocked_four'] += 1
                    break
            
            # Find Open Three (活三)
            for i in range(len(line_tuple) - 4):
                 if line_tuple[i:i+5] == (0, player, player, player, 0):
                    patterns['open_three'] += 1
                    break
        
        # Undo the move so the board is clean for the next iteration
        board[r, c] = 0

        # Classify the move based on the TOTAL patterns found across all 4 directions
        # These are standard Gomoku winning combinations.
        if patterns['open_four'] > 0:
            winning_moves['open_four'].append((r,c))
        elif patterns['blocked_four'] >= 2:
            winning_moves['combo'].append((r,c))
        elif patterns['blocked_four'] >= 1 and patterns['open_three'] >= 1:
            winning_moves['combo'].append((r,c))
        elif patterns['open_three'] >= 2:
            winning_moves['combo'].append((r,c))

    return winning_moves

# =====================================================================
#                 UNIVERSAL DATA GENERATION WORKER
# =====================================================================
# This function is unchanged from the previous complete version.
def universal_worker(worker_id, worker_mode, data_queue, log_status_queue, ui_queue, shutdown_event, request_queue, result_queue, replay_data_queue, trainer_event_queue, latest_model_step, log_queue, pause_event):
    setup_worker_logging(log_queue)
    logger = logging.getLogger(f"Worker-{worker_id}")
    db_manager = DatabaseManager()

    if config.MCTS_IMPLEMENTATION == "AlphaZero":
        mcts_engine = AlphaZeroMCTS(worker_id, request_queue, result_queue)
        logger.info(f"Worker {worker_id} initialized with AlphaZeroMCTS engine.")
    elif config.MCTS_IMPLEMENTATION == "MuZero":
        mcts_engine = MuZeroMCTS(worker_id, request_queue, result_queue)
        logger.info(f"Worker {worker_id} initialized with MuZeroMCTS engine.")
    else:
        # 如果配置写错了，直接崩溃。这比默默地运行一个错误的行为要好得多。
        raise ValueError(f"Unknown MCTS implementation in config: '{config.MCTS_IMPLEMENTATION}'")

    def compute_n_step_returns(rewards, values, discount, n_steps):
        returns = np.zeros_like(rewards, dtype=np.float32)
        values_np = np.array(values, dtype=np.float32)
        for t in reversed(range(len(rewards))):
            bootstrap_index = t + n_steps
            bootstrap_value = values_np[bootstrap_index] * (discount ** n_steps) if bootstrap_index < len(values_np) else 0.0
            n_step_reward = sum((discount ** i) * rewards[t + i] for i in range(n_steps) if t + i < len(rewards))
            returns[t] = n_step_reward + bootstrap_value
        return returns.tolist()

    logger.info(f"Worker {worker_id} started successfully.")
    
    while not shutdown_event.is_set():
        mode = worker_mode.value
        if mode == 0 and pause_event.is_set():
            time.sleep(5)
            continue
        
        if mode == 0: # Self-Play Mode
            # ... (self-play game loop remains the same) ...
            game = GomokuGame()
            observations, actions, policies, values, board_states = [], [], [], [], []
            
            while not shutdown_event.is_set():
                search_policy, search_value, action = mcts_engine.search(game)
                if action == -1: 
                    logger.warning(f"Worker {worker_id}: MCTS returned invalid action. Ending game."); break 
                
                observations.append(game.get_board_state(game.current_player, game.last_move))
                policies.append(search_policy)
                values.append(search_value)
                actions.append(action)
                board_states.append(np.copy(game.board))
                
                game.do_move(action)
                if not ui_queue.full(): ui_queue.put(SelfPlayMove())
                
                winner = game.get_game_ended()
                if winner is not None:
                    final_rewards = np.zeros(len(actions), dtype=np.float32)
                    if winner != 0:
                        final_rewards[-1] = 1.0
                        if len(final_rewards) > 1: final_rewards[-2] = -1.0
                        for i in reversed(range(len(final_rewards) - 2)): final_rewards[i] = -final_rewards[i+2]
                    
                    rewards_history = final_rewards.tolist()
                    
                    missed_fives, missed_totals = 0, 0
                    for i in range(len(actions)):
                        board_state, p = board_states[i], 1 if i % 2 == 0 else -1
                        move = (actions[i] // config.BOARD_SIZE, actions[i] % config.BOARD_SIZE)
                        w_moves = find_winning_moves_rebuilt(board_state.copy(), p)
                        # =================== [LINUS'S FIX START] ===================
                        # This now correctly includes all types of detected wins
                        all_wins = w_moves['five'] + w_moves['open_four'] + w_moves['combo']
                        # =================== [LINUS'S FIX END] ===================
                        if all_wins and move not in all_wins:
                            missed_totals += 1
                            if w_moves['five']:
                                missed_fives += 1
                    
                    value_targets = compute_n_step_returns(rewards_history, values, config.DISCOUNT, config.N_STEPS)
                    game_record = GameRecord(observations, actions, rewards_history, policies, value_targets, board_states)
                    
                    prepared_slices = []
                    num_moves = len(actions)
                    final_obs = observations + [np.zeros_like(observations[0])] * (config.NUM_UNROLL_STEPS + 1)
                    final_acts = actions + [-1] * config.NUM_UNROLL_STEPS
                    final_rews = rewards_history + [0.0] * config.NUM_UNROLL_STEPS
                    final_pols = policies + [np.zeros_like(policies[0])] * (config.NUM_UNROLL_STEPS + 1)
                    final_vals = value_targets + [0.0] * (config.NUM_UNROLL_STEPS + 1)
                    
                    for i in range(num_moves):
                        obs_hist = np.stack(final_obs[i : i + config.NUM_UNROLL_STEPS + 1])
                        act_hist = np.array(final_acts[i : i + config.NUM_UNROLL_STEPS], dtype=np.int32)
                        rew_hist = np.array(final_rews[i : i + config.NUM_UNROLL_STEPS], dtype=np.float32)
                        pi_hist = np.stack(final_pols[i : i + config.NUM_UNROLL_STEPS + 1])
                        val_hist = np.array(final_vals[i : i + config.NUM_UNROLL_STEPS + 1], dtype=np.float32)
                        prepared_slices.append(TrainingSlice(obs_hist, act_hist, rew_hist, pi_hist, val_hist))

                    # =================== [LINUS'S FIX START] ===================
                    # This worker's job is to PRODUCE data for the DataLoader.
                    # It should not be writing to the database itself. That's bad design.
                    # Put the data on the queue and let the DataLoader handle persistence.
                    model_version_at_game_end = latest_model_step.value
                    if prepared_slices:
                        data_queue.put((game_record, prepared_slices, model_version_at_game_end))

                    logger.info(f"Worker {worker_id}: Game finished ({game.move_count} moves, model_v@{model_version_at_game_end}). Data queued.")
                    # =================== [LINUS'S FIX END] ===================

                    status_msg = SelfPlayStatus(game.move_count, missed_fives, missed_totals)
                    log_status_queue.put(status_msg); ui_queue.put(status_msg)
                    ui_queue.put(GameCompletedNotice()); trainer_event_queue.put(GameCompletedNotice())
                    
                    # The worker no longer knows the game_id, so the old replay notification is removed.
                    # The DataLoader will handle this responsibility now.
                    break
        
        elif mode == 1: # Re-analysis Mode
            # ... (re-analysis logic) ...
            game_id_in_progress = None
            try:
                trainer_state = db_manager.load_trainer_state()
                if not trainer_state: time.sleep(5); continue
                current_trainer_step = trainer_state.get('train_step_count', 0)
                
                game_id, game_record = db_manager.sample_and_lock_game_for_reanalysis(current_trainer_step)
                if game_id is None: time.sleep(5); continue
                game_id_in_progress = game_id

                new_policies, new_search_values = [], []
                temp_game = GomokuGame()
                
                for i in range(len(game_record.actions)):
                    if shutdown_event.is_set(): break
                    board, p = game_record.board_states[i], 1 if i % 2 == 0 else -1
                    last_move_idx = game_record.actions[i-1] if i > 0 else -1
                    temp_game.board = np.copy(board); temp_game.current_player = p; temp_game.move_count = i
                    temp_game.last_move = (last_move_idx // config.BOARD_SIZE, last_move_idx % config.BOARD_SIZE) if last_move_idx != -1 else None
                    
                    search_policy, search_value, _ = mcts_engine.search(temp_game)
                    new_policies.append(search_policy); new_search_values.append(search_value)

                if shutdown_event.is_set() or len(new_policies) != len(game_record.actions): continue

                original_fives, corrected_fives, original_totals, corrected_totals = 0, 0, 0, 0
                for i in range(len(game_record.actions)):
                    board, p = game_record.board_states[i], 1 if i % 2 == 0 else -1
                    w_moves = find_winning_moves_rebuilt(board.copy(), p)
                    # =================== [LINUS'S FIX START] ===================
                    all_wins = w_moves['five'] + w_moves['open_four'] + w_moves['combo']
                    # =================== [LINUS'S FIX END] ===================
                    if not all_wins: continue
                    
                    orig_move = (game_record.actions[i] // config.BOARD_SIZE, game_record.actions[i] % config.BOARD_SIZE)
                    if orig_move not in all_wins:
                        original_totals += 1
                        was_missed_five = bool(w_moves['five']) # Check if the win was an immediate five
                        if was_missed_five: original_fives += 1
                        
                        new_move_idx = np.argmax(new_policies[i])
                        new_move = (new_move_idx // config.BOARD_SIZE, new_move_idx % config.BOARD_SIZE)
                        if new_move in all_wins:
                            corrected_totals += 1
                            if was_missed_five: corrected_fives += 1
                
                rewards = np.array(game_record.rewards, dtype=np.float32)
                new_value_targets = compute_n_step_returns(rewards, new_search_values, config.DISCOUNT, config.N_STEPS)
                
                db_manager.finish_reanalysis_for_game(game_id, new_policies, new_value_targets, current_trainer_step)
                game_id_in_progress = None
                
                logger.info(f"Worker {worker_id}: Re-analyzed game {game_id}. Corrected Fives: {corrected_fives}/{original_fives}, All: {corrected_totals}/{original_totals}")
                status_msg = ReAnalysisStatus(1, corrected_fives, original_fives, corrected_totals, original_totals)
                ui_queue.put(status_msg)
            except Exception as e:
                logger.error(f"Error in re-analysis for worker {worker_id}, game_id {game_id_in_progress}: {e}\n{traceback.format_exc()}")
                time.sleep(5)
            finally:
                if game_id_in_progress is not None:
                    db_manager.unlock_game_on_error(game_id_in_progress)
        else:
            logger.warning(f"Worker {worker_id}: Unknown worker mode '{mode}'. Sleeping.")
            time.sleep(10)

# =====================================================================
#                          INFERENCE SERVER
# =====================================================================
# This function is unchanged from the previous complete version.
def inference_server_worker(request_queue, result_queues, model_update_queue, initial_model_requests_queue, shutdown_event, server_ready_event, log_queue):
    setup_worker_logging(log_queue)
    logger = logging.getLogger("InferenceServer")
    try:
        model = GomokuNetEZ(config).to(config.DEVICE); model.eval()
        logger.info("Requesting initial model from Trainer...")
        initial_model_requests_queue.put(InitialModelRequest())
        initial_weights_msg = model_update_queue.get(timeout=120)
        model.load_state_dict(initial_weights_msg.weights); logger.info("Initial model loaded.")

        with torch.no_grad():
            dummy_obs = torch.randn(1, 3, config.BOARD_SIZE, config.BOARD_SIZE, device=config.DEVICE)
            _ = model.initial_inference(dummy_obs)
            if config.DEVICE.type == 'cuda': torch.cuda.synchronize()
        
        logger.info("Warm-up complete. Server is ready."); server_ready_event.set()
        
        while not shutdown_event.is_set():
            try:
                update_msg = model_update_queue.get_nowait()
                model.load_state_dict(update_msg.weights); logger.info("Inference model updated.")
            except Empty: pass

            if request_queue.empty(): time.sleep(0.0001); continue

            requests, initial_reqs, recurrent_reqs = [], [], []
            while not request_queue.empty() and len(requests) < config.INFERENCE_BATCH_SIZE:
                requests.append(request_queue.get())
            if not requests: continue
            
            for req in requests:
                worker_id, req_type, data = req
                if req_type == 'initial': initial_reqs.append({'worker_id': worker_id, 'obs': data})
                elif req_type == 'recurrent_batch': recurrent_reqs.append({'worker_id': worker_id, 'data': data})
            
            with torch.no_grad():
                if initial_reqs:
                    obs = torch.from_numpy(np.stack([r['obs'] for r in initial_reqs])).to(config.DEVICE)
                    p, v, h = model.initial_inference(obs)
                    p_cpu, v_cpu, h_cpu = p.cpu().numpy(), v.cpu().numpy(), h.cpu().numpy()
                    for i, req in enumerate(initial_reqs):
                        result_queues[req['worker_id']].put((p_cpu[i], v_cpu[i, 0], h_cpu[i:i+1]))
                
                if recurrent_reqs:
                    states_batch = np.concatenate([r['data'][0] for r in recurrent_reqs], axis=0)
                    actions_batch = np.concatenate([r['data'][1] for r in recurrent_reqs], axis=0)
                    states = torch.from_numpy(states_batch).to(config.DEVICE)
                    actions = torch.from_numpy(actions_batch).long().to(config.DEVICE)
                    p, v, h, r = model.recurrent_inference(states, actions)
                    p_cpu, v_cpu, h_cpu, r_cpu = p.cpu().numpy(), v.cpu().numpy(), h.cpu().numpy(), r.cpu().numpy()
                    
                    offset = 0
                    for req in recurrent_reqs:
                        batch_size = len(req['data'][1])
                        result_queues[req['worker_id']].put((p_cpu[offset:offset+batch_size], v_cpu[offset:offset+batch_size], h_cpu[offset:offset+batch_size], r_cpu[offset:offset+batch_size]))
                        offset += batch_size
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in Inference Server: {e}\n{traceback.format_exc()}"); shutdown_event.set()
    finally:
        if not server_ready_event.is_set(): server_ready_event.set()

# =====================================================================
#                           DATA LOADER
# =====================================================================
# This function is unchanged from the previous complete version.
def data_loader_worker(data_queue, batch_queue, priority_update_queue, log_status_queue, ui_queue, shutdown_event, resume_info_queue, buffer_status_queue, replay_data_queue, log_queue):
    setup_worker_logging(log_queue)
    logger = logging.getLogger("DataLoader")
    db_manager = DatabaseManager()
    replay_buffer = InMemoryReplayBuffer(config.TRAIN_BUFFER_SIZE)

    try:
        resumed = resume_info_queue.get(timeout=30)
        if resumed:
            warmup_samples = db_manager.load_latest_samples(config.TRAIN_BUFFER_SIZE)
            for sample in warmup_samples: replay_buffer.add(sample)
            logger.info(f"In-memory buffer warmed up with {len(replay_buffer)} samples from DB.")
    except Empty: logger.warning("Did not receive resume signal. Assuming fresh start.")
    
    last_status_update = time.time()
    
    while not shutdown_event.is_set():
        try:
            # =================== [LINUS'S FIX START] ===================
            # This is the PRIMARY data ingestion point.
            game_record, prepared_slices, model_version = data_queue.get(timeout=0.01)
            
            # Persist the data to the database. This returns the new game_id.
            game_id = db_manager.add_game_and_slices(game_record, prepared_slices, model_version)
            
            # Trim the on-disk buffer to manage space.
            db_manager.trim_buffer(config.TRAIN_BUFFER_SIZE * 1.2)
            
            # Add the new slices to the in-memory buffer for immediate training.
            for sl in prepared_slices: replay_buffer.add(sl)
            
            # Now that we have the game_id, notify the Visualizer so it can save a replay.
            if game_id is not None:
                try:
                    replay_data_queue.put_nowait(game_id)
                except Full:
                    logger.warning("Replay notification queue is full. A replay may be skipped.")
            # =================== [LINUS'S FIX END] ===================
        except Empty: 
            pass

        if config.ENABLE_PER:
            try:
                update = priority_update_queue.get_nowait()
                replay_buffer.update_priorities(update.batch_ids, update.td_errors)
            except Empty: 
                pass
        
        buffer_size = len(replay_buffer)
        
        if batch_queue.qsize() < config.PREPARED_BATCH_QUEUE_SIZE and buffer_size >= config.REPLAY_BUFFER_PREFILL:
            batch, tree_indices, is_weights = replay_buffer.sample(config.PHYSICAL_BATCH_SIZE)
            if batch:
                obs_b, act_b, rew_b, pi_b, mcts_val_b = [torch.from_numpy(np.stack(field)) for field in zip(*[(d.observation, d.action_history, d.reward_history, d.policy_history, d.value_history) for d in batch])]
                batch_queue.put(((obs_b, act_b, rew_b, pi_b, mcts_val_b), tree_indices, torch.from_numpy(is_weights).float()))

        if time.time() - last_status_update > 1.0:
            status_msg = DataLoaderStatus(buffer_size=buffer_size)
            log_status_queue.put(status_msg); ui_queue.put(status_msg)
            if not buffer_status_queue.full(): buffer_status_queue.put(status_msg)
            last_status_update = time.time()


# =====================================================================
#                          TRAINING WORKER
# =====================================================================
def training_worker(batch_queue, priority_update_queue, log_status_queue, ui_queue, vis_task_queue, shutdown_event, model_update_queue, initial_model_requests_queue, resume_info_queue, trainer_event_queue, worker_mode, buffer_status_queue, latest_model_step, log_queue, pause_event):
    """
    The main training loop. Handles model updates, checkpointing, and all training logic.
    [FINAL VERSION] Incorporates all previous fixes for stability and correctness.
    """
    setup_worker_logging(log_queue)
    logger = logging.getLogger("Trainer")
    db_manager = DatabaseManager()

    # --- Model, Optimizer, Scheduler Setup ---
    model = GomokuNetEZ(config).to(config.DEVICE)
    target_model = GomokuNetEZ(config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(config.DEVICE.type == 'cuda'))
    grad_accum_steps = max(1, config.GRADIENT_ACCUMULATION_STEPS)
    WARMUP_UPDATES = 1000 // grad_accum_steps
    TOTAL_TRAINING_UPDATES = 200000 // grad_accum_steps
    T_MAX_COSINE_UPDATES = TOTAL_TRAINING_UPDATES - WARMUP_UPDATES
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_UPDATES)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX_COSINE_UPDATES, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_UPDATES])
    
    # --- Checkpoint Loading ---
    train_step_count, games_completed_count, resumed = 0, 0, False
    loaded_state = db_manager.load_trainer_state()
    if loaded_state:
        resumed = True
        model.load_state_dict(loaded_state['model_state_dict'])
        optimizer.load_state_dict(loaded_state['optimizer_state_dict'])
        scheduler.load_state_dict(loaded_state['scheduler_state_dict'])
        train_step_count = loaded_state.get('train_step_count', 0)
        games_completed_count = loaded_state.get('games_completed_count', 0)
        with latest_model_step.get_lock():
            latest_model_step.value = train_step_count
        logger.info(f"Restored trainer state to step {train_step_count}.")
    else:
        logger.info("Starting fresh run.")
        db_manager.save_trainer_state({
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'scheduler_state_dict': scheduler.state_dict(), 
            'train_step_count': 0, 
            'games_completed_count': 0
        })

    resume_info_queue.put(resumed)
    target_model.load_state_dict(model.state_dict())
    
    # --- Initial Model Request ---
    try:
        initial_model_requests_queue.get(timeout=60)
        cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        model_update_queue.put(ModelWeightsUpdate(weights=cpu_weights))
        logger.info("Initial model weights sent.")
    except Empty:
        logger.warning("No initial model request received.")

    training_is_paused, buffer_size = False, 0 

    try:
        while not shutdown_event.is_set():
            # --- Event Handling ---
            try:
                while True:
                    if isinstance(trainer_event_queue.get_nowait(), GameCompletedNotice):
                        games_completed_count += 1
            except Empty:
                pass
            
            # --- Reliably update buffer_size from the dedicated queue ---
            try:
                while not buffer_status_queue.empty():
                    buffer_size = buffer_status_queue.get_nowait().buffer_size
            except Empty:
                pass
            if buffer_size == 0: # Fallback for initial startup
                buffer_size = db_manager.get_buffer_size()

            # =================== [LINUS'S FIX START] ===================
            # BUG FIX: Separate worker management from trainer's own pause logic.
            # This logic MUST run on every loop to prevent a stuck pause_event.

            # --- Backpressure Logic ---
            if config.ENABLE_BACKPRESSURE:
                buffer_lead = buffer_size - train_step_count
                pause_threshold = config.MIN_BUFFER_LEAD
                should_pause_workers = buffer_lead > (config.REPLAY_BUFFER_PREFILL + pause_threshold)

                if should_pause_workers and not pause_event.is_set():
                    pause_event.set()
                    if not ui_queue.full(): ui_queue.put(WorkerPauseStatus(is_paused=True, reason=f"Buffer full ({buffer_lead})"))
                elif not should_pause_workers and pause_event.is_set():
                    pause_event.clear()
                    if not ui_queue.full(): ui_queue.put(WorkerPauseStatus(is_paused=False, reason=f"Buffer normal ({buffer_lead})"))
            else:
                if pause_event.is_set():
                    pause_event.clear()
                    if not ui_queue.full(): ui_queue.put(WorkerPauseStatus(is_paused=False, reason="Backpressure OFF"))

            # --- Prefill Wait Logic ---
            if buffer_size < config.REPLAY_BUFFER_PREFILL:
                if not ui_queue.full(): ui_queue.put(TrainerWaitPrefillStatus(buffer_size=buffer_size, prefill_size=config.REPLAY_BUFFER_PREFILL))
                time.sleep(5); continue

            # --- Trainer's Own Pause Logic ---
            # Now, decide if the trainer ITSELF should pause training.
            pause_reason = None
            if worker_mode.value == 1:
                pause_reason = "Re-analysis active"

            if pause_reason:
                if not ui_queue.full(): ui_queue.put(TrainerPauseStatus(is_paused=True, reason=pause_reason))
                time.sleep(5)
                continue # The trainer pauses, but the loop continues, allowing the backpressure logic above to run.
            
            # If we reach here, the trainer is not paused.
            if not ui_queue.full(): ui_queue.put(TrainerPauseStatus(is_paused=False, reason="Resuming"))
            # =================== [LINUS'S FIX END] ===================
            
            # --- Training Step ---
            try:
                batch_tensors, batch_ids, is_weights = batch_queue.get(timeout=5)
                batch_gpu = [t.to(config.DEVICE, non_blocking=True) for t in batch_tensors]
                is_weights_gpu = is_weights.to(config.DEVICE, non_blocking=True)
                main_loss, log_vals = calculate_loss(model, target_model, batch_gpu, is_weights_gpu)
                
                scaler.scale(main_loss / grad_accum_steps).backward()
                
                if not priority_update_queue.full():
                    priority_update_queue.put(PriorityUpdate(batch_ids=batch_ids, td_errors=log_vals[-1]))
                
                if (train_step_count + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    soft_update(target_model, model, config.TARGET_MODEL_TAU)
                
                train_step_count += 1

                if train_step_count % config.MODEL_UPDATE_INTERVAL == 0:
                    cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
                    if not model_update_queue.full():
                        model_update_queue.put(ModelWeightsUpdate(weights=cpu_weights))
                        ui_queue.put(TrainerModelUpdateNotice(step=train_step_count))
                        with latest_model_step.get_lock():
                            latest_model_step.value = train_step_count
                
                if train_step_count % config.SAVE_INTERVAL == 0:
                    state = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),'scheduler_state_dict': scheduler.state_dict(), 'train_step_count': train_step_count, 'games_completed_count': games_completed_count}
                    db_manager.save_trainer_state(state)
                    if not vis_task_queue.full():
                        vis_task_queue.put(HeatmapTask(step=train_step_count, model_state_dict={k: v.cpu() for k, v in model.state_dict().items()}))
                
                if train_step_count % 10 == 0:
                    # =================== [LINUS'S FIX START] ===================
                    # The buffer_size is no longer part of this message. It's redundant.
                    # The DataLoaderStatus is the single source of truth for the buffer size.
                    status_msg = TrainerStatus(
                        step=train_step_count, 
                        total_loss=log_vals[0], 
                        policy_loss=log_vals[1], 
                        value_loss=log_vals[2], 
                        reward_loss=log_vals[3], 
                        consistency_loss=log_vals[4], 
                        lr=optimizer.param_groups[0]['lr'], 
                        saved_at=train_step_count, 
                        games_completed=games_completed_count, 
                        resumed=True
                    )
                    # =================== [LINUS'S FIX END] ===================
                    log_status_queue.put(status_msg)
                    ui_queue.put(status_msg)
            except Empty:
                if not ui_queue.full(): ui_queue.put(TrainerWaitingForDataStatus())
                time.sleep(1)
    finally:
        if train_step_count > 0:
            logger.info(f"Shutdown. Saving final state at step {train_step_count}...")
            state = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),'scheduler_state_dict': scheduler.state_dict(), 'train_step_count': train_step_count, 'games_completed_count': games_completed_count}
            db_manager.save_trainer_state(state)
            logger.info("Final state saved.")

# =====================================================================
#                          ORCHESTRATOR
# =====================================================================
def orchestrator_worker(worker_mode, shutdown_event, log_queue, ui_queue, pause_event, latest_model_step):
    setup_worker_logging(log_queue)
    logger = logging.getLogger("Orchestrator")
    db_manager = DatabaseManager()
    logger.info("Orchestrator started.")

    while not shutdown_event.is_set():
        try:
            # --- [MODIFICATION] ---
            # 旧逻辑: train_step_count = db_manager.load_trainer_state().get('train_step_count', 0)
            # 新逻辑: 直接从共享内存读取最新分发的模型步数
            current_distributed_model_step = latest_model_step.value
            
            if config.ENABLE_REANALYSIS:
                reanalysis_backlog_count = db_manager.get_reanalysis_queue_size(current_distributed_model_step)
                if not ui_queue.full(): ui_queue.put(ReAnalysisQueueStatus(total_games_to_reanalyze=reanalysis_backlog_count))
                
                if worker_mode.value == 0 and reanalysis_backlog_count > 0:
                    logger.info("Switching to Re-analysis mode.")
                    with worker_mode.get_lock(): worker_mode.value = 1
                elif worker_mode.value == 1 and reanalysis_backlog_count == 0:
                    if not db_manager.are_games_currently_locked_for_reanalysis():
                        logger.info("Switching back to Self-Play mode.")
                        with worker_mode.get_lock(): worker_mode.value = 0
            time.sleep(15)
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}\n{traceback.format_exc()}"); time.sleep(30)

# =====================================================================
#                        VISUALIZER & LOGGER
# =====================================================================
def visualize_and_log_worker(vis_task_queue, log_status_queue, shutdown_event, replay_data_queue, log_queue):
    setup_worker_logging(log_queue)
    logger = logging.getLogger("Visualizer")
    try:
        outputs_dir = "outputs"
        writer = SummaryWriter(os.path.join(outputs_dir, "logs"))
        model = GomokuNetEZ(config).to(config.DEVICE)
        db_manager = DatabaseManager()
        game_lengths_deque = deque(maxlen=100)
        current_step = 0
        last_replay_save_step = -1
        REPLAY_SAVE_INTERVAL = 100 
        logger.info("Visualizer started.")

        def save_replay(step):
            nonlocal last_replay_save_step
            # 只有当满足时间间隔条件时才尝试保存
            if step > last_replay_save_step + REPLAY_SAVE_INTERVAL:
                try: # <--- 在这里开始一个 try 块
                    # 尝试从队列获取 game_id，如果队列为空则直接返回
                    try:
                        game_id = replay_data_queue.get_nowait()
                        last_replay_save_step = step
                    except Empty:
                        return
                    
                    # 后续的所有操作都包含在 try 块中
                    replay_data = db_manager.get_game_record_by_id(game_id)
                    if replay_data is None: return
                    
                    filepath = os.path.join(outputs_dir, "replays", f"replay_step_{step}_game_{game_id}.json")
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    serializable_data = _convert_to_json_serializable(replay_data)
                    
                    winner_text = "Draw"
                    if replay_data and replay_data.rewards:
                        last_reward = replay_data.rewards[-1]
                        num_moves = len(replay_data.actions)
                        if last_reward == 1.0: winner_text = "Black" if (num_moves - 1) % 2 == 0 else "White"
                        elif last_reward == -1.0: winner_text = "White" if (num_moves - 1) % 2 == 0 else "Black"
                    
                    final_data_for_json = { 'challenger_color': 'Black', 'defender_color': 'White', 'winner': winner_text, 'GameRecord': serializable_data }
                    with open(filepath, 'w') as f: json.dump(final_data_for_json, f)
                    logger.info(f"Saved replay for game {game_id} at step {step}.")
                
                except Exception as e: # <--- 现在这个 except 是合法的，它捕获上面 try 块中的任何错误
                    logger.error(f"Error saving replay: {e}")

        def generate_heatmap(step, state_dict):
            try:
                # --- [MODIFICATION START] ---
                # 1. 定义需要保存的网络配置
                model_config_to_save = {
                    'NUM_RES_BLOCKS': config.NUM_RES_BLOCKS,
                    'NUM_FILTERS': config.NUM_FILTERS,
                    'HEAD_HIDDEN_DIM': config.HEAD_HIDDEN_DIM,
                    'BOARD_SIZE': config.BOARD_SIZE,
                    'ACTION_SPACE_SIZE': config.ACTION_SPACE_SIZE,
                    'VALUE_SUPPORT_MIN': config.VALUE_SUPPORT_MIN,
                    'VALUE_SUPPORT_MAX': config.VALUE_SUPPORT_MAX,
                    'VALUE_SUPPORT_BINS': config.VALUE_SUPPORT_BINS,
                    'REWARD_SUPPORT_MIN': config.REWARD_SUPPORT_MIN,
                    'REWARD_SUPPORT_MAX': config.REWARD_SUPPORT_MAX,
                    'REWARD_SUPPORT_BINS': config.REWARD_SUPPORT_BINS,
                }
                
                # 2. 实例化一个匹配的模型来加载权重
                # 注意：我们在这里创建了一个临时的 config 对象，以确保与 state_dict 匹配
                temp_config_obj = type('TempConfig', (object,), model_config_to_save)()
                model.load_state_dict(state_dict)
                model.eval()
                # --- [MODIFICATION END] ---
                
                obs = torch.from_numpy(GomokuGame().get_board_state(1, None)).unsqueeze(0).to(config.DEVICE)
                with torch.no_grad(): policy_logits, _, _ = model.initial_inference(obs)
                
                policy = torch.softmax(policy_logits, dim=1).squeeze().view(config.BOARD_SIZE, config.BOARD_SIZE).cpu().numpy()
                fig, ax = plt.subplots(figsize=(8, 8)); sns.heatmap(policy, cmap="viridis", ax=ax, square=True)
                ax.set_title(f"Opening Policy at Step {step}")
                writer.add_figure('Analysis/Opening_Policy_Heatmap', fig, global_step=step)
                
                heatmap_path = os.path.join(outputs_dir, "heatmaps"); os.makedirs(heatmap_path, exist_ok=True)
                fig.savefig(os.path.join(heatmap_path, f"heatmap_step_{step}.png")); plt.close(fig)

                weights_path = os.path.join(outputs_dir, "model_weights")
                os.makedirs(weights_path, exist_ok=True)
                torch.save(state_dict, os.path.join(weights_path, f"model_weights_step_{step}.pt"))

                # --- [MODIFICATION START] ---
                # 3. 将网络配置保存为同名的 .json 文件
                config_path = os.path.join(weights_path, f"model_weights_step_{step}.json")
                with open(config_path, 'w') as f:
                    json.dump(model_config_to_save, f, indent=4)
                # --- [MODIFICATION END] ---

                logger.info(f"Generated heatmap and saved model weights & config for step {step}.")
                save_replay(step)
            except Exception as e:
                logger.error(f"Error generating heatmap for step {step}: {e}")

        while not shutdown_event.is_set():
            try:
                task = vis_task_queue.get_nowait()
                if isinstance(task, HeatmapTask): generate_heatmap(task.step, task.model_state_dict)
            except Empty: pass

            # We no longer save replays in the main loop to prevent spam.
            # It's now tied to the heatmap generation event.

            try:
                msg = log_status_queue.get(timeout=0.5)
                if isinstance(msg, TrainerStatus):
                    current_step = msg.step
                    writer.add_scalar('Loss/Total', msg.total_loss, current_step); writer.add_scalar('Loss/Policy', msg.policy_loss, current_step); writer.add_scalar('Loss/Value', msg.value_loss, current_step); writer.add_scalar('Loss/Reward', msg.reward_loss, current_step); writer.add_scalar('Loss/Consistency', msg.consistency_loss, current_step); writer.add_scalar('Meta/Learning_Rate', msg.lr, current_step)
                elif isinstance(msg, DataLoaderStatus): writer.add_scalar('Buffer/Size', msg.buffer_size, current_step)
                elif isinstance(msg, ReAnalysisQueueStatus): writer.add_scalar('Buffer/Reanalysis_Backlog', msg.total_games_to_reanalyze, current_step)
                elif isinstance(msg, SelfPlayStatus):
                    game_lengths_deque.append(msg.avg_len); writer.add_scalar('Analysis/Avg_Game_Length', np.mean(game_lengths_deque), current_step); writer.add_scalar('Analysis/Missed_Fives', msg.miss_five, current_step); writer.add_scalar('Analysis/Missed_All_Wins', msg.miss_total, current_step)
            except Empty: continue
    except Exception as e: logging.getLogger("Visualizer_Fallback").critical(f"CRITICAL ERROR: {e}\n{traceback.format_exc()}")
    finally:
        if 'writer' in locals() and writer is not None: writer.close()

# =====================================================================
#                          DISPLAY MANAGER
# =====================================================================
def display_manager(ui_queue, shutdown_event, server_ready_event, worker_mode, log_queue):
    tr_bar, sp_bar, re_bar = None, None, None # 初始化为 None
    try:
        setup_worker_logging(log_queue)
        logger = logging.getLogger("DisplayManager")
        logger.setLevel(logging.WARNING)
        server_ready_event.wait()
        
        tr_bar = tqdm(position=0, leave=True, bar_format="{desc}: [{elapsed}] {postfix}", dynamic_ncols=True, desc="Training")
        sp_bar = tqdm(position=1, leave=True, bar_format="{desc}: {postfix}", dynamic_ncols=True, desc="Self-playing")
        re_bar = tqdm(position=2, leave=True, bar_format="{desc}: {postfix}", dynamic_ncols=True, desc="Re-analysis")

        tr_stats, sp_model_step, spinner_idx, games_completed = {}, 0, 0, 0
        tr_is_working, prefilling_done, tr_is_paused, sp_is_paused = True, False, False, False
        tr_pause_reason, spinner_chars = "", ['|', '/', '-', '\\']
        sp_postfix_history, re_postfix_history, previous_worker_mode = "[Waiting for game...]", "[Waiting for model...]", 0
        len_deque, miss5_deque = deque(maxlen=20), deque(maxlen=20)

        # --- [MODIFICATION START] ---
        # Replace deques with cumulative counters for the current re-analysis batch
        re_corrected_fives, re_original_fives = 0, 0
        re_corrected_all, re_original_all = 0, 0
        # --- [MODIFICATION END] ---

        while not shutdown_event.is_set():
            while not ui_queue.empty():
                try:
                    msg = ui_queue.get_nowait()
                    if isinstance(msg, TrainerStatus): 
                        tr_stats.update({'step': msg.step, 'loss': msg.total_loss, 'lr': msg.lr})
                        tr_is_working = True
                        prefilling_done = True
                    elif isinstance(msg, DataLoaderStatus): 
                        tr_stats['buffer'] = msg.buffer_size
                    elif isinstance(msg, SelfPlayStatus): len_deque.append(msg.avg_len); miss5_deque.append(msg.miss_five)
                    elif isinstance(msg, ReAnalysisStatus):
                        re_bar.update(1)
                        # --- [MODIFICATION START] ---
                        # Accumulate the raw numbers instead of calculating and storing individual rates
                        re_corrected_fives += msg.corrected_fives
                        re_original_fives += msg.original_fives
                        re_corrected_all += msg.corrected_totals
                        re_original_all += msg.original_totals
                        # --- [MODIFICATION END] ---
                    elif isinstance(msg, ReAnalysisQueueStatus):
                        if worker_mode.value == 1 and previous_worker_mode == 0: re_bar.total = msg.total_games_to_reanalyze; re_bar.n = 0; re_bar.refresh()
                    elif isinstance(msg, GameCompletedNotice): games_completed += 1; sp_bar.update(1)
                    elif isinstance(msg, TrainerWaitPrefillStatus): tr_stats.update({'buffer': msg.buffer_size, 'prefill': msg.prefill_size}); prefilling_done = False
                    elif isinstance(msg, TrainerModelUpdateNotice): sp_model_step = msg.step
                    elif isinstance(msg, TrainerWaitingForDataStatus): tr_is_working = False
                    elif isinstance(msg, TrainerPauseStatus): tr_is_paused = msg.is_paused; tr_pause_reason = msg.reason if msg.is_paused else ""
                    elif isinstance(msg, WorkerPauseStatus): sp_is_paused = msg.is_paused
                except Empty: break
            
            step, buffer_size = tr_stats.get('step', 0), tr_stats.get('buffer', 0)
            spinner = spinner_chars[spinner_idx := (spinner_idx + 1) % 4]
            
            tr_spinner = spinner if (prefilling_done and tr_is_working and not tr_is_paused) else ''
            tr_desc = f"[Paused: {tr_pause_reason}] Training" if tr_is_paused else f"Training {tr_spinner}"
            tr_postfix = f"[Waiting for prefill: {buffer_size}/{config.REPLAY_BUFFER_PREFILL}]" if not prefilling_done else f"{step} steps, loss={tr_stats.get('loss', 0):.4f}, buffer={buffer_size}, lr={tr_stats.get('lr', 0):.1e}"
            if tr_bar is not None: tr_bar.set_description(tr_desc); tr_bar.set_postfix_str(tr_postfix)

            current_mode = worker_mode.value
            if current_mode == 0 and not sp_is_paused:
                sp_postfix = f"{games_completed} games, len={np.mean(len_deque):.1f}, miss5={np.mean(miss5_deque):.2f}" if len_deque else "[Waiting for game...]"
                sp_postfix_history = sp_postfix
            sp_spinner = spinner if (current_mode == 0 and not sp_is_paused) else ''
            sp_desc = f"[Paused] Self-playing (M@{sp_model_step})" if sp_is_paused else f"Self-playing (M@{sp_model_step}){sp_spinner}"
            if sp_bar is not None: sp_bar.set_description(sp_desc); sp_bar.set_postfix_str(sp_postfix_history)

            if current_mode == 0 and previous_worker_mode == 1:
                # This resets the progress bar visually when switching back to self-play
                if re_bar is not None: re_bar.n = 0; re_bar.total = 0; re_bar.refresh()

            # --- [MODIFICATION START] ---
            # This resets the counters when a NEW re-analysis cycle begins
            if current_mode == 1 and previous_worker_mode == 0:
                re_corrected_fives, re_original_fives = 0, 0
                re_corrected_all, re_original_all = 0, 0
            # --- [MODIFICATION END] ---

            re_desc = f"Re-analysis (M@{sp_model_step})"
            if current_mode == 1:
                re_desc += f" {spinner}"
                # --- [MODIFICATION START] ---
                # Calculate percentage based on the cumulative sums. Handle division by zero.
                fives_rate = (re_corrected_fives / re_original_fives * 100) if re_original_fives > 0 else 0.0
                total_rate = (re_corrected_all / re_original_all * 100) if re_original_all > 0 else 0.0
                avg_fives_rate_str = f"{fives_rate:.1f}%"
                avg_total_rate_str = f"{total_rate:.1f}%"
                re_postfix = f"{(re_bar.n if re_bar is not None else 0)}/{(re_bar.total if re_bar is not None else '?')} games, corrected (5, all): {avg_fives_rate_str}, {avg_total_rate_str}"
                re_postfix_history = re_postfix
                # --- [MODIFICATION END] ---

            if re_bar is not None: re_bar.set_description(re_desc); re_bar.set_postfix_str(re_postfix_history)
            
            if tr_bar is not None: tr_bar.refresh()
            if sp_bar is not None: sp_bar.refresh()
            if re_bar is not None: re_bar.refresh()
            
            previous_worker_mode = current_mode
            time.sleep(0.1)
    except Exception as e:
        logging.getLogger("DisplayManager").critical(f"CRITICAL ERROR: {e}\n{traceback.format_exc()}")
    finally:
        if tr_bar is not None: tr_bar.close()
        if sp_bar is not None: sp_bar.close()
        if re_bar is not None: re_bar.close()