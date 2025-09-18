# 文件: logger_config.py
import logging
import logging.handlers
from multiprocessing import Queue
import sys

def logger_process(log_queue: Queue, log_file: str):
    # [修复] 移除了 StreamHandler，日志将只写入文件
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(processName)-18s - %(levelname)-8s - %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='a')])
    while True:
        try:
            record = log_queue.get()
            if record is None: break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)

def setup_worker_logging(log_queue: Queue):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    root.addHandler(queue_handler)
    root.setLevel(logging.INFO)