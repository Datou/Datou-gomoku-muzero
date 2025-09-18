# db_manager.py (FINAL, REFACTORED by Linus, with hotfix)
# This module's ONLY job is persistence. All sampling logic has been ripped out.

import sqlite3
import pickle
import threading
import numpy as np
import os
import logging
from collections import deque
from config import config

# [FIX] Import from the single source of truth.
from data_structures import GameRecord, TrainingSlice

thread_local = threading.local()
logger = logging.getLogger("DatabaseManager")

def get_db_connection(db_path):
    """Establishes a thread-local database connection."""
    if not hasattr(thread_local, 'connection'):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        thread_local.connection = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        thread_local.connection.execute('PRAGMA journal_mode=WAL;')
        thread_local.connection.execute('PRAGMA synchronous=NORMAL;')
    return thread_local.connection

class DatabaseManager:
    """
    Manages all database interactions. Its sole responsibility is the persistent
    storage and retrieval of game data and trainer state.
    """
    def __init__(self, db_path="outputs/training_state.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the database schema."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            # Stores game-level metadata and the full game record for re-analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS games (
                    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_record BLOB NOT NULL,
                    analysis_version INTEGER NOT NULL,
                    move_count INTEGER NOT NULL,
                    status TEXT DEFAULT 'PENDING' NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS status_version_idx ON games (status, analysis_version);')

            # Stores individual training slices as blobs. Simplified for persistence only.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS replay_buffer (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER NOT NULL,
                    move_index INTEGER NOT NULL,
                    slice_data BLOB NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES games (game_id) ON DELETE CASCADE
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS game_id_idx ON replay_buffer (game_id);')

            # Stores the trainer's state (model, optimizer, etc.)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trainer_state (
                    key TEXT PRIMARY KEY,
                    state_blob BLOB
                )
            ''')
            conn.commit()

    def add_game_and_slices(self, game_record, prepared_slices, model_version):
        """Adds a completed game and all its training slices to the database."""
        with get_db_connection(self.db_path) as conn:
            try:
                with conn:
                    cursor = conn.cursor()
                    serialized_game = pickle.dumps(game_record, protocol=pickle.HIGHEST_PROTOCOL)
                    cursor.execute(
                        'INSERT INTO games (game_record, analysis_version, move_count) VALUES (?, ?, ?)',
                        (serialized_game, model_version, len(game_record.actions))
                    )
                    game_id = cursor.lastrowid
                    
                    slices_to_insert = [
                        (game_id, i, pickle.dumps(slice_data, protocol=pickle.HIGHEST_PROTOCOL)) 
                        for i, slice_data in enumerate(prepared_slices)
                    ]
                    
                    cursor.executemany(
                        'INSERT INTO replay_buffer (game_id, move_index, slice_data) VALUES (?, ?, ?)',
                        slices_to_insert
                    )
                # =================== [LINUS'S FIX START] ===================
                return game_id # Return the ID of the game we just saved.
                # =================== [LINUS'S FIX END] ===================
            except Exception as e:
                logger.error(f"Failed to add game and slices, transaction rolled back. Error: {e}")
                # =================== [LINUS'S FIX START] ===================
                return None # Return None on failure.
                # =================== [LINUS'S FIX END] ===================

    def get_game_record_by_id(self, game_id: int):
        """Loads a single game record from the database by its ID."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT game_record FROM games WHERE game_id = ?', (game_id,))
            row = cursor.fetchone()
            return pickle.loads(row[0]) if row else None
        
    def load_latest_samples(self, num_samples: int) -> list:
        """Loads the N most recent training slices from the DB to warm up the in-memory buffer."""
        logger.info(f"DB: Loading latest {num_samples} samples for buffer warm-up...")
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT slice_data FROM replay_buffer ORDER BY id DESC LIMIT ?', (num_samples,))
            rows = cursor.fetchall()
            if not rows:
                logger.warning("DB: No samples found in database to load.")
                return []
            
            # Records are returned newest-first (DESC), so we reverse to get oldest-first.
            samples = [pickle.loads(row[0]) for row in reversed(rows)]
            logger.info(f"DB: Successfully loaded {len(samples)} samples from disk.")
            return samples

    def get_buffer_size(self) -> int:
        """Returns the total number of samples currently in the on-disk buffer."""
        with get_db_connection(self.db_path) as conn:
            return conn.execute('SELECT COUNT(*) FROM replay_buffer').fetchone()[0]

    def trim_buffer(self, capacity: int):
        """Deletes the oldest games from the database if the buffer size exceeds capacity."""
        with get_db_connection(self.db_path) as conn:
            current_size = self.get_buffer_size()
            if current_size > capacity:
                to_delete = current_size - capacity
                cursor = conn.cursor()
                cursor.execute('SELECT game_id FROM games ORDER BY timestamp ASC LIMIT ?', (100,)) # Get a batch of old games
                oldest_game_ids = cursor.fetchall()

                if oldest_game_ids:
                    placeholders = ','.join('?' for _ in oldest_game_ids)
                    cursor.execute(f'DELETE FROM games WHERE game_id IN ({placeholders})', [gid[0] for gid in oldest_game_ids])
                    conn.commit()
                    logger.info(f"DB: Trimmed buffer by deleting {len(oldest_game_ids)} old games.")

    def get_reanalysis_queue_size(self, current_trainer_step: int) -> int:
        """Counts how many games are currently eligible for re-analysis."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT COUNT(*) FROM games 
                   WHERE status = 'PENDING' AND ? - analysis_version > ?''',
                (current_trainer_step, config.REANALYSIS_AGE_THRESHOLD)
            )
            return cursor.fetchone()[0]
        
    def sample_and_lock_game_for_reanalysis(self, current_trainer_step: int):
        """Atomically selects the oldest eligible game, locks it, and returns it."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            with conn: # Start transaction
                cursor.execute(
                    '''SELECT game_id, game_record FROM games 
                       WHERE status = 'PENDING' AND ? - analysis_version > ? 
                       ORDER BY analysis_version ASC LIMIT 1''',
                    (current_trainer_step, config.REANALYSIS_AGE_THRESHOLD)
                )
                candidate = cursor.fetchone()
                if not candidate:
                    return None, None
                
                game_id, serialized_game_record = candidate
                cursor.execute("UPDATE games SET status = 'RUNNING' WHERE game_id = ?", (game_id,))
                return game_id, pickle.loads(serialized_game_record)

    def finish_reanalysis_for_game(self, game_id: int, new_policies: list, new_value_targets: list, new_analysis_version: int):
        """Updates the stored training slices with new MCTS results after re-analysis."""
        from collections import deque
        with get_db_connection(self.db_path) as conn:
            try:
                with conn: # Start transaction
                    cursor = conn.cursor()
                    
                    # --- This sliding window logic is still good taste. It's efficient. ---
                    num_moves = len(new_policies)
                    k = config.NUM_UNROLL_STEPS + 1
                    policy_deque, value_deque = deque(maxlen=k), deque(maxlen=k)
                    all_pi_hists, all_val_hists = [], []
                    for i in range(num_moves + k -1):
                        if i < num_moves:
                            policy_deque.append(new_policies[i])
                            value_deque.append(new_value_targets[i])
                        else:
                            policy_deque.append(np.zeros_like(new_policies[0]))
                            value_deque.append(0.0)
                        if i >= k - 1:
                            all_pi_hists.append(np.array(policy_deque))
                            all_val_hists.append(np.array(value_deque, dtype=np.float32))

                    # --- This part is adapted for the new blob-based schema ---
                    cursor.execute("SELECT id, move_index, slice_data FROM replay_buffer WHERE game_id = ? ORDER BY move_index ASC", (game_id,))
                    slices_to_update = cursor.fetchall()
                    
                    update_payloads = []
                    for sample_id, move_index, serialized_slice in slices_to_update:
                        original_slice = pickle.loads(serialized_slice)
                        updated_slice = original_slice._replace(
                            policy_history=all_pi_hists[move_index],
                            value_history=all_val_hists[move_index]
                        )
                        update_payloads.append((pickle.dumps(updated_slice, protocol=pickle.HIGHEST_PROTOCOL), sample_id))

                    cursor.executemany("UPDATE replay_buffer SET slice_data = ? WHERE id = ?", update_payloads)
                    cursor.execute("UPDATE games SET analysis_version = ?, status = 'DONE' WHERE game_id = ?", (new_analysis_version, game_id))
            except Exception as e:
                # If anything fails, roll back the transaction and unlock the game
                with conn: conn.execute("UPDATE games SET status = 'PENDING' WHERE game_id = ?", (game_id,))
                logger.error(f"Failed to finish re-analysis for game {game_id}, transaction rolled back. Error: {e}")

    def unlock_game_on_error(self, game_id: int):
        """Unlocks a game if the worker processing it crashes."""
        if game_id is None: return
        with get_db_connection(self.db_path) as conn:
            conn.execute("UPDATE games SET status = 'PENDING' WHERE game_id = ?", (game_id,))

    def save_trainer_state(self, state_dict: dict):
        """Saves the complete trainer state as a single blob."""
        with get_db_connection(self.db_path) as conn:
            serialized_state = pickle.dumps(state_dict, protocol=pickle.HIGHEST_PROTOCOL)
            conn.execute('INSERT OR REPLACE INTO trainer_state (key, state_blob) VALUES (?, ?)', ('singleton_state', serialized_state))
            conn.commit()

    def load_trainer_state(self) -> dict or None:
        """Loads the trainer state blob from the database."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT state_blob FROM trainer_state WHERE key = ?', ('singleton_state',))
            row = cursor.fetchone()
            return pickle.loads(row[0]) if row else None

    def are_games_currently_locked_for_reanalysis(self):
        """
        Checks if any game's status is 'RUNNING'.
        This tells the orchestrator if any workers are still busy with a re-analysis task.
        """
        with get_db_connection(self.db_path) as conn:
            # This query efficiently checks if any row matches the condition.
            query = "SELECT EXISTS(SELECT 1 FROM games WHERE status = 'RUNNING')"
            is_running = conn.execute(query).fetchone()[0]
            # The query returns 1 if at least one game is running, 0 otherwise.
            return is_running == 1