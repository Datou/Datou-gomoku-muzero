# data_structures.py

from collections import namedtuple
from typing import List
import numpy as np

# This structure holds the complete record of a single game.
# It is used for storage, re-analysis, and logging.
GameRecord = namedtuple('GameRecord', [
    'observations',    # List of board states (np.ndarray)
    'actions',         # List of action indices (int)
    'rewards',         # List of rewards (float)
    'policies',        # List of MCTS policies (np.ndarray)
    'values',          # List of value targets (float)
    'board_states'     # List of raw board states for re-analysis (np.ndarray)
])

# This structure holds a single, ready-to-use sample for the neural network.
# It contains the unrolled history needed for training.
TrainingSlice = namedtuple('TrainingSlice', [
    'observation',     # Stack of observations [U+1, C, H, W] (np.ndarray)
    'action_history',  # History of actions [U] (np.ndarray)
    'reward_history',  # History of rewards [U] (np.ndarray)
    'policy_history',  # History of policy targets [U+1, A] (np.ndarray)
    'value_history'    # History of value targets [U+1] (np.ndarray)
])