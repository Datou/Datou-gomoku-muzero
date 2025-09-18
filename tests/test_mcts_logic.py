# file: test_mcts_logic.py (FINAL, CORRECTED VERSION)

import unittest
import numpy as np
import torch
import sys
import os
import random  # <-- [LINUS'S FIX] THE MISSING IMPORT IS HERE.
from queue import Empty

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# --- Project Imports ---
from mcts import AlphaZeroMCTS, MuZeroMCTS, Node
from game import GomokuGame
from config import config
from utils import MinMaxStats

# =====================================================================
#                      Mock Objects for Predictable Testing
# =====================================================================
# ... (MockInferenceQueue and MockModel classes remain unchanged, they are correct) ...
class MockInferenceQueue:
    def __init__(self, mock_model):
        self.mock_model = mock_model
        self.request_log = []
        self.put_log = []

    def put(self, item):
        self.request_log.append(item)
        self.put_log.append(item)

    def get(self, timeout=None):
        if not self.request_log:
            raise Empty("Mock queue has no requests to process.")
        worker_id, req_type, data = self.request_log.pop(0)
        if req_type == 'initial':
            obs = torch.from_numpy(data).unsqueeze(0).to(config.DEVICE)
            p, v, h = self.mock_model.initial_inference(obs)
            return p.squeeze(0).cpu().numpy(), v.item(), h.cpu().numpy()
        elif req_type == 'recurrent_batch':
            hidden_states, actions = data
            states_tensor = torch.from_numpy(hidden_states).to(config.DEVICE)
            actions_tensor = torch.from_numpy(actions).long().to(config.DEVICE)
            p, v, h, r = self.mock_model.recurrent_inference(states_tensor, actions_tensor)
            return p.cpu().numpy(), v.cpu().numpy(), h.cpu().numpy(), r.cpu().numpy()

    def get_nowait(self):
        if not self.request_log:
            raise Empty
        return self.request_log.pop(0)

    def clear_log(self):
        self.request_log = []
        self.put_log = []

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def initial_inference(self, obs):
        batch_size = obs.shape[0]
        policy_logits = torch.zeros(batch_size, config.ACTION_SPACE_SIZE, device=config.DEVICE)
        value = torch.full((batch_size, 1), 0.5, device=config.DEVICE)
        hidden_state = torch.ones(batch_size, config.NUM_FILTERS, config.BOARD_SIZE, config.BOARD_SIZE, device=config.DEVICE)
        return policy_logits, value, hidden_state

    @torch.no_grad()
    def recurrent_inference(self, hidden_state, action):
        batch_size = hidden_state.shape[0]
        policy_logits = torch.zeros(batch_size, config.ACTION_SPACE_SIZE, device=config.DEVICE)
        value = torch.full((batch_size, 1), 0.5, device=config.DEVICE)
        next_hidden_state = torch.ones_like(hidden_state) * 2
        reward = torch.full((batch_size, 1), 0.0, device=config.DEVICE)
        return policy_logits, value, next_hidden_state, reward
# =====================================================================
#                           MCTS Test Class
# =====================================================================
class TestMCTSLogic(unittest.TestCase):

    def setUp(self):
        self.mock_model = MockModel().to(config.DEVICE)
        self.mock_queue = MockInferenceQueue(self.mock_model)
        self.az_mcts = AlphaZeroMCTS(worker_id=0, request_queue=self.mock_queue, result_queue=self.mock_queue)
        self.mz_mcts = MuZeroMCTS(worker_id=0, request_queue=self.mock_queue, result_queue=self.mock_queue)
        self.original_sims = config.NUM_SIMULATIONS

    def tearDown(self):
        config.NUM_SIMULATIONS = self.original_sims
        self.mock_queue.clear_log()

    def test_node_logic(self):
        node = Node()
        self.assertEqual(node.get_value(), 0.0)
        node.visit_count = 5
        node.value_sum = 2.5
        self.assertEqual(node.get_value(), 0.5)

    def test_backpropagation_logic(self):
        root = Node()
        root.visit_count = 1 
        child1 = root.get_child(action=10)
        leaf_nodes, values = [child1], [0.8]
        mock_stats = MinMaxStats(config.VALUE_MINMAX_DELTA)
        self.az_mcts._backpropagate(leaf_nodes, values, mock_stats)
        self.assertEqual(child1.visit_count, 1)
        self.assertEqual(child1.value_sum, 0.8)
        self.assertEqual(root.visit_count, 2)
        self.assertAlmostEqual(root.value_sum, config.DISCOUNT * 0.8)

    def test_alphazero_search_behavior(self):
        print("\n--- Running test_alphazero_search_behavior ---")
        game = GomokuGame()
        config.NUM_SIMULATIONS = 15
        _, _, _ = self.az_mcts.search(game)
        requests = self.mock_queue.put_log
        request_types = [req[1] for req in requests]
        self.assertEqual(request_types.count('initial'), config.NUM_SIMULATIONS)
        self.assertEqual(request_types.count('recurrent_batch'), 0)
        print(f"OK: Found {request_types.count('initial')} 'initial' calls and 0 'recurrent' calls.")

    def test_muzero_search_behavior(self):
        print("\n--- Running test_muzero_search_behavior ---")
        game = GomokuGame()
        config.NUM_SIMULATIONS = 15
        _, _, _ = self.mz_mcts.search(game)
        requests = self.mock_queue.put_log
        request_types = [req[1] for req in requests]
        self.assertEqual(request_types.count('initial'), 1)
        self.assertGreater(request_types.count('recurrent_batch'), 0)
        print(f"OK: Found 1 'initial' call and {request_types.count('recurrent_batch')} 'recurrent' calls.")
    
    def test_search_output_contract(self):
        print("\n--- Running test_search_output_contract ---")
        game = GomokuGame()
        for _ in range(10): 
            if game.get_game_ended() is not None: break
            valid_moves = game.get_valid_moves()
            if not valid_moves: break
            action_coord = random.choice(valid_moves)
            action = action_coord[0] * config.BOARD_SIZE + action_coord[1]
            game.do_move(action)

        if game.get_game_ended() is not None:
            self.skipTest("Game ended prematurely during setup.")

        engines = {"AlphaZeroMCTS": self.az_mcts, "MuZeroMCTS": self.mz_mcts}
        for name, engine in engines.items():
            with self.subTest(engine=name):
                policy, value, action = engine.search(game)
                self.assertIsInstance(policy, np.ndarray)
                self.assertAlmostEqual(np.sum(policy), 1.0, places=5)
                self.assertIsInstance(action, int)
                valid_actions = {m[0] * config.BOARD_SIZE + m[1] for m in game.get_valid_moves()}
                if valid_actions:
                    self.assertIn(action, valid_actions)
                self.assertIsInstance(value, (float, np.floating))
                self.assertGreaterEqual(value, -1.0, f"Value from {name} must be >= -1, but was {value}")
                self.assertLessEqual(value, 1.0, f"Value from {name} must be <= 1, but was {value}")
                print(f"OK: {name} passed contract validation.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
