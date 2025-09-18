import unittest
import numpy as np
import sys
import os

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# --- Project Imports ---
from data_structures import TrainingSlice
from config import config

# This is a helper function that replicates the slicing logic from your worker
def create_training_slices(observations, actions, rewards, policies, value_targets):
    """Replicates the core slicing logic from the universal_worker."""
    prepared_slices = []
    num_moves = len(actions)
    
    # Pad all data structures to prevent index-out-of-bounds errors
    final_obs = observations + [np.zeros_like(observations[0])] * (config.NUM_UNROLL_STEPS + 1)
    final_acts = actions + [-1] * config.NUM_UNROLL_STEPS
    final_rews = rewards + [0.0] * config.NUM_UNROLL_STEPS
    final_pols = policies + [np.zeros_like(policies[0])] * (config.NUM_UNROLL_STEPS + 1)
    final_vals = value_targets + [0.0] * (config.NUM_UNROLL_STEPS + 1)
    
    for i in range(num_moves):
        # Create the unrolled histories starting from index i
        obs_hist = np.stack(final_obs[i : i + config.NUM_UNROLL_STEPS + 1])
        act_hist = np.array(final_acts[i : i + config.NUM_UNROLL_STEPS], dtype=np.int32)
        rew_hist = np.array(final_rews[i : i + config.NUM_UNROLL_STEPS], dtype=np.float32)
        pi_hist = np.stack(final_pols[i : i + config.NUM_UNROLL_STEPS + 1])
        val_hist = np.array(final_vals[i : i + config.NUM_UNROLL_STEPS + 1], dtype=np.float32)
        prepared_slices.append(TrainingSlice(obs_hist, act_hist, rew_hist, pi_hist, val_hist))
    
    return prepared_slices

class TestDataSlicing(unittest.TestCase):

    def test_slice_alignment(self):
        """
        Verify that the created TrainingSlice has correctly aligned data.
        """
        print("\n--- Running test_slice_alignment ---")
        # --- Create a short, predictable game history ---
        # Let's use simple integers for observations to make them easy to track
        total_moves = 10
        observations = [np.full((3, 15, 15), i, dtype=np.float32) for i in range(total_moves + 1)] # Obs from t=0 to t=10
        actions = list(range(total_moves)) # Action from t=0 to t=9
        rewards = [float(i) for i in range(total_moves)] # Reward from t=0 to t=9
        policies = [np.full(config.ACTION_SPACE_SIZE, i, dtype=np.float32) for i in range(total_moves + 1)] # Policy from t=0 to t=10
        values = [float(i) for i in range(total_moves + 1)] # Value from t=0 to t=10
        
        # --- Generate slices from this history ---
        slices = create_training_slices(observations, actions, rewards, policies, values)
        self.assertEqual(len(slices), total_moves)

        # --- Check a slice from the middle of the game, e.g., the slice for t=3 ---
        t = 3
        test_slice = slices[t]
        
        # 1. The first observation in the slice should be the observation from step t=3.
        # np.testing.assert_array_equal is better for numpy arrays
        np.testing.assert_array_equal(test_slice.observation[0], observations[t])
        
        # 2. The first action in the action_history should be the action taken at step t=3.
        self.assertEqual(test_slice.action_history[0], actions[t])
        
        # 3. The first reward should be the reward received after the action at t=3.
        self.assertEqual(test_slice.reward_history[0], rewards[t])
        
        # 4. The policy history should start with the policy for the state at t=3.
        np.testing.assert_array_equal(test_slice.policy_history[0], policies[t])
        
        # 5. The value history should start with the value for the state at t=3.
        self.assertEqual(test_slice.value_history[0], values[t])

        # 6. Check the NEXT step's alignment
        # The second observation in the slice should be from step t=4
        np.testing.assert_array_equal(test_slice.observation[1], observations[t+1])
        # The second policy should be from step t=4
        np.testing.assert_array_equal(test_slice.policy_history[1], policies[t+1])

if __name__ == '__main__':
    unittest.main(verbosity=2)