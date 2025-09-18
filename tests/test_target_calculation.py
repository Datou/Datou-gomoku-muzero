import unittest
import numpy as np
import sys
import os

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# --- Project Imports ---
from config import config

# Helper function that replicates the target calculation from loss.py
def calculate_value_targets_for_test(rewards, mcts_values, final_bootstrap_value):
    """Replicates the n-step return logic from the calculate_loss function."""
    discount = config.DISCOUNT
    n_steps = config.N_STEPS
    num_unroll_steps = len(rewards)
    
    value_targets = np.zeros(num_unroll_steps + 1)
    
    for i in range(num_unroll_steps + 1):
        n_step_return = 0.0
        for j in range(n_steps):
            if i + j < num_unroll_steps:
                n_step_return += (discount ** j) * rewards[i + j]
            else:
                break
        
        bootstrap_index = i + n_steps
        if bootstrap_index < len(mcts_values):
            bootstrap_value = mcts_values[bootstrap_index]
        else:
            bootstrap_value = final_bootstrap_value
        
        n_step_return += (discount ** n_steps) * bootstrap_value
        value_targets[i] = n_step_return
        
    return value_targets

class TestTargetCalculation(unittest.TestCase):

    def test_n_step_return_calculation(self):
        """
        Verify the n-step return calculation with a predictable sequence.
        """
        print("\n--- Running test_n_step_return_calculation ---")
        # --- Setup a predictable scenario ---
        config.DISCOUNT = 0.9
        config.N_STEPS = 3
        
        rewards =       [0.0, 0.0, 1.0, 0.0, 0.0]
        mcts_values =   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        final_bootstrap_value = 0.5

        # --- Calculate targets ---
        targets = calculate_value_targets_for_test(rewards, mcts_values, final_bootstrap_value)
        
        # --- Manually calculate the target for the first step (i=0) ---
        # Target(0) = r_0*D^0 + r_1*D^1 + r_2*D^2 + D^3 * bootstrap_value_at_t_3
        expected_target_0 = (rewards[0] * config.DISCOUNT**0) + \
                            (rewards[1] * config.DISCOUNT**1) + \
                            (rewards[2] * config.DISCOUNT**2) + \
                            (config.DISCOUNT**3 * mcts_values[3])
        
        self.assertAlmostEqual(targets[0], expected_target_0, places=5)
        
        # --- Manually calculate the target for a later step (i=3) ---
        # [FIXED LOGIC]
        # Target(3) = r_3*D^0 + r_4*D^1 + D^3 * bootstrap_value_at_t_6 (falls back to final_bootstrap)
        #           = 0*1.0   + 0*0.9   + 0.9^3 * 0.5
        #           = 0 + 0.729 * 0.5 = 0.3645
        expected_target_3 = (rewards[3] * config.DISCOUNT**0) + \
                            (rewards[4] * config.DISCOUNT**1) + \
                            (config.DISCOUNT**3 * final_bootstrap_value) # Corrected logic
                            
        self.assertAlmostEqual(targets[3], expected_target_3, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)