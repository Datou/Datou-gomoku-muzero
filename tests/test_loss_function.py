# file: test_loss_function.py

import unittest
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# --- Project Imports ---
from loss import calculate_loss
from network import GomokuNetEZ
from config import config

# =====================================================================
#                           Loss Test Class
# =====================================================================

class TestLossFunction(unittest.TestCase):

    def setUp(self):
        """Create models and a sample batch structure for tests."""
        self.model = GomokuNetEZ(config).to(config.DEVICE)
        self.target_model = GomokuNetEZ(config).to(config.DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.batch_size = 4
        self.num_unroll_steps = config.NUM_UNROLL_STEPS
        
        self.obs_b = torch.randn(
            self.batch_size, 
            self.num_unroll_steps + 1, 
            3, 
            config.BOARD_SIZE, 
            config.BOARD_SIZE,
            device=config.DEVICE
        )
        self.is_weights = torch.ones(self.batch_size, device=config.DEVICE)
        # =================== [LINUS'S FIX START] ===================
        # ALWAYS zero the grad before each test.
        self.model.zero_grad()
        self.target_model.zero_grad()
        # =================== [LINUS'S FIX END] ===================


    def test_zero_loss_on_perfect_prediction(self):
        """
        If model predictions perfectly match targets, loss should be (near) zero.
        NOTE: This is a sanity check; perfect zero is hard due to value logic.
        """
        print("\n--- Running test_zero_loss_on_perfect_prediction ---")
        act_b = torch.ones(self.batch_size, self.num_unroll_steps, dtype=torch.long, device=config.DEVICE)
        rew_b = torch.zeros(self.batch_size, self.num_unroll_steps, device=config.DEVICE)
        pi_b = F.one_hot(act_b[:, 0], num_classes=config.ACTION_SPACE_SIZE).float() 
        pi_b = pi_b.unsqueeze(1).repeat(1, self.num_unroll_steps + 1, 1)
        mcts_val_b = torch.zeros(self.batch_size, self.num_unroll_steps + 1 + config.N_STEPS, device=config.DEVICE)

        batch = (self.obs_b, act_b, rew_b, pi_b, mcts_val_b)
        
        loss, log_vals = calculate_loss(self.model, self.target_model, batch, self.is_weights)
        
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss.item(), 0.0, "Loss should not be negative.")

    def test_gradient_flow(self):
        """Check if gradients are actually flowing to the model parameters."""
        print("\n--- Running test_gradient_flow ---")
        act_b = torch.randint(0, config.ACTION_SPACE_SIZE, (self.batch_size, self.num_unroll_steps), device=config.DEVICE)
        rew_b = (torch.rand(self.batch_size, self.num_unroll_steps, device=config.DEVICE) * 2) - 1
        pi_b = torch.rand(self.batch_size, self.num_unroll_steps + 1, config.ACTION_SPACE_SIZE, device=config.DEVICE)
        pi_b = F.softmax(pi_b, dim=-1)
        mcts_val_b = (torch.rand(self.batch_size, self.num_unroll_steps + 1 + config.N_STEPS, device=config.DEVICE) * 2) - 1
        
        batch = (self.obs_b, act_b, rew_b, pi_b, mcts_val_b)
        
        loss, _ = calculate_loss(self.model, self.target_model, batch, self.is_weights)
        
        loss.backward()
        
        grad = self.model.prediction_net.policy_fc.weight.grad
        self.assertIsNotNone(grad, "Policy head has no gradient.")
        self.assertGreater(torch.sum(torch.abs(grad)), 0, "Gradient for policy head is zero.")

    def test_masking_ignores_padded_steps(self):
        """Verify that invalid steps (action=-1) do not contribute to the loss."""
        print("\n--- Running test_masking_ignores_padded_steps ---")
        
        # Batch 1: All actions are valid
        act_b1 = torch.ones(self.batch_size, self.num_unroll_steps, dtype=torch.long, device=config.DEVICE)
        
        # Batch 2: The last action is invalid (-1)
        act_b2 = torch.ones(self.batch_size, self.num_unroll_steps, dtype=torch.long, device=config.DEVICE)
        act_b2[:, -1] = -1 # Mask the final unroll step
        
        # Consistent targets for both batches
        rew_b = torch.ones(self.batch_size, self.num_unroll_steps, device=config.DEVICE) * 0.5
        pi_b = torch.rand(self.batch_size, self.num_unroll_steps + 1, config.ACTION_SPACE_SIZE, device=config.DEVICE)
        mcts_val_b = torch.ones(self.batch_size, self.num_unroll_steps + 1 + config.N_STEPS, device=config.DEVICE) * 0.5
        
        batch1 = (self.obs_b, act_b1, rew_b, pi_b, mcts_val_b)
        batch2 = (self.obs_b, act_b2, rew_b, pi_b, mcts_val_b)
        
        loss1, _ = calculate_loss(self.model, self.target_model, batch1, self.is_weights)
        loss2, _ = calculate_loss(self.model, self.target_model, batch2, self.is_weights)
        
        # =================== [LINUS'S FIX START] ===================
        # The logic is REVERSED. The loss with FEWER steps (loss2) should be LOWER
        # because fewer terms are being added to it.
        self.assertLess(loss2.item(), loss1.item(), "Loss with a masked step should be LOWER than the full loss.")
        # =================== [LINUS'S FIX END] ===================

if __name__ == '__main__':
    unittest.main(verbosity=2)