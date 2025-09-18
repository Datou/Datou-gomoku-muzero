# loss.py (FINAL, CORRECTED AND MASKED VERSION)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import config
from network import scalar_to_support, support_to_scalar

class BarlowLoss(nn.Module):
    # This class is correct, no changes needed.
    def __init__(self, lmbda, projection_dim):
        super().__init__()
        self.lmbda = lmbda
        self.bn = nn.BatchNorm1d(projection_dim, affine=False, track_running_stats=False)
    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m, "Input must be a square matrix"
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    def forward(self, z1, z2):
        z1, z2 = z1.float(), z2.float()
        z1_norm, z2_norm = self.bn(z1), self.bn(z2)
        c = torch.mm(z1_norm.T, z2_norm)
        c.div_(z1.size(0))
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
        return on_diag + self.lmbda * off_diag

def calculate_loss(model, target_model, batch, is_weights):
    model.train(); target_model.eval()
    obs_b, act_b, rew_b, pi_b, mcts_val_b = batch
    act_b, rew_b, mcts_val_b = act_b.long(), rew_b.float(), mcts_val_b.float()
    LOSS_WEIGHTS = config.LOSS_WEIGHTS
    
    # --- Data Augmentation (This part is fine) ---
    k = np.random.randint(4)
    flip = np.random.choice([True, False])
    obs_b_aug = torch.rot90(obs_b, k, dims=[3, 4])
    if flip: obs_b_aug = torch.flip(obs_b_aug, dims=[4])
    b, u_plus_1, c, h, w = obs_b_aug.shape
    pi_b_aug = pi_b.view(b, u_plus_1, h, w)
    pi_b_aug = torch.rot90(pi_b_aug, k, dims=[2, 3])
    if flip: pi_b_aug = torch.flip(pi_b_aug, dims=[3])
    pi_b_aug = pi_b_aug.flatten(start_dim=2)
    rows, cols = act_b // w, act_b % w
    if k == 1: rows, cols = cols, w - 1 - rows
    elif k == 2: rows, cols = h - 1 - rows, w - 1 - cols
    elif k == 3: rows, cols = h - 1 - cols, rows
    if flip: cols = w - 1 - cols
    act_b_aug = rows * w + cols
    
    # --- Target Value Calculation (This part is fine) ---
    with torch.no_grad():
        _, final_values_tensor, _ = target_model.initial_inference(obs_b_aug[:, -1])
        value_targets = torch.zeros_like(mcts_val_b)
        for i in range(config.NUM_UNROLL_STEPS + 1):
            n_step_return = 0.0
            for j in range(config.N_STEPS):
                if i + j < config.NUM_UNROLL_STEPS: n_step_return += (config.DISCOUNT ** j) * rew_b[:, i + j]
                else: break
            bootstrap_index = i + config.N_STEPS
            bootstrap_value = mcts_val_b[:, bootstrap_index] if bootstrap_index <= config.NUM_UNROLL_STEPS else final_values_tensor.squeeze(-1)
            n_step_return += (config.DISCOUNT ** config.N_STEPS) * bootstrap_value
            value_targets[:, i] = n_step_return
            
    consistency_loss_fn = BarlowLoss(config.BARLOW_LAMBDA, model.projection_net.fc2.out_features).to(config.DEVICE)
    all_dynamic_proj_for_loss = []

    with torch.amp.autocast('cuda', enabled=(config.DEVICE.type == 'cuda')):
        hidden_state = model.representation(obs_b_aug[:, 0])
        policy_logits, value_logits = model.prediction(hidden_state)
        
        policy_loss = F.cross_entropy(policy_logits.float(), pi_b_aug[:, 0], reduction='none')
        value_loss = F.cross_entropy(value_logits.float(), scalar_to_support(value_targets[:, 0], config.VALUE_SUPPORT_MIN, config.VALUE_SUPPORT_MAX, config.VALUE_SUPPORT_BINS), reduction='none')
        
        pred_values_scalar = support_to_scalar(F.softmax(value_logits.float(), dim=1), config.VALUE_SUPPORT_MIN, config.VALUE_SUPPORT_MAX, config.VALUE_SUPPORT_BINS)
        td_errors = torch.abs(pred_values_scalar.detach().squeeze(-1) - value_targets[:, 0])
        reward_loss = torch.zeros(obs_b.size(0), device=config.DEVICE)
        
        valid_unroll_steps = 0
        
        for k_step in range(config.NUM_UNROLL_STEPS):
            mask = (act_b[:, k_step] != -1)
            if not mask.any(): continue
            valid_unroll_steps += 1
            
            valid_hidden_state = hidden_state[mask]
            valid_action_aug = act_b_aug[:, k_step][mask]

            next_hidden_state_k, pred_reward_logits_k = model.dynamics(valid_hidden_state, valid_action_aug)
            policy_logits_k, value_logits_k = model.prediction(next_hidden_state_k)
            
            policy_loss[mask] += F.cross_entropy(policy_logits_k.float(), pi_b_aug[:, k_step + 1][mask], reduction='none')
            value_loss[mask] += F.cross_entropy(value_logits_k.float(), scalar_to_support(value_targets[:, k_step + 1][mask], config.VALUE_SUPPORT_MIN, config.VALUE_SUPPORT_MAX, config.VALUE_SUPPORT_BINS), reduction='none')
            reward_loss[mask] += F.cross_entropy(pred_reward_logits_k.float(), scalar_to_support(rew_b[:, k_step][mask], config.REWARD_SUPPORT_MIN, config.REWARD_SUPPORT_MAX, config.REWARD_SUPPORT_BINS), reduction='none')
            
            
            dynamic_proj = model.project(next_hidden_state_k, with_grad=True)
            
            # [修改] 使用主模型 model 进行真实投影，并用 no_grad 阻止梯度
            with torch.no_grad():
                true_next_hidden_state_k = model.representation(obs_b_aug[:, k_step + 1][mask])
                true_proj = model.project(true_next_hidden_state_k, with_grad=False)
            
            all_dynamic_proj_for_loss.append((dynamic_proj, true_proj))
            
            next_hidden_state = hidden_state.clone()
            next_hidden_state[mask] = next_hidden_state_k
            next_hidden_state.register_hook(lambda grad: grad * 0.5)
            hidden_state = next_hidden_state

    # =================== [LINUS'S FIX START] ===================
    # STOP doing WRONG math. Calculate per-sample and batch-level losses SEPARATELY.
    
    # 1. Finalize and average the PER-SAMPLE losses over unroll steps.
    avg_policy_loss_per_sample = policy_loss / (valid_unroll_steps + 1)
    avg_value_loss_per_sample = value_loss / (valid_unroll_steps + 1)
    
    # Handle reward loss, avoiding division by zero if no valid unroll steps occurred.
    if valid_unroll_steps > 0:
        avg_reward_loss_per_sample = reward_loss / valid_unroll_steps
    else:
        avg_reward_loss_per_sample = torch.zeros_like(reward_loss)

    # 2. Calculate the BATCH-LEVEL consistency loss. It's a SUM of scalars.
    total_consistency_loss_scalar = sum(consistency_loss_fn(dyn_proj, true_proj) for dyn_proj, true_proj in all_dynamic_proj_for_loss)

    # 3. Average the batch-level consistency loss over the steps to make it comparable.
    if valid_unroll_steps > 0:
        avg_consistency_loss = total_consistency_loss_scalar / valid_unroll_steps
    else:
        avg_consistency_loss = torch.tensor(0.0, device=config.DEVICE)

    # 4. Apply PER weights and compute the final mean for each loss component.
    final_policy_loss = (avg_policy_loss_per_sample * is_weights).mean()
    final_value_loss = (avg_value_loss_per_sample * is_weights).mean()
    final_reward_loss = (avg_reward_loss_per_sample * is_weights).mean()
    
    # 5. Combine all weighted losses correctly.
    weighted_loss = (
        LOSS_WEIGHTS['policy'] * final_policy_loss +
        LOSS_WEIGHTS['value'] * final_value_loss +
        LOSS_WEIGHTS['reward'] * final_reward_loss +
        LOSS_WEIGHTS['consistency'] * avg_consistency_loss
    )
    
    # 6. Prepare the log values. They should reflect the final, scaled component losses.
    log_values = (
        weighted_loss.item(),
        final_policy_loss.item(),
        final_value_loss.item(),
        final_reward_loss.item(),
        avg_consistency_loss.item(),
        td_errors.cpu().numpy()
    )
    
    return weighted_loss, log_values
    # =================== [LINUS'S FIX END] ===================