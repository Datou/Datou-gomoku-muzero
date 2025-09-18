# network.py (FINAL, CLEANED VERSION)

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config # Import config directly

# Helper functions are fine, keep them.
def support_to_scalar(logits, support_min, support_max, support_bins):
    if logits.device.type == 'mps': logits = logits.to('cpu')
    support = torch.linspace(support_min, support_max, support_bins, device=logits.device)
    probabilities = F.softmax(logits, dim=1)
    return torch.sum(support * probabilities, dim=1, keepdim=True)

def scalar_to_support(scalar, support_min, support_max, support_bins):
    scalar = scalar.clamp(support_min, support_max)
    scaling = (support_bins - 1) / (support_max - support_min)
    float_idx = (scalar - support_min) * scaling
    low_idx, high_idx = torch.floor(float_idx).long(), torch.ceil(float_idx).long()
    high_weight = float_idx - low_idx.float()
    low_weight = 1 - high_weight
    target_support = torch.zeros(scalar.size(0), support_bins, device=scalar.device)
    target_support.scatter_add_(1, low_idx.unsqueeze(1), low_weight.unsqueeze(1))
    target_support.scatter_add_(1, high_idx.unsqueeze(1), high_weight.unsqueeze(1))
    return target_support

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class EvarResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class RepresentationNetwork(nn.Module):
    def __init__(self, observation_shape, num_blocks, num_channels):
        super().__init__()
        self.conv = conv3x3(observation_shape[0], num_channels)
        self.bn = nn.BatchNorm2d(num_channels, eps=1e-4)
        self.resblocks = nn.Sequential(*[EvarResBlock(num_channels, num_channels) for _ in range(num_blocks)])
    def forward(self, x):
        return self.resblocks(F.relu(self.bn(self.conv(x))))

class PredictionNetwork(nn.Module):
    def __init__(self, num_channels, board_size, policy_output_size, value_support_bins, head_hidden_dim):
        super().__init__()
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2, eps=1e-4)
        self.policy_fc = nn.Linear(2 * board_size * board_size, policy_output_size)
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1, eps=1e-4)
        self.value_fc1 = nn.Linear(board_size * board_size, head_hidden_dim)
        self.value_fc2 = nn.Linear(head_hidden_dim, value_support_bins)
    def forward(self, x):
        p = F.relu(self.policy_bn(self.policy_conv(x))).view(x.size(0), -1)
        policy_logits = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x))).view(x.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value_logits = self.value_fc2(v)
        return policy_logits, value_logits

class DynamicsNetwork(nn.Module):
    def __init__(self, num_channels, board_size, reward_support_bins, head_hidden_dim):
        super().__init__()
        action_embedding_dim = 16 # This is an arbitrary choice, can be tuned.
        self.action_embed_conv = nn.Conv2d(1, action_embedding_dim, kernel_size=1, bias=False)
        self.conv = conv3x3(num_channels + action_embedding_dim, num_channels)
        self.bn = nn.BatchNorm2d(num_channels, eps=1e-4)
        self.resblocks = nn.Sequential(*[EvarResBlock(num_channels, num_channels) for _ in range(config.NUM_RES_BLOCKS)])
        self.reward_fc = nn.Sequential(
            nn.Linear(num_channels * board_size * board_size, head_hidden_dim), 
            nn.ReLU(), 
            nn.Linear(head_hidden_dim, reward_support_bins)
        )
    def forward(self, state, action):
        action_plane = F.one_hot(action, num_classes=state.shape[2] * state.shape[3]).float().view(state.shape[0], 1, state.shape[2], state.shape[3])
        action_embedding = self.action_embed_conv(action_plane)
        x = torch.cat((state, action_embedding), dim=1)
        x = F.relu(self.bn(self.conv(x)))
        next_state = self.resblocks(x)
        reward_logits = self.reward_fc(next_state.view(next_state.size(0), -1))
        return next_state, reward_logits

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=1e-4)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)

class GomokuNetEZ(nn.Module):
    def __init__(self, config_obj):
        super().__init__()
        # Use the passed config object
        self.board_size = config_obj.BOARD_SIZE
        self.action_space_size = config_obj.ACTION_SPACE_SIZE
        num_res_blocks, num_filters, head_hidden_dim = config_obj.NUM_RES_BLOCKS, config_obj.NUM_FILTERS, config_obj.HEAD_HIDDEN_DIM
        self.value_support_min, self.value_support_max, self.value_support_bins = config_obj.VALUE_SUPPORT_MIN, config_obj.VALUE_SUPPORT_MAX, config_obj.VALUE_SUPPORT_BINS
        self.reward_support_min, self.reward_support_max, self.reward_support_bins = config_obj.REWARD_SUPPORT_MIN, config_obj.REWARD_SUPPORT_MAX, config_obj.REWARD_SUPPORT_BINS
        observation_shape = (3, self.board_size, self.board_size)
        
        self.representation_net = RepresentationNetwork(observation_shape, num_res_blocks, num_filters)
        self.prediction_net = PredictionNetwork(num_filters, self.board_size, self.action_space_size, self.value_support_bins, head_hidden_dim)
        self.dynamics_net = DynamicsNetwork(num_filters, self.board_size, self.reward_support_bins, head_hidden_dim)
        self.projection_net = ProjectionHead(num_filters * self.board_size * self.board_size)

        for m in self.modules():
            if isinstance(m, EvarResBlock): nn.init.constant_(m.bn2.weight, 0)

    def representation(self, state_input): return self.representation_net(state_input)
    def prediction(self, hidden_state): return self.prediction_net(hidden_state)
    def dynamics(self, hidden_state, action): return self.dynamics_net(hidden_state, action.squeeze(-1))
    
    def project(self, hidden_state, with_grad=True):
        if with_grad: return self.projection_net(hidden_state)
        else:
            with torch.no_grad(): return self.projection_net(hidden_state)

    @torch.no_grad()
    def initial_inference(self, obs):
        self.eval()
        hidden_state = self.representation(obs)
        policy_logits, value_logits = self.prediction(hidden_state)
        value_scalar = support_to_scalar(value_logits, self.value_support_min, self.value_support_max, self.value_support_bins)
        return policy_logits, value_scalar, hidden_state

    @torch.no_grad()
    def recurrent_inference(self, hidden_state, action):
        self.eval()
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
        policy_logits, value_logits = self.prediction(next_hidden_state)
        value_scalar = support_to_scalar(value_logits, self.value_support_min, self.value_support_max, self.value_support_bins)
        reward_scalar = support_to_scalar(reward_logits, self.reward_support_min, self.reward_support_max, self.reward_support_bins)
        return policy_logits, value_scalar, next_hidden_state, reward_scalar