# mcts.py (REFACTORED BY LINUS)

import torch
import numpy as np
import random
import logging
from queue import Empty
from abc import ABC, abstractmethod

from config import config
from utils import MinMaxStats 
from game import GomokuGame

class Node:
    """
    A standard node in a Monte Carlo Tree Search.
    This class is fine. It does one job and does it simply. No changes needed.
    """
    def __init__(self, action=None, parent=None):
        self.action, self.parent = action, parent
        self.visit_count, self.value_sum, self.reward = 0, 0, 0
        self.policy_logits, self.hidden_state, self.children = None, None, {}
    
    def expand(self, policy_logits, hidden_state, reward):
        self.policy_logits, self.hidden_state, self.reward = policy_logits, hidden_state, reward
    
    def get_child(self, action):
        if action not in self.children:
            self.children[action] = Node(action=action, parent=self)
        return self.children[action]

    def get_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
    
    def get_qsa(self, action: int, discount: float) -> float:
        child = self.children.get(action)
        # The Q-value is the reward from taking the action plus the discounted future value.
        return child.reward + discount * child.get_value() if child and child.visit_count > 0 else 0.0
    
    def is_expanded(self) -> bool:
        return self.hidden_state is not None
    
    def is_root(self) -> bool:
        return self.parent is None

# =================================================================
# 1. Define an Abstract Base Class (the "interface").
#    This declares the contract that all MCTS implementations must follow.
# =================================================================
class MCTS(ABC):
    def __init__(self, worker_id, request_queue, result_queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        # Each implementation gets its own distinct logger name.
        self.logger = logging.getLogger(f"MCTS-{self.__class__.__name__}-{worker_id}")

    @abstractmethod
    def search(self, game: GomokuGame):
        """
        All MCTS implementations MUST provide this method.
        It runs the search and returns a tuple of (policy, value, action).
        """
        pass

# =================================================================
# 2. Create a base class for Gumbel MCTS to hold all shared helper methods.
#    This avoids code duplication.
# =================================================================
class GumbelMCTSBase(MCTS):
    
    # --- Network Communication ---
    def _remote_initial_inference(self, obs):
        self.request_queue.put((self.worker_id, 'initial', obs))
        return self.result_queue.get(timeout=20)

    def _remote_recurrent_inference_batch(self, hidden_states_batch, actions_batch):
        if not actions_batch: return []
        self.request_queue.put((self.worker_id, 'recurrent_batch', (hidden_states_batch, np.array(actions_batch, dtype=np.int32))))
        try:
            p, v, h, r = self.result_queue.get(timeout=20)
            return [(p[i], v[i,0], h[i:i+1], r[i,0]) for i in range(len(actions_batch))]
        except Empty:
            self.logger.warning(f"Worker {self.worker_id} timed out waiting for recurrent inference.")
            return []
            
    # --- Core MCTS Helpers ---
    def _select_leaf(self, root: Node, valid_moves: set, min_max_stats) -> tuple[Node, int]:
        node = root
        while node.is_expanded():
            action = self._select_action(node, valid_moves, min_max_stats)
            node = node.get_child(action)
        return node, node.action

    def _select_action(self, node: Node, valid_moves: set, min_max_stats) -> int:
        if node.is_root():
            min_visits = float('inf')
            best_action = -1
            for action in self.selected_children_actions:
                visits = node.get_child(action).visit_count
                if visits < min_visits:
                    min_visits = visits
                    best_action = action
            return best_action

        transformed_qs = self._get_transformed_completed_Qs(node, min_max_stats)
        improved_policy = self._get_improved_policy(node, valid_moves, transformed_qs)
        
        children_visits = np.array([node.get_child(a).visit_count for a in range(config.ACTION_SPACE_SIZE)])
        total_visits = sum(child.visit_count for child in node.children.values())

        scores = improved_policy - children_visits / (1 + total_visits)
        
        mask = np.full_like(scores, -np.inf)
        valid_indices = list(valid_moves)
        mask[valid_indices] = scores[valid_indices]
        return np.argmax(mask)

    def _backpropagate(self, leaves: list[Node], values: list[float], min_max_stats):
        for i, leaf in enumerate(leaves):
            value, node = values[i], leaf
            # [Linus式修正] 确保初始的 leaf value 就在 [-1, 1] 范围内
            value = np.clip(value, -1.0, 1.0)

            while node is not None:
                node.value_sum += value
                node.visit_count += 1
                if not node.is_root():
                    q_value = node.parent.get_qsa(node.action, config.DISCOUNT)
                    min_max_stats.update(q_value)
                
                # [Linus式修正] 
                # 在更新 value 给父节点之前，再次进行限制。
                # 这确保了无论 reward 是多少，累加的值都不会无限增长。
                value = node.reward + config.DISCOUNT * value
                value = np.clip(value, -1.0, 1.0)

                node = node.parent

    # --- Gumbel-specific Helpers ---
    def _get_transformed_completed_Qs(self, node: Node, min_max_stats) -> np.ndarray:
        qsa_values = np.array([node.get_qsa(a, config.DISCOUNT) for a in range(config.ACTION_SPACE_SIZE)])
        normalized_qsa = np.array([min_max_stats.normalize(q) for q in qsa_values])
        
        max_child_visit = 0
        if node.children:
            max_child_visit = max(child.visit_count for child in node.children.values()) if node.children else 0

        return (config.C_VISIT + max_child_visit) * config.C_SCALE * normalized_qsa

    def _get_improved_policy(self, node: Node, valid_moves: set, transformed_qs: np.ndarray) -> np.ndarray:
        logits = node.policy_logits + transformed_qs
        mask = np.full_like(logits, -np.inf)
        valid_indices = list(valid_moves)
        mask[valid_indices] = logits[valid_indices]
        return torch.softmax(torch.from_numpy(mask), dim=0).numpy()
    
    def _initialize_sequential_halving_schedule(self):
        self.current_phase, self.current_num_top_actions, self.used_visit_num = 0, config.NUM_TOP_ACTIONS, 0
        n, m = config.NUM_SIMULATIONS, config.NUM_TOP_ACTIONS
        if m <= 1 or np.log2(m) <= 0:
            self.visit_num_for_next_phase = n
        else:
            self.visit_num_for_next_phase = int(min(np.floor(n / (np.log2(m) * m)) * m, n))

    def _ready_for_next_gumbel_phase(self, sim_idx: int) -> bool:
        if sim_idx < self.visit_num_for_next_phase: return False
        self.current_phase += 1
        self.current_num_top_actions //= 2
        if self.current_num_top_actions < 1: return False
        n, m, current_m = config.NUM_SIMULATIONS, config.NUM_TOP_ACTIONS, self.current_num_top_actions
        
        if current_m <= 1 or np.log2(m) <= 0:
             extra_visit = n - self.used_visit_num
        else:
             extra_visit = np.floor(n / (np.log2(m) * current_m)) * current_m
        
        self.used_visit_num += extra_visit
        self.visit_num_for_next_phase = min(self.visit_num_for_next_phase + int(extra_visit), n)
        return True

    def _sequential_halving(self, root: Node, valid_moves: set, min_max_stats):
        transformed_qs = self._get_transformed_completed_Qs(root, min_max_stats)
        scores = {a: self.gumbel_noises[a] + root.policy_logits[a] + transformed_qs[a] for a in self.selected_children_actions}
        self.selected_children_actions = sorted(scores, key=scores.get, reverse=True)[:self.current_num_top_actions]

# =================================================================
# 3. Create the two concrete implementations. Each one does ONE thing.
# =================================================================

class AlphaZeroMCTS(GumbelMCTSBase):
    """
    An MCTS implementation that uses the REAL game engine for rollouts.
    It does not trust or use the dynamics network. This is the stable,
    predictable approach.
    """
    def search(self, game: GomokuGame):
        try:
            while True: self.result_queue.get_nowait()
        except Empty:
            pass

        min_max_stats = MinMaxStats(config.VALUE_MINMAX_DELTA)
        root = Node()
        obs = game.get_board_state(game.current_player, game.last_move)
        
        try:
            policy_logits, value, hidden_state = self._remote_initial_inference(obs)
        except Empty:
            self.logger.warning(f"Worker {self.worker_id} timed out on initial inference.")
            return np.zeros(config.ACTION_SPACE_SIZE), 0.0, -1

        valid_moves = {m[0] * config.BOARD_SIZE + m[1] for m in game.get_valid_moves()}
        if not valid_moves:
            return np.zeros(config.ACTION_SPACE_SIZE), 0.0, -1

        root.expand(policy_logits, hidden_state, 0)
        self._backpropagate([root], [value], min_max_stats)
        
        self._initialize_sequential_halving_schedule()
        self.gumbel_noises = np.random.gumbel(0, 1, config.ACTION_SPACE_SIZE)
        valid_moves_list = list(valid_moves)
        priors = root.policy_logits[valid_moves_list]
        scores = self.gumbel_noises[valid_moves_list] + priors
        sorted_actions = [action for _, action in sorted(zip(scores, valid_moves_list), reverse=True)]
        self.selected_children_actions = sorted_actions[:self.current_num_top_actions]
        
        sim_count = 1
        while sim_count < config.NUM_SIMULATIONS:
            
            # --- AlphaZero-style rollout ---
            # 1. Select a leaf node.
            leaf_node, _ = self._select_leaf(root, valid_moves, min_max_stats)

            # 2. Simulate to that leaf using a temporary, REAL game state.
            action_history = []
            temp_node = leaf_node
            while not temp_node.is_root():
                action_history.append(temp_node.action)
                temp_node = temp_node.parent
            action_history.reverse()

            temp_game = GomokuGame(board_size=game.board_size)
            temp_game.board = np.copy(game.board)
            temp_game.current_player = game.current_player
            temp_game.move_count = game.move_count
            for act in action_history:
                temp_game.do_move(act)
            
            # 3. Expand the node using an inference call on the REAL board state.
            leaf_obs = temp_game.get_board_state(temp_game.current_player, temp_game.last_move)
            try:
                policy_logits_res, value_res, hidden_state_res = self._remote_initial_inference(leaf_obs)
            except Empty:
                self.logger.warning(f"Worker {self.worker_id} timed out during MCTS expansion.")
                continue
            
            # In AlphaZero mode, intermediate rewards are always 0.
            reward_res = 0.0
            leaf_node.expand(policy_logits_res, hidden_state_res, reward_res)
            
            # 4. Backpropagate the value from the new leaf.
            self._backpropagate([leaf_node], [value_res], min_max_stats)
            sim_count += 1

            # Gumbel sequential halving logic
            if self._ready_for_next_gumbel_phase(sim_count):
                self._sequential_halving(root, valid_moves, min_max_stats)
        
        # --- Decision Phase ---
        transformed_qs = self._get_transformed_completed_Qs(root, min_max_stats)
        final_policy = self._get_improved_policy(root, valid_moves, transformed_qs)
        
        visit_counts = {a: root.get_child(a).visit_count for a in valid_moves}
        best_action = max(visit_counts, key=visit_counts.get) if visit_counts else -1

        if best_action == -1:
             best_action = random.choice(list(valid_moves)) if valid_moves else -1

        return final_policy, root.get_value(), int(best_action)


class MuZeroMCTS(GumbelMCTSBase):
    """
    An MCTS implementation that uses the learned DYNAMICS network for rollouts.
    This is the experimental, potentially higher-risk approach.
    """
    def search(self, game: GomokuGame):
        try:
            while True: self.result_queue.get_nowait()
        except Empty:
            pass

        min_max_stats = MinMaxStats(config.VALUE_MINMAX_DELTA)
        root = Node()
        obs = game.get_board_state(game.current_player, game.last_move)
        
        try:
            policy_logits, value, hidden_state = self._remote_initial_inference(obs)
        except Empty:
            self.logger.warning(f"Worker {self.worker_id} timed out on initial inference.")
            return np.zeros(config.ACTION_SPACE_SIZE), 0.0, -1

        valid_moves = {m[0] * config.BOARD_SIZE + m[1] for m in game.get_valid_moves()}
        if not valid_moves:
            return np.zeros(config.ACTION_SPACE_SIZE), 0.0, -1

        root.expand(policy_logits, hidden_state, 0)
        self._backpropagate([root], [value], min_max_stats)
        
        self._initialize_sequential_halving_schedule()
        self.gumbel_noises = np.random.gumbel(0, 1, config.ACTION_SPACE_SIZE)
        valid_moves_list = list(valid_moves)
        priors = root.policy_logits[valid_moves_list]
        scores = self.gumbel_noises[valid_moves_list] + priors
        sorted_actions = [action for _, action in sorted(zip(scores, valid_moves_list), reverse=True)]
        self.selected_children_actions = sorted_actions[:self.current_num_top_actions]
        
        sim_count = 1
        while sim_count < config.NUM_SIMULATIONS:
            
            # --- MuZero-style rollout (using the dynamics model) ---
            leaves, actions_to_expand = [], []
            hidden_states_to_expand, actions_for_inference = [], []

            for _ in range(len(self.selected_children_actions)):
                leaf_node, action = self._select_leaf(root, valid_moves, min_max_stats)
                if leaf_node is not None and action != -1:
                    leaves.append(leaf_node)
                    actions_to_expand.append(action)
                    hidden_states_to_expand.append(leaf_node.parent.hidden_state)
                    actions_for_inference.append(leaf_node.action)
            
            if not leaves: break

            results = self._remote_recurrent_inference_batch(np.concatenate(hidden_states_to_expand, axis=0), actions_for_inference)
            if not results: continue

            values_to_backprop = []
            for i, leaf in enumerate(leaves):
                policy_logits, value, hidden_state, reward = results[i]
                leaf.expand(policy_logits, hidden_state, reward)
                values_to_backprop.append(value)
            
            self._backpropagate(leaves, values_to_backprop, min_max_stats)
            sim_count += len(leaves)

            # Gumbel sequential halving logic
            if self._ready_for_next_gumbel_phase(sim_count):
                self._sequential_halving(root, valid_moves, min_max_stats)
        
        # --- Decision Phase ---
        transformed_qs = self._get_transformed_completed_Qs(root, min_max_stats)
        final_policy = self._get_improved_policy(root, valid_moves, transformed_qs)
        
        visit_counts = {a: root.get_child(a).visit_count for a in valid_moves}
        best_action = max(visit_counts, key=visit_counts.get) if visit_counts else -1

        if best_action == -1:
             best_action = random.choice(list(valid_moves)) if valid_moves else -1

        return final_policy, root.get_value(), int(best_action)
