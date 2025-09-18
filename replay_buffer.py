import numpy as np
from config import config # Assuming config is used for PER_BETA, PER_EPSILON

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.write_ptr = 0
        self.count = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority):
        tree_idx = self.write_ptr + self.capacity - 1
        self.update(tree_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.count < self.capacity: self.count += 1
            
    def get_leaf(self, value):
        parent_idx = 0
        while True:
            left_child_idx, right_child_idx = 2 * parent_idx + 1, 2 * parent_idx + 2
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx; break
            if value <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        return leaf_idx

    def total_priority(self):
        return self.tree[0]

class InMemoryReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.sum_tree = SumTree(capacity)
        self.data = [None] * capacity
        self.max_priority = 1.0

    def add(self, training_slice):
        write_ptr = self.sum_tree.write_ptr
        self.data[write_ptr] = training_slice
        # 根据开关决定使用最大优先级还是固定的默认优先级
        priority_to_add = self.max_priority if config.ENABLE_PER else 1.0
        self.sum_tree.add(priority_to_add)

    def sample(self, batch_size):
        if self.sum_tree.count < batch_size: return None, None, None
        
        if config.ENABLE_PER:
            batch, tree_indices, is_weights = [], [], []
            total_p = self.sum_tree.total_priority()
            segment = total_p / batch_size
            
            for i in range(batch_size):
                s = np.random.uniform(segment * i, segment * (i + 1))
                tree_idx = self.sum_tree.get_leaf(s)
                priority = self.sum_tree.tree[tree_idx]
                
                data_idx = tree_idx - self.capacity + 1
                batch.append(self.data[data_idx])
                tree_indices.append(tree_idx)
                
                sampling_prob = priority / total_p
                # [FIX] Beta should be annealed externally, but for now, just use the config value.
                weight = (self.sum_tree.count * sampling_prob) ** -config.PER_BETA
                is_weights.append(weight)

            is_weights = np.array(is_weights, dtype=np.float32)
            # Normalize weights by the maximum weight in the batch.
            max_w = np.max(is_weights)
            if max_w > 0:
                is_weights /= max_w

            return batch, tree_indices, is_weights
        else:
            # =================== [LINUS'S FIX START] ===================
            # Uniform sampling. The old way of calculating tree_indices was WRONG.
            # For uniform sampling, the trainer doesn't update priorities anyway,
            # so we can just pass the data indices themselves.
            indices = np.random.choice(self.sum_tree.count, batch_size, replace=False)
            batch = [self.data[i] for i in indices]
            is_weights = np.ones(batch_size, dtype=np.float32)
            # Pass the DATA indices. It's cleaner and correct. The trainer won't use them.
            return batch, indices, is_weights
            # =================== [LINUS'S FIX END] ===================

    def update_priorities(self, tree_indices, td_errors):
        if not config.ENABLE_PER: return # 如果 PER 关闭，则直接返回
        priorities = np.abs(td_errors) + config.PER_EPSILON
        for idx, p in zip(tree_indices, priorities):
            self.max_priority = max(self.max_priority, p)
            self.sum_tree.update(idx, p)
    
    def __len__(self):
        return self.sum_tree.count