# utils.py

import torch
import numpy as np

class MinMaxStats:
    def __init__(self, minmax_delta):
        self.maximum = -float('inf')
        self.minimum = float('inf')
        self.minmax_delta = minmax_delta

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        # If the range is too small, avoid division by zero.
        if self.maximum > self.minimum:
            normalized = (value - self.minimum) / (self.maximum - self.minimum + self.minmax_delta)
            # =================== [LINUS'S FIX START] ===================
            # THIS IS THE CRITICAL FIX. The value MUST be clamped to [0, 1].
            return max(0.0, min(1.0, normalized))
            # =================== [LINUS'S FIX END] ===================
        # Return a neutral value if no range is established yet.
        return 0.0
    

def soft_update(target_model, source_model, tau):
    """Performs a soft update of the target model's weights."""
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def _convert_to_json_serializable(obj):
    """Recursively converts objects to be JSON serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.int8, np.int16, np.int32, np.int64)):
        return obj.item()
    if isinstance(obj, tuple) and hasattr(obj, '_fields'): # Check for namedtuple
        return {field: _convert_to_json_serializable(getattr(obj, field)) for field in obj._fields}
    if isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    return obj