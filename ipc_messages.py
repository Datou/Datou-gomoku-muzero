# ipc_messages.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import numpy as np

# [REFACTOR 2] 定义清晰、类型安全的消息类，取代"魔法字典"

# ==================== Status & Logging Messages ====================

@dataclass
class TrainerStatus:
    """由Trainer发送，用于更新UI和日志"""
    step: int
    total_loss: float
    policy_loss: float
    value_loss: float
    reward_loss: float
    consistency_loss: float
    lr: float
    saved_at: int
    # [补丁] 新增 games_completed 字段，用于同步和恢复游戏计数
    games_completed: int
    resumed: bool = False
    source: str = 'trainer'

@dataclass
class DataLoaderStatus:
    """由DataLoader发送，报告缓冲区大小"""
    buffer_size: int
    source: str = 'data_loader'

@dataclass
class SelfPlayStatus:
    """由SelfPlayWorker在游戏结束后发送"""
    avg_len: float
    miss_five: int
    miss_total: int
    source: str = 'self_play'

@dataclass
class SelfPlayMove:
    """由SelfPlayWorker在每一步移动后发送，用于UI spinner"""
    source: str = 'self_play_move'

# --- [新增] ---
# 为了解决UI和Logger对SelfPlayStatus消息的竞争消费问题，
# 我们添加一个专门用于UI进度条更新的轻量级通知。
@dataclass
class GameCompletedNotice:
    """由SelfPlayWorker在游戏结束后发送，专门用于通知UI更新游戏计数"""
    source: str = 'self_play_notice'
# --- [新增结束] ---


# ==================== Trainer状态消息 ====================
@dataclass
class TrainerWaitPrefillStatus:
    """由Trainer在等待预填充时发送"""
    buffer_size: int
    prefill_size: int
    source: str = 'trainer_wait_prefill'

@dataclass
class TrainerModelUpdateNotice:
    """由Trainer在更新InferenceServer模型后发送，用于UI提示"""
    # [最终修复] 将没有默认值的 'step' 字段放在前面
    step: int
    # [最终修复] 将有默认值的 'source' 字段放在后面
    source: str = 'trainer_model_update_notice'

# ==================== Task & Control Messages ====================

@dataclass
class ModelWeightsUpdate:
    """由Trainer发送给InferenceServer，包含最新的模型权重"""
    weights: Dict[str, Any]

@dataclass
class InitialModelRequest:
    """由InferenceServer在启动时发送给Trainer，请求初始模型"""
    server_id: int = 0 # 为未来可能的多个推理服务器扩展

@dataclass
class HeatmapTask:
    """由主进程或Trainer发送给Visualizer，请求生成热图"""
    step: int
    model_state_dict: Dict[str, Any]
    task: str = 'generate_heatmap'

@dataclass
class PriorityUpdate:
    """由Trainer发送给DataLoader，用于更新样本优先级"""
    batch_ids: List[int]
    td_errors: np.ndarray

@dataclass
class TrainerWaitingForDataStatus:
    """由Trainer在等待新批次数据超时后发送"""
    source: str = 'trainer_wait_data'

@dataclass
class WorkerPauseStatus: # <--- [LINUS'S FIX] RENAME THIS CLASS
    """由Trainer发送，通知UI所有数据生成进程的暂停/恢复状态"""
    is_paused: bool
    reason: str
    source: str = 'worker_pause_status' # <--- and this source string


@dataclass
class TrainerPauseStatus:
    """由Trainer发送，通知UI训练进程的暂停/恢复状态"""
    is_paused: bool
    reason: str
    source: str = 'trainer_pause_status'

@dataclass
class ReAnalysisStatus:
    """由复盘进程发送到UI，报告其工作进度和核心指标"""
    total_reanalyzed: int
    corrected_fives: int
    original_fives: int
    corrected_totals: int
    original_totals: int
    source: str = 're_analysis'

@dataclass
class ReAnalysisQueueStatus:
    """由Orchestrator发送，报告需要复盘的游戏总数"""
    total_games_to_reanalyze: int
    source: str = 'orchestrator_queue_status'