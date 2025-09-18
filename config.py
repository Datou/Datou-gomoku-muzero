# config.py
import torch

class Config:
    def __init__(self):
        # ================================================================
        #                      系统与环境配置
        # ================================================================
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [REVISED] 统一设置工作进程数量。
        # 这些进程将根据Trainer的指令，在自对弈和复盘模式间动态切换。
        self.NUM_WORKERS = 15
        
        # ================================================================
        #                      游戏与MCTS配置
        # ================================================================
        self.BOARD_SIZE = 6
        self.N_IN_ROW = 5
        self.ACTION_SPACE_SIZE = self.BOARD_SIZE * self.BOARD_SIZE

        self.NUM_SIMULATIONS = 400 #400
        self.NUM_TOP_ACTIONS = 16

        self.MCTS_IMPLEMENTATION = "MuZero"  # 可选项: "AlphaZero", "MuZero"
        
        # MCTS根节点采样策略，以增强探索
        self.POLICY_SAMPLING_FRACTION = 0.75
        self.MIN_POLICY_SAMPLES = 4
        
        self.C_VISIT = 30
        self.C_SCALE = 1.0
        self.VALUE_MINMAX_DELTA = 1e-3
        self.DISCOUNT = 0.997
        
        # ================================================================
        #                      网络结构配置 (为五子棋优化)
        # ================================================================
        # VALUE: 赢, 输, 平局. 没了.
        self.VALUE_SUPPORT_MIN = -1
        self.VALUE_SUPPORT_MAX = 1
        self.VALUE_SUPPORT_BINS = 3 # 分别对应 -1, 0, 1

        # REWARD: 赢, 输.
        self.REWARD_SUPPORT_MIN = -1
        self.REWARD_SUPPORT_MAX = 1
        self.REWARD_SUPPORT_BINS = 3 # 奖励也只有-1, 0, 1
        
        self.NUM_RES_BLOCKS = 8 # 6
        self.NUM_FILTERS = 128 # 128
        self.HEAD_HIDDEN_DIM = 64
        
        # ================================================================
        #                      训练过程配置 (为五子棋优化)
        # ================================================================
        self.PHYSICAL_BATCH_SIZE = 360
        self.GRADIENT_ACCUMULATION_STEPS = 1
        
        self.TRAIN_BUFFER_SIZE = 1000000
        self.REPLAY_BUFFER_PREFILL = 25000
        self.MIN_BUFFER_LEAD = 10000

        self.ENABLE_BACKPRESSURE = True
        
        self.OPTIMIZER_TYPE = 'Adam'
        self.LEARNING_RATE = 5e-6
        self.WEIGHT_DECAY = 1e-5
        self.BARLOW_LAMBDA = 5e-3
        
        self.TARGET_MODEL_TAU = 0.995
        self.NUM_UNROLL_STEPS = 5
        self.GRAD_CLIP_NORM = 5.0
        
        self.LOSS_WEIGHTS = {
            'policy': 1.0,
            'value': 1.0,
            'reward': 0.5,
            'consistency': 5.0
        }
        
        # ================================================================
        #               [REVISED] Re-analysis (浪涌式复盘) 配置
        # ================================================================
        # 是否启用动态的“浪涌式复盘”模式
        self.ENABLE_REANALYSIS = False
        
        # 当一个游戏的分析版本落后于当前训练器超过 N 步时，
        # Trainer将触发全局模式切换，命令所有Worker进入复盘模式。
        self.REANALYSIS_AGE_THRESHOLD = 900
        
        # ================================================================
        #                其他配置
        # ================================================================
        # [新增] 是否启用优先经验回放 (PER) 的总开关
        self.ENABLE_PER = False

        # PER (优先经验回放) 参数
        self.PER_ALPHA = 0.6; self.PER_BETA = 0.4; self.PER_BETA_INCREMENT = 0.00001; self.PER_EPSILON = 1e-6
        
        self.N_STEPS = 10
        self.PREPARED_BATCH_QUEUE_SIZE = 10
        self.INFERENCE_BATCH_SIZE = 15

        self.SAVE_INTERVAL = 1000 # 这是模型保存到磁盘的周期
        self.MODEL_UPDATE_INTERVAL = 1000 # 每*步就告诉worker们有新模型了
        
        self.CURRENT_CONFIG = "Gomoku_Surge_Reanalysis_Final"

config = Config()