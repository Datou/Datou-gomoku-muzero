# Datou的五子棋-MuZero

这是一个使用 Python 和 PyTorch 实现的、高度并行的五子棋（Gomoku）AI 训练项目。它借鉴了 DeepMind 的 AlphaZero 和 MuZero 论文中的核心思想，通过自我对弈（Self-Play）强化学习，从零开始训练一个强大的五子棋模型。

项目内置了一个功能丰富的 Web UI，用于模型分析、人机对战和模型评估。

<img width="1880" height="1310" alt="image" src="https://github.com/user-attachments/assets/ebffda2e-3c93-4a29-b403-a92daac4b70d" />
  
## 核心特性

*   **混合模型架构**: 实现了两种经典的 MCTS (蒙特卡洛树搜索) 模式，可在 `config.py` 中轻松切换：
    *   **`AlphaZero` 模式**: MCTS 的每一步推演都基于真实的游戏引擎状态。
    *   **`MuZero` 模式**: MCTS 在一个由神经网络学习到的“动态模型”中进行推演，不依赖于真实的游戏引擎。
*   **高性能并行架构**: 采用 `torch.multiprocessing` 构建了一个包含多个独立工作进程的复杂系统，实现了数据生成、模型训练和推理的高效并行。
    *   **Self-Play Workers**: 多个并行的自我对弈进程，负责生成大量高质量的游戏数据。
    *   **Inference Server**: 一个独立的推理服务进程，为所有 Self-Play Worker 提供批处理（Batching）过的神经网络推理服务，极大提升了 MCTS 的效率。
    *   **Trainer**: 独立的训练进程，从经验回放池中采样数据并持续更新模型。
    *   **DataLoader**: 负责管理磁盘和内存中的经验回放池（Replay Buffer），并为训练器准备批次数据。
*   **功能强大的 Web UI**: 基于 Flask，提供了一个全面的可视化竞技场和分析工具 (`webui.py`)。
    *   **人机对战**: 与你训练好的任意版本的模型进行对战。
    *   **模型对战 (Live Battle)**: 观察两个不同版本的模型进行实时对战。
    *   **对局复盘 (Replay)**: 加载并回放历史上任何一局自我对弈的棋局。
    *   **模型分析工具**:
        *   **MCTS 测试**: 在预设的战术谜题（如“一步杀”）上测试模型的 MCTS 搜索能力。
        *   **策略热图**: 可视化模型在给定局面下的原始策略网络输出。
        *   **动态函数测试**: 验证 MuZero 模式下动态模型的“想象能力”。
*   **持久化与可恢复训练**: 所有游戏数据和训练状态（模型权重、优化器状态等）都通过 SQLite (`db_manager.py`) 保存，训练可以随时中断和恢复。
*   **浪涌式复盘 (Surge Re-analysis)**: (可选功能) 一个高级特性，当旧数据的质量远低于当前模型水平时，系统会自动切换所有 Worker 进程进入“复盘模式”，使用最新的模型重新分析旧棋局，从而提升数据质量。

## 性能指标特别说明

<img width="3838" height="1886" alt="image" src="https://github.com/user-attachments/assets/6edd4c5f-6958-4757-b616-9d1ba24e7bb5" />

在训练过程中，我们会监控一些指标，例如 **“错过获胜/必杀棋的概率”**。需要特别强调的是：

> **该指标的降低仅仅代表了模型在基础战术层面的能力有所提升，例如能够发现简单的“连五”、“活四”等组合。这并不直接等同于模型在与高水平人类玩家对战时的表现。**
>
> 人类的棋局涉及复杂的长期战略、布局和心理博弈，这些是简单的战术指标无法完全衡量的。因此，请将此指标视为模型训练是否在正轨上的一个基础性参考，而非其最终棋力的直接体现。模型的真实水平需要通过与其它强大 AI 或人类棋手的对战来综合评估。

## 项目结构

下面是项目核心文件的功能简介：

*   `main.py`: **项目入口**。负责初始化和启动所有并行进程。
*   `config.py`: **全局配置文件**。几乎所有的超参数和设置都在这里定义。
*   `workers.py`: **核心业务逻辑**。包含了所有主要进程（Trainer, InferenceServer, universal\_worker 等）的实现。
*   `network.py`: **神经网络定义**。定义了 `GomokuNetEZ` 模型，包括表征、预测和动态三个网络头。
*   `mcts.py`: **MCTS 算法**。实现了 `AlphaZeroMCTS` 和 `MuZeroMCTS` 两种搜索算法。
*   `game.py`: **五子棋游戏逻辑**。定义了棋盘、落子、胜负判断等规则。
*   `db_manager.py`: **数据库管理器**。封装了所有与 SQLite 数据库的交互。
*   `replay_buffer.py`: 实现了支持 **优先经验回放 (PER)** 的内存回放池。
*   `loss.py`: **损失函数**。定义了包含策略、价值、奖励和一致性等多个部分的复杂损失函数。
*   `webui.py`: **Web 后端**。提供了所有 API 接口和 Flask 服务。
*   `index.html`: **Web 前端**。单文件实现了所有可视化交互界面。
*   `ipc_messages.py`: 定义了进程间通信所使用的所有数据结构。

## 如何运行

### 1. 环境配置

建议使用 Conda 或 venv 创建一个独立的 Python 环境。然后安装所需的依赖包。

```bash
# 推荐 Python 3.8+
pip install torch numpy flask tqdm seaborn matplotlib
```
*如果你有支持 CUDA 的 NVIDIA GPU，请务必安装对应版本的 PyTorch 以获得最佳性能。*

### 2. 开始训练

直接运行 `main.py` 即可启动完整的分布式训练流程。

```bash
python main.py
```

你会在终端看到一个由 `tqdm` 生成的动态进度条，实时显示训练状态、自我对弈进度等信息。所有训练的产出（模型权重、日志、棋局复盘文件等）都会被保存在 `outputs/` 目录下。

### 3. 启动 Web UI

当模型训练了一段时间并生成了至少一个模型权重文件 (`.pt`) 后，你可以启动 Web UI。

```bash
python webui.py
```

启动后，在浏览器中打开 `http://127.0.0.1:5000` 即可访问。

## 自定义配置

本项目的所有关键参数都在 `config.py` 文件中，你可以根据你的需求和计算资源进行调整：

*   `DEVICE`: 自动检测使用 `cuda` 或 `cpu`。
*   `NUM_WORKERS`: 设置用于自我对弈的并行工作进程数量，**这是影响数据生成速度的关键参数**。
*   `BOARD_SIZE`, `N_IN_ROW`: 棋盘大小和获胜所需的连子数。
*   `NUM_SIMULATIONS`: 每次落子前 MCTS 的模拟次数，越高棋力越强，但耗时越长。
*   `MCTS_IMPLEMENTATION`: 在 `"AlphaZero"` 和 `"MuZero"` 之间切换。
*   `PHYSICAL_BATCH_SIZE`: 训练时的物理批次大小。
*   `LEARNING_RATE`: 学习率。
*   以及其它所有关于网络结构、MCTS 探索和训练过程的详细参数。
