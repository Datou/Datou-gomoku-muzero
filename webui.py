# webui.py
import os
import glob
import json
import torch
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, abort
from queue import Empty
import random

# 导入你项目中的核心组件
from config import config
from game import GomokuGame
from network import GomokuNetEZ
from mcts import GumbelEzV2MCTS
from workers import find_winning_moves_rebuilt 

# --- 全局变量和初始化 ---
app = Flask(__name__, static_folder='.', static_url_path='')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "outputs")
REPLAYS_DIR = os.path.join(MODELS_DIR, "replays")
MODEL_WEIGHTS_DIR = os.path.join(MODELS_DIR, "model_weights") # 使用更有组织的目录

loaded_models = {}

# --- 战术谜题定义 ---
PUZZLES = {
    "win_in_one": {
        "board": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        "player": 1,
        "solutions": [(6, 4), (6, 9)] # [修复] 使用 "solutions" 列表允许多个正确答案
    },
    "block_in_one": {
        "board": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,-1,-1,-1,-1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        "player": 1,
        "solutions": [(6, 5), (6, 10)] # [修复] 使用 "solutions" 列表允许多个正确答案
    }
}


def get_model(model_path):
    if model_path not in loaded_models:
        print(f"Loading model: {model_path}...")
        try:
            # --- [MODIFICATION START] ---
            # 1. 构造 config.json 的路径
            config_path = model_path.replace('.pt', '.json')
            
            # 2. 读取模型专属的配置文件
            if not os.path.exists(config_path):
                # 如果找不到 .json，就退回旧的加载方式并打印警告
                print(f"Warning: {config_path} not found. Falling back to global config.")
                model_config_obj = config
            else:
                with open(config_path, 'r') as f:
                    model_params = json.load(f)
                # 使用读取的参数创建一个临时的 config 对象
                model_config_obj = type('ModelConfig', (object,), model_params)()

            # 3. 使用正确的 config 对象实例化模型
            model = GomokuNetEZ(model_config_obj).to(config.DEVICE)
            # --- [MODIFICATION END] ---
            
            weights = torch.load(model_path, map_location=config.DEVICE)
            model.load_state_dict(weights)
            model.eval()
            loaded_models[model_path] = model
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None
    return loaded_models[model_path]

class LocalInferenceEngine:
    def __init__(self, model):
        self.model = model
        self.last_result = None

    def put(self, request_data):
        """Performs inference and stores the result internally."""
        worker_id, req_type, data = request_data
        with torch.no_grad():
            if req_type == 'initial':
                obs = torch.from_numpy(data).unsqueeze(0).to(config.DEVICE)
                p, v, h = self.model.initial_inference(obs)
                self.last_result = (p.cpu().numpy()[0], v.cpu().numpy()[0, 0], h.cpu().numpy())
            elif req_type == 'recurrent_batch':
                states, actions = data
                states = torch.from_numpy(states).to(config.DEVICE)
                actions = torch.from_numpy(actions).long().to(config.DEVICE)
                p, v, h, r = self.model.recurrent_inference(states, actions)
                
                p_cpu, v_cpu, h_cpu, r_cpu = p.cpu().numpy(), v.cpu().numpy(), h.cpu().numpy(), r.cpu().numpy()
                
                self.last_result = (p_cpu, v_cpu, h_cpu, r_cpu)

    def get(self, timeout=None):
        """Returns the stored result and consumes it, mimicking a queue."""
        if self.last_result is None:
            raise Empty("LocalInferenceEngine is empty")
        
        result = self.last_result
        self.last_result = None # Consume the result
        return result

    def get_nowait(self):
        """Identical to get() for this mock, as it doesn't block."""
        return self.get()

def run_mcts_search(game, model):
    local_inference = LocalInferenceEngine(model)
    mcts_engine = GumbelEzV2MCTS(0, local_inference, local_inference)
    policy, value, action = mcts_engine.search(game)
    return policy, value, action

# --- API Endpoints ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/api/config')
def get_app_config():
    return jsonify({'board_size': config.BOARD_SIZE, 'n_in_row': config.N_IN_ROW})

@app.route('/get_replay_list')
def get_replay_list():
    if not os.path.exists(REPLAYS_DIR): return jsonify([])
    replay_files = glob.glob(os.path.join(REPLAYS_DIR, "*.json"))
    replays = []
    for rf in sorted(replay_files, reverse=True):
        basename = os.path.basename(rf)
        try:
            parts = basename.replace('.json', '').split('_')
            iteration = int(parts[2])
            game_id = int(parts[4])
            replays.append({'name': f"Game {game_id}", 'iter': iteration, 'path': basename})
        except (IndexError, ValueError):
            replays.append({'name': basename, 'iter': 0, 'path': basename})
    return jsonify(replays)

@app.route('/load_replay/<path:filename>')
def load_replay(filename):
    safe_path = os.path.join(REPLAYS_DIR, os.path.basename(filename))
    if not os.path.exists(safe_path): abort(404)
    return send_from_directory(REPLAYS_DIR, os.path.basename(filename))

@app.route('/get_hof_list')
def get_hof_list():
    if not os.path.exists(MODEL_WEIGHTS_DIR): return jsonify([])
    model_files = glob.glob(os.path.join(MODEL_WEIGHTS_DIR, "model_weights_step_*.pt"))
    models = sorted(model_files, key=lambda p: int(p.split('_')[-1].split('.')[0]), reverse=True)
    return jsonify([{'name': f'Step {os.path.basename(mf).split("_")[-1].split(".")[0]}', 'path': os.path.basename(mf)} for mf in models])

@app.route('/api/move', methods=['POST'])
def handle_play_move():
    data = request.json
    board = np.array(data['board'])
    player_color = data['player_color']
    model_path = os.path.join(MODEL_WEIGHTS_DIR, data['model_path'])
    # [修复] 从请求中获取人类玩家的最后一步棋，可能不存在（例如AI先手）
    last_human_move = data.get('last_move') 

    model = get_model(model_path)
    if not model: return jsonify({'error': 'Failed to load model'}), 500

    game = GomokuGame()
    game.board = board
    game.move_count = np.sum(board != 0)

    # --- [核心修复逻辑] ---
    # 1. 如果前端提供了 last_human_move，这说明是人类刚刚下完棋，我们必须立即检查胜负
    if last_human_move:
        move_tuple = tuple(last_human_move)
        # 假设 GomokuGame 类有 check_win 方法，这在 workers.py 中被使用过
        if game.check_win(move=move_tuple):
            # 胜利者就是刚刚落子的人类玩家
            winner = player_color
            return jsonify({
                'ai_move': None, # AI没有机会移动
                'black_win_rate': 100.0 if winner == 1 else 0.0,
                'game_over': True,
                'winner': int(winner)
            })

    # 2. 如果人类没赢，检查是否平局
    if game.move_count >= config.BOARD_SIZE * config.BOARD_SIZE:
        return jsonify({
            'ai_move': None,
            'black_win_rate': 50.0,
            'game_over': True,
            'winner': 0
        })
    # --- [修复结束] ---

    # 3. 如果游戏没有结束，才轮到 AI 行动
    game.current_player = player_color * -1 
    
    policy, value, action = run_mcts_search(game, model)
    ai_move = (int(action // config.BOARD_SIZE), int(action % config.BOARD_SIZE)) if action != -1 else None
    
    game.do_move(action)
    ended = game.get_game_ended()
    game_over = ended is not None
    winner = ended if game_over else None

    return jsonify({
        'ai_move': ai_move,
        'black_win_rate': float((value + 1) / 2 * 100 if game.current_player == -1 else (1 - (value + 1) / 2) * 100),
        'game_over': game_over,
        'winner': int(winner) if winner is not None else None
    })

@app.route('/api/live_move', methods=['POST'])
def handle_live_move():
    data = request.json
    board = np.array(data['board'])
    current_player = data['current_player']
    model_path = os.path.join(MODEL_WEIGHTS_DIR, data['model_path'])

    model = get_model(model_path)
    if not model: return jsonify({'error': 'Failed to load model'}), 500

    game = GomokuGame()
    game.board = board
    game.current_player = current_player
    game.move_count = np.sum(board != 0)

    policy, value, action = run_mcts_search(game, model)
    ai_move = (int(action // config.BOARD_SIZE), int(action % config.BOARD_SIZE)) if action != -1 else None

    game.do_move(action)
    ended = game.get_game_ended()
    game_over = ended is not None
    winner = ended if game_over else None

    return jsonify({
        'ai_move': ai_move,
        # (之前的 float32 修复保持不变)
        'black_win_rate': float((value + 1) / 2 * 100 if current_player == 1 else (1 - (value + 1) / 2) * 100),
        'game_over': game_over,
        # [修复] 在序列化前，将 winner (可能是 numpy.int64 或 None) 转换为 Python int 或 None
        'winner': int(winner) if winner is not None else None
    })

@app.route('/api/mcts_test', methods=['POST'])
def handle_mcts_test():
    data = request.json
    puzzle_name = data['puzzle']
    model_path = os.path.join(MODEL_WEIGHTS_DIR, data['model_path'])

    if puzzle_name not in PUZZLES: abort(404, "Puzzle not found")
    
    puzzle = PUZZLES[puzzle_name]
    model = get_model(model_path)
    if not model: return jsonify({'error': 'Failed to load model'}), 500

    game = GomokuGame()
    game.board = puzzle['board']
    game.current_player = puzzle['player']
    game.move_count = np.sum(puzzle['board'] != 0)

    policy, value, action = run_mcts_search(game, model)
    ai_move = (int(action // config.BOARD_SIZE), int(action % config.BOARD_SIZE))
    
    # 检查 AI 的移动是否在任何一个正确答案中
    is_correct = (ai_move in puzzle['solutions'])
    
    return jsonify({
        'board': puzzle['board'].tolist(),
        'player': int(puzzle['player']),
        'ai_move': ai_move,
        'solutions': puzzle['solutions'], # [修复] 返回完整的解决方案列表
        'is_correct': bool(is_correct),
        'policy': policy.tolist(),
        'value': float(value)
    })

@app.route('/api/policy_test', methods=['POST'])
def handle_policy_test():
    data = request.json
    puzzle_name = data['puzzle']
    model_path = os.path.join(MODEL_WEIGHTS_DIR, data['model_path'])

    if puzzle_name not in PUZZLES: abort(404, "Puzzle not found")
    
    puzzle = PUZZLES[puzzle_name]
    model = get_model(model_path)
    if not model: return jsonify({'error': 'Failed to load model'}), 500

    game = GomokuGame()
    game.board = puzzle['board']
    
    obs = game.get_board_state(puzzle['player'], None)
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        policy_logits, _, _ = model.initial_inference(obs_tensor)
    
    policy = torch.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
    
    valid_moves_numpy = game.get_valid_moves()
    # =================== [THE FIX IS HERE] ===================
    # Explicitly convert NumPy's int types to Python's native int
    # so the JSON serializer doesn't have a seizure.
    valid_moves_python = [(int(r), int(c)) for r, c in valid_moves_numpy]
    # =========================================================

    return jsonify({
        'board': puzzle['board'].tolist(),
        'policy_heatmap': policy.tolist(),
        'valid_moves': valid_moves_python # Use the converted list
    })

@app.route('/api/dynamics_test', methods=['POST'])
def handle_dynamics_test():
    """
    执行动态分析测试：
    1. 在真实棋盘上随机走一黑一白。
    2. 单独使用模型的动态函数（想象）来推演这两步。
    3. 返回真实棋局和模型想象后给出的策略，用于对比。
    """
    data = request.json
    model_path = os.path.join(MODEL_WEIGHTS_DIR, data['model_path'])
    model = get_model(model_path)
    if not model:
        return jsonify({'error': 'Failed to load model'}), 500

    # --- 步骤1: 在真实棋盘上随机走两步 ---
    game = GomokuGame()
    
    valid_moves_black = game.get_valid_moves()
    if not valid_moves_black: return jsonify({'error': 'No valid moves on empty board'}), 500
    black_move_coords_np = random.choice(valid_moves_black)
    black_action = black_move_coords_np[0] * config.BOARD_SIZE + black_move_coords_np[1]
    game.do_move(black_action)
    
    valid_moves_white = game.get_valid_moves()
    if not valid_moves_white: return jsonify({'error': 'No valid moves after first stone'}), 500
    white_move_coords_np = random.choice(valid_moves_white)
    white_action = white_move_coords_np[0] * config.BOARD_SIZE + white_move_coords_np[1]
    game.do_move(white_action)
    
    final_board_state = game.board.copy()
    final_valid_moves = game.get_valid_moves()

    # --- 步骤2: 使用模型动态函数进行“想象” ---
    with torch.no_grad():
        initial_obs = GomokuGame().get_board_state(player=1, last_move=None)
        obs_tensor = torch.from_numpy(initial_obs).unsqueeze(0).to(config.DEVICE)
        _, _, h0 = model.initial_inference(obs_tensor)
        
        black_action_tensor = torch.tensor([black_action], dtype=torch.long, device=config.DEVICE).unsqueeze(0)
        _, _, h1, _ = model.recurrent_inference(h0, black_action_tensor)
        
        white_action_tensor = torch.tensor([white_action], dtype=torch.long, device=config.DEVICE).unsqueeze(0)
        p_logits, v_predicted, h2, r_predicted = model.recurrent_inference(h1, white_action_tensor)

    imagined_policy = torch.softmax(p_logits, dim=1).squeeze().cpu().numpy()
    imagined_value = v_predicted.squeeze().cpu().item()

    # =================== [THE FIX IS HERE] ===================
    # 显式地将 NumPy 整数 (np.int64) 转换为 Python 原生 int
    black_move_coords = (int(black_move_coords_np[0]), int(black_move_coords_np[1]))
    white_move_coords = (int(white_move_coords_np[0]), int(white_move_coords_np[1]))
    # =========================================================

    return jsonify({
        'final_board': final_board_state.tolist(),
        'black_move': black_move_coords, # 使用转换后的元组
        'white_move': white_move_coords, # 使用转换后的元组
        'policy_heatmap': imagined_policy.tolist(),
        'valid_moves': [(int(r), int(c)) for r, c in final_valid_moves],
        'predicted_value': imagined_value
    })

@app.route('/api/analyze_move', methods=['POST'])
def analyze_move_endpoint():
    """
    接收一个棋盘状态和当前玩家，返回所有潜在的获胜/必杀落子点。
    这是新的“事实标准”API，供前端调用。
    """
    try:
        data = request.json
        board = np.array(data['board'])
        player = int(data['player'])

        winning_moves_numpy = find_winning_moves_rebuilt(board.copy(), player)

        # [修复] 在序列化之前，将所有 NumPy 整数 (int64) 转换为 Python 原生 int
        serializable_winning_moves = {}
        for key, move_list in winning_moves_numpy.items():
            # 使用列表推导式高效地转换每个元组中的坐标
            serializable_winning_moves[key] = [(int(r), int(c)) for r, c in move_list]

        # 返回转换后、可安全序列化的字典
        return jsonify(serializable_winning_moves)
        
    except Exception as e:
        print(f"Error in /api/analyze_move: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(REPLAYS_DIR, exist_ok=True)
    os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
    print("\n--- Gomoku AlphaZero WebUI ---")
    print(f"Serving on http://127.0.0.1:5000")
    print(f"Models expected in: {MODEL_WEIGHTS_DIR}")
    print(f"Replays expected in: {REPLAYS_DIR}")
    app.run(host='0.0.0.0', port=5000, debug=False)