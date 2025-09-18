import numpy as np
from config import config

class GomokuGame:
    def __init__(self, board_size=config.BOARD_SIZE, n_in_row=config.N_IN_ROW):
        self.board_size, self.n_in_row = board_size, n_in_row
        self.reset()
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player, self.last_move, self.move_count = 1, None, 0
        return self
    def get_board_state(self, player, last_move):
        board_state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        board_state[0] = (self.board == player)
        board_state[1] = (self.board == -player)
        if last_move is not None: board_state[2, last_move[0], last_move[1]] = 1
        return board_state
    def get_valid_moves(self):
        return list(zip(*np.where(self.board == 0)))
    def do_move(self, move_idx):
        move = (move_idx // self.board_size, move_idx % self.board_size)
        self.board[move[0], move[1]] = self.current_player
        self.last_move = move; self.current_player = -self.current_player; self.move_count += 1

    def check_win(self, move=None):
        if move is None:
            if self.last_move is None: return False
            r, c = self.last_move
        else:
            r, c = move

        player = self.board[r, c]
        if player == 0: return False # 不应该发生

        # 四个方向: 水平, 垂直, 主对角线, 副对角线
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # 向正方向数
            for i in range(1, self.n_in_row + 2): # 多检查一点以处理长连
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                else:
                    break
            # 向负方向数
            for i in range(1, self.n_in_row + 2):
                nr, nc = r - i * dr, c - i * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                else:
                    break
            
            # 在标准规则 (n_in_row=5) 下，只要大于等于5就算赢
            if count >= self.n_in_row:
                return True
        return False
    
    def get_game_ended(self):
        if self.check_win(): return self.board[self.last_move[0], self.last_move[1]]
        if self.move_count >= self.board_size * self.board_size: return 0
        return None