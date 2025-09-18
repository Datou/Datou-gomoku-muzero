import unittest
import numpy as np
import sys
import os

# Allow direct imports from the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from workers import find_winning_moves_rebuilt
from game import GomokuGame
from config import config

# =====================================================================
#           [FINAL, CORRECTED] Unit Test Class 
# =====================================================================

class TestWinningMoves(unittest.TestCase):
    
    def setUp(self):
        # Use a large, fixed board size for consistent testing of complex patterns
        self.board_size = 15 
        config.BOARD_SIZE = self.board_size # Override config for consistency
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.player = 1
        self.opponent = -1
        self.center = self.board_size // 2

    def test_open_four_win(self):
        """Test: An open-three `_OOO_`, where moves on either side create an open-four."""
        c = self.center
        self.board[c, c - 1: c + 2] = self.player # . . . _PPP_ . . .
        result = find_winning_moves_rebuilt(self.board, self.player)
        self.assertIn((c, c - 2), result['open_four'])
        self.assertIn((c, c + 2), result['open_four'])
        self.assertEqual(len(result['open_four']), 2)

    def test_double_three_combo_win(self):
        """Test: A double-three combo, where the center move creates two open-threes."""
        c = self.center
        # Creates _P_P_ vertically and horizontally
        self.board[c, c - 1] = self.board[c, c + 1] = self.player
        self.board[c - 1, c] = self.board[c + 1, c] = self.player
        winning_move = (c, c)
        result = find_winning_moves_rebuilt(self.board, self.player)
        self.assertIn(winning_move, result['combo'])
        self.assertEqual(len(result['combo']), 1)

    def test_blocked_four_open_three_combo(self):
        """Test: A '冲四活三' combo."""
        c = self.center
        # Horizontal setup for blocked-four: XPP_P
        self.board[c, c - 2] = self.opponent
        self.board[c, c - 1] = self.player
        self.board[c, c + 1] = self.player
        # Vertical setup for open-three: _P_P_
        self.board[c - 1, c] = self.player
        self.board[c + 1, c] = self.player
        # The winning move is (c, c).
        # It creates a blocked-four horizontally: XPPPP
        # And an open-three vertically: _PPP_
        winning_move = (c, c)
        result = find_winning_moves_rebuilt(self.board, self.player)
        self.assertIn(winning_move, result['combo'])
        self.assertEqual(len(result['combo']), 1)

    def test_double_blocked_four_combo(self):
        """Test: A '双冲四' combo."""
        c = self.center
        # Horizontal setup for a blocked-four: XPP_P
        self.board[c, c - 2] = self.opponent
        self.board[c, c - 1] = self.player
        self.board[c, c + 1] = self.player
        # Vertical setup for another blocked-four: XPP_P (transposed)
        self.board[c - 2, c] = self.opponent
        self.board[c - 1, c] = self.player
        self.board[c + 1, c] = self.player
        # The winning move is (c, c).
        # It completes both blocked-fours.
        winning_move = (c, c)
        result = find_winning_moves_rebuilt(self.board, self.player)
        self.assertIn(winning_move, result['combo'])
        self.assertEqual(len(result['combo']), 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)