import sys
import os
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agents.base_agent import BaseAgent

# Heuristic Weight Matrix promoting a corner-focused gradient strategy (bottom-right)
SCORE_MATRIX = [
    [2,  4,  8,  16],
    [32, 64, 128,256],
    [512,1024,2048,4096],
    [8192,16384,32768,65536]
]

class ExpectimaxAgent(BaseAgent):
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def _evaluate_board(self, board):
        """Heuristic evaluation function of the board state"""
        if board.is_game_over():
            return -999999  # Heavy penalty for game over state

        score = 0
        empty_cells = 0

        # Calculate score based on the corner gradient weight matrix
        for r in range(4):
            for c in range(4):
                val = board.grid[r][c]
                if val == 0:
                    empty_cells += 1
                else:
                    # Give higher scores for having large tiles in the corner
                    score += val * SCORE_MATRIX[r][c]

        # Add a bonus for keeping empty cells open to prevent clogging
        score += empty_cells * 500
        return score

    def _expectimax(self, board, depth, player_turn):
        """Recursive expectimax algorithm returning (best_score)"""
        if depth == 0 or board.is_game_over():
            return self._evaluate_board(board)

        if player_turn:
            # Player turn: Maximize the score across all possible valid moves
            best_score = -float('inf')
            available_moves = board.get_available_moves()
            if not available_moves:
                return self._evaluate_board(board)

            for move in available_moves:
                # Simulate move
                temp_board = copy.deepcopy(board)
                temp_board.move(move)
                # Recurse to computer's turn (chance node)
                score = self._expectimax(temp_board, depth - 1, False)
                best_score = max(best_score, score)
            return best_score

        else:
            # Computer turn: Calculate expected value (average weighted score) of random tile spawns
            empty_cells = [(r, c) for r in range(4) for c in range(4) if board.grid[r][c] == 0]
            if not empty_cells:
                return self._evaluate_board(board)

            total_expected_score = 0
            # 2048 spawns 2 with 90% probability and 4 with 10% probability
            p_2 = 0.9 / len(empty_cells)
            p_4 = 0.1 / len(empty_cells)

            for r, c in empty_cells:
                # Simulate spawning a 2
                board_2 = copy.deepcopy(board)
                board_2.grid[r][c] = 2
                total_expected_score += p_2 * self._expectimax(board_2, depth - 1, True)

                # Simulate spawning a 4
                board_4 = copy.deepcopy(board)
                board_4.grid[r][c] = 4
                total_expected_score += p_4 * self._expectimax(board_4, depth - 1, True)

            return total_expected_score

    def select_move(self, board):
        """Select the best move (0:L, 1:U, 2:R, 3:D) using expectimax search tree"""
        available_moves = board.get_available_moves()
        if not available_moves:
            return 0 # Fallback default

        best_move = available_moves[0]
        best_score = -float('inf')

        # Evaluate the immediate next moves and find the one with the highest expectimax score
        for move in available_moves:
            temp_board = copy.deepcopy(board)
            temp_board.move(move)
            # Begin recursion starting at computer's turn (chance node)
            score = self._expectimax(temp_board, self.max_depth - 1, False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move