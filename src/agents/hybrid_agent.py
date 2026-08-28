import sys
import os
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agents.base_agent import BaseAgent
from src.agents.ntuple_agent import NTupleAgent

class HybridAgent(BaseAgent):
    def __init__(self, weights_path="checkpoints/ntuple_weights.npz", max_depth=3):
        self.max_depth = max_depth
        
        # Initialize a helper N-Tuple agent to use its state evaluation function
        self.ntuple_evaluator = NTupleAgent()
        
        if os.path.exists(weights_path):
            print(f"[Hybrid Agent] Loading trained N-Tuple weights from {weights_path}...")
            self.ntuple_evaluator.load_weights(weights_path)
        else:
            print(f"[Hybrid Agent] Warning: No trained weights found at {weights_path}. Running with zero weights.")

    def _evaluate_board(self, board):
        """Use the trained N-Tuple Lookup Tables to evaluate the leaf node state value"""
        if board.is_game_over():
            return -999999.0 # Heavy penalty for losing the game
            
        # Call the neural/tabular state evaluation of N-Tuple
        return self.ntuple_evaluator.evaluate_state(board.grid)

    def _expectimax(self, board, depth, player_turn):
        """Recursive expectimax tree search utilizing N-Tuple evaluation at the leaves"""
        if depth == 0 or board.is_game_over():
            return self._evaluate_board(board)

        if player_turn:
            best_score = -float('inf')
            available_moves = board.get_available_moves()
            if not available_moves:
                return self._evaluate_board(board)

            for move in available_moves:
                temp_board = copy.deepcopy(board)
                temp_board.move(move)
                score = self._expectimax(temp_board, depth - 1, False)
                best_score = max(best_score, score)
            return best_score

        else:
            empty_cells = [(r, c) for r in range(4) for c in range(4) if board.grid[r][c] == 0]
            if not empty_cells:
                return self._evaluate_board(board)

            total_expected_score = 0
            p_2 = 0.9 / len(empty_cells)
            p_4 = 0.1 / len(empty_cells)

            for r, c in empty_cells:
                # Simulate spawn 2
                board_2 = copy.deepcopy(board)
                board_2.grid[r][c] = 2
                total_expected_score += p_2 * self._expectimax(board_2, depth - 1, True)

                # Simulate spawn 4
                board_4 = copy.deepcopy(board)
                board_4.grid[r][c] = 4
                total_expected_score += p_4 * self._expectimax(board_4, depth - 1, True)

            return total_expected_score

    def select_move(self, board):
        """Select the best move (0:L, 1:U, 2:R, 3:D) using expectimax combined with N-Tuple weights"""
        available_moves = board.get_available_moves()
        if not available_moves:
            return 0

        best_move = available_moves[0]
        best_score = -float('inf')

        for move in available_moves:
            temp_board = copy.deepcopy(board)
            temp_board.move(move)
            # Begin the lookahead tree search evaluating leaf nodes using N-Tuple LUT
            score = self._expectimax(temp_board, self.max_depth - 1, False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move