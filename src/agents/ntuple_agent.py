import sys
import os
import random
import numpy as np
import copy

# Resolve project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.agents.base_agent import BaseAgent

# We expand BASE_PATTERNS from 2 to 4 distinct 6-tuple shapes
# This covers corners, outer lines, inner blocks, and zigzag configurations
BASE_PATTERNS = [
    [0, 1, 2, 4, 5, 6],    # Pattern 1: Corner 2x3 block
    [0, 1, 2, 3, 4, 5],    # Pattern 2: Horizontal outer line + adjacent
    [1, 2, 3, 5, 6, 7],    # Pattern 3: Shifted/Inner 2x3 block
    [0, 1, 4, 8, 12, 13]   # Pattern 4: L-shape/Zigzag around the corner
]

class NTupleAgent(BaseAgent):
    def __init__(self, lr=0.1, gamma=0.99):
        self.lr = lr # Learning rate (alpha)
        self.gamma = gamma # Discount factor
        
        # 16^6 = 16,777,216 possible states per 6-tuple LUT (takes ~67MB of RAM each)
        self.lut_size = 16**6 
        
        # Initialize a Lookup Table (LUT) dynamically for each pattern with zeros
        self.luts = [np.zeros(self.lut_size, dtype=np.float32) for _ in BASE_PATTERNS]
        
        # Precompute the 8 symmetric transformations for all patterns
        self.symmetric_tuples = self._precompute_symmetries()

    def _get_symmetric_indices(self, flat_index, transform_id):
        """Map a flat index (0-15) to its symmetric counterpart on a 4x4 grid"""
        r, c = flat_index // 4, flat_index % 4
        for _ in range(transform_id % 4):
            r, c = c, 3 - r
        if transform_id >= 4:
            c = 3 - c
        return r * 4 + c

    def _precompute_symmetries(self):
        """Precompute the 8 isomorphic structures for each base pattern"""
        all_symmetries = []
        for pattern in BASE_PATTERNS:
            pattern_symmetries = []
            for sym_id in range(8):
                sym_tuple = [self._get_symmetric_indices(idx, sym_id) for idx in pattern]
                pattern_symmetries.append(sym_tuple)
            all_symmetries.append(pattern_symmetries)
        return all_symmetries

    def _get_state_feature_indices(self, grid):
        """Convert board state to log2 exponents, then calculate LUT indices for all active tuples"""
        flat_grid = np.array(grid).flatten()
        with np.errstate(divide='ignore'):
            exponents = np.where(flat_grid > 0, np.log2(flat_grid), 0.0).astype(int)
        
        lut_indices = []
        for pattern_id, symmetries in enumerate(self.symmetric_tuples):
            pattern_indices = []
            for sym_tuple in symmetries:
                index = 0
                for k, cell_idx in enumerate(sym_tuple):
                    index += exponents[cell_idx] * (16**k)
                pattern_indices.append(index)
            lut_indices.append(pattern_indices)
        return lut_indices

    def evaluate_state(self, grid):
        """Evaluate the total state value V(s) by summing values across all LUT symmetries"""
        lut_indices = self._get_state_feature_indices(grid)
        total_value = 0.0
        for pattern_id, indices in enumerate(lut_indices):
            for idx in indices:
                total_value += self.luts[pattern_id][idx]
        return total_value

    def select_move(self, board):
        """Select the best move (0:L, 1:U, 2:R, 3:D) that maximizes next state value V(s')"""
        available_moves = board.get_available_moves()
        if not available_moves:
            return 0

        best_move = available_moves[0]
        best_value = -float('inf')

        for move in available_moves:
            temp_board = copy.deepcopy(board)
            temp_board.move(move)
            val = self.evaluate_state(temp_board.grid)
            if val > best_value:
                best_value = val
                best_move = move

        return best_move

    def learn(self, state, reward, next_state, done):
        """Apply TD(0) learning rule to update all active LUT cells"""
        v_s = self.evaluate_state(state)
        v_s_prime = 0.0 if done else self.evaluate_state(next_state)
        
        td_target = reward + self.gamma * v_s_prime
        td_error = td_target - v_s
        
        state_indices = self._get_state_feature_indices(state)
        
        # Total active features to update = 4 base patterns * 8 symmetries = 32
        total_features = len(BASE_PATTERNS) * 8
        update_step = (self.lr * td_error) / total_features

        for pattern_id, indices in enumerate(state_indices):
            for idx in indices:
                self.luts[pattern_id][idx] += update_step

    def save_weights(self, filepath):
        """Save all Lookup Tables (LUTs) dynamically into a compressed .npz file"""
        save_dict = {f"lut{i}": lut for i, lut in enumerate(self.luts)}
        np.savez_compressed(filepath, **save_dict)

    def load_weights(self, filepath):
        """Load all Lookup Tables (LUTs) dynamically from a compressed .npz file"""
        data = np.load(filepath)
        self.luts = [data[f"lut{i}"] for i in range(len(BASE_PATTERNS))]