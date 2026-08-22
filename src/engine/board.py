import random
import copy

class Board2048:
    def __init__(self):
        self.grid = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.reset()

    def reset(self):
        """Reset the grid to empty, set score to 0, and spawn 2 initial tiles"""
        self.grid = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.spawn_tile()
        self.spawn_tile()
        return self.grid

    def spawn_tile(self):
        """Spawn a new tile (90% chance of 2, 10% chance of 4) in a random empty cell"""
        empty_cells = [(r, c) for r in range(4) for c in range(4) if self.grid[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.grid[r][c] = 2 if random.random() < 0.9 else 4

    def rotate_clockwise(self):
        """Rotate the 4x4 grid 90 degrees clockwise"""
        self.grid = [list(x) for x in zip(*self.grid[::-1])]

    def _slide_left_row(self, row):
        """Slide and merge a single row of size 4 to the left. Returns (new_row, score_gained)"""
        # Compress non-zero elements to the left
        non_zeros = [num for num in row if num != 0]
        new_row = []
        score_gained = 0
        skip = False

        # Merge adjacent identical numbers
        for i in range(len(non_zeros)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zeros) and non_zeros[i] == non_zeros[i+1]:
                merged_val = non_zeros[i] * 2
                new_row.append(merged_val)
                score_gained += merged_val
                skip = True
            else:
                new_row.append(non_zeros[i])

        # Fill remaining spots with zeros
        while len(new_row) < 4:
            new_row.append(0)

        return new_row, score_gained

    def move(self, direction):
        """
        Execute a move in one of 4 directions:
        0: Left, 1: Up, 2: Right, 3: Down
        Returns True if the board changed, False otherwise.
        """
        old_grid = copy.deepcopy(self.grid)
        score_gained = 0

        # Implement all 4 directions by rotating, sliding left, and rotating back
        if direction == 0:  # Left
            for r in range(4):
                self.grid[r], gain = self._slide_left_row(self.grid[r])
                score_gained += gain
        elif direction == 1:  # Up
            self.rotate_clockwise()
            self.rotate_clockwise()
            self.rotate_clockwise()
            for r in range(4):
                self.grid[r], gain = self._slide_left_row(self.grid[r])
                score_gained += gain
            self.rotate_clockwise()
        elif direction == 2:  # Right
            for r in range(4):
                self.grid[r] = self.grid[r][::-1]
                self.grid[r], gain = self._slide_left_row(self.grid[r])
                score_gained += gain
                self.grid[r] = self.grid[r][::-1]
        elif direction == 3:  # Down
            self.rotate_clockwise()
            for r in range(4):
                self.grid[r], gain = self._slide_left_row(self.grid[r])
                score_gained += gain
            self.rotate_clockwise()
            self.rotate_clockwise()
            self.rotate_clockwise()

        # If the move changed the board, update score and spawn a new tile
        if self.grid != old_grid:
            self.score += score_gained
            self.spawn_tile()
            return True
        return False

    def get_available_moves(self):
        """Return a list of valid moves (0, 1, 2, 3) that will actually change the board"""
        available = []
        for direction in range(4):
            temp_board = copy.deepcopy(self)
            if temp_board.move(direction):
                available.append(direction)
        return available

    def is_game_over(self):
        """Game is over if there are no empty cells and no valid moves left"""
        if any(0 in row for row in self.grid):
            return False
        if self.get_available_moves():
            return False
        return True

    def print_board(self):
        """Print the board cleanly to the terminal for debugging"""
        print("-" * 21)
        for row in self.grid:
            print("|" + "|".join(f"{num:^4}" if num != 0 else "    " for num in row) + "|")
        print("-" * 21)
        print(f"Score: {self.score}\n")