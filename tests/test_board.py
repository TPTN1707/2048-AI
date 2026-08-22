import unittest
from src.engine.board import Board2048

class TestBoard2048(unittest.TestCase):
    def setUp(self):
        """Set up a fresh board before each test case"""
        self.board = Board2048()

    def test_slide_left_row_simple_merge(self):
        """Test simple merge: [2, 2, 0, 0] -> [4, 0, 0, 0]"""
        row, score = self.board._slide_left_row([2, 2, 0, 0])
        self.assertEqual(row, [4, 0, 0, 0])
        self.assertEqual(score, 4)

    def test_slide_left_row_no_merge(self):
        """Test row with no possible merges: [2, 4, 8, 16] -> [2, 4, 8, 16]"""
        row, score = self.board._slide_left_row([2, 4, 8, 16])
        self.assertEqual(row, [2, 4, 8, 16])
        self.assertEqual(score, 0)

    def test_slide_left_row_triple_merge(self):
        """Test triple merge prioritizing left: [2, 2, 2, 0] -> [4, 2, 0, 0]"""
        row, score = self.board._slide_left_row([2, 2, 2, 0])
        self.assertEqual(row, [4, 2, 0, 0])
        self.assertEqual(score, 4)

    def test_slide_left_row_quadruple_merge(self):
        """Test quadruple merge: [2, 2, 2, 2] -> [4, 4, 0, 0]"""
        row, score = self.board._slide_left_row([2, 2, 2, 2])
        self.assertEqual(row, [4, 4, 0, 0])
        self.assertEqual(score, 8)

    def test_slide_left_row_spaced_merge(self):
        """Test merge with empty spaces in between: [2, 0, 2, 0] -> [4, 0, 0, 0]"""
        row, score = self.board._slide_left_row([2, 0, 2, 0])
        self.assertEqual(row, [4, 0, 0, 0])
        self.assertEqual(score, 4)

    def test_slide_left_row_prevent_double_merge(self):
        """Test that merged tiles cannot merge again in the same turn: [4, 2, 2, 0] -> [4, 4, 0, 0] (no [8, 0, 0, 0])"""
        row, score = self.board._slide_left_row([4, 2, 2, 0])
        self.assertEqual(row, [4, 4, 0, 0])
        self.assertEqual(score, 4)

    def test_is_game_over_not_over(self):
        """Test board is not over when moves are still possible"""
        self.board.grid = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 0] # One empty spot remaining
        ]
        self.assertFalse(self.board.is_game_over())

    def test_is_game_over_is_over(self):
        """Test board is correctly identified as over when no moves/merges are possible"""
        self.board.grid = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2] # No empty spots, no adjacent equal numbers
        ]
        self.assertTrue(self.board.is_game_over())

if __name__ == "__main__":
    unittest.main()