from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_move(self, board):
        """
        Given a Board2048 object, analyze the current grid and select the best move.
        Returns an integer representing the direction (0: L, 1: U, 2: R, 3: D).
        """
        pass