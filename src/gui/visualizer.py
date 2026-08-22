# src/gui/visualizer.py
import sys
import os
# Automatically find the project root directory and add it to Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pygame
from src.engine.board import Board2048

# Colors palette matching the original 2048 game
BG_COLOR = (250, 248, 239)
GRID_COLOR = (187, 173, 160)
EMPTY_CELL_COLOR = (205, 193, 180)

TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}

TEXT_COLORS = {
    2: (119, 110, 101),
    4: (119, 110, 101)
}
DEFAULT_TEXT_COLOR = (249, 246, 242)

WINDOW_SIZE = 500
GRID_SIZE = 400
CELL_SIZE = 85
PADDING = 12

class GameVisualizer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI - 2048")
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        self.board = Board2048()

        # Initialize fonts safely
        self.font_score = pygame.font.SysFont("arial", 24, bold=True)
        self.font_tile = pygame.font.SysFont("arial", 36, bold=True)
        self.font_gameover = pygame.font.SysFont("arial", 48, bold=True)

    def draw_board(self):
        """Render the 2048 board, tiles, score, and game over state"""
        self.screen.fill(BG_COLOR)

        # Draw header / score board
        score_text = self.font_score.render(f"Score: {self.board.score}", True, (119, 110, 101))
        self.screen.blit(score_text, (PADDING + 5, 20))

        # Draw main grid container
        grid_rect = pygame.Rect(PADDING, 80, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, GRID_COLOR, grid_rect, border_radius=8)

        # Draw cells and tiles
        for r in range(4):
            for c in range(4):
                val = self.board.grid[r][c]
                # Calculate coordinates for each tile
                x = PADDING + PADDING + c * (CELL_SIZE + PADDING)
                y = 80 + PADDING + r * (CELL_SIZE + PADDING)
                tile_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                # Select background color based on tile value
                bg_color = TILE_COLORS.get(val, (60, 58, 50)) if val != 0 else EMPTY_CELL_COLOR
                pygame.draw.rect(self.screen, bg_color, tile_rect, border_radius=6)

                # Render tile number inside the cell
                if val != 0:
                    text_color = TEXT_COLORS.get(val, DEFAULT_TEXT_COLOR)
                    text_surface = self.font_tile.render(str(val), True, text_color)
                    text_rect = text_surface.get_rect(center=tile_rect.center)
                    self.screen.blit(text_surface, text_rect)

        # Draw translucent overlay if the game is over
        if self.board.is_game_over():
            overlay = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
            overlay.fill((238, 228, 218, 180)) # Translucent pastel overlay
            self.screen.blit(overlay, (PADDING, 80))

            gameover_text = self.font_gameover.render("Game Over!", True, (119, 110, 101))
            text_rect = gameover_text.get_rect(center=grid_rect.center)
            self.screen.blit(gameover_text, text_rect)

        pygame.display.flip()

    def run_manual_game(self):
        """Infinite loop to run the game manually using keyboard arrow keys"""
        running = True
        while running:
            self.draw_board()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

                # Handle manual keyboard controls
                elif event.type == pygame.KEYDOWN:
                    if self.board.is_game_over():
                        # Press R key to restart when game is over
                        if event.key == pygame.K_r:
                            self.board.reset()
                        continue

                    # Map arrow keys to game directions (0:L, 1:U, 2:R, 3:D)
                    if event.key == pygame.K_LEFT:
                        self.board.move(0)
                    elif event.key == pygame.K_UP:
                        self.board.move(1)
                    elif event.key == pygame.K_RIGHT:
                        self.board.move(2)
                    elif event.key == pygame.K_DOWN:
                        self.board.move(3)

            self.clock.tick(30) # Keep app running at 30 frames per second

if __name__ == "__main__":
    visualizer = GameVisualizer()
    visualizer.run_manual_game()