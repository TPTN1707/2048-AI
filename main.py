import sys
import pygame
from src.gui.visualizer import GameVisualizer
from src.agents.expectimax import ExpectimaxAgent

def run_ai_game():
    """Main execution loop where the Expectimax AI plays 2048 automatically"""
    # Initialize the Pygame visualizer
    visualizer = GameVisualizer()
    
    # Initialize the Expectimax AI Agent (Depth 3 is a sweet spot for speed vs intelligence)
    ai_agent = ExpectimaxAgent(max_depth=3)
    
    # Mapping directions to text for terminal logging
    directions_map = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}
    
    running = True
    print("--- 2048 AI Gym: Expectimax Mode Activated ---")
    
    while running:
        # 1. Handle standard OS window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        # 2. If the game is not over, let the AI make a decision
        if not visualizer.board.is_game_over():
            # AI analyzes the board state and selects the best move
            best_move = ai_agent.select_move(visualizer.board)
            
            # Print decision to terminal for monitoring
            print(f"AI Decision: {directions_map[best_move]} | Current Score: {visualizer.board.score}")
            
            # Execute the move on the board
            visualizer.board.move(best_move)
            
            # Optional small delay (in milliseconds) to make the gameplay watchable
            pygame.time.delay(50) 
            
        # 3. Draw the updated board on Pygame window
        visualizer.draw_board()
        
        # Maintain frame rate
        visualizer.clock.tick(30)

if __name__ == "__main__":
    run_ai_game()