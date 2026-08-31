import sys
import os
import numpy as np
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.board import Board2048
from src.agents.ntuple_agent import NTupleAgent
from src.agents.hybrid_agent import HybridAgent

def evaluate_agent(agent_name="ntuple", num_games=10):
    """Run an automated silent evaluation of the selected agent over multiple games"""
    print(f"\n--- Initiating Evaluation: {agent_name.upper()} AGENT over {num_games} games ---")
    
    board = Board2048()
    
    if agent_name == "ntuple":
        agent = NTupleAgent()
        weights_path = "checkpoints/ntuple_weights.npz"
        if os.path.exists(weights_path):
            agent.load_weights(weights_path)
        else:
            print(f"Error: No trained weights found at {weights_path}")
            return
    elif agent_name == "hybrid":
        # Hybrid agent combines Expectimax depth 3 with N-Tuple weights
        agent = HybridAgent(weights_path="checkpoints/ntuple_weights.npz", max_depth=3)
    else:
        print("Unknown agent name.")
        return

    scores = []
    max_tiles = []

    for game_id in range(1, num_games + 1):
        board.reset()
        done = False
        
        while not done:
            action = agent.select_move(board)
            board.move(action)
            done = board.is_game_over()
            
        final_score = board.score
        final_max_tile = int(np.max(board.grid))
        
        scores.append(final_score)
        max_tiles.append(final_max_tile)
        
        print(f" Game {game_id:2d}/{num_games:2d} | Score: {final_score:5d} | Max Tile: {final_max_tile:4d}")

    # Calculate statistics
    avg_score = int(np.mean(scores))
    best_score = max(scores)
    
    # Calculate tile distribution percentage
    tile_counts = Counter(max_tiles)
    
    print("\n================ EVALUATION REPORT ================")
    print(f"Agent Tested   : {agent_name.upper()}")
    print(f"Total Games    : {num_games}")
    print(f"Average Score  : {avg_score}")
    print(f"Best Score     : {best_score}")
    print("---------------------------------------------------")
    print("Max Tile Distribution Rate:")
    for tile in sorted(tile_counts.keys()):
        percentage = (tile_counts[tile] / num_games) * 100
        print(f" - Tile {tile:4d}: {percentage:5.1f}% ({tile_counts[tile]}/{num_games} games)")
    print("===================================================\n")

if __name__ == "__main__":
    # Choose which agent to evaluate: "ntuple" (fast) or "hybrid" (slower but master-level)
    # Let's run a quick 10-game evaluation on N-Tuple
    evaluate_agent(agent_name="ntuple", num_games=10)