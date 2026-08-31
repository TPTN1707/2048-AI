import sys
import os
import numpy as np
import torch
from collections import Counter

# Resolve project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.board import Board2048
from src.agents.expectimax import ExpectimaxAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.ntuple_agent import NTupleAgent
from src.agents.hybrid_agent import HybridAgent

def evaluate_agent(agent_name="ntuple", num_games=10):
    """Run an automated silent evaluation of any of the 4 agents over multiple games"""
    print(f"\n--- Initiating Evaluation: {agent_name.upper()} AGENT over {num_games} games ---")
    
    board = Board2048()
    
    # 1. Initialize the selected agent with its corresponding weights
    if agent_name == "expectimax":
        # Pure Expectimax search with depth 3
        agent = ExpectimaxAgent(max_depth=3)
        
    elif agent_name == "dqn":
        # PyTorch Deep Q-Network with epsilon=0 (no random exploration)
        agent = DQNAgent(epsilon=0.0)
        weights_path = "checkpoints/dqn_2048.pth"
        if os.path.exists(weights_path):
            agent.policy_net.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            agent.policy_net.eval()
        else:
            print(f"Error: No trained DQN weights found at {weights_path}. Please train first.")
            return
            
    elif agent_name == "ntuple":
        # Tabular TD-Learning agent using lookup tables
        agent = NTupleAgent()
        weights_path = "checkpoints/ntuple_weights.npz"
        if os.path.exists(weights_path):
            agent.load_weights(weights_path)
        else:
            print(f"Error: No trained N-Tuple weights found at {weights_path}. Please train first.")
            return
            
    elif agent_name == "hybrid":
        # Super Hybrid combining Expectimax depth 3 with trained N-Tuple weights
        agent = HybridAgent(weights_path="checkpoints/ntuple_weights.npz", max_depth=3)
        
    else:
        print(f"Error: Unknown agent name '{agent_name}'. Choose from: expectimax, dqn, ntuple, hybrid.")
        return

    scores = []
    max_tiles = []

    # 2. Run the evaluation loop
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

    # 3. Calculate statistics
    avg_score = int(np.mean(scores))
    best_score = max(scores)
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
    # Change the agent_name below to evaluate different models:
    # Options: "expectimax", "dqn", "ntuple", "hybrid"
    # Number of games: 10-20 games is recommended for ntuple/dqn (fast), 3-5 games for expectimax/hybrid (slower)
    evaluate_agent(agent_name="dqn", num_games=10)