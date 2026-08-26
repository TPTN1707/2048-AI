import sys
import os
import pygame
import torch
import numpy as np

os.makedirs("checkpoints", exist_ok=True)

from src.engine.board import Board2048
from src.gui.visualizer import GameVisualizer
from src.agents.expectimax import ExpectimaxAgent
from src.agents.dqn_agent import DQNAgent

# Choose your execution mode:
# - "play_expectimax": Play with GUI using Expectimax Search Agent
# - "train_dqn": Train the Deep Q-Network Agent (No GUI, maximum speed)
# - "play_dqn": Play with GUI using the trained DQN Agent weights
MODE = "play_dqn" 

MODEL_PATH = "checkpoints/dqn_2048.pth"

# main.py (Chỉ cập nhật hàm train_dqn_agent)

def train_dqn_agent(episodes=2000):
    """Train the DQN Agent over multiple episodes with simplified soft rewards"""
    board = Board2048()
    agent = DQNAgent()
    
    print(f"--- Starting DQN Training with Simplified Soft Rewards for {episodes} episodes ---")
    
    for ep in range(1, episodes + 1):
        board.reset()
        state = board.grid
        done = False
        
        while not done:
            # 1. Select action using epsilon-greedy policy
            action = agent.select_move(board)
            
            # 2. Execute action
            prev_score = board.score
            moved = board.move(action)
            next_state = board.grid
            
            # 3. Simplified Soft Reward Design
            # Base Reward: Actual score gained from merges
            reward = board.score - prev_score
            
            # Gentle penalty if AI chose an invalid move
            if not moved:
                reward = -10
            else:
                # Soft encouragement for keeping empty cells open (prevents clogging)
                empty_cells = sum(row.count(0) for row in board.grid)
                reward += empty_cells * 1.0 # Very soft weight
            
            done = board.is_game_over()
            if done:
                reward = -200 # Softened penalty for losing

            # 4. Store transition in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # 5. Train the network weights
            agent.train_step()
            
            # Update current state
            state = next_state

        # Sync target network
        if ep % 10 == 0:
            agent.update_target_network()

        # Log training progress every 10 episodes
        if ep % 10 == 0:
            max_tile = np.max(board.grid)
            print(f"Episode {ep:4d}/{episodes} | Score: {board.score:5d} | Max Tile: {max_tile:4d} | Epsilon: {agent.epsilon:.4f} | Memory: {len(agent.memory)}")

        # Periodically save checkpoints every 100 episodes
        if ep % 100 == 0:
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)
            print(f"--> Saved checkpoint to {MODEL_PATH} at episode {ep}")

    print("--- Training Complete ---")

def play_expectimax():
    """Play 2048 with Pygame GUI using Expectimax Agent"""
    visualizer = GameVisualizer()
    ai_agent = ExpectimaxAgent(max_depth=3)
    directions_map = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}
    
    print("--- Running Expectimax Agent with GUI ---")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        if not visualizer.board.is_game_over():
            best_move = ai_agent.select_move(visualizer.board)
            print(f"AI Decision: {directions_map[best_move]} | Score: {visualizer.board.score}")
            visualizer.board.move(best_move)
            pygame.time.delay(50) 
            
        visualizer.draw_board()
        visualizer.clock.tick(30)


def play_dqn():
    """Load trained DQN weights and play 2048 with Pygame GUI"""
    visualizer = GameVisualizer()
    ai_agent = DQNAgent(epsilon=0.0) # Set epsilon to 0 to force 100% exploitation (no random moves)
    
    # Check if a trained model checkpoint exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No trained model checkpoint found at {MODEL_PATH}. Please train the model first using MODE = 'train_dqn'.")
        pygame.quit()
        return

    # Load the trained weights
    print(f"Loading trained DQN model weights from {MODEL_PATH}...")
    ai_agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    ai_agent.policy_net.eval()

    directions_map = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}
    
    print("--- Running Trained DQN Agent with GUI ---")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        if not visualizer.board.is_game_over():
            best_move = ai_agent.select_move(visualizer.board)
            print(f"DQN Decision: {directions_map[best_move]} | Score: {visualizer.board.score}")
            visualizer.board.move(best_move)
            pygame.time.delay(100) # Slightly slower delay to observe the network's behavior
            
        visualizer.draw_board()
        visualizer.clock.tick(30)


if __name__ == "__main__":
    if MODE == "train_dqn":
        train_dqn_agent()
    elif MODE == "play_dqn":
        play_dqn()
    elif MODE == "play_expectimax":
        play_expectimax()