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
from src.agents.ntuple_agent import NTupleAgent

# Execution modes:
# - "play_expectimax": Play with GUI using Expectimax Agent
# - "train_dqn": Train the DQN Agent (No GUI)
# - "play_dqn": Play with GUI using trained DQN Agent
# - "train_ntuple": Train the N-Tuple Agent (No GUI, ultra high-speed)
# - "play_ntuple": Play with GUI using trained N-Tuple Agent
MODE = "train_ntuple" 

DQN_MODEL_PATH = "checkpoints/dqn_2048.pth"
NTUPLE_MODEL_PATH = "checkpoints/ntuple_weights.npz"

def train_ntuple_agent(episodes=50000):
    """Train the N-Tuple TD-Learning agent over thousands of episodes at ultra-high speed"""
    board = Board2048()
    agent = NTupleAgent(lr=0.1, gamma=0.99)
    
    print(f"--- Starting N-Tuple TD-Learning Training for {episodes} episodes ---")
    
    for ep in range(1, episodes + 1):
        board.reset()
        state = board.grid
        done = False
        
        while not done:
            # 1. Select best move based on current LUT state values
            action = agent.select_move(board)
            
            # 2. Execute move
            prev_score = board.score
            moved = board.move(action)
            next_state = board.grid
            
            # 3. Reward is the exact score gained from merges
            reward = board.score - prev_score
            
            # 4. Perform TD(0) learning update
            agent.learn(state, reward, next_state, board.is_game_over())
            
            state = next_state
            done = board.is_game_over()

        # Log training progress every 1000 episodes
        if ep % 1000 == 0:
            max_tile = np.max(board.grid)
            print(f"Episode {ep:5d}/{episodes} | Score: {board.score:5d} | Max Tile: {max_tile:5d}")

        # Save weights every 5000 episodes
        if ep % 5000 == 0:
            agent.save_weights(NTUPLE_MODEL_PATH)
            print(f"--> Saved N-Tuple weights to {NTUPLE_MODEL_PATH}")

    print("--- N-Tuple Training Complete ---")


def play_ntuple():
    """Play 2048 with Pygame GUI using the trained N-Tuple Agent"""
    visualizer = GameVisualizer()
    ai_agent = NTupleAgent()
    
    if not os.path.exists(NTUPLE_MODEL_PATH):
        print(f"Error: No weights found at {NTUPLE_MODEL_PATH}. Please train the model first.")
        pygame.quit()
        return

    # Load weights
    print(f"Loading N-Tuple weights from {NTUPLE_MODEL_PATH}...")
    ai_agent.load_weights(NTUPLE_MODEL_PATH)

    directions_map = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}
    
    print("--- Running Trained N-Tuple Agent with GUI ---")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        if not visualizer.board.is_game_over():
            best_move = ai_agent.select_move(visualizer.board)
            print(f"N-Tuple Decision: {directions_map[best_move]} | Score: {visualizer.board.score}")
            visualizer.board.move(best_move)
            pygame.time.delay(50) 
            
        visualizer.draw_board()
        visualizer.clock.tick(30)


def train_dqn_agent(episodes=2000):
    board = Board2048()
    agent = DQNAgent()
    print(f"--- Starting DQN Training with Simplified Soft Rewards for {episodes} episodes ---")
    for ep in range(1, episodes + 1):
        board.reset()
        state = board.grid
        done = False
        while not done:
            action = agent.select_move(board)
            prev_score = board.score
            moved = board.move(action)
            next_state = board.grid
            reward = board.score - prev_score
            if not moved:
                reward = -10
            else:
                empty_cells = sum(row.count(0) for row in board.grid)
                reward += empty_cells * 1.0
            done = board.is_game_over()
            if done:
                reward = -200
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
        if ep % 10 == 0:
            agent.update_target_network()
        if ep % 10 == 0:
            max_tile = np.max(board.grid)
            print(f"Episode {ep:4d}/{episodes} | Score: {board.score:5d} | Max Tile: {max_tile:4d} | Epsilon: {agent.epsilon:.4f} | Memory: {len(agent.memory)}")
        if ep % 100 == 0:
            torch.save(agent.policy_net.state_dict(), DQN_MODEL_PATH)
            print(f"--> Saved checkpoint to {DQN_MODEL_PATH} at episode {ep}")
    print("--- Training Complete ---")

def play_expectimax():
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
    visualizer = GameVisualizer()
    ai_agent = DQNAgent(epsilon=0.0)
    if not os.path.exists(DQN_MODEL_PATH):
        print(f"Error: No trained model checkpoint found at {DQN_MODEL_PATH}.")
        pygame.quit()
        return
    ai_agent.policy_net.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=torch.device('cpu')))
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
            pygame.time.delay(100)
        visualizer.draw_board()
        visualizer.clock.tick(30)


if __name__ == "__main__":
    if MODE == "train_ntuple":
        train_ntuple_agent()
    elif MODE == "play_ntuple":
        play_ntuple()
    elif MODE == "train_dqn":
        train_dqn_agent()
    elif MODE == "play_dqn":
        play_dqn()
    elif MODE == "play_expectimax":
        play_expectimax()