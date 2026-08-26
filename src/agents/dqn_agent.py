import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Resolve project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.agents.base_agent import BaseAgent

# 1. Define the Deep Q-Network (Neural Network Architecture)
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # Input size: 16 flat features representing the 4x4 grid
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        # Output size: 4 actions representing (0:L, 1:U, 2:R, 3:D)
        self.out = nn.Linear(128, 4)

    def forward(self, x):
        """Forward pass through the neural network layers"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)


# 2. Implement the DQN Agent
class DQNAgent(BaseAgent):
    def __init__(self, memory_size=20000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01, lr=0.0005):
        self.batch_size = batch_size
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Experience Replay Buffer using a double-ended queue
        self.memory = deque(maxlen=memory_size)
        
        # Initialize Primary and Target networks for Double-DQN stability
        self.policy_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is kept in evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def _normalize_state(self, grid):
        """Convert standard 2048 grid values to log2 scaled flat array of size 16"""
        flat_grid = np.array(grid).flatten()
        # Suppress the divide by zero warning during log2(0) calculation
        with np.errstate(divide='ignore'):
            normalized = np.where(flat_grid > 0, np.log2(flat_grid), 0.0)
        return torch.tensor(normalized, dtype=torch.float32)

    def select_move(self, board):
        """Select a move using an epsilon-greedy strategy"""
        available_moves = board.get_available_moves()
        if not available_moves:
            return 0 # Fallback

        # Epsilon-greedy: Exploration vs Exploitation
        if random.random() < self.epsilon:
            # Explore: pick a random valid move
            return random.choice(available_moves)
        else:
            # Exploit: pick the valid move with the highest predicted Q-value
            state = self._normalize_state(board.grid).unsqueeze(0) # Add batch dimension
            with torch.no_grad():
                q_values = self.policy_net(state).squeeze(0) # Get prediction
                
            # Filter and select only valid moves from predictions
            valid_q_values = {move: q_values[move].item() for move in available_moves}
            best_move = max(valid_q_values, key=valid_q_values.get)
            return best_move

    def remember(self, state, action, reward, next_state, done):
        """Store transitions in the experience replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_step(self):
        """Train the neural network using a random batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return

        # Sample a random batch of transitions
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to PyTorch Tensors
        states = torch.stack([self._normalize_state(s) for s in states])
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([self._normalize_state(ns) for ns in next_states])
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Calculate current Q-values predicted by the policy network
        current_q = self.policy_net(states).gather(1, actions)

        # Calculate target Q-values using Double-DQN formulation
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Optimize the loss
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon (reduce random exploration rate over time)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay