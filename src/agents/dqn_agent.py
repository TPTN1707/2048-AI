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
        # Input size is now 256 (16 cells * 16 one-hot representation states)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
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
        """Convert standard 4x4 grid into a flat 256-sized one-hot encoded vector"""
        flat_grid = np.array(grid).flatten()
        # Initialize a 16x16 one-hot matrix (16 cells, each has 16 possible power states)
        one_hot = np.zeros((16, 16), dtype=np.float32)
        
        for i, val in enumerate(flat_grid):
            # Map tile values to power exponents: 0->0, 2->1, 4->2, 8->3... 32768->15
            power = int(np.log2(val)) if val > 0 else 0
            power = min(power, 15) # Clip to max supported index 15
            one_hot[i, power] = 1.0
            
        return torch.tensor(one_hot.flatten(), dtype=torch.float32)

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