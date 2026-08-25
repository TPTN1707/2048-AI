# 2048 AI Trainer 🎮🤖

A professional Python-based AI trainer for the game 2048. This repository contains a fully automated modular framework implementing two distinct AI approaches: a heuristic-based **Expectimax Search Agent** and a deep reinforcement learning **Double-DQN Agent** built with PyTorch.

Users can play the game manually, observe the Expectimax agent solve the grid at 4096+ tile capacity, or train a neural network from scratch using Deep Q-Learning.

---

## 🚀 Key Features

- **Robust Game Core (`src/engine/board.py`):** A zero-dependency 2048 game engine utilizing an elegant rotate-and-slide matrix formulation, verified by automated unit tests.
- **Expectimax Agent (`src/agents/expectimax.py`):** An advanced game tree search agent modeling random environment spawns (probabilities of 2 and 4 tiles) combined with a corner-gradient heuristic matrix to push high tiles to the corner.
- **Double-DQN Agent (`src/agents/dqn_agent.py`):** A deep reinforcement learning agent utilizing experience replay and target networks to learn optimal game-playing policies through reward feedback.
- **Interactive GUI Visualizer (`src/gui/visualizer.py`):** A clean Pygame-based graphical interface used to play manually or watch the AI make decisions in real-time.

---

## 📁 Project Directory Structure

    2048-ai-gym/
    ├── .gitignore                   # Excludes caches, venv, and PyTorch model checkpoints
    ├── pyproject.toml               # Package management (managed by uv)
    ├── uv.lock                      # Locked dependency versions
    ├── main.py                      # Main entry point (Run, Train, or Watch AI)
    │
    ├── src/                         # Source directory
    │   ├── engine/                  # Core 2048 game logic
    │   │   └── board.py
    │   ├── gui/                     # Pygame interface
    │   │   └── visualizer.py
    │   ├── agents/                  # AI brain modular implementations
    │   │   ├── base_agent.py        # Abstract base agent class
    │   │   ├── expectimax.py        # Heuristic search tree agent
    │   │   └── dqn_agent.py         # PyTorch RL agent
    │   └── utils/
    │       └── metrics.py
    │
    └── tests/                       # Automated unit tests
        └── test_board.py            # Verification tests for grid merge logic

---

## 💻 Getting Started

### Installation

This project manages packages using Astral's fast Python package installer, `uv`.

1. Clone or download this repository, navigate to the directory:

    cd 2048-ai-gym

2. Install dependencies (including PyTorch and Pygame):

    uv sync

### How to Run

Open the `main.py` file and configure the `MODE` variable near the top to one of the following options:

1. **To Watch the Expectimax Agent Play (GUI):**
   Set `MODE = "play_expectimax"` and run:

    uv run python main.py

2. **To Train the DQN Agent (No GUI - Maximum Speed):**
   Set `MODE = "train_dqn"` and run:

    uv run python main.py

3. **To Watch the Trained DQN Agent Play (GUI):**
   Set `MODE = "play_dqn"` (requires a trained checkpoint in `checkpoints/dqn_2048.pth`) and run:

    uv run python main.py