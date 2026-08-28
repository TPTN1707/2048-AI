# 2048-AI 🎮🤖

An AI system designed to play the game 2048. This repository implements and compares three distinct AI paradigms: a heuristic-based **Expectimax Search Agent**, a deep reinforcement learning **Double-DQN Agent** (PyTorch), and a state-of-the-art **N-Tuple Network TD-Learning Agent**.

Users can play manually, watch the search tree agent solve the grid at 4096+ capacity, or train reinforcement learning models from scratch.

---

## 🚀 Key Features

- **Robust Game Core (`src/engine/board.py`):** A zero-dependency 2048 game engine utilizing an elegant rotate-and-slide matrix formulation, verified by automated unit tests.
- **Expectimax Agent (`src/agents/expectimax.py`):** A game tree search agent modeling random environment spawns (probabilities of 2 and 4 tiles) combined with a corner-gradient heuristic matrix to push high tiles to the corner. Consistently achieves **4096** tiles.
- **Double-DQN Agent (`src/agents/dqn_agent.py`):** A deep reinforcement learning agent utilizing experience replay, target networks, and log2-scaled state representations.
- **N-Tuple TD-Learning Agent (`src/agents/ntuple_agent.py`):** A high-performance tabular reinforcement learning agent using Temporal Difference $TD(0)$ learning. Taps into **8-way board symmetries** (rotations/reflections) to accelerate learning. Consistently achieves **1024** tiles within minutes of CPU training.
- **Interactive GUI Visualizer (`src/gui/visualizer.py`):** A clean Pygame-based graphical interface used to play manually or watch the selected AI make decisions in real-time.

---

## 📁 Project Directory Structure

    2048-AI/
    ├── .gitignore                   # Excludes caches, venv, and model checkpoints
    ├── pyproject.toml               # Package management (managed by uv)
    ├── uv.lock                      # Locked dependency versions
    ├── main.py                      # Main entry point (Run, Train, or Watch AI)
    │
    └── src/                         # Source directory
        ├── engine/                  # Core 2048 game logic
        │   └── board.py             # Board matrix and merge logic
        ├── gui/                     # Pygame interface
        │   └── visualizer.py        # Grid drawing and manual game loop
        ├── agents/                  # AI brain modular implementations
        │   ├── base_agent.py        # Abstract base agent class
        │   ├── expectimax.py        # Heuristic search tree agent
        │   ├── dqn_agent.py         # PyTorch RL agent
        │   └── ntuple_agent.py      # High-speed N-Tuple TD-learning agent
        └── utils/
            └── metrics.py

## 🛠️ Tech Stack & Architecture

The 2048-AI project uses a highly optimized, modular architecture running on the following technology stack:

- **Frontend/GUI:** Pygame (clean 500x500 grid visualization with custom pastel color palettes).
- **Core AI Models:**
  - **Expectimax Agent:** Uses a recursive decision tree search combined with a corner-gradient evaluation matrix to maximize next-state values.
  - **Double-DQN Agent:** Implemented with PyTorch (`torch.nn`), learning via Experience Replay and Mean Squared Error (MSE) loss minimization.
  - **N-Tuple TD-Learning Agent:** Built using NumPy, implementing Temporal Difference $TD(0)$ learning with isomorphic state value lookups.

---

## 💻 Getting Started

### Prerequisites

To run this project, you need:
- Python 3.11 or higher.
- Astral's fast Python package installer `uv` (optional, but highly recommended).

### Installation

This project manages packages using `uv`.

1. Navigate to your project directory:

    cd 2048-AI

2. Sync and install all required dependencies (including PyTorch and Pygame):

    uv sync

### Running the Application

Open the `main.py` file in the root directory and configure the `MODE` variable near the top to one of the following options:

1. **To Watch the Expectimax Agent Play:**
   Set `MODE = "play_expectimax"` and run:

    uv run python main.py

2. **To Train the DQN Agent (No GUI):**
   Set `MODE = "train_dqn"` and run:

    uv run python main.py

3. **To Watch the Trained DQN Agent Play:**
   Set `MODE = "play_dqn"` (requires a trained checkpoint in `checkpoints/dqn_2048.pth`) and run:

    uv run python main.py

4. **To Train the N-Tuple Agent (No GUI - Ultra-fast):**
   Set `MODE = "train_ntuple"` and run:

    uv run python main.py

5. **To Watch the Trained N-Tuple Agent Play:**
   Set `MODE = "play_ntuple"` (requires a trained checkpoint in `checkpoints/ntuple_weights.npz`) and run:

    uv run python main.py