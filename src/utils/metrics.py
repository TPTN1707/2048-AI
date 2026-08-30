import os
import csv
import numpy as np
import matplotlib.pyplot as plt

class MetricsTracker:
    def __init__(self, csv_filepath="checkpoints/ntuple_metrics.csv"):
        self.csv_filepath = csv_filepath
        self.episodes = []
        self.scores = []
        self.max_tiles = []
        
        os.makedirs(os.path.dirname(self.csv_filepath), exist_ok=True)
        
        if not os.path.exists(self.csv_filepath):
            with open(self.csv_filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Score", "MaxTile"])

    def log_episode(self, episode, score, max_tile):
        """Append a single episode's metrics to local lists and save to CSV file"""
        self.episodes.append(episode)
        self.scores.append(score)
        self.max_tiles.append(max_tile)
        
        with open(self.csv_filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, score, max_tile])

    def generate_learning_plot(self, image_filepath="checkpoints/ntuple_training_plot.png"):
        """Generate and save a clean learning curve plot showing Score and Max Tile trends"""
        if not self.episodes:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Scores trend
        ax1.plot(self.episodes, self.scores, color='tab:blue', alpha=0.6, label='Episode Score')
        
        # Calculate and plot running average of 100 episodes to smooth out noise
        if len(self.scores) >= 100:
            running_avg = [np.mean(self.scores[max(0, idx-100):idx+1]) for idx in range(len(self.scores))]
            ax1.plot(self.episodes, running_avg, color='darkblue', linewidth=2, label='100-Ep Running Avg')
            
        ax1.set_title("2048 AI Training Progress - Scores")
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Score")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Plot 2: Max Tile achieved
        ax2.plot(self.episodes, self.max_tiles, color='tab:orange', alpha=0.8)
        ax2.set_title("Max Tile Reached")
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Tile Value")
        ax2.set_yscale('log', base=2) # Use logarithmic scale base 2
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(image_filepath, dpi=150)
        plt.close()
        print(f"--> Saved training progress plot to {image_filepath}")