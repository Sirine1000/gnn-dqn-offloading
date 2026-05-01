"""
Main entry point for the GNN-DQN Task Offloading project.

This project implements AI-based task offloading in Edge-Fog-Cloud environments
using:
- Graph Neural Networks (GNN) for topology representation
- Deep Q-Network (DQN) for offloading decision-making
- RayCloudSim for simulation and evaluation
"""

import argparse
import os
import subprocess
import sys


PROJECT_NAME = "GNN-DQN Task Offloading in Edge-Fog-Cloud"


NOTEBOOKS = [
    "Preprocessing Dataset.ipynb",
    "GNN Graph.ipynb",
    "Integrate GNN with DQN.ipynb",
    "Simulation.ipynb",
    "1000T_NEW_4_TRAINING_GNN_DQN.ipynb",
]


def show_project_info():
    print("=" * 70)
    print(PROJECT_NAME)
    print("=" * 70)

    print("\nProject description:")
    print("- AI-based task offloading for IoT applications")
    print("- Edge-Fog-Cloud infrastructure modeled as a graph")
    print("- GNN used to generate node and graph embeddings")
    print("- DQN used to learn the best offloading decision")
    print("- RayCloudSim used for simulation and evaluation")

    print("\nMain notebooks:")
    for i, notebook in enumerate(NOTEBOOKS, start=1):
        print(f"{i}. {notebook}")

    print("\nInstallation:")
    print("pip install -r requirements.txt")

    print("\nExample usage:")
    print("python main.py --mode info")
    print("python main.py --mode train")


def run_training_script():
    training_script = "GNN_DQN_Training.py"

    if not os.path.exists(training_script):
        print(f"Training script not found: {training_script}")
        print("Please make sure GNN_DQN_Training.py exists in the project folder.")
        return

    print("Starting GNN-DQN training...")
    subprocess.run([sys.executable, training_script], check=True)


def list_notebooks():
    print("Available notebooks:")

    for notebook in NOTEBOOKS:
        if os.path.exists(notebook):
            print(f"[OK] {notebook}")
        else:
            print(f"[MISSING] {notebook}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the GNN-DQN Task Offloading project"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="info",
        choices=["info", "train", "notebooks"],
        help="Execution mode: info, train, or notebooks",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "info":
        show_project_info()

    elif args.mode == "train":
        run_training_script()

    elif args.mode == "notebooks":
        list_notebooks()


if __name__ == "__main__":
    main()
