# Task Offloading in Edge-Fog-Cloud Environment for IoT applications
# Requirements & Installation
# Main Dependent Modules:

    python >= 3.8: Previous versions might be OK but without testing.
    networkx: NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
    simpy: SimPy is a process-based discrete-event simulation framework based on standard Python.
    numpy: NumPy is a Python library used for working with arrays.
    pandas: Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool.


Users are recommended to use Anaconda to configure RayCloudSim:

```bash
conda create --name raycloudsim python=3.12
conda activate raycloudsim
pip install -r requirements.txt
```

# AI-Based Task Offloading using GNN-DQN in Edge–Fog–Cloud Computing

## Overview

This project proposes a **hybrid Graph Neural Network (GNN) and Deep Q-Network (DQN)** framework for intelligent task offloading in **Edge–Fog–Cloud environments**.

The objective is to optimize **Quality of Service (QoS)** for IoT applications by minimizing:

* Latency
* Energy consumption

The approach combines:

* **GNN** → to capture infrastructure topology
* **DQN** → to learn optimal offloading decisions

---

## System Architecture

Nodes: Edge, Fog, Cloud
Edges: Network links (bandwidth-aware)
Simulator: RayCloudSim (discrete-event simulation)

The infrastructure is modeled as a **graph**:

* Nodes = computing resources
* Edges = communication links

---

## Methodology

### Graph Modeling

* Graph built from a JSON scenario file
* Adjacency matrix normalized (GCN-style)
* Self-loops added

**Node features (9 total):**

**Dynamic:**

* Available CPU (normalized)
* Available buffer (normalized)

**Static:**

* Max CPU
* Max buffer
* Idle energy coefficient
* Execution energy coefficient
* Node type (Edge / Fog / Cloud → one-hot)

Node features are dynamically updated during simulation.

---

### GNN Encoder

* Model: Graph Convolutional Network (GCN)
* Layers: 3 GraphConv layers
* Hidden dimension: 64
* Activation: ReLU + Dropout (0.1)

Outputs:

* Node embeddings
* Graph embedding (mean pooling)

Key idea:

GNN embeddings are recomputed dynamically for each task to capture context

---

### DQN Agent

**State representation:**

```
[node_embedding | graph_embedding | task_features]
```

**Architecture:**

* Fully connected network: 256 → 128 → 1 (Q-value per node)

**Training details:**

* Loss: SmoothL1Loss (Huber)
* Optimizer: Adam (lr = 3e-4)
* Discount factor: γ = 0.99

**Reinforcement Learning techniques:**

* Experience Replay (capacity = 20,000)
* Target Network (updated every 50 steps)
* ε-greedy exploration

---

### Reward Function

The reward is defined as:

```
R = - (α × Latency + β × Energy)
```

Optional:

* Deadline penalty
* Failure penalty

A normalized version can be used for better stability.

---

## Installation

### Requirements

* Python ≥ 3.8
* numpy
* pandas
* torch
* networkx
* simpy

---

### Setup (Recommended)

```bash
conda create --name raycloudsim python=3.12
conda activate raycloudsim
pip install -r requirements.txt
```

---

## Usage

To run the simulation:

```bash
python main.py --config configs/Pakistan/DQL/NOTE.yaml
```

You can modify the configuration file to test different scenarios and policies.

---

## Experimental Setup

* Dataset: Pakistan IoT task dataset
* Simulator: RayCloudSim

**Training parameters:**

* Epochs: 5–30
* Batch size: 32
* Replay buffer: 20,000
* Discount factor: 0.99

---

## Results

The proposed GNN-DQN achieves:

* latency reduction vs Random
* Near-optimal performance close to Greedy
* Improved energy efficiency in several scenarios

### Additional insights:

* Learns a **QoS-aware offloading policy**
* High **deadline satisfaction (~98%)**
* Robust and stable behavior



## Baselines

Compared methods:

* Random
* Greedy (oracle baseline)
* DQN-only (MLP without GNN)

---

## Project Structure

```
├── main.py
├── configs/
├── graph_utils.py
├── preprocessing_utils.py
├── GNN_DQN_Training.py
├── GNN Graph.py
├── dataset/
├── models/
└── README.md
```

---

## Key Features

* ✅ Dynamic GNN embeddings (task-aware)
* ✅ Integration with RayCloudSim
* ✅ Realistic latency & energy modeling
* ✅ End-to-end training + evaluation pipeline
* ✅ Visualization (learning curves, node selection, QoS metrics)

---

## Tech Stack

* Python
* PyTorch
* NetworkX
* NumPy / Pandas
* RayCloudSim

---

## References

* M. Aazam et al., *Cloud of Things (CoT)*, IEEE Transactions on Sustainable Computing, 2022
* GARON et al., *NATE / T-NATE: Efficient Transformers for IoT Task Offloading*

---

## Author

Developed as part of a research project in:

**M2 Data Science & Machine Learning**
CY Cergy Paris Université / ENSEA

---

## Contact

For questions or collaborations:

* Open an issue
* Submit a pull request

---


    



