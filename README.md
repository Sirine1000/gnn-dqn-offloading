# gnn-dqn-offloading
AI-Based Task Offloading using GNN-DQN in Edge–Fog–Cloud

 Overview
This project proposes a hybrid Graph Neural Network (GNN) and Deep Q-Network (DQN) framework for intelligent task offloading in Edge–Fog–Cloud computing environments.

The objective is to optimize Quality of Service (QoS) by minimizing:
-  Latency
-  Energy consumption



 Methodology

 Graph Modeling
- Infrastructure modeled as a graph:
  - Nodes → Edge / Fog / Cloud resources
  - Edges → Network connections (bandwidth-aware)
- Node features:
  - CPU capacity (available + max)
  - Buffer capacity
  - Energy coefficients
  - Node type (one-hot encoding)

 GNN Encoder
- 3 Graph Convolution layers
- Hidden dimension: 64
- Activation: ReLU
- Output: Node embeddings capturing topology

 DQN Agent
- Input: GNN embeddings
- Architecture: 256 → 128 → Output (Q-values per node)
- Strategy: ε-greedy exploration
- Replay buffer + target network

---

 Objective Function

The reward is defined as:

R = - (α × Latency + β × Energy)

The model learns to select the optimal node for each task.



 Experimental Setup
  Dataset: Pakistan IoT task dataset
  Simulator: RayCloudSim
  Training:
  - Epochs: 30
  - Batch size: 32
  - Learning rate: 3e-4
  - Discount factor: 0.99

 Results
Significant latency reduction vs Random (-70%)  
Near-optimal performance close to Greedy  


 Tech Stack

- Python
- PyTorch
- NetworkX
- NumPy / Pandas
- RayCloudSim


