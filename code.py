"""
Main script for GNN-DQN based task offloading in Edge-Fog-Cloud environments.

This script is based on the experimental notebooks used in this project:
- Preprocessing Dataset
- GNN Graph Construction
- GNN-DQN Integration
- Training and Evaluation

The goal is to learn an offloading policy that selects the best Edge/Fog/Cloud node
for each IoT task by optimizing latency and energy consumption.
"""

import copy
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

from graph_utils import load_graph_artifacts
from preprocessing_utils import preprocess_datasets

from core.env import Env
from core.task import Task
from scenario_impl import PakistanScenario


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


TASK_NUMERIC_COLS = [
    "GenerationTime",
    "TaskSize",
    "CyclesPerBit",
    "TransBitRate",
    "DDL",
]


class NodeQNetwork(nn.Module):
    """
    DQN model that outputs one Q-value per candidate node.
    """

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    """

    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def make_env(json_scenario_file):
    scenario = PakistanScenario(json_scenario_file)
    env = Env(
        scenario=scenario,
        config_file="core/configs/env_config_null.json",
        verbose=False,
        decimal_places=3,
    )
    return scenario, env


def make_task_from_row(row, default_src_name="e0"):
    return Task(
        task_id=int(row["TaskID"]),
        task_size=int(row["TaskSize"]),
        cycles_per_bit=float(row["CyclesPerBit"]),
        trans_bit_rate=int(row["TransBitRate"]),
        src_name=default_src_name,
        ddl=float(row["DDL"]),
        task_name=str(row.get("TaskName", f"task_{row['TaskID']}")),
    )


def run_task(json_scenario_file, task, dst_name):
    _, env = make_env(json_scenario_file)
    env.process(task=task, dst_name=dst_name)
    env.run(until=100000)
    info = env.logger.task_info[task.task_id]
    return info


def extract_metrics(info):
    status_code = info[0]
    src_name, dst_name = info[1]
    trans_time, wait_time, exe_time = info[2]
    exe_energy, trans_energy = info[3]

    return {
        "status_code": status_code,
        "dst_name": dst_name,
        "total_latency": trans_time + wait_time + exe_time,
        "total_energy": exe_energy + trans_energy,
    }


def compute_reward(metrics, task, alpha=1.0, beta=1.0, fail_penalty=-10.0):
    """
    Reward function:
        R = - (alpha * latency + beta * energy)
    """

    if metrics["status_code"] != 0:
        return fail_penalty

    return -((alpha * metrics["total_latency"]) + (beta * metrics["total_energy"]))


def get_node_embeddings(gnn, x_dummy, a_hat):
    """
    Compute GNN node embeddings and graph embedding.
    """

    gnn.eval()

    with torch.no_grad():
        node_embeddings = gnn(x_dummy, a_hat)
        graph_embedding = node_embeddings.mean(dim=0)

    return node_embeddings, graph_embedding


def build_task_features(task_row, device):
    values = task_row[TASK_NUMERIC_COLS].values.astype(np.float32)
    return torch.tensor(values, dtype=torch.float32, device=device)


def build_q_input(node_embeddings, graph_embedding, task_features):
    """
    DQN input:
        [node_embedding | graph_embedding | task_features]
    """

    num_nodes = node_embeddings.shape[0]

    graph_expanded = graph_embedding.unsqueeze(0).repeat(num_nodes, 1)
    task_expanded = task_features.unsqueeze(0).repeat(num_nodes, 1)

    return torch.cat(
        [node_embeddings, graph_expanded, task_expanded],
        dim=1,
    )


def build_state(task_row_norm, node_embeddings, graph_embedding, device):
    task_features = build_task_features(task_row_norm, device)
    q_input = build_q_input(node_embeddings, graph_embedding, task_features)

    return {
        "task_features": task_features,
        "q_input": q_input,
    }


def choose_best_node(qnet, state, scenario):
    with torch.no_grad():
        q_values = qnet(state["q_input"]).squeeze(-1)

    best_id = torch.argmax(q_values).item()
    best_name = scenario.node_id2name[best_id]

    return best_id, best_name, q_values


def select_action(qnet, state, scenario, epsilon, num_nodes):
    if random.random() < epsilon:
        action = random.randint(0, num_nodes - 1)
        node_name = scenario.node_id2name[action]

        with torch.no_grad():
            q_values = qnet(state["q_input"]).squeeze(-1)

        return action, node_name, q_values, "random"

    best_id, best_name, q_values = choose_best_node(qnet, state, scenario)

    return best_id, best_name, q_values, "policy"


def update_dqn(qnet, target_qnet, replay_buffer, optimizer, criterion, device, gamma=0.99, batch_size=32):
    if len(replay_buffer) < batch_size:
        return None

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    state_action_vectors = torch.stack(
        [states[i]["q_input"][actions[i]] for i in range(batch_size)]
    ).to(device)

    q_sa = qnet(state_action_vectors).squeeze(-1)

    with torch.no_grad():
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        max_next_q = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            if next_states[i] is not None and not dones[i]:
                next_q = target_qnet(next_states[i]["q_input"].to(device))
                max_next_q[i] = next_q.max()

        targets = rewards_t + gamma * max_next_q * (1.0 - dones_t)

    loss = criterion(q_sa, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=5.0)
    optimizer.step()

    return loss.item()


def train(config):
    print("=" * 70)
    print("Training GNN-DQN Task Offloading Policy")
    print("=" * 70)

    json_scenario_file = config["scenario_file"]

    artifacts = load_graph_artifacts(json_scenario_file)

    device = artifacts["device"]
    a_hat = artifacts["A_hat"].to(device)
    x_dummy = artifacts["X_dummy"].to(device)
    gnn = artifacts["gnn"].to(device)
    num_nodes = artifacts["num_nodes"]

    datasets = preprocess_datasets(
        config["train_path"],
        config["test_path"],
    )

    train_df = datasets["train_df_raw"]
    train_df_norm = datasets["train_df_norm"]
    test_df = datasets["test_df_raw"]
    test_df_norm = datasets["test_df_norm"]

    scenario_train, _ = make_env(json_scenario_file)

    node_embeddings, graph_embedding = get_node_embeddings(gnn, x_dummy, a_hat)
    example_state = build_state(train_df_norm.iloc[0], node_embeddings, graph_embedding, device)
    input_dim = example_state["q_input"].shape[1]

    qnet = NodeQNetwork(input_dim=input_dim, hidden_dim=config["hidden_dim"]).to(device)
    target_qnet = copy.deepcopy(qnet).to(device)
    target_qnet.eval()

    optimizer = optim.Adam(qnet.parameters(), lr=config["learning_rate"])
    criterion = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=config["replay_buffer_capacity"])

    epsilon = config["epsilon_start"]

    reward_history = []
    loss_history = []

    global_step = 0

    for epoch in range(config["num_epochs"]):
        epoch_rewards = []
        epoch_losses = []

        pbar = tqdm(range(config["max_train_tasks"]), desc=f"Epoch {epoch + 1}/{config['num_epochs']}")

        for idx in pbar:
            row_raw = train_df.iloc[idx]
            row_norm = train_df_norm.iloc[idx]

            node_embeddings, graph_embedding = get_node_embeddings(gnn, x_dummy, a_hat)
            state = build_state(row_norm, node_embeddings, graph_embedding, device)

            action, node_name, q_values, policy_type = select_action(
                qnet=qnet,
                state=state,
                scenario=scenario_train,
                epsilon=epsilon,
                num_nodes=num_nodes,
            )

            task = make_task_from_row(row_raw)
            info = run_task(json_scenario_file, task, node_name)
            metrics = extract_metrics(info)
            reward = compute_reward(metrics, task)

            next_idx = idx + 1

            if next_idx < config["max_train_tasks"]:
                next_row_norm = train_df_norm.iloc[next_idx]
                next_embeddings, next_graph_embedding = get_node_embeddings(gnn, x_dummy, a_hat)
                next_state = build_state(next_row_norm, next_embeddings, next_graph_embedding, device)
                done = False
            else:
                next_state = None
                done = True

            replay_buffer.push(state, action, reward, next_state, done)

            loss = update_dqn(
                qnet=qnet,
                target_qnet=target_qnet,
                replay_buffer=replay_buffer,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                gamma=config["gamma"],
                batch_size=config["batch_size"],
            )

            if loss is not None:
                loss_history.append(loss)
                epoch_losses.append(loss)

            reward_history.append(reward)
            epoch_rewards.append(reward)

            global_step += 1

            if global_step % config["target_sync_every"] == 0:
                target_qnet.load_state_dict(qnet.state_dict())

            pbar.set_postfix({
                "reward": f"{np.mean(epoch_rewards):.4f}",
                "epsilon": f"{epsilon:.3f}",
            })

        epsilon = max(config["epsilon_min"], epsilon * config["epsilon_decay"])

        print(
            f"Epoch {epoch + 1}/{config['num_epochs']} | "
            f"Mean Reward: {np.mean(epoch_rewards):.4f} | "
            f"Mean Loss: {np.mean(epoch_losses) if epoch_losses else 0:.4f} | "
            f"Epsilon: {epsilon:.4f}"
        )

    torch.save(
        {
            "qnet_state_dict": qnet.state_dict(),
            "input_dim": input_dim,
            "task_numeric_cols": TASK_NUMERIC_COLS,
            "config": config,
        },
        config["model_output"],
    )

    print(f"\nModel saved to: {config['model_output']}")

    evaluate(
        config=config,
        qnet=qnet,
        gnn=gnn,
        a_hat=a_hat,
        x_dummy=x_dummy,
        test_df=test_df,
        test_df_norm=test_df_norm,
        device=device,
    )


def evaluate(config, qnet, gnn, a_hat, x_dummy, test_df, test_df_norm, device):
    print("\n" + "=" * 70)
    print("Evaluation: GNN-DQN vs Greedy vs Random")
    print("=" * 70)

    json_scenario_file = config["scenario_file"]
    scenario_eval, _ = make_env(json_scenario_file)
    node_names = list(scenario_eval.node_id2name.values())

    results_gnn_dqn = []
    results_greedy = []
    results_random = []

    max_eval_tasks = min(config["max_eval_tasks"], len(test_df))

    for idx in tqdm(range(max_eval_tasks), desc="Evaluating"):
        row_raw = test_df.iloc[idx]
        row_norm = test_df_norm.iloc[idx]

        task = make_task_from_row(row_raw)

        node_embeddings, graph_embedding = get_node_embeddings(gnn, x_dummy, a_hat)
        state = build_state(row_norm, node_embeddings, graph_embedding, device)

        _, dqn_node, _ = choose_best_node(qnet, state, scenario_eval)
        info = run_task(json_scenario_file, task, dqn_node)
        metrics = extract_metrics(info)
        reward = compute_reward(metrics, task)

        results_gnn_dqn.append({
            "reward": reward,
            "latency": metrics["total_latency"],
            "energy": metrics["total_energy"],
            "node": dqn_node,
        })

        best_reward = float("-inf")
        best_metrics = None
        best_node = None

        for node in node_names:
            task_greedy = make_task_from_row(row_raw)
            info = run_task(json_scenario_file, task_greedy, node)
            metrics = extract_metrics(info)
            reward = compute_reward(metrics, task_greedy)

            if reward > best_reward:
                best_reward = reward
                best_metrics = metrics
                best_node = node

        results_greedy.append({
            "reward": best_reward,
            "latency": best_metrics["total_latency"],
            "energy": best_metrics["total_energy"],
            "node": best_node,
        })

        random_node = random.choice(node_names)
        task_random = make_task_from_row(row_raw)
        info = run_task(json_scenario_file, task_random, random_node)
        metrics = extract_metrics(info)
        reward = compute_reward(metrics, task_random)

        results_random.append({
            "reward": reward,
            "latency": metrics["total_latency"],
            "energy": metrics["total_energy"],
            "node": random_node,
        })

    summary = pd.DataFrame([
        summarize(results_gnn_dqn, "GNN-DQN"),
        summarize(results_greedy, "Greedy"),
        summarize(results_random, "Random"),
    ])

    print("\nEvaluation Summary:")
    print(summary.to_string(index=False))

    summary.to_csv(config["results_output"], index=False)
    print(f"\nResults saved to: {config['results_output']}")


def summarize(results, method):
    return {
        "method": method,
        "mean_reward": np.mean([r["reward"] for r in results]),
        "mean_latency": np.mean([r["latency"] for r in results]),
        "mean_energy": np.mean([r["energy"] for r in results]),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GNN-DQN task offloading experiment"
    )

    parser.add_argument(
        "--scenario_file",
        type=str,
        default="New_Feature_nodes.json",
        help="Path to the Edge-Fog-Cloud JSON scenario file",
    )

    parser.add_argument(
        "--train_path",
        type=str,
        default="data/trainset.csv",
        help="Path to training dataset",
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default="data/testset.csv",
        help="Path to test dataset",
    )

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--max_train_tasks", type=int, default=1000)
    parser.add_argument("--max_eval_tasks", type=int, default=500)

    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "scenario_file": args.scenario_file,
        "train_path": args.train_path,
        "test_path": args.test_path,

        "num_epochs": args.num_epochs,
        "max_train_tasks": args.max_train_tasks,
        "max_eval_tasks": args.max_eval_tasks,

        "hidden_dim": 256,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "replay_buffer_capacity": 20000,
        "target_sync_every": 50,

        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.80,

        "model_output": "gnn_dqn_trained.pt",
        "results_output": "evaluation_summary.csv",
    }

    train(config)


if __name__ == "__main__":
    main()
