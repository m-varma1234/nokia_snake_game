"""
Train DQN, PPO, and A2C on Snake and save checkpoints for comparison.

Usage:
    python train_all.py                        # all three algos, 1000 episodes
    python train_all.py --episodes 500 --algos dqn ppo
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--grid",     type=int, default=20)
    p.add_argument("--algos",    nargs="+", default=["dqn", "ppo", "a2c"],
                   choices=["dqn", "ppo", "a2c"])
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--log-interval", type=int, default=50)
    return p.parse_args()


# ──────────────────────────────────────── DQN training ───────────────────────

def train_dqn(episodes: int, grid: int, save_dir: Path, log_interval: int):
    env   = SnakeEnv(grid_size=grid)
    agent = DQNAgent(state_dim=11, action_dim=3)
    history = {"scores": [], "steps": []}
    recent  = deque(maxlen=100)
    best_mean = -float("inf")
    t0 = time.time()

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, float(done))
            agent.learn()
            state = next_state
            score = info["score"]
            steps += 1

        agent.decay_epsilon()
        recent.append(score)
        mean = np.mean(recent)
        history["scores"].append(score)
        history["steps"].append(steps)

        if mean > best_mean and len(recent) >= 20:
            best_mean = mean
            agent.save(save_dir / "dqn_best.pt")

        if ep % log_interval == 0:
            print(f"  [DQN] Ep {ep:>5}/{episodes} | Score {score:>4} | Mean100 {mean:>6.2f} | "
                  f"ε {agent.epsilon:.4f} | {time.time()-t0:.1f}s")

    agent.save(save_dir / "dqn_final.pt")
    env.close()
    return history


# ──────────────────────────────────────── PPO training ───────────────────────

def train_ppo(episodes: int, grid: int, save_dir: Path, log_interval: int):
    env   = SnakeEnv(grid_size=grid)
    agent = PPOAgent(state_dim=11, action_dim=3)
    history = {"scores": [], "steps": []}
    recent  = deque(maxlen=100)
    best_mean = -float("inf")
    t0 = time.time()

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward, done)
            state = next_state
            score = info["score"]
            steps += 1

        # Learn at end of episode; bootstrap = 0 since episode ended
        agent.learn(last_value=0.0)
        recent.append(score)
        mean = np.mean(recent)
        history["scores"].append(score)
        history["steps"].append(steps)

        if mean > best_mean and len(recent) >= 20:
            best_mean = mean
            agent.save(save_dir / "ppo_best.pt")

        if ep % log_interval == 0:
            print(f"  [PPO] Ep {ep:>5}/{episodes} | Score {score:>4} | Mean100 {mean:>6.2f} | "
                  f"{time.time()-t0:.1f}s")

    agent.save(save_dir / "ppo_final.pt")
    env.close()
    return history


# ──────────────────────────────────────── A2C training ───────────────────────

def train_a2c(episodes: int, grid: int, save_dir: Path, log_interval: int):
    env   = SnakeEnv(grid_size=grid)
    agent = A2CAgent(state_dim=11, action_dim=3)
    history = {"scores": [], "steps": []}
    recent  = deque(maxlen=100)
    best_mean = -float("inf")
    t0 = time.time()

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done  = False
        score = 0
        steps = 0
        last_state = state

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward, done)
            last_state = next_state
            state = next_state
            score = info["score"]
            steps += 1

            # Update every n_steps mid-episode
            if len(agent._rewards) >= agent.n_steps and not done:
                agent.learn(last_state=last_state)

        # Flush remaining steps
        if agent._rewards:
            agent.learn(last_state=None)

        recent.append(score)
        mean = np.mean(recent)
        history["scores"].append(score)
        history["steps"].append(steps)

        if mean > best_mean and len(recent) >= 20:
            best_mean = mean
            agent.save(save_dir / "a2c_best.pt")

        if ep % log_interval == 0:
            print(f"  [A2C] Ep {ep:>5}/{episodes} | Score {score:>4} | Mean100 {mean:>6.2f} | "
                  f"{time.time()-t0:.1f}s")

    agent.save(save_dir / "a2c_final.pt")
    env.close()
    return history


# ──────────────────────────────────────── Comparison plot ────────────────────

def save_comparison_plot(all_history: dict, save_dir: Path, episodes: int):
    colors = {"dqn": "steelblue", "ppo": "darkorange", "a2c": "seagreen"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Snake RL — Algorithm Comparison", fontsize=14)
    window = 20

    for algo, history in all_history.items():
        scores = history["scores"]
        eps    = range(1, len(scores) + 1)
        color  = colors.get(algo, "gray")
        axes[0].plot(eps, scores, alpha=0.2, color=color)
        if len(scores) >= window:
            ma = np.convolve(scores, np.ones(window) / window, mode="valid")
            axes[0].plot(range(window, len(scores) + 1), ma, color=color, label=algo.upper(), linewidth=2)

    axes[0].set_title(f"Score ({window}-ep moving average)")
    axes[0].set_xlabel("Episode")
    axes[0].legend()

    for algo, history in all_history.items():
        steps = history["steps"]
        eps   = range(1, len(steps) + 1)
        color = colors.get(algo, "gray")
        axes[1].plot(eps, steps, alpha=0.3, color=color, label=algo.upper())

    axes[1].set_title("Steps per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].legend()

    plt.tight_layout()
    path = save_dir / "comparison_curves.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Comparison plot saved → {path}")


# ─────────────────────────────────────── Entry point ─────────────────────────

if __name__ == "__main__":
    args     = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainers = {"dqn": train_dqn, "ppo": train_ppo, "a2c": train_a2c}
    all_history = {}

    for algo in args.algos:
        print(f"\n{'='*60}")
        print(f"  Training {algo.upper()} — {args.episodes} episodes  |  grid {args.grid}x{args.grid}")
        print(f"{'='*60}")
        hist = trainers[algo](args.episodes, args.grid, save_dir, args.log_interval)
        all_history[algo] = hist
        with open(save_dir / f"{algo}_history.json", "w") as f:
            json.dump(hist, f)

    if len(all_history) > 1:
        save_comparison_plot(all_history, save_dir, args.episodes)

    print("\nAll done! Run  python record_comparison.py  to generate the video.")
