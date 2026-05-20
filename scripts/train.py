"""
Training script for Snake DQN agent.

Usage:
    python train.py                       # default settings
    python train.py --episodes 300 --grid 15
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
matplotlib.use("Agg")          # headless — no display required during training
import matplotlib.pyplot as plt

from env.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent


# ─────────────────────────────────────────── CLI args ────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DQN on Snake")
    p.add_argument("--episodes",       type=int,   default=500,   help="Number of training episodes")
    p.add_argument("--grid",           type=int,   default=20,    help="Grid size (NxN)")
    p.add_argument("--lr",             type=float, default=1e-3,  help="Learning rate")
    p.add_argument("--gamma",          type=float, default=0.9,   help="Discount factor")
    p.add_argument("--epsilon-start",  type=float, default=1.0)
    p.add_argument("--epsilon-end",    type=float, default=0.01)
    p.add_argument("--epsilon-decay",  type=float, default=0.995)
    p.add_argument("--buffer",         type=int,   default=100_000, help="Replay buffer size")
    p.add_argument("--batch",          type=int,   default=1000,   help="Mini-batch size")
    p.add_argument("--target-update",  type=int,   default=10,    help="Target net sync freq (episodes)")
    p.add_argument("--save-dir",       type=str,   default="checkpoints")
    p.add_argument("--log-interval",   type=int,   default=20,    help="Print stats every N episodes")
    return p.parse_args()


# ─────────────────────────────────────────── Plotting ────────────────────────

def save_plots(history: dict, save_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Snake DQN — Training Progress", fontsize=14)

    episodes = range(1, len(history["scores"]) + 1)

    # Score
    axes[0, 0].plot(episodes, history["scores"], alpha=0.4, color="steelblue", label="per-episode")
    if len(history["scores"]) >= 20:
        ma = np.convolve(history["scores"], np.ones(20) / 20, mode="valid")
        axes[0, 0].plot(range(20, len(history["scores"]) + 1), ma, color="navy", label="20-ep MA")
    axes[0, 0].set_title("Score")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend()

    # Steps per episode
    axes[0, 1].plot(episodes, history["steps"], alpha=0.5, color="darkorange")
    axes[0, 1].set_title("Steps per Episode")
    axes[0, 1].set_xlabel("Episode")

    # Loss
    losses = [l for l in history["losses"] if l is not None]
    axes[1, 0].plot(losses, color="crimson", alpha=0.6)
    axes[1, 0].set_title("Training Loss")
    axes[1, 0].set_xlabel("Learn call")

    # Epsilon
    axes[1, 1].plot(episodes, history["epsilons"], color="seagreen")
    axes[1, 1].set_title("Epsilon (exploration)")
    axes[1, 1].set_xlabel("Episode")

    plt.tight_layout()
    plot_path = save_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Plots saved → {plot_path}")


# ─────────────────────────────────────────── Training loop ───────────────────

def train(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = SnakeEnv(grid_size=args.grid)
    agent = DQNAgent(
        state_dim=11,
        action_dim=3,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer,
        batch_size=args.batch,
        target_update_freq=args.target_update,
    )

    history = {"scores": [], "steps": [], "losses": [], "epsilons": []}
    recent_scores: deque = deque(maxlen=100)
    best_mean = -float("inf")
    total_losses = []

    print(f"\n{'='*60}")
    print(f"  Snake DQN Training")
    print(f"  Episodes : {args.episodes}")
    print(f"  Grid     : {args.grid}x{args.grid}")
    print(f"  Device   : {agent.device}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        ep_score = 0
        ep_steps = 0
        ep_losses = []

        done = False
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, float(done))
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)
                total_losses.append(loss)

            state = next_state
            ep_score = info["score"]
            ep_steps += 1

        agent.decay_epsilon()
        recent_scores.append(ep_score)
        mean_score = np.mean(recent_scores)

        avg_loss = float(np.mean(ep_losses)) if ep_losses else None
        history["scores"].append(ep_score)
        history["steps"].append(ep_steps)
        history["losses"].append(avg_loss)
        history["epsilons"].append(agent.epsilon)

        # Save best model
        if mean_score > best_mean and len(recent_scores) >= 20:
            best_mean = mean_score
            agent.save(save_dir / "best_model.pt")

        # Periodic logging
        if ep % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"Ep {ep:>5}/{args.episodes} | "
                f"Score {ep_score:>4} | "
                f"Mean100 {mean_score:>6.2f} | "
                f"ε {agent.epsilon:.4f} | "
                f"Loss {f'{avg_loss:.4f}' if avg_loss is not None else 'N/A':>8} | "
                f"Time {elapsed:>6.1f}s"
            )

    # Final save
    agent.save(save_dir / "final_model.pt")

    # Save history as JSON
    history_path = save_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"History saved → {history_path}")

    save_plots(history, save_dir)

    print(f"\nTraining complete. Best 100-ep mean score: {best_mean:.2f}")
    env.close()
    return agent


# ─────────────────────────────────────────── Entry point ─────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
