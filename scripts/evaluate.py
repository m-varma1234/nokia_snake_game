"""
Evaluate a trained Snake DQN agent.

Modes:
  --mode terminal   : watch the game in the terminal (ANSI art)
  --mode video      : save an MP4 using matplotlib animation
  --mode benchmark  : run N silent episodes and print stats
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from env.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Snake DQN agent")
    p.add_argument("--model",    type=str, default="checkpoints/best_model.pt")
    p.add_argument("--mode",     type=str, default="terminal",
                   choices=["terminal", "video", "benchmark"])
    p.add_argument("--episodes", type=int, default=5,  help="Episodes for terminal/benchmark")
    p.add_argument("--grid",     type=int, default=20)
    p.add_argument("--delay",    type=float, default=0.05, help="Seconds between frames (terminal)")
    p.add_argument("--video-ep", type=int, default=1, help="Which episode to record as video")
    p.add_argument("--output",   type=str, default="snake_play.mp4")
    return p.parse_args()


# ─────────────────────────────────────── Helpers ─────────────────────────────

def load_agent(model_path: str, grid: int) -> DQNAgent:
    agent = DQNAgent(state_dim=11, action_dim=3)
    agent.load(model_path)
    agent.epsilon = 0.0   # pure exploitation
    return agent


def run_episode(env: SnakeEnv, agent: DQNAgent, capture_frames: bool = False):
    state, _ = env.reset()
    done = False
    score = 0
    frames = []

    while not done:
        if capture_frames:
            frames.append(env._render_rgb())
        action = agent.select_action(state, training=False)
        state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score = info["score"]

    if capture_frames:
        frames.append(env._render_rgb())
    return score, frames


# ──────────────────────────────────────── Terminal mode ──────────────────────

def play_terminal(agent: DQNAgent, args):
    env = SnakeEnv(grid_size=args.grid, render_mode="ansi")
    scores = []

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        done = False

        while not done:
            print("\033[H\033[J", end="")   # clear terminal
            print(f"Episode {ep}/{args.episodes}  |  Score: {scores[-1] if scores else 0}")
            print(env.render())
            time.sleep(args.delay)
            action = agent.select_action(state, training=False)
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        scores.append(info["score"])
        print(f"\nGame over — Score: {info['score']}")
        time.sleep(0.5)

    print(f"\n{'─'*40}")
    print(f"Episodes: {args.episodes}")
    print(f"Scores:   {scores}")
    print(f"Mean:     {np.mean(scores):.2f}")
    print(f"Max:      {max(scores)}")
    env.close()


# ──────────────────────────────────────── Video mode ─────────────────────────

def record_video(agent: DQNAgent, args):
    matplotlib.use("Agg")
    env = SnakeEnv(grid_size=args.grid, render_mode="rgb_array")

    print(f"Recording episode {args.video_ep}…")
    _, frames = run_episode(env, agent, capture_frames=True)
    env.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        ax.set_title(f"Frame {frame_idx+1}/{len(frames)}", fontsize=10)
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=80, blit=True
    )

    writer = animation.FFMpegWriter(fps=12, metadata={"title": "Snake DQN"})
    ani.save(args.output, writer=writer)
    plt.close()
    print(f"Video saved → {args.output}")


# ──────────────────────────────────────── Benchmark mode ─────────────────────

def benchmark(agent: DQNAgent, args):
    env = SnakeEnv(grid_size=args.grid)
    scores = []

    print(f"Benchmarking {args.episodes} episodes…")
    for ep in range(args.episodes):
        score, _ = run_episode(env, agent)
        scores.append(score)
        print(f"  Episode {ep+1:>4}: score = {score}")

    env.close()

    print(f"\n{'─'*40}")
    print(f"Episodes : {args.episodes}")
    print(f"Mean     : {np.mean(scores):.2f}")
    print(f"Std      : {np.std(scores):.2f}")
    print(f"Min      : {min(scores)}")
    print(f"Max      : {max(scores)}")
    print(f"Median   : {np.median(scores):.1f}")


# ─────────────────────────────────────── Entry point ─────────────────────────

if __name__ == "__main__":
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first:  python train.py")
        raise SystemExit(1)

    agent = load_agent(str(model_path), args.grid)

    if args.mode == "terminal":
        play_terminal(agent, args)
    elif args.mode == "video":
        record_video(agent, args)
    elif args.mode == "benchmark":
        benchmark(agent, args)
