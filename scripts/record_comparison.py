"""
Record a side-by-side comparison video of DQN, PPO, and A2C playing Snake.

Usage:
    python record_comparison.py                        # uses best checkpoints
    python record_comparison.py --episodes 3 --output comparison.mp4
    python record_comparison.py --algos dqn ppo        # subset of algos
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

from env.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent
from scripts.plot_convergence import load_histories, save_convergence_plot


COLORS = {
    "dqn": "#4DA6FF",
    "ppo": "#FF944D",
    "a2c": "#5CD65C",
}

CHECKPOINT_MAP = {
    "dqn": "checkpoints/dqn_best.pt",
    "ppo": "checkpoints/ppo_best.pt",
    "a2c": "checkpoints/a2c_best.pt",
}

# Fall back to original DQN checkpoint for backward compat
DQN_FALLBACK = "checkpoints/best_model.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Record side-by-side RL comparison video")
    p.add_argument("--algos",    nargs="+", default=["dqn", "ppo", "a2c"],
                   choices=["dqn", "ppo", "a2c"])
    p.add_argument("--episodes", type=int, default=1,
                   help="Number of full episodes to record per agent (they play simultaneously)")
    p.add_argument("--grid",     type=int, default=20)
    p.add_argument("--fps",      type=int, default=15)
    p.add_argument("--output",   type=str, default="comparison.mp4")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    return p.parse_args()


# ──────────────────────────────── Agent loader ───────────────────────────────

def load_agents(algos: list[str], ckpt_dir: str) -> dict:
    agents = {}
    for algo in algos:
        ckpt = Path(ckpt_dir) / f"{algo}_best.pt"
        if not ckpt.exists() and algo == "dqn":
            ckpt = Path(DQN_FALLBACK)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                f"Train first:  python train_all.py --algos {algo}"
            )

        if algo == "dqn":
            a = DQNAgent(state_dim=11, action_dim=3)
            a.load(str(ckpt))
            a.epsilon = 0.0
        elif algo == "ppo":
            a = PPOAgent(state_dim=11, action_dim=3)
            a.load(str(ckpt))
        else:  # a2c
            a = A2CAgent(state_dim=11, action_dim=3)
            a.load(str(ckpt))

        agents[algo] = a
    return agents


# ──────────────────────────────── Frame capture ──────────────────────────────

def capture_parallel_episodes(agents: dict, grid: int, n_episodes: int):
    """
    Run all agents simultaneously for n_episodes and capture per-step frames.
    Returns list of composite (side-by-side) RGB frames and per-step score dicts.
    """
    envs   = {algo: SnakeEnv(grid_size=grid, render_mode="rgb_array") for algo in agents}
    states = {algo: envs[algo].reset()[0] for algo in agents}
    dones  = {algo: False for algo in agents}
    scores = {algo: 0 for algo in agents}
    episode_counts = {algo: 0 for algo in agents}

    frames = []          # list of {algo: rgb_array}
    score_history = []   # list of {algo: score}

    print(f"Capturing {n_episodes} episode(s) per agent…")

    while not all(episode_counts[a] >= n_episodes for a in agents):
        frame_dict = {}
        for algo, agent in agents.items():
            if episode_counts[algo] >= n_episodes:
                # Render a static "done" screen
                rgb = envs[algo]._render_rgb()
                frame_dict[algo] = rgb
                continue

            action = agent.select_action(states[algo], training=False)
            next_state, _, terminated, truncated, info = envs[algo].step(action)
            scores[algo] = info["score"]
            frame_dict[algo] = envs[algo]._render_rgb()
            states[algo] = next_state

            if terminated or truncated:
                episode_counts[algo] += 1
                if episode_counts[algo] < n_episodes:
                    states[algo], _ = envs[algo].reset()

        frames.append(frame_dict)
        score_history.append(dict(scores))

    for env in envs.values():
        env.close()

    print(f"Captured {len(frames)} frames total.")
    return frames, score_history


# ──────────────────────────────── Video render ───────────────────────────────

def render_video(
    frames: list,
    score_history: list,
    algos: list[str],
    output: str,
    fps: int,
    grid: int,
):
    n_algos  = len(algos)
    cell     = 20
    img_size = grid * cell   # pixels per panel

    padding_top  = 70        # title + score bar height (pixels in data-space we fake with subplot margins)
    fig_w = 4.0 * n_algos
    fig_h = 5.0

    fig, axes = plt.subplots(1, n_algos, figsize=(fig_w, fig_h))
    if n_algos == 1:
        axes = [axes]

    fig.patch.set_facecolor("#1A1A2E")

    # Title
    fig.suptitle(
        "Snake RL — Algorithm Comparison",
        color="white", fontsize=13, fontweight="bold", y=0.97,
    )

    ims    = []
    titles = []

    for ax, algo in zip(axes, algos):
        ax.set_facecolor("#1A1A2E")
        ax.axis("off")
        rgb0 = frames[0][algo]
        im   = ax.imshow(rgb0, aspect="equal")
        ims.append(im)

        title = ax.set_title(
            f"{algo.upper()}\nScore: 0",
            color=COLORS[algo], fontsize=11, fontweight="bold", pad=6,
        )
        titles.append(title)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    def update(frame_idx):
        fd  = frames[frame_idx]
        sc  = score_history[frame_idx]
        for im, title, algo in zip(ims, titles, algos):
            im.set_data(fd[algo])
            title.set_text(f"{algo.upper()}\nScore: {sc[algo]}")
        return ims + titles

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={"title": "Snake RL Comparison"},
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )
    ani.save(output, writer=writer)
    plt.close()
    print(f"\nVideo saved → {output}")


# ─────────────────────────────────────── Entry point ─────────────────────────

if __name__ == "__main__":
    args   = parse_args()
    agents = load_agents(args.algos, args.ckpt_dir)

    frames, score_history = capture_parallel_episodes(
        agents, args.grid, args.episodes
    )

    render_video(
        frames, score_history, args.algos,
        output=args.output, fps=args.fps, grid=args.grid,
    )

    # Print final scores
    print("\nFinal scores per episode:")
    final = score_history[-1]
    for algo in args.algos:
        print(f"  {algo.upper()}: {final[algo]}")

    # Auto-generate convergence plot from training histories
    conv_out = str(Path(args.ckpt_dir) / "convergence.png")
    print(f"\nGenerating convergence plot → {conv_out}")
    histories = load_histories(args.algos, Path(args.ckpt_dir))
    if histories:
        save_convergence_plot(histories, conv_out, window=30)
