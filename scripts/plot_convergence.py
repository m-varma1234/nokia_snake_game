"""
Unified convergence plot — score, steps, and smoothed curves for all trained algos.

Usage:
    python plot_convergence.py                     # loads all *_history.json from checkpoints/
    python plot_convergence.py --algos dqn ppo     # subset
    python plot_convergence.py --ckpt-dir checkpoints --output convergence.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


COLORS = {
    "dqn": "#4DA6FF",
    "ppo": "#FF944D",
    "a2c": "#5CD65C",
}

WINDOW = 30   # moving-average window for smoothing


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algos",    nargs="+", default=None,
                   help="Algorithms to include (default: auto-detect from checkpoints/)")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--output",   type=str, default="checkpoints/convergence.png")
    p.add_argument("--window",   type=int, default=WINDOW,
                   help="Moving-average window size")
    return p.parse_args()


def moving_average(arr, window):
    if len(arr) < window:
        return np.array(arr)
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def load_histories(algos, ckpt_dir: Path):
    histories = {}
    for algo in algos:
        path = ckpt_dir / f"{algo}_history.json"
        if not path.exists():
            # Try original DQN history file name
            if algo == "dqn":
                path = ckpt_dir / "history.json"
            if not path.exists():
                print(f"  Warning: no history for {algo.upper()} ({path}) — skipping")
                continue
        with open(path) as f:
            histories[algo] = json.load(f)
        print(f"  Loaded {algo.upper()} history ({len(histories[algo]['scores'])} episodes)")
    return histories


def save_convergence_plot(histories: dict, output: str, window: int):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#1A1A2E")
    ax.spines[:].set_color("#333355")
    ax.tick_params(colors="#AAAACC")
    ax.xaxis.label.set_color("#AAAACC")
    ax.yaxis.label.set_color("#AAAACC")
    ax.title.set_color("white")

    fig.suptitle("Snake RL — Convergence Comparison", color="white", fontsize=14,
                 fontweight="bold", y=1.02)

    legend_patches = []

    for algo, history in histories.items():
        color  = COLORS.get(algo, "gray")
        scores = history["scores"]
        eps    = np.arange(1, len(scores) + 1)

        if len(scores) >= window:
            ma     = moving_average(scores, window)
            eps_ma = np.arange(window, len(scores) + 1)
            ax.plot(eps_ma, ma, color=color, linewidth=2.5, label=algo.upper())
            ax.fill_between(eps_ma, ma, alpha=0.10, color=color)
        else:
            ax.plot(eps, scores, color=color, linewidth=2.5, label=algo.upper())

        legend_patches.append(
            plt.Line2D([0], [0], color=color, linewidth=3, label=algo.upper())
        )

    ax.set_title(f"Score (Food Eaten) vs Episode  [{window}-ep moving avg]", fontsize=11)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score (food eaten)")
    ax.legend(handles=legend_patches, facecolor="#1A1A2E", labelcolor="white",
              edgecolor="#333355")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.grid(color="#222244", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Convergence plot saved → {out}")


if __name__ == "__main__":
    args     = parse_args()
    ckpt_dir = Path(args.ckpt_dir)

    if args.algos:
        algos = args.algos
    else:
        algos = [p.stem.replace("_history", "")
                 for p in sorted(ckpt_dir.glob("*_history.json"))]
        if not algos:
            # Fall back: try history.json as dqn
            if (ckpt_dir / "history.json").exists():
                algos = ["dqn"]
        if not algos:
            print("No history JSON files found. Train first:  python train_all.py")
            raise SystemExit(1)

    print(f"Loading histories for: {', '.join(a.upper() for a in algos)}")
    histories = load_histories(algos, ckpt_dir)

    if not histories:
        print("Nothing to plot.")
        raise SystemExit(1)

    save_convergence_plot(histories, args.output, args.window)
