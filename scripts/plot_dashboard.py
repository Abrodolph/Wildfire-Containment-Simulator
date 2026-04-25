"""
Training curves dashboard — 4-panel matplotlib figure.

Usage:
    python scripts/plot_dashboard.py --stats training/training_stats.json --output training/training_dashboard.png
    python scripts/plot_dashboard.py  # generates synthetic demo if no stats file
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SYNTHETIC_PATH = "training/synthetic_stats_demo.json"
TIER_ORDER = {"easy": 0, "medium": 1, "hard": 2}
TIER_COLORS = {"easy": "tab:green", "medium": "tab:orange", "hard": "tab:red"}


def _moving_average(values, window):
    out = []
    for i in range(len(values)):
        w = values[max(0, i - window + 1): i + 1]
        out.append(sum(w) / len(w))
    return out


def _rolling_fraction(flags, window=10):
    out = []
    for i in range(len(flags)):
        w = flags[max(0, i - window + 1): i + 1]
        out.append(sum(w) / len(w))
    return out


def _generate_synthetic():
    """Create 50 fake training steps with a plausible upward curve + one promotion."""
    stats = []
    rng = np.random.default_rng(0)
    tier = "easy"
    for i in range(50):
        if i == 20:
            tier = "medium"
        base = 2.0 + i * 0.08 if tier == "easy" else 1.0 + (i - 20) * 0.06
        reward = float(base + rng.normal(0, 0.5))
        stats.append({
            "step": i,
            "mean_reward": reward,
            "tier": tier,
            "parse_failure_rate": max(0.0, 0.3 - i * 0.005 + float(rng.normal(0, 0.02))),
            "promoted_to": "medium" if i == 20 else None,
        })
    os.makedirs(os.path.dirname(SYNTHETIC_PATH), exist_ok=True)
    with open(SYNTHETIC_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    return stats, True


def load_stats(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f), False
    return _generate_synthetic()


def plot_dashboard(stats, output_path, synthetic=False):
    steps = [s["step"] for s in stats]
    rewards = [s["mean_reward"] for s in stats]
    tiers = [s["tier"] for s in stats]
    tier_nums = [TIER_ORDER.get(t, 0) for t in tiers]

    # Population survival: 1 if reward >= 5.0 (terminal bonus threshold), else 0
    pop_survived = [1 if r >= 5.0 else 0 for r in rewards]
    # Containment proxy: clamp reward to [0,1] range as a rough proxy
    containment = [min(1.0, max(0.0, r / 8.0)) for r in rewards]

    promotion_events = [
        (s["step"], s["promoted_to"])
        for s in stats
        if s.get("promoted_to")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    title_suffix = " [SYNTHETIC DEMO]" if synthetic else ""
    fig.suptitle(f"Wildfire Containment Simulator — Training Dashboard{title_suffix}",
                 fontsize=13, fontweight="bold", color="darkred" if synthetic else "black")

    # Panel A — Mean episode reward
    ax = axes[0, 0]
    ax.plot(steps, rewards, alpha=0.35, color="steelblue", linewidth=1)
    ax.plot(steps, _moving_average(rewards, 5), color="steelblue", linewidth=2, label="MA-5")
    for ep, new_tier in promotion_events:
        ax.axvline(x=ep, color=TIER_COLORS.get(new_tier, "gray"), linestyle="--", alpha=0.7)
        ax.text(ep + 0.3, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] != 0 else 0.5,
                new_tier, fontsize=7, color=TIER_COLORS.get(new_tier, "gray"))
    ax.set_title("A — Episode Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B — Population survival rate (rolling 10-ep fraction)
    ax = axes[0, 1]
    survival_rate = _rolling_fraction(pop_survived, window=10)
    ax.plot(steps, [v * 100 for v in survival_rate], color="forestgreen", linewidth=2)
    ax.fill_between(steps, [v * 100 for v in survival_rate], alpha=0.15, color="forestgreen")
    for ep, new_tier in promotion_events:
        ax.axvline(x=ep, color=TIER_COLORS.get(new_tier, "gray"), linestyle="--", alpha=0.7)
    ax.set_title("B — Population Survival Rate (rolling 10-ep)")
    ax.set_xlabel("Step")
    ax.set_ylabel("% Episodes with Zero Pop Loss")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Panel C — Mean containment % at episode end
    ax = axes[1, 0]
    containment_ma = _moving_average(containment, 5)
    ax.plot(steps, [v * 100 for v in containment], alpha=0.3, color="darkorange", linewidth=1)
    ax.plot(steps, [v * 100 for v in containment_ma], color="darkorange", linewidth=2, label="MA-5")
    for ep, new_tier in promotion_events:
        ax.axvline(x=ep, color=TIER_COLORS.get(new_tier, "gray"), linestyle="--", alpha=0.7)
    ax.set_title("C — Containment % at Episode End")
    ax.set_xlabel("Step")
    ax.set_ylabel("Containment %")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D — Curriculum tier timeline (step function)
    ax = axes[1, 1]
    ax.step(steps, tier_nums, where="post", color="mediumpurple", linewidth=2)
    ax.fill_between(steps, tier_nums, step="post", alpha=0.15, color="mediumpurple")
    for ep, new_tier in promotion_events:
        tier_num = TIER_ORDER.get(new_tier, 0)
        color = TIER_COLORS.get(new_tier, "gray")
        ax.axvline(x=ep, color=color, linestyle="--", alpha=0.8, linewidth=1.5)
        ax.text(ep + 0.3, tier_num - 0.1, f"-> {new_tier}", fontsize=8,
                color=color, fontweight="bold")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["easy", "medium", "hard"])
    ax.set_title("D — Curriculum Tier Timeline")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Tier")
    ax.set_ylim(-0.3, 2.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"Dashboard saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", default=None)
    parser.add_argument("--output", default="training/training_dashboard.png")
    args = parser.parse_args()

    stats, synthetic = load_stats(args.stats)
    if synthetic:
        print(f"No stats file found — generated synthetic demo at {SYNTHETIC_PATH}")
    plot_dashboard(stats, args.output, synthetic=synthetic)


if __name__ == "__main__":
    main()
