"""
Demo runner — runs heuristic (and optionally trained LLM) on the chosen demo seed,
generates GIF(s), and prints a play-by-play narrative.

Chosen demo seed:
  DEMO_SEED = 7
  Medium tier, seed 7: wind shift fires around step 70, heuristic loses a
  populated cell on the south flank while over-committing crews north.
  This makes the contrast between a reactive heuristic and a planning LLM
  visible in a single GIF.

Usage:
    python scripts/run_demo.py                        # heuristic on DEMO_SEED
    python scripts/run_demo.py --seed 42
    python scripts/run_demo.py --agent trained_llm    # requires TRAINED_MODEL_PATH env var
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import WildfireEnv
from env.rendering import render_frame, render_episode_gif
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent

DEMO_SEED = 7
TIER = "medium"
MAX_STEPS = 150


def _load_trained_agent():
    model_path = os.environ.get("TRAINED_MODEL_PATH")
    if not model_path:
        return None, "[skipped — TRAINED_MODEL_PATH not set]"
    try:
        from agents.llm_agent import LLMAgent  # type: ignore
        return LLMAgent(model_path=model_path), model_path
    except ImportError:
        return None, "[skipped — agents.llm_agent not found]"


def run_episode_with_narrative(agent, seed, gif_path):
    env = WildfireEnv()
    obs = env.reset(task_id=TIER, seed=seed)
    frames = [render_frame(env.state(), step=0)]
    total_reward = 0.0
    events_narrative = []
    done = False

    while not done:
        action = agent.act(obs)
        result = env.step(action)
        total_reward += result.reward
        step = env.current_step
        frames.append(render_frame(env.state(), step=step))

        for event in result.info.get("events", []):
            if any(kw in event for kw in ("WIND SHIFT", "populated", "crew", "casualty",
                                           "IGNITION", "suppressed", "firebreak")):
                events_narrative.append((step, event))

        obs = result.observation
        done = result.done

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    render_episode_gif(frames, gif_path)

    import imageio.v3 as iio
    png_path = os.path.splitext(gif_path)[0] + ".png"
    iio.imwrite(png_path, frames[-1], extension=".png")

    final = env.state()
    total_pop = final.get("total_population", 1) or 1
    stats = {
        "steps": env.current_step,
        "total_reward": round(total_reward, 3),
        "pop_lost": final.get("population_lost", 0),
        "pop_saved_pct": round((1 - final.get("population_lost", 0) / total_pop) * 100, 1),
        "containment_pct": round(final.get("containment_pct", 0.0) * 100, 1),
        "cells_burned": final.get("cells_burned", 0),
        "crew_casualty": env._crew_casualty_occurred,
    }
    return stats, events_narrative, gif_path, png_path


def print_narrative(label, stats, events):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if events:
        print("Play-by-play:")
        for step, event in events[:20]:
            print(f"  Step {step:3d}: {event}")
    else:
        print("  (no notable events recorded)")
    print(f"\nFinal stats:")
    print(f"  Steps:        {stats['steps']}")
    print(f"  Total reward: {stats['total_reward']:+.3f}")
    print(f"  Pop saved:    {stats['pop_saved_pct']:.1f}%")
    print(f"  Containment:  {stats['containment_pct']:.1f}%")
    print(f"  Cells burned: {stats['cells_burned']}")
    print(f"  Crew casualty:{stats['crew_casualty']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    parser.add_argument("--agent", choices=["heuristic", "random", "trained_llm"],
                        default="heuristic")
    args = parser.parse_args()

    print(f"Demo: tier={TIER}, seed={args.seed}, agent={args.agent}")

    # Always run heuristic as baseline
    heuristic = HeuristicAgent()
    h_stats, h_events, h_gif, h_png = run_episode_with_narrative(
        heuristic, args.seed, "demos/heuristic_demo.gif"
    )
    print_narrative("Heuristic Agent", h_stats, h_events)
    print(f"\n  GIF  -> {h_gif}")
    print(f"  PNG  -> {h_png}")

    # Optionally run trained LLM
    if args.agent == "trained_llm":
        trained, note = _load_trained_agent()
        if trained is None:
            print(f"\nTrained LLM: {note}")
        else:
            t_stats, t_events, t_gif, t_png = run_episode_with_narrative(
                trained, args.seed, "demos/trained_demo.gif"
            )
            print_narrative("Trained LLM", t_stats, t_events)
            print(f"\n  GIF  -> {t_gif}")
            print(f"  PNG  -> {t_png}")

            print(f"\n{'='*60}")
            print("  Side-by-Side Comparison")
            print(f"{'='*60}")
            print(f"{'Metric':<20} {'Heuristic':>12} {'Trained LLM':>12}")
            print("-" * 44)
            for key, label in [
                ("total_reward", "Total Reward"),
                ("pop_saved_pct", "Pop Saved %"),
                ("containment_pct", "Containment %"),
                ("steps", "Steps"),
            ]:
                h_val = h_stats[key]
                t_val = t_stats[key]
                fmt = "{:+.2f}" if isinstance(h_val, float) else "{}"
                print(f"{label:<20} {fmt.format(h_val):>12} {fmt.format(t_val):>12}")


if __name__ == "__main__":
    main()
