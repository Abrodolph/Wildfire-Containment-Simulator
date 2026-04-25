"""
Generate supervised fine-tuning (SFT) training examples by running the
HeuristicAgent through episodes and recording (prompt, action) pairs.

Usage:
    python scripts/generate_sft_data.py
    python scripts/generate_sft_data.py --output training/sft_data.jsonl --easy-seeds 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from env.wildfire_env import WildfireEnv
from env.serialization import serialize_observation
from env.models import TIER_EASY, TIER_MEDIUM, TIER_HARD, ActionType
from agents.heuristic_agent import HeuristicAgent

SYSTEM_PROMPT = (
    "You are an AI Incident Commander managing wildfire containment. "
    "You will receive a situation briefing each step. "
    "Respond with ONLY a valid JSON action object and nothing else. "
    'Example: {"action_type": "idle"}'
)

TIER_CONFIGS = {
    "easy":   {"max_steps": TIER_EASY.episode_length,   "target": 2000},
    "medium": {"max_steps": TIER_MEDIUM.episode_length, "target": 1500},
    "hard":   {"max_steps": TIER_HARD.episode_length,   "target": 800},
}


def run_episode(tier: str, seed: int) -> list[dict] | None:
    """Run a full episode with the HeuristicAgent.

    Returns a list of raw (prompt, action, step) records for the episode,
    or None if the episode is unsuccessful (population lost > 0).
    """
    max_steps = TIER_CONFIGS[tier]["max_steps"]
    env = WildfireEnv()
    obs = env.reset(task_id=tier, seed=seed)
    agent = HeuristicAgent()

    offset = random.randint(0, min(30, max_steps // 4))

    prev_cells_burning = 0
    records: list[dict] = []
    step_num = 0

    while not env.done:
        action = agent.act(obs)

        if step_num >= offset:
            prompt_text = serialize_observation(
                obs, step_num, max_steps,
                tier=tier, prev_cells_burning=prev_cells_burning,
            )
            action_json = action.model_dump_json(exclude_none=True)
            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "completion": action_json,
                "tier": tier,
                "seed": seed,
                "step": step_num,
                "action_type": action.action_type.value,
            })

        prev_cells_burning = obs.stats.cells_burning
        result = env.step(action)
        obs = result.observation
        step_num += 1

    state = env.state()
    if state["population_lost"] != 0:
        return None

    return records


def filter_idle(records: list[dict]) -> list[dict]:
    """Keep all non-IDLE steps, then cap IDLE steps at 20% of total."""
    non_idle = [r for r in records if r["action_type"] != "idle"]
    idle = [r for r in records if r["action_type"] == "idle"]

    if not non_idle:
        return idle

    max_idle = max(1, int(len(non_idle) * 0.25))
    if len(idle) > max_idle:
        random.shuffle(idle)
        idle = idle[:max_idle]

    combined = non_idle + idle
    combined.sort(key=lambda r: r["step"])
    return combined


def strip_internal_fields(records: list[dict]) -> list[dict]:
    """Remove the action_type helper field before writing."""
    for r in records:
        r.pop("action_type", None)
    return records


def generate(output_path: str, max_seeds: dict[str, int]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_examples: list[dict] = []
    tier_counts = {t: 0 for t in TIER_CONFIGS}

    for tier in ["easy", "medium", "hard"]:
        target = TIER_CONFIGS[tier]["target"]
        limit = max_seeds[tier]
        seed = 0

        print(f"\n{'='*50}")
        print(f"Generating {tier} tier  (target={target}, max_seeds={limit})")
        print(f"{'='*50}")

        while tier_counts[tier] < target and seed < limit:
            records = run_episode(tier, seed)

            if records is not None:
                filtered = filter_idle(records)
                remaining = target - tier_counts[tier]
                if len(filtered) > remaining:
                    filtered = filtered[:remaining]
                all_examples.extend(strip_internal_fields(filtered))
                tier_counts[tier] += len(filtered)

            seed += 1
            if seed % 50 == 0:
                print(f"  [{tier}] seed={seed}, examples={tier_counts[tier]}/{target}")

        print(f"  [{tier}] DONE — {tier_counts[tier]} examples from {seed} seeds")

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = len(all_examples)
    print(f"\n{'='*50}")
    print(f"SFT data saved to {output_path}")
    print(f"Total examples: {total}")
    print(f"Tier distribution:")
    for tier in ["easy", "medium", "hard"]:
        print(f"  {tier}: {tier_counts[tier]}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data from HeuristicAgent episodes")
    parser.add_argument("--output", default="training/sft_data.jsonl",
                        help="Output JSONL file path (default: training/sft_data.jsonl)")
    parser.add_argument("--easy-seeds", type=int, default=500,
                        help="Max seeds to try for easy tier")
    parser.add_argument("--medium-seeds", type=int, default=500,
                        help="Max seeds to try for medium tier")
    parser.add_argument("--hard-seeds", type=int, default=500,
                        help="Max seeds to try for hard tier")
    args = parser.parse_args()

    max_seeds = {
        "easy": args.easy_seeds,
        "medium": args.medium_seeds,
        "hard": args.hard_seeds,
    }
    generate(args.output, max_seeds)


if __name__ == "__main__":
    main()
