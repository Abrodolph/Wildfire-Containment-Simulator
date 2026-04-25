"""
Evaluate a trained HF adapter model against heuristic and random baselines
on the Wildfire Containment Simulator.

Saves results to scripts/trained_results.json.

Usage:
    python scripts/eval_trained_model.py --model-path Eshit/wildfire-grpo-7b
    python scripts/eval_trained_model.py --model-path ./grpo_final --num-seeds 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from env.wildfire_env import WildfireEnv
from env.serialization import serialize_observation
from env.action_parser import parse_action
from env.models import TIER_EASY, TIER_MEDIUM, TIER_HARD
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent

TIER_MAX_STEPS = {
    "easy": TIER_EASY.episode_length,
    "medium": TIER_MEDIUM.episode_length,
    "hard": TIER_HARD.episode_length,
}

SYSTEM_PROMPT = (
    "You are an AI Incident Commander managing wildfire containment. "
    "You will receive a situation briefing each step. "
    "Respond with ONLY a valid JSON action object and nothing else. "
    'Example: {"action_type": "idle"}'
)


class LLMAgent:
    """
    Wraps the trained model for grader compatibility.
    Must be re-instantiated for every episode — _step and _prev_burning
    are per-episode state and will produce wrong prompts if reused.
    """

    def __init__(self, model, tokenizer, tier, max_steps):
        self.model = model
        self.tokenizer = tokenizer
        self.tier = tier
        self.max_steps = max_steps
        self._step = 0
        self._prev_burning = 0
        self.json_success = self.regex_fallback = self.safe_idle = 0

    def act(self, obs):
        import torch

        prompt = serialize_observation(
            obs, self._step, self.max_steps,
            tier=self.tier,
            prev_cells_burning=self._prev_burning,
        )
        self._prev_burning = obs.stats.cells_burning
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids, max_new_tokens=128,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        action, status = parse_action(text, obs)
        if status == "json_success":
            self.json_success += 1
        elif status == "regex_fallback":
            self.regex_fallback += 1
        else:
            self.safe_idle += 1
        self._step += 1
        return action


def run_llm_episode(model, tokenizer, tier, seed):
    """Run a full episode with a fresh LLMAgent. Returns (reward, details)."""
    max_steps = TIER_MAX_STEPS[tier]
    agent = LLMAgent(model, tokenizer, tier, max_steps)
    env = WildfireEnv()
    obs = env.reset(task_id=tier, seed=seed)
    total_reward = 0.0

    while not env.done:
        action = agent.act(obs)
        result = env.step(action)
        total_reward += result.reward
        obs = result.observation

    final = env.state()
    total_pop = final.get("total_population", 1) or 1
    pop_lost = final.get("population_lost", 0)

    details = {
        "total_reward": round(total_reward, 4),
        "containment_pct": round(
            final.get("reward_breakdown", {}).get("containment", 0.0), 4
        ),
        "pop_saved_pct": round(1.0 - pop_lost / total_pop, 4),
        "steps": env.current_step,
        "crew_casualty": env._crew_casualty_occurred,
        "json_success": agent.json_success,
        "regex_fallback": agent.regex_fallback,
        "safe_idle": agent.safe_idle,
    }
    return total_reward, details


def load_model(model_path: str, base_model: str):
    """Load a trained model, handling both full repos and PEFT adapters."""
    from unsloth import FastLanguageModel

    # Try loading directly (works for merged models and HF adapter repos
    # that embed base_model_name_or_path in adapter_config.json)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        print(f"Loaded model directly from: {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Direct load failed ({e}), trying base + adapter...")

    # Fallback: load base model then attach adapter (for standalone PEFT adapters)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model.load_adapter(model_path, adapter_name="default")
    print(f"Loaded base model ({base_model}) + adapter ({model_path})")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model vs baselines")
    parser.add_argument("--model-path", required=True,
                        help="HF hub ID or local path to the trained adapter")
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-7B-Instruct",
                        help="Base model for PEFT adapter loading "
                             "(default: unsloth/Qwen2.5-7B-Instruct)")
    parser.add_argument("--num-seeds", type=int, default=15,
                        help="Evaluation seeds per tier (default: 15, uses seeds 200+)")
    parser.add_argument("--tiers", nargs="+", default=["easy", "medium", "hard"],
                        help="Tiers to evaluate (default: easy medium hard)")
    args = parser.parse_args()

    seeds = list(range(200, 200 + args.num_seeds))

    # Load trained model (Issue 1 fix: uses --base-model for adapter fallback)
    print(f"Loading model: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    print("Model ready for inference.\n")

    # Load existing baselines (Issue 3 fix: use stored values for comparison table)
    baselines_path = os.path.join(os.path.dirname(__file__), "results.json")
    if not os.path.exists(baselines_path):
        print(f"WARNING: {baselines_path} not found. Run scripts/evaluate.py first.")
        sys.exit(1)
    with open(baselines_path, "r") as f:
        baselines = json.load(f)

    # Output in same shape as results.json: {agent: {tier: {...}}}  (Issue 2 fix)
    all_results = {"trained": {}}

    for tier in args.tiers:
        max_steps = TIER_MAX_STEPS[tier]
        print(f"{'='*60}")
        print(f"  Tier: {tier}  |  Seeds: {seeds[0]}-{seeds[-1]}  |  Max steps: {max_steps}")
        print(f"{'='*60}")

        tier_rewards = []
        tier_pop_saved = []
        tier_containment = []
        tier_json_success = 0
        tier_total_actions = 0
        tier_casualty_count = 0
        tier_times = []

        for seed in seeds:
            start = time.time()
            reward, details = run_llm_episode(model, tokenizer, tier, seed)
            elapsed = time.time() - start

            tier_rewards.append(reward)
            tier_pop_saved.append(details["pop_saved_pct"])
            tier_containment.append(details["containment_pct"])
            tier_json_success += details["json_success"]
            tier_total_actions += (details["json_success"]
                                   + details["regex_fallback"]
                                   + details["safe_idle"])
            if details["crew_casualty"]:
                tier_casualty_count += 1
            tier_times.append(elapsed)

            print(f"  seed={seed}: reward={reward:+.2f}, "
                  f"pop_saved={details['pop_saved_pct']*100:.0f}%, "
                  f"steps={details['steps']}, time={elapsed:.1f}s")

        json_rate = (100.0 * tier_json_success / tier_total_actions
                     if tier_total_actions > 0 else 0)

        all_results["trained"][tier] = {
            "scores": [round(r, 4) for r in tier_rewards],
            "mean": round(float(np.mean(tier_rewards)), 4),
            "std": round(float(np.std(tier_rewards)), 4),
            "mean_containment_pct": round(float(np.mean(tier_containment)), 4),
            "mean_pop_saved_pct": round(float(np.mean(tier_pop_saved)), 4),
            "crew_casualty_rate": round(tier_casualty_count / len(seeds), 2),
            "mean_time_s": round(float(np.mean(tier_times)), 3),
            "json_success_rate": round(json_rate, 2),
        }
        print()

    # ── Print comparison table using stored baselines ──
    print()
    print("=" * 65)
    print("=== Evaluation: Trained Model vs Baselines ===")
    print(f"Model:  {args.model_path}")
    print(f"Seeds:  {seeds[0]}-{seeds[-1]}  ({len(seeds)} per tier)")
    print("=" * 65)
    print(f"{'Tier':<10} {'Trained':>12} {'Heuristic':>12} {'Random':>12} {'vs Heuristic':>14}")
    print("-" * 65)

    for tier in args.tiers:
        t = all_results["trained"][tier]
        h_mean = baselines["heuristic"][tier]["mean"]
        h_std = baselines["heuristic"][tier]["std"]
        r_mean = baselines["random"][tier]["mean"]
        r_std = baselines["random"][tier]["std"]
        delta = t["mean"] - h_mean
        marker = " OK" if delta >= -1.0 else ""
        print(
            f"{tier:<10} "
            f"{t['mean']:+.2f}+/-{t['std']:.1f}  "
            f"{h_mean:+.2f}+/-{h_std:.1f}  "
            f"{r_mean:+.2f}+/-{r_std:.1f}  "
            f"{delta:+.2f}{marker}"
        )

    print()
    print("JSON success rate:  ", end="")
    print("  ".join(
        f"{t}={all_results['trained'][t]['json_success_rate']:.1f}%"
        for t in args.tiers
    ))
    print("Pop saved rate:     ", end="")
    print("  ".join(
        f"{t}={all_results['trained'][t]['mean_pop_saved_pct']*100:.0f}%"
        for t in args.tiers
    ))
    print("=" * 65)

    # ── Save results (same top-level shape as results.json) ──
    output_path = os.path.join(os.path.dirname(__file__), "trained_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
