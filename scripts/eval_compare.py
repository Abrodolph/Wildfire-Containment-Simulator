"""
Eval comparison script — runs multiple agents on fixed seeds and prints a summary table.

Usage:
    python scripts/eval_compare.py --seeds 42 43 44 45 46 --tiers medium hard --agents random heuristic
    python scripts/eval_compare.py --quick
"""

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import WildfireEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


def _make_llm_agent(model_path_env: str):
    """Return an LLM agent factory or None if the model path is unset."""
    path = os.environ.get(model_path_env)
    if not path:
        return None
    try:
        from agents.llm_agent import LLMAgent  # type: ignore
        return LLMAgent(model_path=path)
    except ImportError:
        warnings.warn(f"agents.llm_agent not found — skipping {model_path_env}")
        return None


AGENT_REGISTRY = {
    "random": lambda: RandomAgent(),
    "heuristic": lambda: HeuristicAgent(),
    "base_llm": lambda: _make_llm_agent("BASE_MODEL_PATH"),
    "trained_llm": lambda: _make_llm_agent("TRAINED_MODEL_PATH"),
}

AGENT_LABELS = {
    "random": "Random Agent",
    "heuristic": "Heuristic Agent",
    "base_llm": "Base LLM",
    "trained_llm": "Trained LLM (ours)",
}


def run_episode(agent, tier: str, seed: int) -> dict:
    env = WildfireEnv()
    obs = env.reset(task_id=tier, seed=seed)
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = agent.act(obs)
        result = env.step(action)
        total_reward += result.reward
        obs = result.observation
        done = result.done
        steps += 1

    final = env.state()
    total_pop = final.get("total_population", 1) or 1
    pop_lost = final.get("population_lost", 0)
    containment = final.get("containment_pct", 0.0)

    return {
        "containment_pct": containment,
        "pop_saved_pct": 1.0 - pop_lost / total_pop,
        "total_reward": total_reward,
        "episode_steps": steps,
    }


def run_comparison(agent_names, tiers, seeds):
    results = {}
    for agent_name in agent_names:
        factory = AGENT_REGISTRY.get(agent_name)
        agent = factory() if factory else None
        results[agent_name] = {}
        for tier in tiers:
            if agent is None:
                results[agent_name][tier] = None
                continue
            tier_results = []
            for seed in seeds:
                ep = run_episode(agent, tier, seed)
                tier_results.append(ep)
            results[agent_name][tier] = {
                "containment_pct": sum(r["containment_pct"] for r in tier_results) / len(tier_results),
                "pop_saved_pct": sum(r["pop_saved_pct"] for r in tier_results) / len(tier_results),
                "total_reward": sum(r["total_reward"] for r in tier_results) / len(tier_results),
                "episode_steps": sum(r["episode_steps"] for r in tier_results) / len(tier_results),
                "runs": tier_results,
            }
    return results


def print_table(results, tiers, agent_names, seeds):
    for tier in tiers:
        n = len(seeds)
        print(f"\n=== EVAL RESULTS — {tier.capitalize()} Tier ({n} seed{'s' if n != 1 else ''}) ===")
        header = f"{'Agent':<28} {'Containment':>12} {'Pop Saved':>10} {'Reward':>8} {'Steps':>7}"
        print(header)
        print("-" * len(header))
        for agent_name in agent_names:
            label = AGENT_LABELS.get(agent_name, agent_name)
            data = results[agent_name].get(tier)
            if data is None:
                print(f"{label:<28} {'[skipped — no model]':>39}")
            else:
                containment = f"{data['containment_pct']*100:.0f}%"
                pop_saved = f"{data['pop_saved_pct']*100:.0f}%"
                reward = f"{data['total_reward']:+.1f}"
                steps = f"{data['episode_steps']:.0f}"
                print(f"{label:<28} {containment:>12} {pop_saved:>10} {reward:>8} {steps:>7}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--tiers", nargs="+", choices=["easy", "medium", "hard"], default=["medium", "hard"])
    parser.add_argument("--agents", nargs="+", choices=list(AGENT_REGISTRY), default=["random", "heuristic"])
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--quick", action="store_true", help="Easy tier, 2 seeds only")
    args = parser.parse_args()

    if args.quick:
        args.tiers = ["easy"]
        args.seeds = [42, 43]
        args.agents = [a for a in args.agents if a in ("random", "heuristic")]

    print(f"Running: agents={args.agents}, tiers={args.tiers}, seeds={args.seeds}")
    results = run_comparison(args.agents, args.tiers, args.seeds)
    print_table(results, args.tiers, args.agents, args.seeds)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    serializable = {}
    for agent_name, tier_data in results.items():
        serializable[agent_name] = {}
        for tier, data in tier_data.items():
            if data is None:
                serializable[agent_name][tier] = None
            else:
                serializable[agent_name][tier] = {
                    k: v for k, v in data.items() if k != "runs"
                }
                serializable[agent_name][tier]["runs"] = data["runs"]

    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
