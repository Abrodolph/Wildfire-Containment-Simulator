"""
Wildfire Containment Simulator — Evaluation Script.

Runs both agents (random + heuristic) on all 3 difficulty tiers,
reports scores, and saves results to JSON.
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from graders.grader_easy import grade as grade_easy
from graders.grader_medium import grade as grade_medium
from graders.grader_hard import grade as grade_hard


def run_evaluation(num_runs: int = 5) -> dict:
    graders = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }

    agents = {
        "random": lambda seed: RandomAgent(seed=seed),
        "heuristic": lambda seed: HeuristicAgent(),
    }

    results = {}

    print("=" * 80)
    print("WILDFIRE CONTAINMENT SIMULATOR — Evaluation")
    print("=" * 80)
    print()

    for agent_name, agent_factory in agents.items():
        results[agent_name] = {}
        for tier_name, grader_fn in graders.items():
            scores = []
            detail_rows = []
            times = []

            for run in range(num_runs):
                seed = 42 + run
                agent = agent_factory(seed)

                start = time.time()
                score, details = grader_fn(agent, seed=seed)
                elapsed = time.time() - start

                scores.append(score)
                detail_rows.append(details)
                times.append(elapsed)

            mean_score = sum(scores) / len(scores)
            std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
            mean_containment = sum(d["containment_pct"] for d in detail_rows) / len(detail_rows)
            mean_pop_saved = sum(d["pop_saved_pct"] for d in detail_rows) / len(detail_rows)
            mean_steps = sum(d["steps"] for d in detail_rows) / len(detail_rows)
            casualty_rate = sum(1 for d in detail_rows if d["crew_casualty"]) / len(detail_rows)

            results[agent_name][tier_name] = {
                "scores": [round(s, 4) for s in scores],
                "mean": round(mean_score, 4),
                "std": round(std_score, 4),
                "mean_containment_pct": round(mean_containment, 4),
                "mean_pop_saved_pct": round(mean_pop_saved, 4),
                "mean_steps": round(mean_steps, 1),
                "crew_casualty_rate": round(casualty_rate, 2),
                "mean_time_s": round(sum(times) / len(times), 3),
            }

            print(f"  {agent_name:12s} | {tier_name:8s} | "
                  f"reward={mean_score:+.2f}+-{std_score:.2f} | "
                  f"contain={mean_containment*100:.0f}% | "
                  f"pop_saved={mean_pop_saved*100:.0f}% | "
                  f"steps={mean_steps:.0f}")

        print()

    print("=" * 80)
    print(f"{'Agent':>12s} | {'Easy':>10s} | {'Medium':>10s} | {'Hard':>10s}")
    print("-" * 80)
    for agent_name in agents:
        easy = results[agent_name]["easy"]["mean"]
        medium = results[agent_name]["medium"]["mean"]
        hard = results[agent_name]["hard"]["mean"]
        print(f"{agent_name:>12s} | {easy:>+10.2f} | {medium:>+10.2f} | {hard:>+10.2f}")
    print("=" * 80)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_evaluation(num_runs=num_runs)
