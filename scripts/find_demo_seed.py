"""
Find demo seeds on medium tier where the heuristic struggles interestingly.

Filters for seeds where:
  (a) a wind shift fires between step 60-90
  (b) heuristic loses at least one populated cell
  (c) heuristic total_reward is between -4.0 and +2.0

Usage:
    python scripts/find_demo_seed.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import WildfireEnv
from agents.heuristic_agent import HeuristicAgent

TIER = "medium"
MAX_SEED = 500


def scan_seed(seed):
    env = WildfireEnv()
    agent = HeuristicAgent()
    obs = env.reset(task_id=TIER, seed=seed)

    total_reward = 0.0
    wind_shift_step = None
    done = False

    while not done:
        action = agent.act(obs)
        result = env.step(action)
        total_reward += result.reward

        for event in result.info.get("events", []):
            if "WIND SHIFT" in event and wind_shift_step is None:
                wind_shift_step = env.current_step

        obs = result.observation
        done = result.done

    final = env.state()
    pop_lost = final.get("population_lost", 0)
    total_pop = final.get("total_population", 1) or 1

    return {
        "seed": seed,
        "total_reward": round(total_reward, 3),
        "pop_lost": pop_lost,
        "pop_saved_pct": round(1.0 - pop_lost / total_pop, 3),
        "wind_shift_step": wind_shift_step,
        "steps": env.current_step,
        "containment_pct": round(final.get("containment_pct", 0.0), 3),
    }


def main():
    candidates = []
    print(f"Scanning seeds 0-{MAX_SEED - 1} on {TIER} tier...")

    for seed in range(MAX_SEED):
        if seed % 50 == 0:
            print(f"  seed {seed}...")
        info = scan_seed(seed)

        wind_ok = (info["wind_shift_step"] is not None
                   and 60 <= info["wind_shift_step"] <= 90)
        pop_ok = info["pop_lost"] >= 1
        reward_ok = -4.0 <= info["total_reward"] <= 2.0

        if wind_ok and pop_ok and reward_ok:
            candidates.append(info)

    candidates.sort(key=lambda x: x["total_reward"], reverse=True)
    top5 = candidates[:5]

    for c in top5:
        ws = c["wind_shift_step"]
        print(f"  seed={c['seed']:3d}  reward={c['total_reward']:+.2f}  "
              f"pop_lost={c['pop_lost']}  wind_shift=step {ws}  "
              f"steps={c['steps']}")

    os.makedirs("demos", exist_ok=True)
    with open("demos/candidate_seeds.json", "w") as f:
        json.dump(top5, f, indent=2)
    print(f"\nTop {len(top5)} candidates saved -> demos/candidate_seeds.json")

    if not top5:
        print("No candidates matched all 3 filters — relaxing pop_lost filter...")
        fallback = [scan_seed(s) for s in [42, 7, 13, 99, 123]]
        fallback.sort(key=lambda x: x["total_reward"], reverse=True)
        with open("demos/candidate_seeds.json", "w") as f:
            json.dump(fallback, f, indent=2)
        print("Saved fallback candidates.")


if __name__ == "__main__":
    main()
