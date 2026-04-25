"""Grader for Task 3 (Hard): Full production chaos."""

from __future__ import annotations
from env import WildfireEnv


def grade(agent, seed: int = 42):
    """
    Run a full episode on Hard tier.

    Returns:
        Tuple of (total_reward: float, details: dict)
    """
    env = WildfireEnv()
    obs = env.reset(task_id="hard", seed=seed)
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
        "containment_pct": round(final.get("reward_breakdown", {}).get("containment", 0.0), 4),
        "pop_saved_pct": round(1.0 - pop_lost / total_pop, 4),
        "steps": env.current_step,
        "crew_casualty": env._crew_casualty_occurred,
    }
    return total_reward, details
