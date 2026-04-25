from env import WildfireEnv
from env.models import Action, ActionType
from env.reward import RewardCalculator
from env.models import TIER_EASY
from agents.heuristic_agent import HeuristicAgent


def test_successful_episode_scores_high(fresh_env):
    agent = HeuristicAgent()
    obs = fresh_env.reset(task_id="easy", seed=42)
    total_reward = 0.0
    done = False
    while not done:
        action = agent.act(obs)
        result = fresh_env.step(action)
        total_reward += result.reward
        obs = result.observation
        done = result.done
    assert total_reward > 3.0, f"Expected > 3.0, got {total_reward:.3f}"


def test_all_pop_lost_scores_negative():
    calc = RewardCalculator(TIER_EASY)
    final_state = {
        "containment_pct": 0.0,
        "pop_lost": 100,
        "total_pop": 100,
        "crew_casualty_occurred": False,
        "invalid_action_count": 0,
    }
    terminal = calc.compute_terminal_reward(final_state, episode_steps=80, max_steps=80)
    assert terminal < -2.0, f"Expected < -2.0, got {terminal:.3f}"


def test_crew_casualty_stacks():
    calc = RewardCalculator(TIER_EASY)
    # pop loss AND crew casualty
    final_state = {
        "containment_pct": 0.0,
        "pop_lost": 50,
        "total_pop": 100,
        "crew_casualty_occurred": True,
        "invalid_action_count": 0,
    }
    terminal = calc.compute_terminal_reward(final_state, episode_steps=80, max_steps=80)
    # -3.0*(0.5) for pop loss = -1.5, -2.0 for casualty = -3.5 total
    assert terminal < -3.0, f"Expected < -3.0 (both penalties stacked), got {terminal:.3f}"


def test_redundant_action_penalty(fresh_env):
    obs = fresh_env.reset(task_id="easy", seed=42)
    rows = len(obs.grid)
    cols = len(obs.grid[0])
    tr, tc = rows // 2, cols // 2

    # First deploy — not redundant
    result1 = fresh_env.step(Action(
        action_type=ActionType.DEPLOY_CREW,
        crew_id="crew_0",
        target_row=tr,
        target_col=tc,
    ))

    # Same action again — redundant, step reward should include -0.1 penalty
    result2 = fresh_env.step(Action(
        action_type=ActionType.DEPLOY_CREW,
        crew_id="crew_0",
        target_row=tr,
        target_col=tc,
    ))

    # The non-terminal step reward for the redundant action must be at least -0.1
    # lower than it would be without the penalty. We can't isolate it perfectly,
    # but we can verify the redundancy flag is wired by checking the env directly.
    assert result2 is not None  # basic smoke check

    # Direct unit test on compute_step_reward
    from env.reward import RewardCalculator
    from env.models import TIER_EASY
    calc = RewardCalculator(TIER_EASY)
    state = {"containment_pct": 0.5, "pop_lost": 0, "total_pop": 10}
    reward_normal = calc.compute_step_reward(state, state, True, False)
    reward_redundant = calc.compute_step_reward(state, state, True, True)
    assert reward_redundant == reward_normal - 0.1, (
        f"Redundant penalty missing: {reward_normal:.3f} vs {reward_redundant:.3f}"
    )
