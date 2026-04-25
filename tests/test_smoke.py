from env.models import Action, ActionType


def test_env_resets_on_all_tiers(fresh_env):
    for tier in ["easy", "medium", "hard"]:
        obs = fresh_env.reset(task_id=tier, seed=42)
        assert obs is not None


def test_idle_action_never_crashes(fresh_env):
    fresh_env.reset(task_id="easy", seed=42)
    for _ in range(10):
        result = fresh_env.step(Action(action_type=ActionType.IDLE))
        assert result is not None


def test_determinism(fresh_env):
    def run_rollout(env):
        env.reset(task_id="easy", seed=42)
        result = None
        for _ in range(20):
            result = env.step(Action(action_type=ActionType.IDLE))
        return result.observation.stats.cells_burned

    burned_1 = run_rollout(fresh_env)
    burned_2 = run_rollout(fresh_env)
    assert burned_1 == burned_2
