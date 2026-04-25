# Contributing

## Adding a new tier

1. Define a new `TierConfig` instance in `env/models.py` (follow the pattern of `TIER_EASY/MEDIUM/HARD`).
2. Register it in `WildfireEnv.TIER_MAP` in `env/wildfire_env.py`.
3. Add a grader in `graders/grader_<name>.py` returning `(total_reward, details_dict)`.
4. Add the task to `openenv.yaml` under `tasks:`.

## Adding a new action type

1. Add the enum value to `ActionType` in `env/models.py`.
2. Add parameter validation to `Action.validate_params()` in the same file.
3. Handle the new action in `WildfireEnv._execute_action()` in `env/wildfire_env.py`.
4. Add regex extraction for the new type in `env/action_parser.py` Layer 2.
5. Add at least one test in `tests/test_action_parser.py`.

## Where tests live

All tests are in `tests/`. Run with:

```bash
pytest tests/ -v --cov=env
```

Each prompt has a corresponding test file (e.g. `test_reward.py`, `test_briefing.py`). Add new tests to the relevant file or create a new one if the feature is standalone.
