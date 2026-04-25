# Repository Guidelines

## Project Structure & Module Organization
Core simulation code lives in `env/`, including fire spread, weather, rewards, rendering, serialization, and the main `WildfireEnv`. Baseline agents are in `agents/`, difficulty graders in `graders/`, and HTTP serving code in `server/` with the entrypoint at `server/app.py`. Utility scripts such as evaluation, replay, and demo generation live in `scripts/`. Tests are centralized in `tests/`, training material is under `training/`, and generated media belongs in `demos/`.

## Build, Test, and Development Commands
Install dependencies with `uv pip install -r requirements.txt` and `uv pip install -e .`. Run the test suite with `pytest tests -v` or include coverage via `pytest tests -v --cov=env`. Start the local API with `python app.py` or `python -m server.app`; both serve FastAPI on port `7860`. Common workflows:

- `python scripts/evaluate.py 5` runs baseline evaluation across tiers.
- `python scripts/eval_compare.py --seeds 42 43 44 --tiers medium hard --agents random heuristic` compares agents.
- `python scripts/run_demo.py` generates the demo GIF.
- `python scripts/replay.py --tier medium --seed 42 --agent heuristic --output demos/replay.gif` replays one episode.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for Pydantic models and classes, and descriptive enum names such as `ActionType.DEPLOY_CREW`. Keep validation close to models in `env/models.py` and environment execution logic in `env/wildfire_env.py`. No formatter config is checked in, so preserve the surrounding style and keep imports straightforward.

## Testing Guidelines
Use `pytest`; test discovery is configured in `pyproject.toml` to read from `tests/`. Name files `test_<feature>.py` and add focused cases near related coverage, for example parser changes in `tests/test_action_parser.py`. For new actions or tiers, add both behavioral tests and at least one regression test for invalid or edge-case inputs.

## Commit & Pull Request Guidelines
This workspace does not include `.git`, so repository history is not available for direct inspection. Use short, imperative commit subjects such as `Add hard-tier recon regression tests`. In pull requests, include a concise summary, list affected modules, note test commands run, and attach screenshots or GIFs when changing rendering, replay, or demo output.

## Configuration & Contribution Notes
Update `openenv.yaml` when adding tasks, and keep grader/task IDs aligned with `WildfireEnv.TIER_MAP`. When adding a new action, update `env/models.py`, `env/wildfire_env.py`, `env/action_parser.py`, and the corresponding tests together to avoid contract drift.
