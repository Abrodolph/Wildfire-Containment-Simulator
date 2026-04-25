# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"   # editable mode with test deps

# Run tests
pytest                           # all tests
pytest tests/test_graders.py     # single test file
pytest -k "test_reward"          # tests matching a pattern

# Run baseline evaluation (both agents, all 3 tiers, default 5 runs)
python scripts/evaluate.py [num_runs]

# Compare evaluation results against saved baselines
python scripts/eval_compare.py

# Start the REST API server on port 7860
python server/app.py
serve                     # via pyproject.toml entry point

# Docker
docker build -t wildfire-sim .
docker run -p 7860:7860 wildfire-sim
```

Validate environment changes by running `scripts/evaluate.py` and comparing scores against `scripts/results.json` baselines. The `HeuristicAgent` score is the primary reference for difficulty scaling.

## Architecture

The simulator is an OpenEnv-compliant RL environment where AI agents dispatch firefighting resources on a grid to protect populated zones from wildfire.

**Core environment** (`env/`): Components orchestrated by `wildfire_env.py`:
- `wildfire_env.py` — Main entry point implementing OpenEnv API (`reset`, `step`, `state`). Manages the 11-step tick sequence, action validation (invalid actions return penalty reward, never crash), and event logging.
- `models.py` — All Pydantic schemas: `Action`, `Observation`, `StepResult`, `TierConfig`. The three `TierConfig` instances (easy/medium/hard) define grid size, resource counts, episode length, and reward weights.
- `grid.py` — Terrain generation (elevation, fuel types, water, populated zones), cell state management, smoke propagation, fog-of-war.
- `fire_spread.py` — Rothermel-inspired cellular automaton. Each burning cell ignites 8 Moore-neighborhood cells based on: `P(ignite) = base_rate × fuel × wind × slope × (1 − moisture) × (1 − suppression) × tier_scale`. Tier scale: easy=1.0, medium=0.7, hard=0.55.
- `weather.py` — Stochastic wind (random walk + shift events), sinusoidal humidity cycle, Poisson rain events.
- `resources.py` — Crew deployment/movement (adjacent cells only), tanker drops (5-step cooldown), firebreak construction, recon budget tracking.
- `reward.py` — Weighted composite of 5 components: containment, population safety, efficiency, speed, area saved. Also computes per-step delta rewards and a terminal reward on episode end.
- `briefing.py` — Generates a structured `OperationalBriefing` on `reset()`, attached to the first `Observation`. Provides incident cause, priority zones, infrastructure labels, and wind forecast for LLM context.
- `serialization.py` — Converts an `Observation` into a structured text prompt for LLM agents via `serialize_observation(obs, step_num, max_steps)`.
- `action_parser.py` — 3-layer LLM output → `Action` parser: direct JSON → regex field extraction → safe IDLE fallback.
- `curriculum.py` — `CurriculumController` for auto-promoting agents across tiers based on a rolling 10-episode average reward.
- `rendering.py` — Renders ground-truth state dicts into RGB frames for episode replay GIFs.

**Agents** (`agents/`): `RandomAgent` (lower-bound baseline) and `HeuristicAgent` (priority-based: evacuate endangered crews → protect population → air support → contain perimeter → recon → idle). New agents implement `act(obs: Observation) -> Action`.

**Graders** (`graders/`): `grade(agent, seed=42) -> float` for each tier. Called by `scripts/evaluate.py` to benchmark.

**Server** (`server/app.py`): FastAPI wrapping a singleton `WildfireEnv`. Endpoints: `POST /reset?task_id=easy&seed=42`, `POST /step` (Action JSON body), `GET /state`, `GET /health`.

**LLM inference** (`inference.py`): Runs an OpenAI-compatible client against the three tasks. Requires env vars `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`.

**Scripts** (`scripts/`): `evaluate.py` (benchmark), `eval_compare.py` (diff vs baselines), `replay.py` (GIF generation), `plot_dashboard.py` (metrics visualization), `find_demo_seed.py` (search for visually interesting seeds), `run_demo.py`.

## Key Conventions

- All external data uses Pydantic models — never bypass validation at the `env/` boundary.
- Invalid actions return a penalty reward and continue the episode; they never raise exceptions.
- All env components use the 8-cell Moore neighborhood consistently.
- `reset(task_id, seed)` must be fully deterministic — use `np.random.default_rng(seed)` and pass the RNG down to all components.
- Agents must not access `state()` (ground truth) during normal execution — only the `Observation` returned by `reset`/`step`.
- Hard tier enables staggered ignition (a third fire spawns mid-episode) and crew loss events; both are configured via `TierConfig` fields.
