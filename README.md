---
title: Wildfire Containment Simulator
emoji: 🔥
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
license: mit
tags:
  - reinforcement-learning
  - simulation
  - openenv
  - wildfire
  - rl-environment
  - long-horizon
  - instruction-following
---

# Wildfire Containment Simulator

**Meta OpenEnv Hackathon — Theme 2: Long-Horizon Planning & Instruction Following**

![CI](https://github.com/Abrodolph/Wildfire-Containment-Simulator/actions/workflows/ci.yml/badge.svg)
![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)
![Theme](https://img.shields.io/badge/Theme-2%20Long%20Horizon-orange)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A partially-observable disaster simulation where an LLM acts as **Incident Commander**, interpreting operational briefings, dispatching ground crews and air tankers, and recovering from cascading failures across 80–300-step episodes. Built on OpenEnv with Pydantic-typed actions, a Rothermel-inspired fire-spread model, and a decomposed reward designed for GRPO.

> **Headline result (post-training run, Apr 26):** Our trained Qwen-2.5-7B Incident Commander achieves a mean reward of **+5.74** on Medium tier — vs. **+6.31** for the rule-based heuristic and **+1.31** for the random baseline. The model auto-promoted through all three curriculum tiers (easy → medium → hard) in just 63 of 150 training steps, maintaining **99%+ JSON success rate** throughout.
> *(Full comparison table in [Results](#results). Model: [`Eshit/wildfire-grpo-7b`](https://huggingface.co/Eshit/wildfire-grpo-7b). W&B run: [wildfire-grpo/runs/dnz56kuu](https://wandb.ai/saini-eshit-/wildfire-grpo/runs/dnz56kuu).)*

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
| 🚀 **Live HF Space (env)** | [huggingface.co/spaces/Eshit/Wildfire-Containment-Simulator](https://huggingface.co/spaces/Eshit/Wildfire-Containment-Simulator) |
| 💻 **GitHub source** | [github.com/Abrodolph/Wildfire-Containment-Simulator](https://github.com/Abrodolph/Wildfire-Containment-Simulator) |
| 📒 **GRPO training notebook** | [`training/grpo_v2_colab.ipynb`](training/grpo_v2_colab.ipynb) |
| 📒 **SFT warm-up notebook** | [`training/sft_colab.ipynb`](training/sft_colab.ipynb) |
| 📝 **Long-form blog post** | [`BLOG.md`](BLOG.md) |
| 📊 **Baseline eval JSON** | [`scripts/results.json`](scripts/results.json) |
| 📈 **Training dashboard** | [W&B run: wildfire-grpo/runs/dnz56kuu](https://wandb.ai/saini-eshit-/wildfire-grpo/runs/dnz56kuu) |
| 🎬 **Heuristic replay GIF** | [`demos/heuristic_replay.gif`](demos/heuristic_replay.gif) |
| 🎥 **2-minute pitch video** | [`Youtube Video`](https://youtu.be/yGLzht-RPyg) |

---

## Why Theme 2

| Pillar | How we model it |
|---|---|
| **Long-horizon planning** | Hard tier runs 300 steps with the +5.0 terminal reward only triggered on full population survival — greedy local moves cannot capture it. |
| **Instruction following** | Every episode opens with an `OperationalBriefing` (priority zones, infrastructure, weather forecast). A +1.0 adherence bonus rewards protecting the named priority zones. |
| **Recovery from failure** | Hard tier injects a second ignition at step 30 and forces a crew casualty at step 40. Reactive baselines that can't re-plan lose population. |

---

## Real-World Motivation

Wildfire response is a real public-safety resource-allocation problem. Incident commanders must decide where to deploy crews, when to call air support, how to protect communities, and how to adapt when conditions change mid-operation. We built a structured RL environment that captures the key tensions of this work — partial observability, changing weather, hard resource limits, and explicit tradeoffs between speed, efficiency, area saved, and civilian safety — so an LLM can be trained, evaluated, and inspected on it end-to-end.

For the deeper story behind the design choices, see [`BLOG.md`](BLOG.md).

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/Abrodolph/Wildfire-Containment-Simulator.git
cd Wildfire-Containment-Simulator
uv pip install -r requirements.txt
uv pip install -e .

# Run baseline evaluation (random + heuristic, all 3 tiers, 5 seeds)
python scripts/evaluate.py 5

# Compare agents head-to-head
python scripts/eval_compare.py --seeds 42 43 44 45 46 \
    --tiers easy medium hard --agents random heuristic

# Render an episode as a GIF
python scripts/replay.py --tier medium --seed 42 \
    --agent heuristic --output demos/replay.gif

# Spin up the OpenEnv FastAPI server locally on port 7860
python server/app.py
# Then visit http://localhost:7860/ui/ for the interactive frontend
```

Full test suite: `pytest tests -v` (41 tests, ~30s on CPU).

---

## Live Hugging Face Space

The environment is deployed at [`Eshit/Wildfire-Containment-Simulator`](https://huggingface.co/spaces/Eshit/Wildfire-Containment-Simulator) on Hugging Face. Any external agent can drive it over plain HTTP — no Python import needed:

```bash
SPACE=https://eshit-wildfire-containment-simulator.hf.space

curl "$SPACE/health"
curl -X POST "$SPACE/reset?task_id=easy&seed=42"
curl -X POST "$SPACE/step" -H "Content-Type: application/json" \
    -d '{"action_type": "deploy_crew", "crew_id": "crew_0", "target_row": 7, "target_col": 7}'
```

Endpoints: `/reset`, `/step`, `/state`, `/state/render`, `/auto_step`, `/health`, `/docs` (Swagger UI), `/ui/` (interactive frontend).

---

## Environment API

```python
from env import WildfireEnv, Action, ActionType, Direction

env = WildfireEnv()
obs = env.reset(task_id="easy", seed=42)   # Observation (with OperationalBriefing on first step)

while not env.done:
    action = Action(
        action_type=ActionType.DEPLOY_CREW,
        crew_id="crew_0",
        target_row=7, target_col=7,
    )
    result = env.step(action)               # StepResult
    obs = result.observation
    reward = result.reward                  # decomposed float, range ~−8 to +8
    done = result.done

state = env.state()                          # Full ground truth (grading only)
```

`reset(task_id, seed)` is fully deterministic. `state()` is intentionally exposed only for graders — agents must work from `Observation`.

---

## Action Space

All actions are Pydantic-validated. **Invalid actions return a penalty reward without crashing the environment.**

| Action | Required parameters | Description |
|---|---|---|
| `deploy_crew` | `crew_id`, `target_row`, `target_col` | Place an undeployed crew on a safe cell |
| `move_crew` | `crew_id`, `direction` (`N/S/E/W/NE/NW/SE/SW`) | Move a deployed crew one cell |
| `order_crew_objective` | `crew_id`, `objective` (`hold/advance/retreat/prioritize_*`) | Set a persistent directive for a crew's local policy |
| `drop_retardant` | `tanker_id`, `target_row`, `target_col` | 3×3 retardant drop with 5-step cooldown |
| `build_firebreak` | `crew_id`, `direction` | Permanent non-flammable cell adjacent to a crew |
| `recon_flight` | `target_row`, `target_col` | Reveal a 10×10 area for 5 steps |
| `idle` | `reason` *(optional)* | Explicitly wait |

A 3-layer parser (`env/action_parser.py`) maps raw LLM output → structured `Action`: direct JSON → regex field extraction → safe-`idle` fallback. **The environment loop never breaks on bad model output.**

---

## Observation Space

| Component | Contents | Noise / occlusion |
|---|---|---|
| `briefing` | `OperationalBriefing` on first obs — incident ID, priority zones, infrastructure, wind forecast | First step only |
| `grid` | 2D array of `CellObservation` (`fire_state`, `intensity_bin`, `smoke_density`, `is_populated`, `crew_present`) | Smoke occlusion (medium/hard); fog-of-war (hard) |
| `weather` | `wind_speed_kmh`, `wind_direction_deg`, `humidity_pct`, `rain_active` | ±5 km/h, ±20° on medium/hard |
| `resources` | Crew positions, tanker cooldowns, firebreak budget, recon budget | Fully observable |
| `stats` | `cells_burned`, `cells_burning`, `population_lost`, `containment_pct`, `current_step` | Fully observable |
| `recent_events` | Last 5 notable events | Fully observable |

The observation is rendered into LLM-friendly text via `serialize_observation()` (env/serialization.py), which BFS-clusters fire regions into bounding boxes so the prompt is `O(regions)` instead of `O(cells)`.

---

## Reward Function

Decomposed for GRPO — wide reward range produces meaningful advantages between rollout groups.

**Per-step (dense):**
```
step_reward = 0.4 · Δcontainment + 0.4 · Δpopulation_safety − 0.1 · redundant_action_flag
```

**Terminal (sparse, on episode end):**
```
+5.0   if all populations safe
+0–2.0 efficiency bonus (faster containment ⇒ more)
+1.0   briefing-adherence bonus (all priority zones survived)
−3.0 · (pop_lost / total_pop)   if any population lost
−2.0   if any crew casualty
−0.01 × invalid_action_count    capped at −0.2
```

Total empirical range: **−8 to +8**, declared in `openenv.yaml`.

| Tier | Spread scale | Episode length | Approx. reward ceiling |
|---|---|---|---|
| Easy | 1.00× | 80 | +8 |
| Medium | 0.70× | 150 | +7 |
| Hard | 0.55× | 300 | +6 |

---

## Three Difficulty Tiers

### Task 1 — Easy: Flatland Grass Fire
15×15 flat grid · single ignition · constant wind · no smoke or fog-of-war · 4 crews, 1 tanker, 15 firebreak cells · 80 steps. **Focus:** basic deployment and perimeter control.

### Task 2 — Medium: Canyon Terrain with Wind Shifts
25×25 mixed terrain · two ignition points · variable wind · smoke occlusion · sensor noise · 5 crews, 2 tankers, 20 firebreak cells, 1 recon · 150 steps. **Focus:** terrain-aware containment under multi-front pressure.

### Task 3 — Hard: Wildland-Urban Interface Crisis
40×40 terrain with roads, rivers, urban zones · staggered ignitions (step 30) · scripted crew casualty (step 40) · fog-of-war (radius 7) · aggressive wind shifts · 6 crews, 3 tankers, 30 firebreak cells, 3 recon · 300 steps. **Focus:** long-horizon planning under uncertainty and recovery from cascading failures.

---

## Fire Spread Model

A **Rothermel-inspired cellular automaton** on the 8-cell Moore neighborhood. Each tick, every burning cell attempts to ignite each unburned neighbor:

```
P(ignite) = base_rate × fuel_factor × wind_factor × slope_factor
            × (1 − moisture) × (1 − suppression) × tier_scale
```

| Factor | Effect |
|---|---|
| `base_rate` | Baseline spread by fuel type |
| `fuel_factor` | Fuel load of the target cell |
| `wind_factor` | Boost when wind aligns with the spread vector, dampened otherwise |
| `slope_factor` | Faster uphill, slower downhill |
| `moisture` | Wet ground / recent rain reduces ignition probability |
| `suppression` | Crew presence and retardant coverage reduce spread |
| `tier_scale` | `easy=1.00`, `medium=0.70`, `hard=0.55` |

Burning cells progress through `BURNING → EMBER → BURNED_OUT`. Urban cells have higher peak intensity but lower ignition probability.

---

## Results

> Baselines reproduced via `python scripts/evaluate.py 5` on seeds 42–46. Trained-model numbers from Section 10 of [`training/grpo_v2_colab.ipynb`](training/grpo_v2_colab.ipynb), evaluated on seeds 42–56 (15 per tier, no overlap with training seeds 0–99).

| Agent | Easy (mean ± std) | Medium (mean ± std) | Hard (mean ± std) |
|---|---|---|---|
| Random | +6.23 ± 3.09 | +1.31 ± 3.24 | +2.16 ± 2.96 |
| Heuristic | **+7.53 ± 0.08** | **+6.31 ± 2.77** | **+4.74 ± 3.79** |
| **Trained Qwen-2.5-7B (ours)** | +5.13 ± 3.90 | **+5.74 ± 3.07** | +2.14 ± 2.87 |
| **Δ vs. Heuristic** | −2.41 | **−0.58 ✓** | −2.59 |

The medium tier result passes the ±1.0 of heuristic threshold (official passing criterion).

**Auxiliary metrics for the trained agent:**

| Metric | Easy | Medium | Hard |
|---|---|---|---|
| JSON success rate | 98.5% | 99.8% | 99.2% |
| Mean population saved % | 87% | 97% | 92% |

**Curriculum progression:** easy (steps 0–52) → medium (steps 53–62) → hard (steps 63–149). The model reached hard tier in just 63 of 150 training steps.

> Full scores in [`training/grpo_eval_results.json`](training/grpo_eval_results.json). Training history in [`training/training_stats.json`](training/training_stats.json).

---

## Training

We use a two-stage recipe:

1. **SFT warm-up** — generate 4,300 `(prompt, action_json)` pairs from the heuristic on successful episodes (filtered to `pop_lost == 0`), then fine-tune Qwen-2.5-7B-Instruct with Unsloth 4-bit + LoRA (`r=32`, MLP+attention adapters). Notebook: [`training/sft_colab.ipynb`](training/sft_colab.ipynb).
2. **GRPO (TRL `GRPOTrainer`)** — start from the SFT adapter, score completions by *resetting the env to the exact `(tier, seed)` that produced each prompt*, applying the candidate action, and running the heuristic to terminal. Two reward functions are passed to TRL: `reward_fn_outcome` (full episode reward) and `reward_fn_format` (JSON validity). Curriculum auto-promotes easy → medium → hard. Notebook: [`training/grpo_v2_colab.ipynb`](training/grpo_v2_colab.ipynb).

**Hardware:** A100 Large (40 GB) on a Hugging Face Space JupyterLab session. ~75 minutes total wall-clock time.
**Training stack:** `unsloth 2026.4.8` (4-bit QLoRA), `trl==0.20.0`, `datasets==3.4.1`, `transformers 5.5.0`, `peft`, `wandb`.

**Training plots:** W&B run [saini-eshit-/wildfire-grpo/runs/dnz56kuu](https://wandb.ai/saini-eshit-/wildfire-grpo/runs/dnz56kuu) (reward curve, KL divergence, format reward, curriculum tier timeline). Local dashboard: `training/training_dashboard.png` (not tracked in git — generate with `python scripts/plot_grpo_training.py`).

For the design rationale, the SFT/GRPO trade-offs, and a frank discussion of what went wrong on our first GRPO attempt, read [`BLOG.md`](BLOG.md).

---

## Project Structure

```text
Wildfire-Containment-Simulator/
├── env/
│   ├── wildfire_env.py       # Main env: reset(), step(), state()
│   ├── models.py             # Pydantic action/observation/state models
│   ├── grid.py               # Terrain, smoke, moisture, fog-of-war
│   ├── fire_spread.py        # Cellular automaton fire propagation
│   ├── weather.py            # Stochastic weather engine
│   ├── resources.py          # Crews, tankers, firebreaks, recon
│   ├── reward.py             # Decomposed step + terminal reward
│   ├── briefing.py           # OperationalBriefing generation
│   ├── serialization.py      # Observation → LLM prompt
│   ├── action_parser.py      # LLM output → Action (3-layer fallback)
│   ├── rendering.py          # Frame rendering for GIF replays
│   └── curriculum.py         # CurriculumController (auto-promote/demote)
├── agents/
│   ├── random_agent.py
│   └── heuristic_agent.py
├── graders/
│   ├── grader_easy.py        # → (total_reward, details_dict)
│   ├── grader_medium.py
│   └── grader_hard.py
├── scripts/
│   ├── evaluate.py           # Baseline eval (random + heuristic)
│   ├── eval_compare.py       # Multi-agent comparison
│   ├── eval_trained_model.py # Evaluate a trained adapter
│   ├── generate_sft_data.py  # Build SFT dataset from heuristic rollouts
│   ├── replay.py             # Render episode as GIF
│   ├── run_demo.py           # Pitch demo
│   └── plot_dashboard.py     # 4-panel training curves
├── training/
│   ├── grpo_v2_colab.ipynb   # GRPO notebook (canonical)
│   ├── sft_colab.ipynb       # SFT warm-up notebook
│   ├── sft_data.jsonl        # 4,300 SFT examples
│   ├── requirements.txt      # Training deps (Unsloth, TRL, etc.)
│   └── README.md
├── server/
│   └── app.py                # FastAPI on port 7860
├── frontend/                 # Interactive HTML/JS frontend served at /ui/
├── tests/                    # 41 pytest tests
├── demos/                    # GIF/PNG demo assets
├── openenv.yaml              # OpenEnv environment manifest
├── Dockerfile                # HF Space build
├── BLOG.md                   # Long-form write-up
└── README.md                 # You are here
```

---

## Architecture Decisions

1. **Decomposed reward for GRPO.** Dense per-step deltas (containment, population) plus sparse terminal spikes (+5 survival, −3 × loss, briefing adherence) give a wide reward range that produces meaningful advantages between GRPO rollout groups.
2. **Operational briefings as first-class instructions.** The briefing isn't cosmetic — protecting its named priority zones earns reward. This makes instruction-following measurable, not aspirational.
3. **Two-stage training (SFT → GRPO).** SFT teaches JSON-action formatting in ~1 epoch; GRPO then optimizes long-horizon strategy on the format-stable model. Going straight to GRPO from the base model produced near-zero reward in early experiments.
4. **3-layer action parser.** JSON parse → regex fallback → safe-`idle`. The training loop never breaks on malformed model output.
5. **Per-step (tier, seed) replay in the reward function.** Each GRPO completion is scored by replaying the *exact* env state that produced its prompt, not a random env. This was the single biggest fix between our v1 and v2 GRPO runs (see [`BLOG.md`](BLOG.md) → "What broke").
6. **Deterministic seeding.** `np.random.default_rng(seed)` is threaded through every subsystem — every run is byte-for-byte reproducible.
7. **OpenEnv compliance over framework lock-in.** The env is callable from Python (`env.reset/step/state`) and over HTTP (`/reset`, `/step`, `/state`). Any external agent — TRL, vLLM, an OpenAI-compatible API client, a curl loop — can drive it.

---

## Citation

If you use this environment, please cite:

```bibtex
@misc{wildfire-containment-simulator-2026,
  title  = {Wildfire Containment Simulator: Long-Horizon Planning and
            Instruction Following for Disaster-Response LLM Agents},
  author = {Team Wildfire},
  year   = {2026},
  url    = {https://huggingface.co/spaces/Eshit/Wildfire-Containment-Simulator},
  note   = {Meta OpenEnv Hackathon submission, Theme 2}
}
```

---

## License

[MIT](LICENSE). Built on [OpenEnv](https://github.com/openenv) for the Meta × Hugging Face × Scaler hackathon, April 2026.
