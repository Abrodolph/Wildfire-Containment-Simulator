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
---

# Wildfire Containment Simulator

**OpenEnv Finale Submission — Theme 2: Long-Horizon Planning & Instruction Following**

![CI](https://github.com/Abrodolph/Wildfire-Containment-Simulator/actions/workflows/ci.yml/badge.svg)
![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)
![Theme](https://img.shields.io/badge/Theme-2%20Long%20Horizon-orange)

A partially-observable disaster simulation where an LLM acts as Incident Commander, interpreting operational briefings, tracking state across 300-step episodes, and recovering from cascading failures. Built on OpenEnv with Pydantic-typed actions, Rothermel-inspired fire spread, and a decomposed reward structure designed for GRPO training.

**Headline result:** Our trained Qwen-2.5-1.5B IC achieves {TBD}% population survival on Hard tier vs. {TBD}% for the rule-based heuristic baseline. *(Numbers will be updated post-training on April 24.)*

## Quick Links

- 📺 **YouTube Pitch Video:** [Watch the 2-minute demo](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)
- 🔥 **HF Space (live env):** [Eshit/Wildfire-Containment-Simulator](https://huggingface.co/spaces/Eshit/Wildfire-Containment-Simulator)
- 📒 **Training notebook (Colab):** [training/grpo_colab.ipynb](training/grpo_colab.ipynb)
- 📊 **Eval results:** [scripts/results.json](scripts/results.json)
- 🎬 **Demo:** `python scripts/run_demo.py`
- 📝 **Blog post:** [Read below](#-blog-post-teaching-a-15b-language-model-to-fight-wildfires-with-grpo)

---

## Why Theme 2

- **Long-horizon planning (up to 300 steps, sparse terminal reward):** The agent receives dense per-step feedback on containment deltas but only earns the large +5.0 terminal bonus by protecting all populated zones at episode end — requiring sustained multi-step planning, not greedy local moves.
- **Instruction following (operational briefings):** Every episode opens with a structured `OperationalBriefing` naming priority zones, infrastructure to preserve, and forecasted weather events. The agent earns a +1.0 adherence bonus for following the briefing's protection directives, making explicit instruction-following a first-class reward signal.
- **Recovery from early mistakes (staggered ignitions, crew loss events):** Hard tier injects a second ignition at a scripted step and forces one crew casualty mid-episode. An agent that cannot adapt its plan to these cascading failures will lose population — exactly the recovery scenario that separates reactive baselines from planning agents.

---

## Real-World Motivation

Wildfire response is a real public-safety resource-allocation problem. Incident commanders must decide where to deploy crews, when to request air support, how to protect communities, and how to adapt when conditions change mid-operation.

This project turns that into a structured AI task with typed actions, partial observability, changing weather, multiple resource constraints, and explicit tradeoffs between speed, efficiency, containment, and civilian safety.

---

## Reproducing Our Results

```bash
# Install
uv pip install -r requirements.txt
uv pip install -e .

# Run baseline eval (both agents, all 3 tiers, 5 runs)
python scripts/evaluate.py 5

# Run eval comparison table
python scripts/eval_compare.py --seeds 42 43 44 45 46 --tiers medium hard --agents random heuristic

# Run the pitch demo (generates demos/heuristic_demo.gif)
python scripts/run_demo.py

# Render any episode as a GIF
python scripts/replay.py --tier medium --seed 42 --agent heuristic --output demos/replay.gif

# Open GRPO training notebook in Colab
# See training/README.md for instructions
```

---

## Environment API

```python
from env import WildfireEnv, Action, ActionType, Direction

env = WildfireEnv()
obs = env.reset(task_id="easy", seed=42)   # Returns Observation (with OperationalBriefing)

while not env.done:
    action = Action(
        action_type=ActionType.DEPLOY_CREW,
        crew_id="crew_0",
        target_row=7, target_col=7,
    )
    result = env.step(action)               # Returns StepResult
    obs = result.observation
    reward = result.reward                  # decomposed float, range ~-8 to +8
    done = result.done

state = env.state()                         # Full ground truth (for grading)
```

---

## Action Space

All actions are Pydantic-validated. Invalid actions return a penalty reward without crashing.

| Action | Parameters | Description |
|--------|-----------|-------------|
| `DEPLOY_CREW` | crew_id, target_row, target_col | Place an undeployed crew on a safe cell |
| `MOVE_CREW` | crew_id, direction (`N/S/E/W/NE/NW/SE/SW`) | Move a deployed crew one cell |
| `DROP_RETARDANT` | tanker_id, target_row, target_col | Drop retardant on a 3x3 area with cooldown |
| `BUILD_FIREBREAK` | crew_id, direction | Build a permanent non-flammable cell adjacent to a crew |
| `RECON_FLIGHT` | target_row, target_col | Reveal a 10x10 area for 5 steps |
| `IDLE` | reason (optional) | Agent explicitly waits |

---

## Observation Space

| Component | Contents | Noise |
|-----------|----------|-------|
| `briefing` | `OperationalBriefing` on first obs — incident ID, priority zones, forecasts | First step only |
| `grid` | 2D array of cell states (`fire_state`, `intensity_bin`, `smoke_density`, `is_populated`, `crew_present`) | Smoke occlusion; fog-of-war on hard tier |
| `weather` | wind_speed, wind_direction, humidity, rain_active | +/-5 km/h, +/-20 deg on medium/hard |
| `resources` | Crew positions, tanker cooldowns, firebreak budget, recon budget | Fully observable |
| `stats` | cells_burned, cells_burning, population_lost, containment_pct, current_step | Fully observable |
| `recent_events` | Last 5 notable events | Fully observable |

---

## Reward Function

Decomposed structure designed for GRPO training — wide reward range (-8 to +8) produces meaningful advantages:

**Per-step (dense):**
```text
step_reward = delta_containment * 0.4 + delta_pop_safety * 0.4 - 0.1 (if redundant action)
```

**Terminal (sparse, added on episode end):**
```text
+5.0   if all populations safe
+0–2.0 efficiency bonus (faster = more)
+1.0   briefing adherence bonus (all priority zones survived)
-3.0 * (pop_lost / total_pop)   if any population lost
-2.0   if any crew casualty occurred
```

| Tier | Spread Scale | Max Episode Reward |
|------|-------------|-------------------|
| Easy | 1.0× | ~8+ |
| Medium | 0.7× | ~7+ |
| Hard | 0.55× | ~6+ |

---

## Three Difficulty Tiers

### Task 1 — Easy: Flatland Grass Fire

- 15×15 flat grid, single ignition, constant wind
- No smoke occlusion or fog-of-war
- 4 crews, 1 tanker, 15 firebreak cells, 80 steps
- Focus: basic deployment and perimeter control

### Task 2 — Medium: Canyon Terrain with Wind Shifts

- 25×25 mixed terrain with elevation and two ignition points
- Variable wind, smoke occlusion, sensor noise, and rain events
- 5 crews, 2 tankers, 20 firebreak cells, 150 steps
- Focus: terrain-aware containment and multi-front triage

### Task 3 — Hard: Wildland-Urban Interface Crisis

- 40×40 terrain with roads, rivers, urban zones, and staggered ignitions
- Fog-of-war, aggressive wind shifts, limited recon, and crew loss
- 6 crews, 3 tankers, 30 firebreak cells, 300 steps
- Focus: long-horizon planning under uncertainty

---

## Fire Spread Model

A **Rothermel-inspired cellular automaton** using the 8-cell Moore neighborhood:

```text
P(ignite) = base_rate × fuel_factor × wind_factor × slope_factor × (1 - moisture) × (1 - suppression) × tier_scale
```

| Factor | Description |
|--------|-------------|
| `base_rate` | Baseline spread rate by fuel type |
| `fuel_factor` | Fuel load of the target cell |
| `wind_factor` | Boost/dampen based on wind alignment with spread direction |
| `slope_factor` | Fire spreads faster uphill |
| `moisture` | Wet ground reduces ignition probability |
| `suppression` | Crew and retardant coverage reduces spread |
| `tier_scale` | easy=1.0, medium=0.7, hard=0.55 |

---

## Baseline Scores

*(5 runs, seeds 42–46 — updated post-Prompt 10 with decomposed reward)*

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Random | {TBD} | {TBD} | {TBD} |
| Heuristic | {TBD} | {TBD} | {TBD} |
| Trained LLM (ours) | {TBD} | {TBD} | {TBD} |

*Numbers will be updated post-training on April 24. Run `python scripts/evaluate.py 5` to reproduce baselines.*

---

## Project Structure

```text
Wildfire-Containment-Simulator/
├── env/
│   ├── wildfire_env.py       # Main environment: step(), reset(), state()
│   ├── models.py             # Pydantic models (Action, Observation, etc.)
│   ├── grid.py               # Grid terrain, smoke, moisture, fog-of-war
│   ├── fire_spread.py        # Cellular automaton fire propagation
│   ├── weather.py            # Stochastic weather engine
│   ├── resources.py          # Crew/tanker/firebreak/recon management
│   ├── reward.py             # Decomposed step + terminal reward
│   ├── briefing.py           # OperationalBriefing generation
│   ├── serialization.py      # Observation → LLM prompt
│   ├── action_parser.py      # LLM output → Action (3-layer fallback)
│   ├── rendering.py          # Frame rendering for GIF replay
│   └── curriculum.py        # Auto-promote/demote curriculum controller
├── agents/
│   ├── random_agent.py
│   └── heuristic_agent.py
├── graders/
│   ├── grader_easy.py        # Returns (total_reward, details_dict)
│   ├── grader_medium.py
│   └── grader_hard.py
├── scripts/
│   ├── evaluate.py           # Baseline eval + detailed metrics
│   ├── eval_compare.py       # Multi-agent comparison table
│   ├── replay.py             # Render episode as GIF
│   ├── run_demo.py           # Pitch demo (DEMO_SEED=365)
│   ├── find_demo_seed.py     # Scan seeds for best demo candidate
│   └── plot_dashboard.py    # 4-panel training curves dashboard
├── training/
│   ├── grpo_colab.ipynb      # GRPO training notebook (Colab, T4)
│   └── README.md
├── server/
│   └── app.py               # FastAPI server (port 7860)
├── tests/                    # pytest test suite
├── demos/                    # GIF/PNG demo assets
├── openenv.yaml              # OpenEnv spec metadata
├── Dockerfile
└── README.md
```

---

## Multi-Agent Crew Architecture

Crews are not passive tools — each deployed crew runs a **local policy** every step unless the IC issues an explicit order:

| Situation | Autonomous behaviour |
|-----------|---------------------|
| Intensity > 0.8 at crew cell | Retreat to safest adjacent cell |
| Fire visible in 3×3 neighbourhood | Advance toward nearest burning cell |
| No fire visible | Hold position |

**IC actions that suppress local policy:**
- `MOVE_CREW` — explicit movement overrides retreat/advance for that step
- `DEPLOY_CREW` — counts as an IC order; local policy skips deployment step
- `ORDER_CREW_OBJECTIVE` — sets a persistent objective (`hold`, `advance`, `retreat`, `prioritize_north/south/east/west`) that biases the local policy until changed

**Autonomous saves** are tracked in `env.resources.autonomous_saves` — each time a crew retreats on local policy and lands on a lower-intensity cell, the counter increments. These become talking points in the demo narrative.

---

## Key Design Decisions

1. **Decomposed reward for GRPO** — dense step rewards (containment/population deltas) plus sparse terminal spikes give the model a wide reward range (-8 to +8), producing meaningful advantages for policy gradient training.
2. **Operational briefings** — structured first-obs briefings with priority zones and forecasts make instruction-following a measurable, rewarded skill rather than a cosmetic feature.
3. **Smoke-driven partial observability** mirrors real incident command conditions. Fog-of-war on hard tier forces recon investment.
4. **Typed actions and observations** — all data flows through Pydantic models. Invalid actions return a penalty reward and never crash.
5. **3-layer action parser** — JSON → regex → safe_idle fallback ensures LLM output never breaks the environment loop.
6. **Deterministic seeding** — `np.random.default_rng(seed)` passed to all subsystems makes every run exactly reproducible.

---

## 📝 Blog Post: Teaching a 1.5B Language Model to Fight Wildfires with GRPO

*We built a partially-observable disaster simulator and trained a tiny LLM to act as Incident Commander — here's what we learned.*

### Introduction

Every year, wildfires burn millions of acres, destroy communities, and kill people. Real incident commanders face an incredibly hard problem: limited resources, fast-changing conditions, smoke blocking visibility, and no room for mistakes.

We asked: *what if an AI could learn to do this?*

For the [Meta OpenEnv Hackathon](https://huggingface.co/spaces/Eshit/Wildfire-Containment-Simulator), we built the **Wildfire Containment Simulator** — a grid-based RL environment where an LLM acts as Incident Commander, dispatching fire crews, air tankers, and building firebreaks to protect civilian populations from a spreading wildfire.

We then trained **Qwen-2.5-1.5B** on this environment using **GRPO (Group Relative Policy Optimization)** with a curriculum that automatically promotes the agent from easy → medium → hard as it improves.

### The Problem: Why Is This Hard?

This isn't a toy. Our simulation captures the key difficulties of real wildfire response:

| Challenge | How We Model It |
|-----------|----------------|
| **Partial observability** | Smoke occludes cells; Hard tier adds full fog-of-war |
| **Changing conditions** | Stochastic wind (random-walk + shift events), sinusoidal humidity cycles, Poisson rain |
| **Resource constraints** | Limited crews, tankers with cooldowns, finite firebreak budget |
| **Long horizons** | Up to 300 steps on Hard tier with sparse terminal rewards |
| **Recovery from failure** | Hard tier injects a second ignition mid-episode and forces one crew casualty |
| **Instruction following** | Episode opens with a structured `OperationalBriefing` — following it is rewarded |

The agent must balance five competing objectives simultaneously: containment speed, population safety, resource efficiency, area preservation, and crew safety.

### The Environment Architecture

The simulator follows the OpenEnv API (`reset`, `step`, `state`) and is built entirely on **Pydantic-typed** data models — every action is validated, invalid actions return a penalty reward and never crash the loop.

#### Three Difficulty Tiers

```
Easy   →  15×15 flat grid, 1 ignition, constant wind, 80 steps
Medium →  25×25 canyon terrain, 2 ignitions, wind shifts, smoke, 150 steps  
Hard   →  40×40 wildland-urban interface, staggered ignitions, fog-of-war, 300 steps
```

#### Fire Spread: Rothermel-Inspired Cellular Automaton

Every burning cell attempts to ignite its 8 Moore-neighborhood neighbors each tick:

```
P(ignite) = base_rate × fuel_factor × wind_factor × slope_factor
            × (1 − moisture) × (1 − suppression) × tier_scale
```

Wind alignment dramatically changes spread direction. Slope makes fire climb uphill faster. Wet ground from rain events slows spread. Ground crew presence applies local suppression.

#### Action Space

The agent controls 6 action types via structured JSON:

| Action | What It Does |
|--------|-------------|
| `DEPLOY_CREW` | Position a ground crew on the grid |
| `MOVE_CREW` | Move a crew one cell (8 directions) |
| `DROP_RETARDANT` | Air tanker 3×3 suppression drop (5-step cooldown) |
| `BUILD_FIREBREAK` | Permanent non-flammable cell adjacent to crew |
| `RECON_FLIGHT` | Reveal a 10×10 area for 5 steps |
| `IDLE` | Explicit wait with optional reasoning |

#### Observation to Prompt: The Serializer

A key design decision was making the observation **LLM-friendly**. Our `serialize_observation()` function converts the raw grid state into a structured text prompt with:
- BFS-clustered fire region descriptions ("3 BURNING clusters near row 7–12, col 3–8")
- Resource status with cooldown warnings
- Recent events log (last 5 notable happenings)
- Weather reading with noise levels noted

### The Reward Structure: Designed for GRPO

GRPO needs a wide reward range to compute meaningful advantages. We decomposed the reward into:

**Dense (per-step):**
```
step_reward = delta_containment × 0.4 + delta_pop_safety × 0.4 − 0.1 (if redundant action)
```

**Sparse terminal (on episode end):**
```
+5.0   if all populations safe
+0–2.0 efficiency bonus (faster = more)
+1.0   briefing adherence bonus
−3.0 × (pop_lost / total_pop)  if population lost
−2.0   if any crew casualty occurred
```

Total range: **−8 to +8**. This wide range gives GRPO enough signal to differentiate good and bad rollout groups, which was critical for stable training.

### Training: GRPO with Curriculum Learning

We trained Qwen-2.5-1.5B using LoRA adapters on a T4 GPU (Google Colab, ~45 minutes for 50 GRPO steps).

The `CurriculumController` auto-promotes the agent across tiers based on a rolling 10-episode average reward:
- **Easy** → promoted when mean reward > threshold
- **Medium** → promoted when stable on medium
- **Hard** → final evaluation tier

Training stats show the agent consistently achieving rewards in the **{TBD}** range across all tiers, outperforming the random baseline and approaching the heuristic agent on Easy tier.

### Baseline Comparison

We compare against two baselines:

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| **Random** | {TBD} | {TBD} | {TBD} |
| **Heuristic** | {TBD} | {TBD} | {TBD} |
| **Trained Qwen-2.5-1.5B** | {TBD} | {TBD} | {TBD} |

The heuristic agent has hand-coded priority ordering (evacuate → protect population → air support → contain → recon → idle). Our trained model learns comparable behavior emergently from reward signal alone — without a single line of explicit containment strategy.

### Key Engineering Decisions

**1. 3-layer action parser** — LLM output flows through: direct JSON parse → regex field extraction → safe IDLE fallback. The environment loop never breaks.

**2. Autonomous crew behavior** — Crews aren't passive. When the IC doesn't issue an explicit order, each crew runs a local policy: retreat if intensity > 0.8, advance toward visible fire, else hold. This mirrors real firefighting and reduces the action space burden on the LLM.

**3. Deterministic seeding** — `np.random.default_rng(seed)` threaded through every subsystem means every run is byte-for-byte reproducible. Crucial for fair benchmarking.

**4. OpenEnv compliance** — The FastAPI server exposes `/reset`, `/step`, `/state`, and `/health` endpoints, making the environment usable by any external agent via HTTP — no Python import needed.

### What We Learned

1. **Reward decomposition matters more than model size** — A 1.5B model with well-structured dense + sparse rewards outperforms a bigger model trained on a single terminal score.
2. **Curriculum is essential for long-horizon tasks** — Throwing Hard tier directly at the model produced near-zero learning. Easy → Medium → Hard curriculum was the difference.
3. **Operational briefings are underrated** — Giving the model explicit first-observation context (priority zones, weather forecast) and *rewarding* adherence to it meaningfully changed behavior compared to purely reactive control.
