# Wildfire Containment Simulator — Agent Prompt Sequence

**Usage:** Feed these prompts **one at a time** to your coding agent (Claude Code or Antigravity). After each prompt finishes, run its acceptance test yourself before moving to the next. Each prompt assumes the prior ones completed successfully.

**Global context to paste once at the start of every new agent session** (if the agent loses context between prompts):

> You are working on the Wildfire Containment Simulator — an OpenEnv-compatible RL environment for the Meta × PyTorch × HuggingFace OpenEnv Hackathon finale (April 25–26, 2026). The repo is at `https://github.com/Abrodolph/Wildfire-Containment-Simulator`. Core packages: `env/` (simulation), `agents/` (baselines), `graders/` (one per tier), `scripts/` (evaluation). The env exposes `reset()`, `step()`, `state()` with Pydantic-validated `Action`, `Observation`, `StepResult` models defined in `env/models.py`. Three tiers exist: easy (15×15), medium (25×25), hard (40×40). You can run `pytest`, `python scripts/evaluate.py`, and any other command. Iterate on failures until tests pass. Never skip the acceptance test at the end of each prompt.

---

## Prompt 1 — Repo Cleanup & Test Scaffolding ✅ DONE

```
Clean up repo cruft and set up a test scaffold before we make any functional changes.

Tasks:
1. Delete the nested `Wildfire-Containment-Simulator/` directory at repo root (leftover HF Space metadata).
2. Delete the literal `{env,graders,agents,scripts}` directory at repo root (shell-brace artifact).
3. Delete all committed `__pycache__/` directories and `*.egg-info/` folders.
4. Delete the `venv/` directory if it's committed.
5. Update `.gitignore` to include: `__pycache__/`, `*.egg-info/`, `venv/`, `.venv/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `checkpoints/`, `results/`.
6. Consolidate server entry points: keep `server/app.py` as the single source of truth. Update the root `app.py` to be a one-line shim that imports and runs `server.app:main`. Update `Dockerfile` CMD to match.
7. Create `tests/` directory with `tests/__init__.py` and `tests/conftest.py`. In conftest, add a fixture `fresh_env` that yields a `WildfireEnv()` instance.
8. Create `tests/test_smoke.py` with three tests:
   - `test_env_resets_on_all_tiers` — calls `env.reset(task_id=t, seed=42)` for t in ["easy", "medium", "hard"] and asserts obs is not None.
   - `test_idle_action_never_crashes` — resets env, calls `env.step(Action(action_type=ActionType.IDLE))` 10 times, asserts no exception.
   - `test_determinism` — runs a fixed 20-step idle rollout twice with seed=42 on easy tier, asserts the final `stats.cells_burned` matches.
9. Add `pytest` and `pytest-cov` to `requirements.txt` if missing.

Acceptance test:
- `pytest tests/ -v` passes with 3 tests green.
- `python app.py` still starts the server on port 7860.
- `git status` shows no `__pycache__` or `{env,...}` cruft.
- Output the diff summary of deleted files and new files.
```

---

## Prompt 2 — Reward Restructuring (Decomposed Terminal + Dense Step) ✅ DONE

```
Replace the current normalized [0,1] composite reward with a decomposed terminal + dense step structure. This is critical for GRPO training — the current reward is too flat to produce meaningful advantages.

Read `env/reward.py` first. Understand the current RewardCalculator class. Also read `env/wildfire_env.py` to see how reward is called per step.

Tasks:
1. In `env/reward.py`, add a new method `compute_step_reward(prev_state, current_state, action_was_valid, action_was_redundant) -> float` that returns:
   - (delta_containment_pct * 0.4) + (delta_population_safety * 0.4) + (-0.1 if action_was_redundant else 0.0)
   - where delta_containment_pct is (current_containment - prev_containment) in [0, 1] units
   - delta_population_safety is (1 - current_pop_lost/total_pop) - (1 - prev_pop_lost/total_pop)
   - redundant = same action_type + same target coords as the immediately prior action

2. Add a method `compute_terminal_reward(final_state, episode_steps, max_steps) -> float`:
   - start at 0
   - if all_populations_safe (pop_lost == 0): add +5.0
   - else: add -3.0 * (pop_lost / total_pop)
   - if any crew_casualty occurred in the episode: add -2.0 (stacks with above)
   - efficiency_bonus = (max_steps - episode_steps) / max_steps * 2.0 — ONLY applied if pop_lost == 0
   - invalid_action_penalty_total = min(0.2, 0.01 * invalid_action_count) — subtract this

3. In `env/wildfire_env.py`:
   - Track `self._prev_action` and `self._invalid_action_count` and `self._crew_casualty_occurred` across the episode (reset them in `reset()`).
   - Replace the current reward computation in `step()` with: step_reward from above, plus terminal_reward ONLY when `done == True`.
   - The StepResult.reward should be `step_reward + (terminal if done else 0.0)`.

4. Keep the OLD composite reward accessible as `info["legacy_reward"]` in StepResult for backward compatibility with existing graders. (Graders get updated in Prompt 10.)

5. Add `tests/test_reward.py` with:
   - `test_successful_episode_scores_high` — run heuristic agent on easy tier seed=42, assert total reward > +3.0
   - `test_all_pop_lost_scores_negative` — construct a scenario (or mock state) where all population is lost, assert terminal < -2.0
   - `test_crew_casualty_stacks` — scenario with pop loss AND crew casualty, assert terminal includes both penalties
   - `test_redundant_action_penalty` — call the same DEPLOY_CREW twice, assert second call's step_reward includes -0.1

Acceptance test:
- `pytest tests/test_reward.py -v` passes all 4 tests.
- Run `python scripts/evaluate.py 20` on easy tier with the heuristic agent. Report mean + std of total rewards. Successful episodes should cluster in the +5 to +8 range, failed episodes in the -2 to -5 range. If the ranges overlap by more than 20% of episodes, the reward isn't separated enough — report that and DO NOT proceed.
```

---

## Prompt 3 — Observation-to-Text Serializer ✅ DONE

```
Write a serializer that converts a Pydantic Observation into a structured text prompt that an LLM can reason over. This is required because OpenEnv is an LLM-training framework — the agent is a language model, not a numeric policy.

Read `env/models.py` to understand the Observation schema. Read the README section "Observation Space" for the intended structure.

Tasks:
1. Create `env/serialization.py` with a function `serialize_observation(obs: Observation, step_num: int, max_steps: int) -> str`.

2. Output format (match this structure exactly — the LLM will be trained on it):

```
=== WILDFIRE INCIDENT COMMAND — STEP {step}/{max_steps} ===

SITUATION:
- Fire active on {N} cells. Containment: {pct}%. Population at risk: {N} zones.
- Wind: {speed} km/h {dir} (±{noise} km/h noise). Humidity: {h}%. Rain: {active|inactive}.
- Last event: {most_recent_event or "None"}

GRID SUMMARY (smoke-obscured cells marked [?]):
{bounding_box_descriptions_of_fire_regions}
{populated_zone_descriptions}
{firebreak_descriptions_if_any}

RESOURCES:
- crew_0: {deployed at (r,c) | undeployed available}. Status: {active|casualty}.
- crew_1: ...
- tanker_0: {ready | cooldown N steps remaining}
- Firebreaks remaining: {N}. Recon flights remaining: {N}.

RECENT EVENTS:
- Step {N}: {event description}
- ... (last 3 events max)

Available actions: deploy_crew, move_crew, drop_retardant, build_firebreak, recon_flight, idle
Produce your action as JSON: {"action_type": "...", ...}
```

3. Helper functions inside the module (keep private with leading underscore):
   - `_summarize_grid_regions(obs.grid) -> List[str]` — detect rectangular bounding boxes of (a) active fire cells clustered together, (b) populated cells, (c) built firebreaks. Output as "Row X-Y, Col A-B: description". Cap at 5 regions per category, prioritize by size.
   - `_format_resources(obs.resources) -> str`
   - `_format_events(obs.recent_events) -> str`

4. Add `tests/test_serialization.py`:
   - `test_serialize_produces_all_sections` — reset env, serialize, assert the output contains "SITUATION:", "GRID SUMMARY:", "RESOURCES:", "RECENT EVENTS:", "Available actions:".
   - `test_serialize_handles_fog_of_war` — hard tier reset, assert "[?]" appears somewhere in output (smoke or fog-obscured cells).
   - `test_serialize_length_under_2048_tokens` — run on all 3 tiers, assert `len(tokenizer.encode(output))` < 1800 using tiktoken's cl100k_base (if tiktoken not installed, use `len(text.split()) < 1500` as a proxy).

Acceptance test:
- `pytest tests/test_serialization.py -v` passes all 3 tests.
- Run a manual sanity check: `python -c "from env import WildfireEnv; from env.serialization import serialize_observation; env = WildfireEnv(); obs = env.reset(task_id='medium', seed=42); print(serialize_observation(obs, 0, 150))"` — paste the output and confirm it reads like a realistic incident briefing.
```

---

## Prompt 4 — LLM Action Parser with 3-Layer Fallback ✅ DONE

```
Build a robust parser that converts LLM text output into a validated Action object. LLMs produce malformed JSON, hallucinated fields, and out-of-range coords — we need to never crash.

Tasks:
1. Create `env/action_parser.py` with a function `parse_action(llm_output: str, obs: Observation) -> Tuple[Action, str]` returning the action AND a status string ("json_success", "regex_fallback", "safe_idle").

2. Three layers, in order:

   LAYER 1 — Direct JSON parse:
   - Extract JSON from output using a helper `_extract_json_block(text)` that finds content between first `{` and matching `}` (handles ```json fences, handles leading/trailing text).
   - Try `json.loads` then `Action(**data)` — Pydantic validates fields.
   - On success return (action, "json_success").

   LAYER 2 — Regex extraction:
   - Search for action_type via regex: `action_type["\s:]+["']?(deploy_crew|move_crew|drop_retardant|build_firebreak|recon_flight|idle)`
   - Based on detected action_type, extract required fields with regex patterns (e.g., `crew_id["\s:]+["']?(crew_\d+)`, `target_row["\s:]+(\d+)`, `direction["\s:]+["']?(N|S|E|W|NE|NW|SE|SW)`).
   - Construct Action; if Pydantic validates, return (action, "regex_fallback").

   LAYER 3 — Safe fallback:
   - Return `(Action(action_type=ActionType.IDLE, reason="parse_failure"), "safe_idle")`.

3. Add coordinate sanity check: after any layer succeeds, if target_row or target_col is outside the current grid dimensions (infer from obs.grid shape), downgrade to safe_idle. Never trust LLM-provided coords blindly.

4. Add `tests/test_action_parser.py` with 8 test cases covering:
   - Clean JSON output
   - JSON wrapped in ```json fences
   - JSON with extra surrounding commentary
   - Malformed JSON (missing quotes) that regex can save
   - Completely garbage output → safe_idle
   - Out-of-bounds coords → safe_idle
   - Hallucinated action_type (e.g., "nuke_fire") → safe_idle
   - Empty string → safe_idle

Acceptance test:
- `pytest tests/test_action_parser.py -v` passes all 8 tests.
- Zero crashes across the test suite.
- Status string is correctly reported for each case.
```

---

## Prompt 5 — Replay / GIF Renderer ✅ DONE

```
Build a replay script that renders any episode as an animated GIF. This is critical for the storytelling score — every demo asset depends on it.

Tasks:
1. Add `imageio` and `matplotlib` to `requirements.txt` if not present.

2. Create `scripts/replay.py` with CLI: `python scripts/replay.py --tier {easy|medium|hard} --seed {int} --agent {random|heuristic} --output {path.gif}`.

3. The script should:
   - Instantiate the env, run the agent, capture the full ground-truth `env.state()` at every step.
   - For each step, render a matplotlib figure (8x8 inches, 100 dpi) with:
     * Main panel (80% area): grid colored by cell state. Burning = red (intensity → color saturation), burned = dark gray, populated = blue square outline, firebreak = brown, crew = green circle with crew_id label, tanker drop zone = translucent cyan overlay.
     * Bottom strip: step number, cells burning, containment %, pop lost, wind arrow + speed.
   - Save all frames, stitch to GIF at 5 fps, write to output path.
   - Also save final-frame PNG to same path with `.png` extension.

4. Keep the rendering code in `env/rendering.py` (importable helpers), not inline in the script. Functions:
   - `render_frame(state: EnvState, step: int, stats: dict) -> np.ndarray` — returns RGB array.
   - `render_episode_gif(frames: List[np.ndarray], output_path: str, fps: int = 5)`.

5. Add `tests/test_rendering.py`:
   - `test_render_frame_produces_rgb` — reset env on easy, render frame, assert shape is (H, W, 3) and dtype is uint8.
   - `test_gif_creation` — run 20 steps of random agent, call `render_episode_gif`, assert output file exists and is > 10KB.

Acceptance test:
- `pytest tests/test_rendering.py -v` passes both tests.
- Run: `python scripts/replay.py --tier medium --seed 42 --agent heuristic --output demos/heuristic_medium_42.gif`
- Open the GIF. Confirm it shows fire spreading, crews moving, and the stats strip updating. Paste the final-frame stats as confirmation.
```

---

## Prompt 6 — Curriculum Controller ✅ DONE

```
Add a curriculum controller that auto-promotes through tiers based on rolling performance. This produces the characteristic "dip-and-recover" pattern on training curves that makes for compelling demo visuals.

Tasks:
1. Create `env/curriculum.py` with class `CurriculumController`:
   - `__init__(self, start_tier: str = "easy", thresholds: Optional[dict] = None)` — default thresholds: easy→medium at 4.0 avg over 10 eps, medium→hard at 3.5 avg over 10 eps (these are total episode rewards under the new reward scheme, NOT [0,1]).
   - `after_episode(self, total_reward: float) -> Optional[str]` — returns the new tier name if a promotion just fired, else None.
   - `get_tier(self) -> str` — current tier.
   - `get_history(self) -> List[Tuple[int, str, float]]` — list of (episode_idx, tier, reward) for plotting.
   - `promotion_log: List[Tuple[int, str]]` — list of (episode_idx, new_tier) for marking vertical lines on plots.

2. Demote behavior: if recent 10-ep avg drops below (threshold * 0.5) after a promotion, demote back. Log this too.

3. Add `tests/test_curriculum.py`:
   - `test_promotion_fires_at_threshold` — feed 10 rewards of 5.0, assert promotion to medium.
   - `test_no_premature_promotion` — feed 5 rewards of 5.0, assert still on easy.
   - `test_demotion_on_collapse` — promote to medium, then feed 10 rewards of 0.5, assert demoted to easy.
   - `test_history_tracking` — run 20 episodes, assert history length is 20 and promotion_log is correctly populated.

Acceptance test:
- `pytest tests/test_curriculum.py -v` passes all 4 tests.
- The controller is not yet wired into the env itself (that happens in the training notebook, Prompt 8). This prompt just builds the component.
```

---

## Prompt 7 — Eval Comparison Script ✅ DONE

```
Build the eval comparison script that generates the headline comparison table for the pitch. This runs multiple agents on fixed seeds and outputs a clean comparison.

Tasks:
1. Create `scripts/eval_compare.py` with CLI: `python scripts/eval_compare.py --seeds 42 43 44 45 46 --tiers medium hard --agents random heuristic base_llm trained_llm --output eval_results.json`.

2. Agent registry — a dict mapping agent name to a factory function:
   - `random` → existing RandomAgent
   - `heuristic` → existing HeuristicAgent
   - `base_llm` → LLM agent using `env/serialization.py` + `env/action_parser.py`, calling a base model (stub this for now — read model path from env var `BASE_MODEL_PATH`, default to None which skips this agent with a warning).
   - `trained_llm` → same pattern, env var `TRAINED_MODEL_PATH`.

3. For each (agent, tier, seed) combination:
   - Run the episode.
   - Record: final containment_pct, pop_saved_pct (= 1 - pop_lost/total_pop), total_reward, episode_steps.

4. Output:
   - A JSON file at the specified path with full results.
   - A printed table to stdout formatted like:
     ```
     === EVAL RESULTS — Medium Tier (5 seeds) ===
                         Containment   Pop Saved   Reward   Steps
     Random Agent            41%          60%       -1.2     150
     Heuristic Agent         49%          71%       +1.8     143
     Base LLM (Qwen)         38%          55%       -0.9     150  [skipped — no model]
     Trained LLM (ours)      67%          89%       +4.1     121  [skipped — no model]
     ```
   - Use mean across seeds for each column. Mark skipped agents clearly.

5. Add `--quick` flag that runs only easy tier with 2 seeds for smoke testing.

6. Add `tests/test_eval_compare.py`:
   - `test_quick_mode_runs` — invoke with --quick, assert eval_results.json exists, assert at least random and heuristic have non-null entries.

Acceptance test:
- `python scripts/eval_compare.py --quick` completes in under 2 minutes.
- `pytest tests/test_eval_compare.py -v` passes.
- Full run `python scripts/eval_compare.py --seeds 42 43 44 45 46 --tiers medium hard --agents random heuristic` produces the table. The LLM columns will show "[skipped — no model]" which is expected at this stage.
```

---

## Prompt 8 — GRPO Training Notebook (Colab) ✅ DONE

```
Build the GRPO training notebook. This is a hackathon minimum requirement — without it we're technically DQ'd.

Tasks:
1. Create `training/grpo_colab.ipynb` (a Jupyter notebook — JSON format). Use `nbformat` to construct it programmatically to avoid JSON escaping errors.

2. Notebook sections (each a separate cell with a markdown header cell above it):

   **Section 1: Setup**
   - pip install: `unsloth trl openenv-core pydantic numpy imageio matplotlib`
   - Clone the repo or install from path.
   - Import FastLanguageModel from unsloth, load `unsloth/Qwen2.5-1.5B-Instruct` in 4-bit with max_seq_length=2048.
   - Apply LoRA: r=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"].

   **Section 2: Environment & Rollout**
   - Import WildfireEnv, serialize_observation, parse_action.
   - Define `collect_rollout(env, model, tokenizer, tier, seed) -> List[Dict]` that:
     * resets env
     * for each step: serializes obs → generates completion → parses action → steps env → records (prompt, completion, reward, step_status).
     * returns trajectory list.
   - Define `system_prompt` — a short, firm instruction to always output action as JSON only.

   **Section 3: GRPO Training Loop**
   - Use TRL's GRPOTrainer. Config: num_generations=8 per prompt, learning_rate=5e-6, max_steps=50, save_steps=10, per_device_train_batch_size=1, gradient_accumulation_steps=4.
   - Reward function: for each generation, run a mini-rollout (fresh env, same seed, sample action from the completion) and return the 1-step reward + discounted terminal if done. Cache seeds so generations for the same prompt see the same env state.
   - Wire in the CurriculumController from Prompt 6: after each full episode, call `controller.after_episode(total_reward)` and switch tier for the next episode.

   **Section 4: Checkpointing & Recovery**
   - Save LoRA adapter to `./checkpoints/step_{N}` every 10 steps.
   - Save a JSON of training stats (step, mean_reward, tier, parse_failure_rate) to `./training_stats.json` every step.
   - Add a "resume from checkpoint" cell at the top of Section 3 that loads the latest checkpoint if present.

   **Section 5: Plot Reward Curve**
   - Load training_stats.json, plot mean_reward vs step with matplotlib.
   - Save as `reward_curve.png`.
   - Mark tier promotions as vertical lines using controller.promotion_log.

3. Add `training/README.md` with:
   - How to open in Colab (a badge link).
   - Which cells to run in order.
   - Expected runtime on T4: ~45 min for 50 steps.
   - How to download the trained adapter.

4. Add `training/test_notebook_imports.py` — a plain Python file (not pytest) that imports every module the notebook uses and instantiates the env + tokenizer (skipping the model load). This catches broken imports before you open Colab.

Acceptance test:
- `python training/test_notebook_imports.py` runs without error.
- Open the notebook locally with `jupyter nbconvert --to notebook --execute training/grpo_colab.ipynb --ExecutePreprocessor.timeout=600` — skip this if no GPU available locally (which is expected). Instead, validate notebook JSON with `jupyter nbconvert --to script training/grpo_colab.ipynb` and confirm the generated .py file has no syntax errors.
- Confirm the notebook has exactly the 5 sections described, each with a markdown header.
```

---

## Prompt 9 — Training Curves Dashboard ✅ DONE

```
Build a 4-panel training dashboard. Panel D (curriculum transitions) is the storytelling hook.

Tasks:
1. Create `scripts/plot_dashboard.py` with CLI: `python scripts/plot_dashboard.py --stats training/training_stats.json --output training/training_dashboard.png`.

2. Layout: 2x2 matplotlib grid, figsize=(12, 8), dpi=100.

   - Panel A (top-left): Mean episode reward vs training step. Line plot with moving average (window=5) as a thicker overlay.
   - Panel B (top-right): Population survival rate (% of eps with zero pop loss) vs training step. Computed as rolling 10-ep fraction.
   - Panel C (bottom-left): Mean containment % at episode end, vs training step.
   - Panel D (bottom-right): Curriculum tier timeline. X-axis = episode index, Y-axis = tier (easy=0, medium=1, hard=2) drawn as a step function. Vertical dashed lines at promotion events with tier labels.

3. Handle missing data gracefully — if `training_stats.json` is absent, generate a synthetic stats file at `training/synthetic_stats_demo.json` with 50 fake training steps showing a plausible upward curve + one tier promotion, then plot from that (clearly label it "SYNTHETIC DEMO" in the figure title). This lets us test the plot script without a real training run.

4. Add `tests/test_dashboard.py`:
   - `test_synthetic_dashboard` — run `plot_dashboard.py` with no stats file, assert the synthetic PNG is created and > 50KB.

Acceptance test:
- `pytest tests/test_dashboard.py -v` passes.
- Open the generated PNG. Confirm all 4 panels render, Panel D has visible vertical promotion lines, and the synthetic warning label is visible.
```

---

## Prompt 10 — Grader Alignment & Legacy Reward Cleanup ✅ DONE

```
The existing graders in graders/ were written against the old [0,1] composite reward. Align them with the new decomposed reward so eval numbers are consistent between training and grading.

Tasks:
1. Read `graders/grader_easy.py`, `graders/grader_medium.py`, `graders/grader_hard.py`. Identify where each one reads `result.reward` or computes a final score.

2. Update each grader:
   - Sum step rewards + terminal reward across the episode using the new decomposed structure.
   - Return the total episode reward as the grader's score.
   - Add a `details` dict to the grader return value: `{"total_reward": float, "containment_pct": float, "pop_saved_pct": float, "steps": int, "crew_casualty": bool}`.

3. Remove any references to `legacy_reward` that are no longer needed. Keep `legacy_reward` in StepResult.info for one more cycle (delete later), but graders should NOT use it.

4. Update `scripts/evaluate.py` to print the new detailed metrics alongside the reward.

5. Update `README.md` "Baseline Scores" table with the new reward scale. Re-run `python scripts/evaluate.py 5` and paste the new numbers. Expected pattern: heuristic should now clearly beat random on ALL tiers under the new reward. If it doesn't, flag it — this is a diagnostic signal that the reward or the heuristic needs work.

6. Add `tests/test_graders.py`:
   - `test_each_grader_returns_float_and_details` — run each of the 3 graders with the heuristic agent, assert return structure.
   - `test_grader_scores_are_in_expected_range` — assert easy total_reward > 3.0 for heuristic, medium > 1.0, hard > 0.0 (generous lower bounds).

Acceptance test:
- `pytest tests/test_graders.py -v` passes.
- `python scripts/evaluate.py 5` produces a table where heuristic beats random on every tier. Paste the output. If heuristic loses on any tier, investigate before proceeding — this likely indicates a variance or reward issue.
- README "Baseline Scores" section is updated with new numbers.
```

---

## Prompt 11 — Demo Seed Finder + Demo Runner ✅ DONE

```
Find a fixed seed that produces a clean, visually obvious contrast between heuristic and (eventually) trained-LLM behavior on medium tier. This becomes the 3-minute pitch demo.

Tasks:
1. Create `scripts/find_demo_seed.py`:
   - Iterate seeds 0..500 on medium tier.
   - For each seed, run the HEURISTIC agent, record: total_reward, pop_saved_pct, wind_shift_step (if any), and whether at least one populated cell was lost.
   - Filter for seeds where: (a) a wind shift fires between step 60–90, (b) heuristic loses at least one populated cell, (c) heuristic total_reward is between 0.0 and +2.0 (i.e., a flawed but not catastrophic baseline — gives room for improvement).
   - Output top 5 candidate seeds to `demos/candidate_seeds.json` with a short description of each.

2. Create `scripts/run_demo.py` with CLI: `python scripts/run_demo.py --seed {int}`:
   - Runs heuristic on medium tier with that seed, generates GIF to `demos/heuristic_demo.gif` using the Prompt 5 renderer.
   - Prints a play-by-play narrative: "Step 45: fire approaches populated cell (12, 8). Step 60: wind shifts. Step 75: crew committed to wrong flank. Step 89: populated cell burns."
   - If `--agent trained_llm` is passed and a TRAINED_MODEL_PATH env var exists, also runs the trained model and saves `demos/trained_demo.gif` + a second narrative.
   - Print a clean side-by-side comparison at the end: both agents' final stats.

3. Pick ONE seed from the top 5 as `DEMO_SEED`. Hardcode it as a constant in `scripts/run_demo.py`: `DEMO_SEED = <chosen_seed>`. The `--seed` flag defaults to this. Document the narrative for this specific seed in a comment block at the top of the file.

4. Add `demos/README.md` explaining how to regenerate demo assets.

Acceptance test:
- `python scripts/find_demo_seed.py` completes in under 10 minutes, outputs candidate_seeds.json.
- `python scripts/run_demo.py` (with default seed) produces heuristic_demo.gif and prints a coherent narrative. Confirm the narrative matches what the GIF actually shows.
- Paste the chosen DEMO_SEED value.
```

---

## Prompt 12 — Theme 2 Framing: Operational Briefing System

```
Add a structured operational briefing that the env produces on reset(). The agent receives this as part of its first observation. This is what pivots the environment into Theme 2 (Long-Horizon Planning & Instruction Following) framing — judges need to see instruction-following as a first-class feature.

Tasks:
1. Create `env/briefing.py` with:
   - Pydantic model `OperationalBriefing` with fields: `incident_id: str`, `ignition_cause: str`, `priority_populated_zones: List[Tuple[int, int]]` (cells the agent must prioritize protecting), `priority_infrastructure: List[Tuple[int, int]]` (e.g., road cells, optional), `forecast_events: List[str]` (e.g., "Wind shift southwest expected by step 60"), `declared_time: str` (narrative time like "04:00").
   - Function `generate_briefing(tier_config, rng) -> OperationalBriefing` that synthesizes a plausible briefing from the tier config. For populated priorities, pick the top 2 largest pop clusters. For forecast events, derive from the weather schedule if the engine exposes scheduled wind shifts; otherwise generate 1-2 plausible generic forecasts.
   - Function `briefing_to_text(briefing: OperationalBriefing) -> str` — formats as a natural-language briefing block:
     ```
     === OPERATIONAL BRIEFING ===
     Incident {incident_id} declared at {declared_time}.
     Cause: {ignition_cause}.
     
     PRIORITY 1: Protect populated zones at {coords list with cell names}.
     PRIORITY 2: Maintain {infrastructure} open where possible.
     
     FORECAST:
     - {forecast_1}
     - {forecast_2}
     
     Commander's intent: Contain fire with zero civilian casualties. Preserve crew safety.
     ```

2. Update `env/models.py`:
   - Add `briefing: Optional[OperationalBriefing]` field to `Observation`. Populated only on the first observation after reset; subsequent observations can reuse or omit.

3. Update `env/wildfire_env.py`:
   - On reset, generate a briefing and attach to the first observation.
   - Store `self.active_briefing` for the episode so reward logic can reference it.

4. Update `env/reward.py` compute_terminal_reward:
   - Add a `briefing_adherence_bonus`: +1.0 if all priority_populated_zones survived, 0 otherwise.
   - Stack this on top of the existing terminal reward.

5. Update `env/serialization.py` serialize_observation:
   - If `obs.briefing` is present, prepend `briefing_to_text(obs.briefing)` above the SITUATION block.
   - Subsequent steps: include a shortened reminder like "Priority zones: (r1,c1), (r2,c2) — still standing" or "— 1 LOST".

6. Add `tests/test_briefing.py`:
   - `test_briefing_generated_on_reset` — reset on medium, assert obs.briefing is not None and has ≥1 priority zone.
   - `test_briefing_adherence_bonus` — run heuristic successfully saving priority zones, assert terminal includes the +1.0.
   - `test_briefing_in_serialized_prompt` — serialize first obs, assert "OPERATIONAL BRIEFING" substring is present.

Acceptance test:
- `pytest tests/test_briefing.py -v` passes all 3 tests.
- Run the serializer manually on a fresh medium reset and confirm the briefing reads coherently. Paste the output.
- Re-run `python scripts/evaluate.py 5`. Reward numbers will shift slightly due to the new bonus — that's expected. Paste the new numbers.
```

---

## Prompt 13 — README Rewrite for Finale Framing

```
Rewrite the README to frame this as a finale submission aligned with Theme 2 (Long-Horizon Planning & Instruction Following). Keep all the technical depth but re-lead with the finale narrative.

Tasks:
1. Replace the current README.md top section (above "Real-World Motivation") with:

```markdown
# Wildfire Containment Simulator

**OpenEnv Finale Submission — Theme 2: Long-Horizon Planning & Instruction Following**

![Training Demo](demos/heuristic_demo.gif)

A partially-observable disaster simulation where an LLM acts as Incident Commander, interpreting operational briefings, tracking state across 300-step episodes, and recovering from cascading failures. Built on OpenEnv with Pydantic-typed actions, Rothermel-inspired fire spread, and a decomposed reward structure designed for GRPO training.

**Headline result:** Our trained Qwen-2.5-1.5B IC achieves {X}% population survival on Hard tier vs. {Y}% for the rule-based heuristic baseline. See [HF blog post]({link}) for details.

## Quick Links
- 🔥 **HF Space (live env):** {link}
- 📒 **Training notebook (Colab):** [training/grpo_colab.ipynb]({link})
- 📊 **Eval results:** [eval_results.json]({link})
- 🎬 **Demo:** `python scripts/run_demo.py`
- 📝 **Blog post:** {link}
```

2. Add a new section right after the quick links called **"Why Theme 2"**:
   - 3 bullets explaining long-horizon planning (300 steps, sparse terminal reward), instruction following (operational briefings), and recovery from early mistakes (staggered ignitions, crew loss events).

3. Keep all existing sections (Environment API, Action Space, Observation Space, Reward Function, Tiers, Fire Spread Model, Project Structure, Key Design Decisions).

4. Update the **Reward Function** section to describe the new decomposed structure (step rewards + terminal spikes), not the old [0,1] composite.

5. Add a new **"Baseline Scores"** table with post-training numbers. If training hasn't completed yet, use placeholder `{TBD}` and add a prominent note: "Numbers will be updated post-training on April 24."

6. Add a **"Reproducing Our Results"** section:
   - How to run baseline evals.
   - How to open the Colab notebook.
   - How to run the demo seed.
   - How to render replays.

Acceptance test:
- README renders cleanly on GitHub (preview via VSCode or `grip`).
- All links are either live or clearly marked as placeholders.
- The first screenful (hero + quick links + theme justification) is self-contained — a judge can get the pitch in 30 seconds without scrolling.
```

---

## Prompt 14 — CI & Final Repo Polish

```
Add CI and final-mile polish. This is the "looks professional on GitHub" pass.

Tasks:
1. Create `.github/workflows/ci.yml`:
   - Triggers: push to main, PRs.
   - Runs: setup Python 3.10, install requirements, run `pytest tests/ -v --cov=env --cov-report=term`.
   - Cache pip dependencies.
   - Required checks: all tests pass.

2. Add a coverage badge and CI badge to the top of README (below the title):
   ```
   ![CI](https://github.com/Abrodolph/Wildfire-Containment-Simulator/actions/workflows/ci.yml/badge.svg)
   ![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)
   ![Theme](https://img.shields.io/badge/Theme-2%20Long%20Horizon-orange)
   ```

3. Create `LICENSE` file with MIT license (to match the README frontmatter).

4. Audit `openenv.yaml` against the latest OpenEnv spec — fetch the latest spec from the openenv repo (github.com/meta-pytorch/openenv if that's the canonical URL at the time of writing) and verify field names, required properties, and schema version. Report any discrepancies and fix them.

5. Clean up `pyproject.toml`:
   - Pin Python to `>=3.10`.
   - Ensure all console_scripts point to existing entry points (no dead references).
   - Move `pytest`, `pytest-cov` to `[project.optional-dependencies]` under a `dev` extra.

6. Add `CONTRIBUTING.md` (brief — 15 lines is fine) explaining how to add a new tier, how to add a new action type, and where tests live.

7. Run `python -c "import env; import server; from env.wildfire_env import WildfireEnv; WildfireEnv().reset(task_id='easy', seed=0)"` as a final smoke test.

Acceptance test:
- CI badge appears (may show pending until the first push).
- `pytest tests/ -v --cov=env` runs clean locally and reports >60% coverage on env/.
- OpenEnv spec audit is completed — paste any discrepancies found and confirm they're fixed.
- Repo root looks clean: no `__pycache__`, no `{env,...}` artifacts, no nested duplicate folder.
```

---

## Prompt 15 (OPTIONAL — Only if P1 complete by April 24 evening) — Multi-Agent Crew Architecture

```
OPTIONAL: Only execute this if Prompts 1-14 are complete AND the training run has produced a working reward curve. Otherwise skip — this is a high-risk refactor close to deadline.

Convert crews from passive tools into semi-autonomous sub-agents. This legitimizes the Halluminate sub-theme claim (Theme 1: Multi-Actor Environments) as a secondary pitch angle.

Tasks:
1. Add `local_observation` method to Crew in `env/resources.py`:
   - Returns a 3×3 neighborhood view centered on the crew's position (fire_state, intensity, smoke), plus crew's own health state.

2. Add a `local_policy` function per crew:
   - Rule-based: if intensity at current cell > 0.8, retreat one cell away from fire center. Otherwise move toward nearest visible fire in the 3×3 window. If no fire visible, hold position.
   - Crews execute this policy automatically each step UNLESS the IC's most recent order overrides.

3. Change IC action space:
   - Keep existing `MOVE_CREW(crew_id, direction)` but re-label semantically as `ORDER_CREW_MOVE`.
   - Add `ORDER_CREW_OBJECTIVE(crew_id, objective: Literal["hold", "advance", "retreat", "prioritize_north", "prioritize_south", "prioritize_east", "prioritize_west"])` — the crew's local policy then biases toward that objective.
   - If the IC issues no order in a given step, crews follow their local_policy autonomously.

4. Reward impact:
   - Add tracking for "autonomous saves" — when a crew retreats on its own local_policy and avoids a casualty that would have otherwise occurred. Log these; they become a talking point ("our crews saved themselves 3 times in this episode without IC instruction").

5. Add `tests/test_multi_agent.py`:
   - `test_crew_retreats_from_high_intensity` — construct scenario with intensity spike at crew's cell, assert crew moves away next step even with no IC order.
   - `test_ic_order_overrides_local_policy` — assert `ORDER_CREW_MOVE` still works when issued.
   - `test_autonomous_save_tracking` — count autonomous_saves after a scripted scenario.

6. Update `env/serialization.py` to include crew local observations in the prompt under a new `CREW REPORTS` section (each crew reports what they see and what they're doing).

7. Update README to add a "Multi-Agent Architecture" section describing the IC/crew decomposition.

Acceptance test:
- `pytest tests/test_multi_agent.py -v` passes all 3 tests.
- Run `python scripts/run_demo.py` — confirm the narrative now includes autonomous crew moments.
- If ANYTHING breaks the existing test suite, revert the changes immediately. This prompt must not destabilize P1 deliverables.
```

---

## Final Checklist (Run Before Submission)

Run these commands sequentially. All must pass.

```bash
# 1. All tests green
pytest tests/ -v

# 2. Baseline eval produces expected pattern
python scripts/evaluate.py 5

# 3. Eval comparison runs
python scripts/eval_compare.py --seeds 42 43 44 45 46 --tiers medium hard --agents random heuristic

# 4. Demo runs cleanly
python scripts/run_demo.py

# 5. Dashboard generates
python scripts/plot_dashboard.py --stats training/training_stats.json --output training/training_dashboard.png

# 6. Replay generates
python scripts/replay.py --tier medium --seed 42 --agent heuristic --output demos/heuristic_medium_42.gif

# 7. Notebook imports work
python training/test_notebook_imports.py

# 8. Env still serves
python app.py &
sleep 3
curl http://localhost:7860/health
kill %1
```
