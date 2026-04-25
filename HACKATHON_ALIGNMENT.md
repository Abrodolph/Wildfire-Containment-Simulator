# Hackathon Alignment — Wildfire Containment Simulator

This document walks through every topic in the organizers' **Hackathon Self-Serve Guide** PDF and describes, topic by topic, what our project currently does, where the gaps are, and concrete changes that would strengthen our submission. It is written to be directly actionable during the final sprint — each "Approach" passage reflects the code on disk today (not aspiration), and each "Potential issues / improvements" list points at specific files, specific lines of behavior, and specific hackathon judging criteria.

Stack reminder (from `pyproject.toml`, `training/grpo_colab.ipynb`, `server/app.py`):

- **Environment:** OpenEnv-style `WildfireEnv` in `env/wildfire_env.py` with Pydantic-typed `Action`/`Observation`/`StepResult`.
- **Trainer:** TRL `GRPOTrainer` + Unsloth 4-bit LoRA on `unsloth/Qwen2.5-1.5B-Instruct`, 50 GRPO steps, 8 generations per prompt.
- **Deployment:** FastAPI at port 7860 (`server/app.py`), Dockerized, deployable as a Hugging Face Space.
- **Baselines:** `RandomAgent` and `HeuristicAgent` in `agents/`, scored by `graders/grader_{easy,medium,hard}.py`.

---

## 0) What you are building

### Approach
We are building exactly the system the guide describes end-to-end: an OpenEnv-compliant RL environment (`env/`) with verifier/reward functions (`env/reward.py`, graders), a TRL `GRPOTrainer` loop (`training/grpo_colab.ipynb`), Unsloth 4-bit quantization + LoRA for efficiency, and a FastAPI/Docker deployment suitable for a Hugging Face Space (`server/app.py`, `Dockerfile`, `openenv.yaml`). The task is a long-horizon disaster-response decision problem — an LLM acts as an Incident Commander dispatching crews, tankers, firebreaks and recon flights over 80–300 steps per episode. Every piece in the "Environment → verifier → TRL → Unsloth → OpenEnv/Spaces" pipeline the PDF specifies has a real implementation in this repo.

### Potential issues / improvements
- **Pipeline is technically complete but not yet empirically closed.** `README.md` still has `{TBD}` placeholders for the trained-model numbers, and `scripts/results.json` only contains random/heuristic baselines. Judges will discount a project whose headline claim (trained LLM beats heuristic) is not demonstrated. Highest-leverage action: run the training notebook's Section 6 evaluation, paste the numbers into README, and back them with `demos/` GIFs.
- **No "value model is verifier" narrative is explicit in the repo.** Write a one-paragraph "Why GRPO + RLVR fits this env" blurb into the README so the judges can see, in 30 seconds, that we understood the intended stack.

---

## 1) Start with the right project idea

### Approach
The task satisfies all three properties in the guide:

1. **Step-by-step action** — `env/wildfire_env.py:step()` executes exactly one `Action` per tick, cycling through 11 deterministic sub-steps (validate → execute → spread → suppress → weather → moisture → smoke → cooldowns → reveal expiry → hard-tier events → reward).
2. **Programmatic verification** — The grader family (`graders/grader_easy.py` and siblings) computes `total_reward`, `containment_pct`, `pop_saved_pct`, `crew_casualty` from the ground-truth `env.state()` — no human judgment required.
3. **Difficulty calibrated so success probability > 0** — The heuristic baseline currently scores +7.0 ± 0.26 on easy, +3.93 on medium, +5.32 on hard (`scripts/results.json`); the `tier_scale` in `env/fire_spread.py` (1.0 / 0.7 / 0.55) is explicitly tuned so that rollouts routinely produce non-zero reward.

### Potential issues / improvements
- **Medium-tier variance is huge** — heuristic gets `[-0.85, 7.09, 7.08, 0.01, 6.31]` across seeds 42–46, std 3.57. That bimodal distribution (either total save or near-total loss) is exactly the pattern the PDF warns about under "so hard that the model never succeeds." If the model sees mostly the bad mode early in training, learning will stall. **Fix:** inspect why seeds 42 and 45 fail for the heuristic — likely the two ignition points spawn on opposite sides of a populated cluster — and tighten `_find_ignition_candidate` in `env/wildfire_env.py` to guarantee at least one winning crew-deployment plan exists.
- **Hard tier scores look suspiciously uniform at 6.7** — four of five seeds returning the identical value 6.7 strongly suggests the episode terminates at the same early exit condition (probably "all population lost" or "fire self-extinguishes before staggered ignition"). The variance should be investigated before training on hard.

---

## 2) Understand the minimum RL loop before you build

### Approach
The 5-step RL loop is implemented cleanly and discoverable inside `training/grpo_colab.ipynb` cell `code-rollout` and `code-grpo-setup`:

1. **Prompt** — `serialize_observation(obs, step, max_steps)` in `env/serialization.py` formats the observation into a structured LLM prompt (SITUATION / GRID SUMMARY / RESOURCES / RECENT EVENTS / Available actions).
2. **Generation** — model.generate inside `collect_rollout` and inside `reward_fn`.
3. **Execute** — `parse_action()` → `env.step(action)` returns a `StepResult`.
4. **Reward** — decomposed step reward plus terminal spike, produced inside `env/reward.py` and assembled in `wildfire_env.step()`.
5. **Update** — `GRPOTrainer.train()` does the gradient step; 8 generations per prompt, 50 steps, lr 5e-6.

### Potential issues / improvements
- **The loop inside `reward_fn` is not the same as the loop inside `collect_rollout`.** `reward_fn` uses the candidate completion only at step 0 and then runs the **heuristic** for 14 more steps. That is a legitimate variance-reduction trick (rollouts dominated by the heuristic have less noise), but it means the gradient signal mostly measures "how good is this single first action followed by a scripted policy" — not "how good is this model's long-horizon plan." Consider a hybrid: sample 50% of rewards from heuristic-continuation and 50% from pure-model continuation.
- **`collect_rollout` is defined in the notebook but never actually called during training.** It's essentially dead code. Either delete it or wire it into an evaluation loop so it earns its keep.

---

## 3) Decide whether you need SFT first

### Approach
We are following the guide's "usually RL from a capable base" path: start from `unsloth/Qwen2.5-1.5B-Instruct` (already instruction-tuned on general chat), add no SFT warm-up, and rely on the base model's JSON-formatting ability plus our tolerant 3-layer parser (`env/action_parser.py`) to get non-zero reward on the very first rollout. The easy tier's high reward ceiling (~+8) and the heuristic continuation inside `reward_fn` both substantially raise the probability that the first few rollouts see positive reward, which is the precondition the PDF calls out.

### Potential issues / improvements
- **No evidence we measured what the pre-RL model actually outputs.** We should run 10 rollouts of the un-trained Qwen-2.5-1.5B through `collect_rollout`, count: (a) JSON parse success rate, (b) semantically-valid action rate, (c) mean episode reward. If JSON success is below ~70%, do a tiny SFT pass (even 50 heuristic-generated examples) purely for format priming — that is exactly the "light SFT first" pattern the guide endorses for hackathons.
- **Heuristic trajectory harvesting is cheap and we already own the heuristic.** A simple script that runs `HeuristicAgent` over seeds 0–199 and logs `(prompt, action_json)` pairs would yield ~10k-30k supervised examples for a warm-up pass. This is optional but high-return insurance.
- **The system prompt in the notebook is minimal** (`'Respond with ONLY a valid JSON action object and nothing else.'`). A richer system prompt that includes the full action schema (as `inference.py` already has in `SYSTEM_PROMPT`) would cut early format failures.

---

## 4) Design the environment before you design the trainer

### Approach
The environment is the first-class artifact in this repo: `env/` has 13 modules, the trainer is a single notebook. The `WildfireEnv` class in `env/wildfire_env.py` exposes the four methods the guide requires:

- `reset(task_id, seed)` → `Observation` (deterministic from seed via `np.random.default_rng(seed)` passed to every sub-system).
- `step(action)` → `StepResult` (11-step tick pipeline, never crashes on bad input).
- `state()` → full ground truth dict (used only by graders, documented as "NOT for agent use").
- Reward is computed inside `step()` via `RewardCalculator.compute_step_reward` + `compute_terminal_reward`.

The five design questions the guide poses are each answered explicitly:

- **What does the agent observe?** — `Observation` in `env/models.py:298` (grid, weather, resources, stats, recent_events, briefing).
- **What actions can it take?** — `ActionType` enum (7 types) with Pydantic per-type field validation.
- **What ends an episode?** — `_check_termination` in `wildfire_env.py:470` — time limit, fire extinguished (with staggered-ignition protection), or total population lost.
- **Reward?** — documented in the README and `openenv.yaml`.
- **Abuse/infinite-loop prevention?** — `episode_length` hard cap, `_validate_action` returns safe messages without exceptions, `parse_action` has a 3-layer fallback that can never return a non-Pydantic-valid `Action`.

### Potential issues / improvements
- **Observation is enormous.** On hard tier (40×40), the grid alone is 1600 `CellObservation` objects, which `serialize_observation` then summarizes via BFS clustering in `env/serialization.py`. Verify the prompt token count is comfortably under `MAX_SEQ_LENGTH=2048`; on hard tier with many fire clusters it may be tight. Add a `len(tokenizer(prompt).input_ids)` assertion at the top of `reward_fn` for the first few calls.
- **Sensor noise is asymmetric.** Wind speed/direction get ±5 km/h / ±20° noise (`env/weather.py`), but moisture, smoke, fire intensity are exact. That is fine, but we should document it so judges don't assume we forgot.
- **`state()` can be accessed at any time** — there is no enforcement that agents only see `Observation`. A malicious agent author could just call `env.state()` during the grader loop. For competition integrity, lock down `state()` when the caller is an `Agent` interface, or at minimum document that it is a grading-only hook.

---

## 5) Build the environment using OpenEnv

### Approach
The project is structured as a Python package exposing the OpenEnv contract:

- `action` / `observation` / `state` dataclasses live in `env/models.py` as Pydantic models (stricter than dataclasses — the guide's recommendation is satisfied and exceeded).
- `WildfireEnv.reset`/`.step`/`.state` implement the methods.
- `server/app.py` wraps the env in a FastAPI app with `/reset`, `/step`, `/state`, `/health`, `/` (HTML landing), `/docs` (Swagger).
- `openenv.yaml` declares the environment class, action space, observation space, reward range (-8 to +8), and three tasks (`easy`/`medium`/`hard`).
- A root `app.py` shim and `Dockerfile` publish it as a Space on port 7860.

The separation the guide calls for — "environment handles world dynamics and scoring, trainer handles optimization, model just learns to act" — is honored: `env/` has no trainer dependency, and the trainer notebook only imports from `env/`, `agents/`, `graders/` via the public surface.

### Potential issues / improvements
- **`_env` is a module-level singleton in `server/app.py`.** Concurrent `/reset` calls from different clients will clobber each other's episode state. Fine for a demo, but a judge running two browser tabs will see garbled behavior. Either switch to a per-request env factory, or document the single-tenant assumption on the HTML landing page.
- **`openenv.yaml` is slightly out of sync with `env/models.py`.** The YAML lists six action types (`deploy_crew, move_crew, drop_retardant, build_firebreak, recon_flight, idle`) but `ActionType` defines seven — `ORDER_CREW_OBJECTIVE` is missing from the YAML. Add it before pushing to Space so the env manifest matches reality.
- **No `openenv init` scaffold in tree** — we hand-built the package. That is fine, but run `openenv push` (or the equivalent `git push` to the Space repo) *now*, not the night before the deadline, so any manifest-mismatch surprises surface early. This is the guide's Topic 13 point restated.

---

## 6) Keep the task simple at first

### Approach
The three-tier curriculum is a literal implementation of the "easy → medium → hard" progression the guide describes, and `env/curriculum.py`'s `CurriculumController` auto-promotes the trainer from easy → medium when a rolling 10-episode average crosses 4.0, and medium → hard at 3.5. It also auto-demotes if average falls below 50% of the prior threshold. The training notebook wires this into `reward_fn` so every batch updates the tier. Heuristic scores confirm success is possible at every tier (means of 7.0 / 3.93 / 5.32).

### Potential issues / improvements
- **Curriculum promotion happens inside `reward_fn`, but the training dataset is frozen at `build_prompt_dataset(50)` *before* `trainer.train()` is called.** Concretely: even if the controller promotes `easy → medium` at step 10, the prompts being scored from step 10 onward are still the **easy** prompts generated up front. The `tier` column in each dataset row is the tier that was active at dataset-build time, not the current tier. **This is a real bug** and it partially explains why `training_stats.json` shows the model spending steps 0-9 on `easy`, then the `tier` field flips to `medium` at step 10 — but every rollout is still running on easy-generated prompts. Fix: rebuild the dataset (or use a dataset-generating callback) whenever the controller returns a promotion.
- **Curriculum thresholds are hard-coded and have not been validated.** 4.0 and 3.5 were picked to match the heuristic's scores, but the *initial* model scores before RL starts are unknown. If Qwen-2.5-1.5B starts at e.g. 5.5 on easy, it will promote on step 1 — too fast. Log the first 20 rewards before enabling promotion.
- **No rollouts ever happen on `hard` during the first 20 steps** according to `training_stats.json` — only at step 20+ does `hard` appear. Given the 50-step budget, only ~30 of the 50 GRPO steps ever see hard-tier gradients. If hard is the theme's centerpiece (long-horizon planning), that is too few.

---

## 7) Design rewards carefully

### Approach
Our reward was intentionally restructured during "Prompt 2" (see `Summary.txt` and `prompts.md`) to match exactly the guide's multi-component advice:

- **Dense step reward** (`compute_step_reward`) — `0.4·Δ containment + 0.4·Δ population_safety − 0.1·redundant_action_flag`.
- **Sparse terminal reward** (`compute_terminal_reward`) — `+5.0` for zero population lost, an efficiency bonus up to `+2.0` for finishing early, `−3.0·loss_pct` for partial loss, `−2.0` for any crew casualty, `+1.0` briefing-adherence bonus if all priority zones survive, and `−0.01·invalid_action_count` capped at `−0.2`.

Reward range is ~`−8` to `+8`, documented in `openenv.yaml:100`. This produces meaningfully-separated advantages for GRPO (the guide's whole justification for wide rewards).

### Potential issues / improvements
- **`containment_pct` is reported as an integer percentage (0–100) in `ClusterStats` but as a fraction (0–1) inside `_snapshot_state` / `compute_step_reward`.** Verify we aren't accidentally multiplying by 100 somewhere — a single unit error here means the delta-containment term dominates or vanishes entirely.
- **Only two delta components drive the dense reward.** The guide stresses "multiple independent reward functions." Good candidates that we already compute but don't reward: resource efficiency (wasted vs. total actions), area-saved ratio, briefing-compliance (already terminal-only — promote it to a per-step signal).
- **Briefing adherence stuffs the raw `Grid` object into `terminal_state["_grid_ref"]` inside `wildfire_env.py:229`.** That leaks a mutable handle into the reward calculator. If the grader ever serializes the state dict (e.g. for logging), it will explode on the non-JSON-serializable Grid. Cleaner: compute the priority-zone survival boolean inline in `wildfire_env.py` and pass only a bool.
- **Redundant-action detection is too shallow.** `_is_redundant` only compares `action_type + target_row + target_col` of the immediately prior action. A model that alternates `DEPLOY_CREW(0,0) / MOVE_CREW(crew_0, N)` in a loop gets no penalty. Either widen the window or add a format-compliance signal (next bullet).
- **Missing reward component: `action_validity`.** Right now an invalid action silently costs `0.02·count` inside the legacy reward, and `0.01·count` (capped 0.2) inside the terminal reward. For GRPO this is too subtle. Add a per-step `-0.05 if not action_was_valid` signal so that producing syntactically valid JSON is itself rewarded — this is the single highest-leverage fix for the early training regime where most LLM outputs are malformed.
- **Missing reward component: `parse_status` bonus.** `parse_action` returns one of `json_success`, `regex_fallback`, `safe_idle`. Reward `json_success` with a small bonus (`+0.02`) so the model learns clean JSON, not just barely-parseable regex output. This is genuinely an independent signal and directly matches the guide's "reward format compliance" example.

---

## 8) Protect yourself against reward hacking

### Approach
Several anti-hacking defenses are already in place:

- **Action validation is enforced twice** — Pydantic validates field presence/types at construction time, and `_validate_action` in `wildfire_env.py:407` enforces bounds. A malformed action never crashes the env; it just returns a penalty.
- **Hard timeouts** — each tier has a fixed `episode_length`, and `_check_termination` guarantees termination.
- **No unrestricted global state** — the env is fully seeded via `np.random.default_rng(seed)`. The agent has no way to mutate the RNG or grid from within the LLM's output channel.
- **No arbitrary code execution** — the parser takes strings and produces a constrained Pydantic `Action`; it does not `eval` anything.
- **Fog-of-war and smoke occlusion are computed on the server side** — the agent cannot read hidden cells through the observation API.
- **Grader is separate from the env** — `graders/grader_*.py` calls `env.reset/step/state`; the agent never touches the grader, so it cannot alter how it's being scored.

### Potential issues / improvements
- **The biggest hacking surface we haven't closed: `parse_action` silently downgrades to IDLE.** A model that outputs garbage constantly will get IDLE-reward (mostly 0 + whatever deltas the environment produces on its own), which on easy tier can still exceed `+5` because the heuristic-scale fire spread is mild. Confirm this with a controlled experiment: run 50 episodes with a "pure garbage agent" (returns random strings) and see what the episode reward is. If it's > +2, that is a free-reward exploit and we need a per-step "safe_idle fallback" penalty of at least `-0.2`.
- **`reward_fn` continues with the heuristic for 14 steps after the model's one action.** Under adversarial framing, *the model doesn't even need to act well* — the heuristic will recover most episodes. The gradient signal will be dominated by the heuristic's rollout, not the model's policy. Add at least some pure-model rollouts (say, 2 of every 8 generations) so the model is directly scored on its full trajectory.
- **The `redundant_action` penalty is bypassable** by inserting an IDLE action between two identical real actions — the comparison is only against `_prev_action`. Fix by tracking a short sliding window of recent actions.
- **Human inspection is not wired in.** The training notebook logs `mean_reward` per step to `training_stats.json` and prints to stdout, but completions are never sampled to disk. Add a `if step % 10 == 0: print(first_completion)` block so we can eyeball what the model is actually generating and catch reward hacking the moment it starts.
- **No seed diversity audit** — `SEED_POOL = list(range(100))` in the notebook. If 100 seeds are cycled through 50 steps × 8 generations = 400 rollouts, some seeds repeat ~4×. That's fine, but a model that memorizes seed-specific fire patterns would look good in training and fall apart on eval seeds 42-46. Use a larger pool or sample seeds without replacement per batch.
- **`_ignite_initial_fires` is seed-deterministic.** That is great for reproducibility, bad for generalization. Evaluate on held-out seeds (say 200-250) to confirm no over-fitting.

---

## 9) Use process-aware feedback when you can

### Approach
Our reward is primarily outcome-based (delta containment, terminal survival), but the guide's "lightweight process checks" category has footholds already:

- The 3-layer parser (`json_success` / `regex_fallback` / `safe_idle`) is a ready-made process signal — we just aren't using it as a reward component yet.
- `recent_events` in the observation shows the agent what happened immediately after its last action, giving the LLM in-context process feedback even if the gradient signal doesn't encode it.
- `info["reward_breakdown"]` on every step (see `wildfire_env.py:244`) already decomposes containment / population / efficiency / speed / area / invalid_actions — a perfect vector for process-aware per-step rewards.

### Potential issues / improvements
- **`reward_breakdown` is computed every step but never used for gradients.** Promote it to a multi-head reward list that `GRPOTrainer` consumes — TRL's `reward_funcs` parameter accepts a **list** of callables. Splitting our current scalar into `[containment_reward, population_reward, efficiency_reward, format_reward]` would give us the "multiple independent reward functions" the guide explicitly recommends, with near-zero code cost.
- **No LLM-as-a-judge anywhere.** The guide says "lightweight" process checks are the hackathon sweet spot — we are on the right side of that warning. Do not add judges under deadline pressure.
- **Briefing adherence is only terminal.** Convert it to per-step: at each step, count priority zones currently safe vs. burning, and reward the delta. That gives the model sub-episode feedback on instruction-following, which is explicitly Theme 2's whole point.

---

## 10) Pick the right training stack

### Approach
We are running the exact stack the guide recommends:

- **TRL `GRPOTrainer`** pinned to `0.12.1` in the install cell (chosen to avoid the mergekit/llm_blender eager-import issues in newer TRL releases — see notebook cell `code-install`).
- **Unsloth** with `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` and `get_peft_model(..., r=16, lora_alpha=32)` on `q/k/v/o_proj`.
- **OpenEnv** shape — `reset/step/state` with FastAPI wrapper per Topic 5.

### Potential issues / improvements
- **LoRA target modules are minimal.** `['q_proj', 'k_proj', 'v_proj', 'o_proj']` covers attention only. For Qwen-2.5 you'd typically also adapt `gate_proj`, `up_proj`, `down_proj` (the MLP) to meaningfully shift behavior. That bumps trainable parameter count ~2× but is well within T4 memory for a 1.5B model. Strongly recommend adding these — our model is probably under-expressive right now.
- **`num_generations=8, batch=1, grad_accum=4`** means each optimizer step sees 32 trajectories of gradient signal. Fine, but `max_completion_length=128` is tight — actions with `reason` strings or edge-case JSON formatting may get truncated. Bump to 192 if latency allows.
- **TRL version pin is a ticking clock.** 0.12.1 is several releases behind. Document exactly *why* in a comment (we already do) so a reviewer doesn't assume carelessness.
- **Unsloth install line is heavy.** `unsloth[colab-new] @ git+https://...` pulls the latest, which may break on the day of judging. If we can, pin to a specific commit.

---

## 11) Prefer GRPO / RLVR style training for verifiable tasks

### Approach
Our task is verifiable end-to-end — `env.state()` returns fully objective population-lost / cells-burned / containment-pct numbers, and our reward is computed from those numbers plus deterministic action-history counters. No learned reward model is used anywhere. We are using TRL's GRPO (`GRPOTrainer`), which the guide specifically endorses over older PPO setups. The verifier (`env/reward.py`) was built before the training notebook — the right order.

### Potential issues / improvements
- **The "verifier" is de facto the env itself, not a separate module.** That is fine but it means if the env has a bug, the verifier has the same bug. Add a small `tests/test_reward_is_deterministic.py` that runs a fixed heuristic-agent rollout twice on the same seed and asserts the reward sequence is bitwise identical — guards against accidentally-stochastic reward paths.
- **No unit test asserts that `compute_terminal_reward` lies in `[-8, +8]`.** Add one. The advertised range in `openenv.yaml` is a judging talking point; violating it quietly would be embarrassing.
- **GRPO reference model is implicit.** TRL's `GRPOTrainer` uses the pre-training model as the KL reference. On Unsloth 4-bit that KL-reference setup is sometimes subtly broken if the `ref_model` path isn't configured. Confirm the `kl_coef` and reference model are actually contributing to the loss by logging the `kl` column; if it's exactly 0 throughout training, GRPO has degenerated to REINFORCE.

---

## 12) Keep inference fast

### Approach
Several efficiency choices align with this guidance:

- **Unsloth 4-bit** roughly halves memory vs. a standard 8-bit load and gives ~2× generation speedup on T4.
- **`max_completion_length=128`** caps generation time per rollout at a few hundred ms.
- **Heuristic continuation in `reward_fn`** — the single most effective speed trick in the notebook. Only the first of 15 rollout steps runs the LLM; the remaining 14 run the hand-coded heuristic at zero GPU cost. That directly addresses the guide's "inference dominates runtime" warning.
- **Vectorized grid ops** — `env/grid.py` and `env/fire_spread.py` use NumPy for all per-cell loops, not Python.
- **Observation serialization clusters cells** into bounding boxes before sending to the LLM, so the grid-summary string is O(regions), not O(cells).

### Potential issues / improvements
- **The LoRA model is loaded without `FastLanguageModel.for_inference()` in some call paths.** Double-check: `collect_rollout` does the switch, but `reward_fn` does *not* — it calls `model.generate` in training mode. This can be 2-4× slower than inference mode on Unsloth. Either (a) move the `FastLanguageModel.for_inference` toggle into `reward_fn` and restore `for_training` before the optimizer step, or (b) use `with torch.no_grad():` around the generate call if mode switches are awkward.
- **No batched generation.** `reward_fn` iterates over `completions` list serially. If TRL passes them as a batch, great — but if it sends one at a time and we could otherwise batch them, we're leaving throughput on the floor.
- **`model.device` is called but the notebook doesn't assert GPU.** On a T4 it will always be CUDA, but on a reviewer's CPU-only env it will silently run and take hours. Add `assert torch.cuda.is_available()` at the top of Section 1 to fail fast.
- **`serialize_observation` on hard tier is O(rows·cols) for every rollout step**. For 40×40×300 steps per episode × 8 generations × 50 training steps that's 192M cell touches. Profile it — if it's > 5% of wall time, cache the previous serialization and diff.

---

## 13) Deploy your environment early

### Approach
We have the deployment artifacts ready: `server/app.py` (FastAPI on port 7860), `Dockerfile` (`python server/app.py`), `openenv.yaml` manifest, and a Hugging Face Space reference in the README (`Eshit/Wildfire-Containment-Simulator`). The README claims it is live. The root `app.py` is a shim that forwards to `server.app:main` so both `docker run` and `python app.py` start the same server. Endpoints provided: `/`, `/health`, `/reset`, `/step`, `/state`, `/docs` (auto-generated Swagger).

### Potential issues / improvements
- **Verify the Space is actually live and up-to-date.** If the repo has drifted since the last push (e.g., new actions in `models.py`), the Space will 500. Before the demo, run `curl https://<space-url>/health`, then `curl -X POST .../reset?task_id=easy&seed=42`, then step. A working remote demo is a judging multiplier.
- **No automated deploy pipeline.** GitHub Actions is wired for CI tests (per the README badge) but not for pushing to the Space. At minimum, document the push command in `training/README.md` or `AGENTS.md` so any teammate can redeploy in 30 seconds.
- **`_env` singleton means we can't easily show a second demo tab in parallel.** For the demo video this may be fine; for a live judge interaction it could embarrass. If time allows, switch `server/app.py` to a `dict[session_id -> WildfireEnv]` keyed on a cookie or header.
- **The HTML landing page uses `&#128293;` instead of UTF-8 🔥** — that is fine but looks dated. Polish up the `/` endpoint HTML — a sharper landing page is free product polish.

---

## 14) Scale only after the environment is stable

### Approach
The repo shows we followed this order. `prompts.md` confirms a sequence: (1) "Repo Cleanup & Test Scaffolding" with smoke tests → (2) "Reward Restructuring" with reward tests → (3) "Observation-to-Text Serializer" → (4) "LLM Action Parser" → (5) "Replay / GIF Renderer" → ... *then* training. The test suite (`tests/test_smoke.py`, `test_reward.py`, `test_serialization.py`, `test_action_parser.py`, `test_rendering.py`, `test_briefing.py`, `test_curriculum.py`, `test_graders.py`, `test_dashboard.py`, `test_eval_compare.py`) verifies reset works, step works, rewards are sensible, the parser never crashes, graders run to completion, and renderings are non-empty — exactly the "before you scale" checklist the guide prescribes. Logs are visible via `recent_events` on every observation and via `info["events"]` in the StepResult.

### Potential issues / improvements
- **Batch size was not stepped up after stabilization.** We are still at `per_device_train_batch_size=1, grad_accum=4` — the guide's "only then, increase batch sizes" step hasn't happened. On T4 we can probably go to `batch=2, grad_accum=2` (same effective batch, faster wall clock) without running out of VRAM. Try it.
- **Prompt dataset is 50 rows and never resampled.** After the environment stabilized we should have diversified prompts — e.g., starting each prompt from a random step offset, not always step 0. Right now every training prompt is a *fresh reset* — so the model never learns mid-episode state recognition.
- **No throughput benchmark is checked in.** Add a `scripts/bench_rollout.py` that times 10 full rollouts and prints steps/sec. That number going into judging ("our env runs 480 steps/sec on T4") is an objective achievement to cite.

---

## 15) Monitor the right things during training

### Approach
`training_stats.json` already logs `(step, tier, mean_reward)` per GRPO step, and `scripts/plot_dashboard.py` renders training curves with tier-promotion markers. The GRPO trainer's own `logging_steps=1` means we see reward, loss, and KL every step in stdout. `scripts/eval_compare.py` produces a multi-agent comparison table against saved baselines.

### Potential issues / improvements
- **We only log `mean_reward`** — the guide explicitly warns against watching a single scalar. We should also log: (a) `json_success_rate`, (b) `regex_fallback_rate`, (c) `safe_idle_rate`, (d) `invalid_action_count`, (e) `pop_lost_rate`, (f) `crew_casualty_rate`, (g) `mean_episode_length`. All of these are already computed in `info["reward_breakdown"]` on every step — we just need to aggregate them.
- **No generation sampling to disk.** The guide's last bullet under this topic: "inspect actual generations during training." Right now we never save any completion strings. Add a step where every 10 training steps, we save the first completion of each of the 8 generations to `training/samples/step_{n}.txt`. That alone would let us catch reward hacking within 10 steps instead of at epoch end.
- **No TensorBoard / W&B hook.** TRL supports both out of the box; 3 lines in `GRPOConfig(report_to="tensorboard")`. Worth it for the judge-demo screenshot alone.

---

## 16) Save models correctly

### Approach
We are doing the right thing according to the guide's warning: `model.save_pretrained('checkpoints/final')` on the **LoRA-adapted 4-bit model**, followed by an explicit verification step that reloads via `FastLanguageModel.from_pretrained(final_ckpt, load_in_4bit=True)` and prints success. We do not attempt a 4-bit → 16-bit upcast and merge. The `checkpints-140/` directory in the repo shows the checkpoint format is the HuggingFace adapter-only layout (`adapter_config.json`, `adapter_model.safetensors`) — exactly the "adapters directly" path the guide recommends.

### Potential issues / improvements
- **Directory name is misspelled: `checkpints-140` (missing an 'o')**. Not a bug, but if any download script or README link uses the correct spelling it will 404. Rename to `checkpoints-140` or at least document the typo.
- **Post-training inference is only tested inside the notebook, not via the server path.** After saving, the next logical step is: (a) bake the adapter into the `Dockerfile` so the Space serves the trained model, or (b) leave the server as a pure env and let the LLM live elsewhere. We have implicitly chosen (b) — confirm that decision in the README so judges understand the architecture.
- **No test that the checkpoint actually improves behavior.** Save and reload succeed even if the adapter is all zeros. Add an assertion: `assert trained_mean_easy > untrained_mean_easy + 0.5` inside Section 6 of the notebook.
- **Adapter export as a downloadable zip is documented** but not scripted. A single `scripts/package_adapter.py` would be more reliable than a Colab-specific recipe in the README.

---

## 17) How to structure your team over the hackathon

### Approach
The four roles in the guide all have owners' fingerprints in the repo:

- **Person A (Environment)** — `env/` (13 modules), `server/app.py`, `Dockerfile`, `openenv.yaml`. Every component is separated cleanly.
- **Person B (Verifier / Rewards)** — `env/reward.py`, `graders/` (one per tier), `env/action_parser.py` (anti-corruption layer between LLM and env).
- **Person C (Training)** — `training/grpo_colab.ipynb`, `training/README.md`, `training_stats.json`, `scripts/plot_dashboard.py`, `scripts/eval_compare.py`.
- **Person D (Demo / Product)** — `scripts/run_demo.py`, `scripts/find_demo_seed.py`, `scripts/replay.py`, `env/rendering.py`, the HTML landing page in `server/app.py`, README narrative.

### Potential issues / improvements
- **The demo person has the thinnest deliverables right now.** `demos/` GIFs and the `{TBD}` rows in the README are the weakest part of the current submission. Carve out explicit time for: (a) a 60-second demo video with heuristic-vs-trained side-by-side, (b) final benchmark numbers, (c) a polished Space landing page.
- **Verifier role is sharing code with the environment role.** `env/reward.py` is the verifier — that is fine in a small team, but when iterating on reward we should version-tag the reward file (e.g., a `REWARD_VERSION = "v2_decomposed"` constant) so we can tell from a checkpoint which reward it was trained against.

---

## 18) A practical 1-day execution plan

### Approach
Mapping our current state to the guide's 9 phases:

- **Phase 1 (narrow task):** Done — easy tier is the narrow task, hard is the stretch goal.
- **Phase 2 (build the env):** Done — `env/` is feature-complete.
- **Phase 3 (build rewards):** Done — decomposed step+terminal; 4+ reward components.
- **Phase 4 (deploy):** Partially done — Docker + FastAPI work locally; Space reference exists but needs verification.
- **Phase 5 (train small):** Done once — `training_stats.json` shows 50 GRPO steps completed, final checkpoint in `checkpints-140/`.
- **Phase 6 (inspect for hacking):** **Not done** — no completions have been saved to disk during training.
- **Phase 7 (add curriculum):** Done — `CurriculumController` with known caveat (dataset is frozen, see Topic 6).
- **Phase 8 (train bigger):** **Not done** — no second training run with larger batch / more steps / diversified prompts.
- **Phase 9 (save and demo):** Partially done — checkpoint saved; demo video and eval-table numbers outstanding.

### Potential issues / improvements
- **Priority for the remaining time, in order:**
  1. Fix the frozen-dataset bug (Topic 6) and run a second training pass with a *live* curriculum.
  2. Generate training-time completion samples and eyeball them (Topic 8 / 15).
  3. Populate the `{TBD}` rows in the README with real numbers (Topic 19).
  4. Record the demo video and push the Space.
  5. (If time remains) Expand LoRA target modules and bump `max_steps` to 150.
- **Do not start any new feature.** Every incomplete feature at submission time is a judge-question risk.

---

## 19) What judges or reviewers will likely find compelling

### Approach
We have five of the six compelling-project elements in place:

- **Clear environment design** — `env/` with separated subsystems, documented Pydantic models, `openenv.yaml` manifest.
- **Objective reward functions** — verifiable from `env.state()`, no LLM judge.
- **Evidence of model improvement** — `training_stats.json` shows ~+4 to +5 across training (noisy but present).
- **Prevention against reward hacking** — typed actions, 3-layer parser, episode timeouts.
- **Reproducible deployment story** — Dockerfile + openenv.yaml + Space reference.

The one missing element is **a sharp demo**. The 5-beat demo format the guide recommends (baseline attempt → verifier output → trained attempt → measurable improvement → safeguards) maps naturally onto our `scripts/run_demo.py` + `scripts/replay.py` + README narrative. The GIF renderer (`env/rendering.py`) is ready; we just need to produce the three clips.

### Potential issues / improvements
- **The README's banner claim uses `{TBD}%` for both heuristic and trained numbers.** This is the first thing judges see. Replace it with real numbers or soften the phrasing.
- **Nothing visually distinguishes a trained run from an untrained run** in the demo assets today. A side-by-side GIF (two panels, same seed) would be a 10x force multiplier for our 60-second pitch.
- **The "safeguards" story isn't spelled out anywhere user-facing.** Turn this document's Topic 8 section into two bullet points on the README so judges can see we thought about reward hacking.
- **We have a unique selling point the guide doesn't: Theme 2 (long-horizon + instruction following).** The `OperationalBriefing` is a genuinely novel element — we should call it out explicitly in the pitch. "Most RL agents don't follow instructions. Ours reads a briefing, identifies priority zones, and gets rewarded for obeying the commander's intent."

---

## 20) Suggested problem statement theme directions

### Approach
The README declares **Theme 2: Long-Horizon Planning & Instruction Following**, and every design decision flows from that choice:

- **Long-horizon:** 300-step hard episodes, sparse terminal reward (+5 only on full survival), rewarding recovery from staggered ignition and crew loss.
- **Instruction following:** `OperationalBriefing` on reset, explicit per-episode priority zones, briefing-adherence reward term.

Both pillars are first-class features of the environment, not after-thoughts.

### Potential issues / improvements
- **The briefing currently contains two priority zones and some infrastructure, but only the priority zones contribute to the adherence reward.** If "instruction following" is our pitch, the briefing should have *more* followable directives that the reward tracks — e.g., "maintain Corridor X open" → reward if no fire ever crosses the corridor; "conserve recon for mid-episode" → reward if recon is used after step 50.
- **The "long horizon" claim is only hard-tier.** On easy tier, 80 steps is not "long horizon" by RL standards. Be precise in the pitch: mention that easy is a proving ground and the headline number is hard.

---

## 21) Common mistakes to avoid

### Approach
Evaluating our project against the guide's blacklist:

| Mistake | Our status |
|---------|-----------|
| Task so hard success is zero | ✅ Avoided — heuristic routinely scores positive on all tiers. |
| Using only one reward function | ⚠️ Partially — we have multiple components but one combined scalar. |
| Not checking for reward hacking | ⚠️ Partially — structural defenses in place, but no completion inspection loop. |
| Training before env is stable | ✅ Avoided — see `prompts.md` ordering. |
| Relying only on average reward | ❌ This is what we are currently doing. |
| Forgetting timeouts / sandbox | ✅ Avoided — `episode_length` cap, Pydantic validation, parser fallback. |
| Saving LoRA/QLoRA models incorrectly | ✅ Avoided — adapter-only save, explicit reload test. |

### Potential issues / improvements
- **The two partial-credit items (multiple reward functions; completion inspection) are the cheapest wins left.** Both can be added in under an hour:
  - Split the reward scalar into a list of callables for TRL — see Topic 9.
  - Dump 1-2 completions per 10 training steps — see Topic 15.
- **Relying only on average reward is the worst of the three issues.** Fix this before the final training run. Grep `training_stats.json`: the current file has exactly `mean_reward` and nothing else. This is the guide's single most-warned-against failure mode and we're committing it directly.

---

## 22) Learning Resources

### Approach
The 5 video modules in the guide are aligned with code we've already written:

- **Module 1 (Why OpenEnv?):** Our env implements the Gymnasium-like `reset/step/state` contract and is Dockerized — matches Sanyam's argument for a universal interface.
- **Module 2 (Using existing envs):** Not directly applicable (we are *producing* an env, not consuming one), but Ben's three Space interfaces (server / repo / registry) are all reachable from our Space.
- **Module 3 (Deploying envs):** `openenv init`-style scaffold exists (we hand-built it), local Uvicorn works (`python server/app.py`), Docker run works.
- **Module 4 (Building your own):** `env/wildfire_env.py` + `env/models.py` are the business logic + models files Ben demonstrates; our parser and serializer are the "client" glue.
- **Module 5 (Training + TRL / Wordle GRPO walkthrough):** Our `training/grpo_colab.ipynb` is the direct parallel — `reward_fn` is our `rollout function`, reward shaping is in `env/reward.py`, `GRPOTrainer` is used the same way.

### Potential issues / improvements
- **We have not confirmed alignment with the Wordle walkthrough's exact `reward_fn` signature.** TRL changed the callback convention twice between 0.10 and 0.12; verify the `(completions, prompts, **kwargs)` form we use is the one 0.12.1 expects.
- **No one on the team has watched Module 5 recently.** Put a 15-minute rewatch on the schedule tonight — Lewis's Wordle GRPO walkthrough was the direct blueprint for what we're doing and will expose any pattern we diverged from.
- **Share this document with all four role-owners** before the final push. Every improvement listed above has an owner in Topic 17's table; making ownership explicit reduces coordination cost in the last 24 hours.

---

## Final summary — the three highest-leverage changes

If we make exactly three code changes in the remaining time, they should be:

1. **Fix the frozen-dataset-vs-live-curriculum bug** (Topic 6). Regenerate the prompt dataset each time the `CurriculumController` returns a promotion. Without this, the model never actually trains on medium or hard prompts.
2. **Split the reward scalar into a list of reward functions and sample completions to disk every 10 steps** (Topics 7, 9, 15, 21). Cheap, directly addresses the guide's single most-repeated advice, and gives us the "multiple independent reward functions + human inspection" talking points for judging.
3. **Populate `{TBD}` rows in the README with real trained-model numbers and produce a side-by-side demo GIF** (Topics 19, 20). The narrative collapse without this — the whole submission relies on a measurable improvement claim that we have not yet measured.

Everything else in this document is polish; those three are the difference between "we have the right architecture" and "we demonstrably won."
