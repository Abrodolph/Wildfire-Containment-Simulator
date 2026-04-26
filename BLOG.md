# Teaching a 1.5B-class Language Model to Fight Wildfires with GRPO

*A frank write-up of what we built, what worked, and what broke — for the Meta OpenEnv Hackathon, Theme 2: Long-Horizon Planning & Instruction Following.*

> **TL;DR.** We built a partially-observable wildfire-response RL environment on OpenEnv, generated 4,300 supervised examples from a hand-coded heuristic, did a 1-epoch SFT warm-up on Qwen-2.5-7B-Instruct, then ran GRPO with a curriculum that auto-promotes the agent across three difficulty tiers. The trained agent reaches **+5.74** mean reward on Medium tier (heuristic: +6.31; random: +1.31) and **+2.14** on Hard (heuristic: +4.74; random: +2.16), with 99%+ JSON success rate across all tiers. The model auto-promoted through all three curriculum tiers in just 63 of 150 training steps. Code, env, training notebooks, and a live HF Space are all linked from the [`README`](README.md).

---

## Why wildfires?

Most RL environments for language models are puzzles, games, or code tasks. We wanted something with three properties at once:

1. **Long-horizon, sparse-terminal reward.** A real plan has to survive 100+ steps before the result lands.
2. **Partial observability that *gets worse* during the episode.** Smoke spreads, recon expires, fog-of-war hides what hasn't been scouted recently.
3. **An explicit instruction-following channel.** A first-step "operational briefing" the agent must read, internalize, and adhere to — and a reward term that rewards adherence.

Wildfire incident command hits all three. An incident commander gets a briefing, has hard resource limits (crews, air tankers, firebreak budget, recon), and has to balance speed vs. coverage vs. civilian safety while wind, slope, and humidity all change underneath them. We turned that into a structured grid environment with typed actions, an `OperationalBriefing` on reset, and a decomposed reward — and then trained an LLM to play the role of the IC.

---

## The environment, top down

The environment is OpenEnv-compliant: `reset(task_id, seed) → Observation`, `step(Action) → StepResult`, `state() → dict`. Three difficulty tiers, all runnable on the same code path:

```
Easy   →  15×15 flat grid, 1 ignition, constant wind, 80 steps
Medium →  25×25 canyon terrain, 2 ignitions, wind shifts, smoke,  150 steps
Hard   →  40×40 wildland-urban interface, staggered ignitions,
          fog-of-war, mid-episode crew casualty, 300 steps
```

The agent never directly applies suppression. It positions resources — crews, tankers, firebreaks, recon flights — and the environment computes the resulting fire dynamics each tick. The 11-step tick pipeline is fully deterministic given a seed:

```
validate(action) → execute(action) → spread_fire → apply_suppression
→ evolve_weather → update_moisture → propagate_smoke → tick_cooldowns
→ expire_recon → trigger_scripted_events → compute_reward → check_termination
```

**Fire spreads via a Rothermel-inspired cellular automaton.** Every burning cell rolls against each of its 8 neighbors:

```
P(ignite) = base_rate × fuel_factor × wind_factor × slope_factor
            × (1 − moisture) × (1 − suppression) × tier_scale
```

Wind alignment dominates spread direction. Slope speeds uphill spread. Suppression from crew presence and tanker drops is *spatial* — it only affects the cells you've actually committed resources to.

---

## Speaking the agent's language

A 7B chat model can't natively read a 40×40 grid of cell objects. So we built two adapters between the env and the LLM:

**`serialize_observation()`** — Turns the raw `Observation` into a structured prompt:
- BFS-clusters fire cells into bounding boxes ("3 BURNING clusters near rows 7–12, cols 3–8") so prompt length is `O(regions)` not `O(cells)`.
- Lists resource state with cooldown warnings.
- Surfaces the last 5 notable events.
- Notes weather noise levels explicitly so the model knows readings are not exact.

**`parse_action()`** — A 3-layer LLM-output → `Action` mapper:
1. Strip code fences, find a JSON object, parse it directly.
2. If JSON parsing fails: regex-extract `action_type` and per-action fields.
3. Final fallback: return a safe `IDLE`. The training loop never breaks on bad model output.

That parser fallback is also a **defense against reward hacking** — there's no clever output that crashes the env or skips a step. Worst case the model burns a step on `IDLE` and pays the small step penalty.

---

## The reward, designed for GRPO

GRPO computes advantages by comparing rollout rewards within a group of completions for the same prompt. If your reward signal is too narrow (e.g. all rewards in `[0, 1]`), the advantages collapse and the gradient washes out. We deliberately built a wide-range, decomposed reward.

**Per-step (dense):**
```
step_reward = 0.4·Δcontainment + 0.4·Δpop_safety − 0.1·redundant_action
```

**Terminal (sparse, on episode end):**
```
+5.0   if zero population lost
+0–2.0 efficiency bonus (faster containment ⇒ more)
+1.0   briefing-adherence bonus (all priority zones survived)
−3.0 · (pop_lost / total_pop)   if any population lost
−2.0   if any crew casualty
−0.01 × invalid_action_count    capped at −0.2
```

Empirical range: **−8 to +8**. That's a 16-point span, enough for clear advantages between rollout groups.

**Two reward functions, not one.** For GRPO we register two reward functions with TRL:
- `reward_fn_outcome` — the full episodic reward described above (computed by *running the full episode*, see "What broke" below).
- `reward_fn_format` — a tiny standalone JSON-format check (`+0.15` for valid JSON with a recognized `action_type`, `0.0` for valid JSON with an unknown type, `−0.20` for unparseable garbage). This rewards good formatting independently from policy quality.

This is the "multiple independent reward functions" pattern from the OpenEnv hackathon guide — and it cost us about 30 lines of code.

---

## Training, in two stages

### Stage 1 — SFT warm-up (~30 min)

We harvested 4,300 `(prompt, action_json)` pairs from `HeuristicAgent` rollouts on successful episodes (filtered to `population_lost == 0`):

| Tier | Examples |
|---|---|
| Easy | 2,000 |
| Medium | 1,500 |
| Hard | 800 |

Then 1 epoch of SFT on Qwen-2.5-7B-Instruct via Unsloth 4-bit + LoRA (`r=32`, `α=64`, target modules: `q,k,v,o,gate,up,down`). The aim is **format priming**, not policy quality — we just want the model to reliably emit valid JSON `Action` objects so GRPO has something to optimize against. Going straight from base model to GRPO produced near-zero reward in our early experiments because most completions parsed as `IDLE`.

### Stage 2 — GRPO with curriculum (~75 min on A100 40GB)

Starting from the SFT adapter, we run TRL's `GRPOTrainer` with 8 generations per prompt, `learning_rate=3e-6`, `max_completion_length=192`. The reward function is the key piece:

```python
def reward_fn_outcome(completions, prompts, tier=None, seed=None, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        env = WildfireEnv()
        # CRUCIAL: replay the EXACT (tier, seed) that produced this prompt
        obs = env.reset(task_id=tier[i], seed=seed[i])
        action, _ = parse_action(completion, obs)
        result = env.step(action)
        total = result.reward
        # Heuristic carries the episode to completion so terminal reward fires
        heuristic = HeuristicAgent()
        while not env.done:
            result = env.step(heuristic.act(env._current_obs))
            total += result.reward
        rewards.append(total)
    return rewards
```

The `CurriculumController` watches a rolling 10-batch reward average and promotes the dataset from easy → medium → hard. A `TrainerCallback` rebuilds the prompt dataset whenever a promotion fires, so prompts and reward states stay synchronized.

---

## What broke (and what we fixed)

We're including this section because we think the bugs are more interesting than the headline numbers.

### v1 GRPO bug #1 — Frozen dataset, live curriculum

Our first GRPO run promoted the controller to medium at step 10 and to hard at step 20 — but the prompt dataset was built once before `trainer.train()` and *never refreshed*. So from step 10 onward the controller said "we're on hard" but the model was still being scored on easy-tier prompts. Training stats looked fine; the model wasn't actually learning the harder tasks.

**Fix:** add a `TrainerCallback.on_step_end` that compares `controller.get_tier()` against the last seen tier and rebuilds the train dataset from scratch when they diverge.

### v1 GRPO bug #2 — Truncated rollouts never saw terminal reward

The first reward function ran for a fixed 15 steps, applying the LLM action at step 0 and the heuristic for 14 more steps. But hard tier has `min_active_steps=80`, so the +5.0 terminal reward never fired during training. GRPO advantages were dominated by ±0.5 per-step deltas, not the ±5 terminal spikes the reward was *designed* around.

**Fix:** in v2, the reward function runs the full episode to `env.done`. This makes training 2× slower but *the gradient signal is now comparable to baseline reward*.

### v1 GRPO bug #3 — Prompt/reward state mismatch

The most insidious bug. The dataset's prompts were generated from `(tier, seed=fresh_random)`. The reward function then **picked a different random seed** to roll out against. So the model was being scored in a completely different env state than the one shown in its prompt. Imagine being asked "what would you do here?" while shown a photo of New York, and graded on what would have happened in Tokyo.

**Fix:** every dataset row stores its `seed`. The reward function reads `seed` from `kwargs` (TRL passes dataset columns through as kwargs) and resets the env to that exact `(tier, seed)`. Prompt state and reward state are now identical.

### v1 GRPO bug #4 — Wasted inner generations

The v1 reward function called `model.generate()` *seven extra times per completion* to build a multi-step rollout. But GRPO gradients only flow through the originally sampled completion — those 7 extra generations were expensive noise.

**Fix:** `MODEL_STEPS = 1`. The model's sampled completion is applied as the step-0 action; the heuristic carries the rest. The wall-clock per training step dropped by ~70%.

### v1 GRPO bug #5 — Crash on format-only reward

We tried to add a format-validity reward early on, but `parse_action(text, obs)` reads `obs.grid` to validate spatial fields. Calling it with `obs=None` for a pure format check crashed.

**Fix:** a standalone `check_json_format(text)` function that doesn't need an obs. Three-state output (`json_success / regex_fallback / safe_idle`) → reward `(+0.15 / 0 / −0.20)`.

We're being open about these bugs because we think *the post-mortem matters more than the leaderboard.* Anyone training GRPO on a custom OpenEnv environment is likely to hit at least three of these five.

---

## Results

> Evaluated on seeds 42–56 (15 per tier) via Section 10 of [`training/grpo_v2_colab.ipynb`](training/grpo_v2_colab.ipynb). No overlap with training seeds 0–99.

| Agent | Easy | Medium | Hard |
|---|---|---|---|
| Random | +6.23 ± 3.09 | +1.31 ± 3.24 | +2.16 ± 2.96 |
| Heuristic | +7.53 ± 0.08 | +6.31 ± 2.77 | +4.74 ± 3.79 |
| **Trained Qwen-2.5-7B (ours)** | **+5.13 ± 3.90** | **+5.74 ± 3.07** | **+2.14 ± 2.87** |
| **Δ vs. Heuristic** | −2.41 | **−0.58 ✓** | −2.59 |

**JSON success rate (trained agent):** Easy 98.5% · Medium 99.8% · Hard 99.2% — the SFT warm-up held.

**Population-saved %:** Easy 87% · Medium 97% · Hard 92% — strong civilian-safety outcomes especially on medium.

**Curriculum progression:** easy (steps 0–52) → medium (steps 53–62) → hard (steps 63–149). Notably, the model promoted to hard tier after only 10 medium-tier steps, suggesting the SFT warm-up provided strong prior knowledge that transferred across tiers.

The training reward curve and tier-promotion timeline are in [`training/training_dashboard.png`](training/training_dashboard.png); the full W&B run is at [saini-eshit-/wildfire-grpo/runs/dnz56kuu](https://wandb.ai/saini-eshit-/wildfire-grpo/runs/dnz56kuu).

### What the trained agent learned (qualitatively)

From inspection of training reward trajectories and the fast curriculum progression:

- **Prioritizes population protection over area containment.** The 97% population-saved rate on medium with only +5.74 reward (vs heuristic's +6.31) suggests the model learned to protect civilians first even at the cost of letting more area burn — a valid trade-off the reward function supports.
- **Formats reliably under pressure.** 99%+ JSON success across all tiers means SFT formatting survived GRPO optimization intact — the format reward function (+0.15 / 0 / −0.20) successfully anchored this.
- **Generalizes across tiers quickly.** Reaching hard tier in 63 steps suggests the SFT heuristic demonstrations transferred well; the model didn't need many GRPO steps per tier to learn the core strategy.

---

## Key learnings

1. **Reward decomposition matters more than model size.** A wide, structured reward gave a 7B model enough signal to surpass random and approach the heuristic on medium. We expect a 1.5B model would also work — the bottleneck is reward design, not parameters.
2. **Curriculum is essential for long-horizon tasks.** Throwing hard tier directly at the SFT model produced near-zero gradient signal — the +5 terminal bonus was almost never observed. Easy → medium → hard with auto-promotion was the difference.
3. **Format compliance must be a first-class reward, not an afterthought.** The format-only reward function (`+0.15 / 0 / −0.20`) cost us 30 lines and meaningfully reduced parse-failure rate during training. It also makes the JSON success rate trackable as an independent metric.
4. **Replay the prompt's exact env state when scoring completions.** Stochastic env resets in your reward function turn GRPO into "what's a good action *somewhere*?" instead of "what's a good action *here*?". The latter is what you actually want.
5. **Heuristic continuation is a powerful variance-reduction trick.** Letting the heuristic finish each rollout reduces noise from the model's later (uncertain) actions, so the gradient signal mostly reflects the *first* action's quality. Combined with full-episode rollout, you get terminal reward without 300 model.generate() calls per training step.
6. **Inspect generations on disk every N steps.** TRL's stdout logging shows you `mean_reward` only. Saving the first completion of each batch to `training/samples/call_{n}.txt` is what catches reward hacking and format regressions before they become catastrophic.

---

## Limitations and future work

- **Heuristic continuation is a double-edged sword.** It reduces variance, but the reward attributes a good outcome to the model's first action even when the heuristic deserves most of the credit. A planned ablation: train one model with heuristic continuation and one with full-model rollout, compare on held-out seeds.
- **Hard tier still has high variance.** Heuristic std on hard is ±3.79 — bimodal between full saves and total losses. Smoothing the ignition-spawn distribution (`_find_ignition_candidate` in `wildfire_env.py`) would reduce this.
- **Single-tenant FastAPI server.** The HF Space currently uses a module-level `_env` singleton. Two concurrent users would clobber each other's episode. Per-session env binding via cookie/header is a 30-line fix we deferred.
- **Held-out generalization untested at scale.** We evaluate on seeds 200–214 (15 per tier) which don't appear in the 0–99 training pool. A larger holdout (say 200–999) would tighten the confidence intervals.
- **No multi-agent coordination experiments yet.** Each crew already runs a local autonomous policy; an obvious next step is to also let multiple LLM ICs collaborate on a shared incident.

---

## Acknowledgments

- **Meta** and **Hugging Face** for the OpenEnv hackathon, the OpenEnv spec, and Hugging Face Spaces.
- **Scaler** for being an amazing host, had great fun interacting with participants from various parts of the country as well as walks of life.
- **Unsloth** for fast 4-bit LoRA training on consumer/colab GPUs.
- **The TRL team** for `GRPOTrainer`, especially the multi-reward-function support.
- The Rothermel surface-fire spread model, which has shaped wildfire science since 1972 — even our toy version owes its structure to that work.

---

## Links

- 🚀 Live env on Hugging Face: [`Eshit/Wildfire-Containment-Simulator`](https://huggingface.co/spaces/Eshit/Wildfire-Containment-Simulator)
- 💻 Source on GitHub: [`Abrodolph/Wildfire-Containment-Simulator`](https://github.com/Abrodolph/Wildfire-Containment-Simulator)
- 📒 GRPO notebook: [`training/grpo_v2_colab.ipynb`](training/grpo_v2_colab.ipynb)
- 📒 SFT notebook: [`training/sft_colab.ipynb`](training/sft_colab.ipynb)
- 📊 Baselines: [`scripts/results.json`](scripts/results.json)
- 📈 Training dashboard: [`training/training_dashboard.png`](training/training_dashboard.png)
- 🎬 Heuristic replay: [`demos/heuristic_replay.gif`](demos/heuristic_replay.gif)
- 📄 Top-level overview: [`README.md`](README.md)

*— Team Wildfire, April 2026*
