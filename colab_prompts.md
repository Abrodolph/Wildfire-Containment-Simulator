# Colab training prompts (feed to Claude, in order)

Each prompt is self-contained — paste as a fresh message with no prior context.

---

## Prompt 1 — SFT data generator script

```
Write a standalone Python script `scripts/generate_sft_data.py` for the Wildfire Containment Simulator project.

PURPOSE: Generate supervised fine-tuning (SFT) training examples by running the HeuristicAgent through episodes and recording (prompt, action) pairs at every step.

REPO STRUCTURE (files that exist):
- env/wildfire_env.py — WildfireEnv with reset(task_id, seed) and step(action)
- env/serialization.py — serialize_observation(obs, step_num, max_steps, tier="", prev_cells_burning=0) -> str
- agents/heuristic_agent.py — HeuristicAgent with act(obs) -> Action
- env/models.py — TIER_EASY(episode_length=80), TIER_MEDIUM(episode_length=150), TIER_HARD(episode_length=300)
- env/action_parser.py — parse_action(text, obs) -> (Action, status)

SYSTEM_PROMPT constant to use in every example:
"You are an AI Incident Commander managing wildfire containment. You will receive a situation briefing each step. Respond with ONLY a valid JSON action object and nothing else. Example: {\"action_type\": \"idle\"}"

REQUIREMENTS:
1. For each tier ("easy", "medium", "hard"), for each seed in a configurable range:
   a. Reset the env
   b. Run the heuristic for a random offset (0 to min(30, max_steps//4)) steps to get mid-episode states
   c. Run the heuristic to EPISODE COMPLETION (env.done == True), recording every step
   d. After the episode is complete, check env.state()["population_lost"] == 0. Only keep examples
      from successful episodes (pop_lost == 0 at end). Discard the whole episode otherwise.
   e. From the kept episodes, record every step as a training example EXCEPT: filter out IDLE
      actions unless they represent more than 30% of the episode's actions (keep a realistic idle rate).
      Concretely: keep all non-IDLE steps, then randomly sample IDLE steps to reach at most 20% of
      total examples per episode.
   f. Each example: {"messages": [{"role": "system", ...}, {"role": "user", "content": prompt_text}],
      "completion": action_json_string, "tier": tier, "seed": seed, "step": step_num}
   g. The "completion" field is the action serialised as compact JSON (action.model_dump_json(exclude_none=True))

2. Track prev_cells_burning across steps to pass to serialize_observation for spread delta.

3. Target counts after filtering: easy=2000 examples, medium=1500, hard=800.
   Iterate seeds starting from 0, incrementing by 1, until targets are met.

4. Save to training/sft_data.jsonl (one JSON object per line). Print progress every 50 seeds.
   Print final tier distribution before exiting.

5. Add argparse: --output (default training/sft_data.jsonl), --easy-seeds N (max seeds to try),
   --medium-seeds N, --hard-seeds N

IMPORTANT:
- The script runs locally, not in Colab. Use sys.path.insert(0, project_root) to make env/ importable.
- No GPU needed.
- Do NOT filter mid-episode observations — they are intentionally included for training diversity.
  The per-episode success filter (pop_lost==0) applies to the whole episode, not individual steps.
```

---

## Prompt 2 — SFT training notebook

```
Write a complete Google Colab notebook `training/sft_colab.ipynb` for supervised fine-tuning of
Qwen2.5-7B-Instruct on wildfire incident command data.

CONTEXT:
- Input: training/sft_data.jsonl, where each line has:
  {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}],
   "completion": "{\"action_type\":...}", "tier": "easy", "seed": 42, "step": 5}
- Goal: teach the model to output valid JSON action objects given wildfire observations
- Hardware target: A100 40GB on Colab (HF credits)

NOTEBOOK SECTIONS:

Section 1 — Install
- pip install: unsloth[colab-new] from git, trl==0.15.2, datasets==3.4.1
- assert torch.cuda.is_available(), print GPU name and total memory

Section 2 — Load Model
- unsloth FastLanguageModel.from_pretrained("unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048, load_in_4bit=True)
- FastLanguageModel.get_peft_model with r=32, lora_alpha=64, lora_dropout=0.05
- target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
- Use pad_token = eos_token if no pad token exists

Section 3 — Load Data
- Read sft_data.jsonl
- Format each example: apply tokenizer.apply_chat_template to the messages list, then append the
  completion string as the assistant turn. The final string is the full conversation for causal LM loss.
- Use datasets.Dataset.from_list
- Print tier distribution (counts per tier)
- Train/val split: 95/5

Section 4 — Train
- Use trl SFTTrainer with:
  - per_device_train_batch_size=2, gradient_accumulation_steps=4 (effective batch 8)
  - num_train_epochs=1
  - learning_rate=2e-4, warmup_ratio=0.05, lr_scheduler_type="cosine"
  - logging_steps=10, save_steps=100, save_total_limit=2
  - output_dir="./sft_checkpoints"
  - report_to="none"
  - max_seq_length=2048, packing=True

Section 5 — Quick Eval (runs in Colab, requires env imports)
- Add sys.path and import WildfireEnv, serialize_observation, parse_action
- Run 10 full episodes (seeds 42–51) on easy tier with the trained model driving EVERY step:
  - FastLanguageModel.for_inference(model)
  - For each step: build messages, apply_chat_template, model.generate(max_new_tokens=128),
    decode, parse_action(completion, obs), env.step(action)
  - Accumulate total_reward; track parse_status counts
- Print: mean reward, std, json_success_rate, mean pop_saved_pct
- assert mean_reward > 2.0, "SFT warm-up insufficient — do not proceed to GRPO"
- FastLanguageModel.for_training(model) before returning

Section 6 — Save
- model.save_pretrained("./sft_final")
- tokenizer.save_pretrained("./sft_final")
- model.push_to_hub("YOUR_HF_USERNAME/wildfire-sft-7b")  # leave as placeholder
- !zip -r sft_final.zip ./sft_final
- from google.colab import files; files.download("sft_final.zip")

IMPORTANT NOTES:
- parse_action(text, obs) requires a real obs object (it reads obs.grid). Always pass the current obs.
- serialize_observation signature: (obs, step_num, max_steps, tier="", prev_cells_burning=0)
- Instantiate a fresh HeuristicAgent (if used) for each episode — it has step_count state.
```

---

## Prompt 3 — GRPO training notebook

```
Write a complete Google Colab notebook `training/grpo_v2_colab.ipynb` for GRPO reinforcement
learning of a wildfire incident command model. This is a redesigned version that fixes five
critical issues from the previous attempt.

FIVE ISSUES FIXED IN THIS VERSION (do not reintroduce them):

Issue 1 — Prompt/reward state mismatch (critical):
  Previous: dataset used mid-episode prompts; reward_fn picked a random seed → model was scored
  in a completely different env state than the one that produced its prompt.
  Fix: Dataset uses step-0 prompts ONLY. Each row stores the seed used. The reward_fn resets the
  env to that exact (tier, seed) pair before scoring the completion. Prompt state = reward state.

Issue 2 — Truncated rollout reward incomparable to curriculum thresholds (critical):
  Previous: 15-step rollouts never reached min_active_steps=25, so terminal reward (+5.0) never
  fired. GRPO rewards capped at ~1-2 while thresholds were set to 7.0/5.5. Promotion never happened.
  Fix: The reward function runs the FULL episode to completion (model's 1 action at step 0, then
  heuristic until env.done). Terminal reward is always included. Reward is comparable to baselines.

Issue 3 — Wasted inner model generations:
  Previous: reward_fn called model.generate() 7 extra times per completion inside the reward loop.
  GRPO gradients only flow through the originally sampled completion, making inner model steps
  expensive noise with no gradient benefit.
  Fix: MODEL_STEPS = 1. Only the sampled completion is applied. Heuristic drives everything after.

Issue 4 — GRPO loop too slow:
  Consequence of Issue 3. Fix is same: MODEL_STEPS = 1 reduces reward_fn generate calls to 0.

Issue 5 — parse_action(text, None) crashes:
  The parser reads obs.grid at line 1. Cannot pass None.
  Fix: Use a standalone check_json_format(text) function in the format reward that does its own
  JSON validation without needing an obs.

CORRECT FULL-EPISODE BASELINES (from scripts/results.json):
  random:    easy=+6.23  medium=+1.31  hard=+2.16
  heuristic: easy=+7.53  medium=+6.31  hard=+4.74

STARTING POINT: SFT checkpoint at "YOUR_HF_USERNAME/wildfire-sft-7b" (or local sft_final.zip)

EXISTING ENV FILES (correct and working — do not reimplement):
- env/wildfire_env.py: WildfireEnv, reset(task_id, seed), step(action)->StepResult(observation,reward,done,info)
- env/serialization.py: serialize_observation(obs, step_num, max_steps, tier="", prev_cells_burning=0)->str
- env/action_parser.py: parse_action(text, obs)->(Action, status); status in ["json_success","regex_fallback","safe_idle"]
- agents/heuristic_agent.py: HeuristicAgent().act(obs)->Action  [stateful: re-instantiate per episode]
- env/curriculum.py: CurriculumController(start_tier, thresholds); after_episode(reward)->Optional[str]; get_tier()->str
- env/models.py: TIER_EASY(episode_length=80), TIER_MEDIUM(episode_length=150), TIER_HARD(episode_length=300)

NOTEBOOK SECTIONS:

Section 1 — Install and assert GPU
- pip install: unsloth[colab-new] from git, trl==0.15.2, datasets==3.4.1, wandb
- assert torch.cuda.is_available()
- print GPU name and total VRAM

Section 2 — Load SFT checkpoint
- FastLanguageModel.from_pretrained("YOUR_HF_USERNAME/wildfire-sft-7b", load_in_4bit=True, max_seq_length=2048)
  OR if loading from local zip: load base model first, then model.load_adapter(sft_path, adapter_name="default")
- Same LoRA: r=32, lora_alpha=64, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

Section 3 — Constants and controller setup

```python
import os, random, json
import torch
from env import WildfireEnv
from env.serialization import serialize_observation
from env.action_parser import parse_action
from agents.heuristic_agent import HeuristicAgent
from env.curriculum import CurriculumController
from datasets import Dataset

SEED_POOL = list(range(100))       # training seeds; eval uses 200+
TIER_MAX_STEPS = {'easy': 80, 'medium': 150, 'hard': 300}
SYSTEM_PROMPT = (
    'You are an AI Incident Commander managing wildfire containment. '
    'You will receive a situation briefing each step. '
    'Respond with ONLY a valid JSON action object and nothing else. '
    'Example: {"action_type": "idle"}'
)

# Thresholds calibrated to full-episode reward with heuristic continuation.
# Promote easy→medium once model's first action consistently beats random (+6.23).
# Promote medium→hard once model demonstrates meaningful improvement over random (+1.31).
controller = CurriculumController(
    start_tier='easy',
    thresholds={'easy': 6.5, 'medium': 3.5},
)

os.makedirs('training/samples', exist_ok=True)
_reward_call_count = 0
```

Section 4 — Standalone JSON format checker (replaces parse_action for format reward)

```python
import json as _json
from env.models import ActionType as _AT

_VALID_ACTION_TYPES = {a.value for a in _AT}

def check_json_format(text: str) -> str:
    """
    Validate LLM output format without needing an obs object.
    Returns "json_success", "regex_fallback", or "safe_idle".
    Does NOT use parse_action — avoids the obs.grid dependency.
    """
    # Strip code fences
    import re
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")
    start = text.find("{")
    if start == -1:
        return "safe_idle"
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return "safe_idle"
    try:
        obj = _json.loads(text[start:end+1])
        if not isinstance(obj, dict):
            return "safe_idle"
        at = str(obj.get("action_type", "")).lower()
        if at in _VALID_ACTION_TYPES:
            return "json_success"
        return "regex_fallback"   # valid JSON but unrecognised action_type
    except Exception:
        return "regex_fallback"   # JSON parse failed but had braces
```

Section 5 — Two reward functions

reward_fn_outcome(completions, prompts, tier=None, seed=None, **kwargs):
  """
  Score each GRPO completion by:
    1. Resetting the env to the EXACT (tier, seed) that generated the prompt (Issue 1 fix).
    2. Applying the sampled completion as the single first action (MODEL_STEPS=1, Issue 3/4 fix).
    3. Running HeuristicAgent until episode completion (Issue 2 fix — captures terminal reward).
  
  tier and seed are dataset columns forwarded by GRPOTrainer.
  """
  global _reward_call_count
  _reward_call_count += 1
  rewards = []

  for i, completion in enumerate(completions):
      ep_tier = tier[i] if tier is not None else controller.get_tier()
      ep_seed = seed[i] if seed is not None else random.choice(SEED_POOL)

      env = WildfireEnv()
      obs = env.reset(task_id=ep_tier, seed=ep_seed)   # step-0: matches prompt state exactly
      total_reward = 0.0

      # Apply the sampled completion as step 0
      text = completion if isinstance(completion, str) else completion[0]['content']
      action, _ = parse_action(text, obs)
      result = env.step(action)
      total_reward += result.reward
      obs = result.observation

      # Heuristic drives everything after (full episode to capture terminal reward)
      heuristic = HeuristicAgent()   # fresh instance per episode (stateful step_count)
      while not env.done:
          action = heuristic.act(obs)
          result = env.step(action)
          total_reward += result.reward
          obs = result.observation

      rewards.append(total_reward)

  # Update curriculum (once per batch, not per completion)
  mean_r = sum(rewards) / len(rewards)
  promoted = controller.after_episode(mean_r)
  if promoted:
      print(f"  *** Curriculum promoted to: {promoted} (mean batch reward={mean_r:.2f}) ***")

  # Sample completions to disk for inspection (Issue 4 in HACKATHON_ALIGNMENT.md)
  if _reward_call_count % 10 == 0:
      sample_path = f'training/samples/call_{_reward_call_count}.txt'
      with open(sample_path, 'w') as f:
          f.write(f"call={_reward_call_count}  tier={tier[0] if tier else '?'}  reward={rewards[0]:.3f}\n")
          f.write("---\n")
          c = completions[0]
          f.write(c if isinstance(c, str) else c[0]['content'])
          f.write("\n")

  return rewards


reward_fn_format(completions, prompts, **kwargs):
  """
  Scores JSON formatting quality using check_json_format() (no obs needed).
  Runs independently of the env — fast and always well-defined.
  """
  rewards = []
  for completion in completions:
      text = completion if isinstance(completion, str) else completion[0]['content']
      status = check_json_format(text)
      if status == "json_success":     r = 0.15
      elif status == "regex_fallback": r = 0.0
      else:                            r = -0.20   # safe_idle / garbage
      rewards.append(r)
  return rewards

Section 6 — Dataset builder (step-0 only; stores seed for reward alignment)

```python
def build_prompt_dataset(n=200):
    """
    Build step-0 prompts for the current curriculum tier.
    Stores the seed in each row so reward_fn can replay the exact same env state.
    No mid-episode offset — GRPO prompt and reward state are always step-0.
    Mid-episode diversity is handled by SFT, not GRPO.
    """
    rows = []
    env_tmp = WildfireEnv()
    tier = controller.get_tier()
    max_steps = TIER_MAX_STEPS[tier]

    for i in range(n):
        seed = SEED_POOL[i % len(SEED_POOL)]
        obs = env_tmp.reset(task_id=tier, seed=seed)   # step-0
        prompt = serialize_observation(obs, 0, max_steps, tier=tier, prev_cells_burning=0)
        rows.append({
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user',   'content': prompt},
            ],
            'tier': tier,
            'seed': seed,   # forwarded to reward_fn_outcome for exact state replay
        })
    return rows
```

Section 7 — CurriculumDatasetCallback

Implement a trl TrainerCallback subclass that rebuilds the training dataset whenever the
curriculum controller promotes to a new tier:

```python
from trl import TrainerCallback

class CurriculumDatasetCallback(TrainerCallback):
    def __init__(self, trainer_ref):
        self._trainer = trainer_ref
        self._last_tier = controller.get_tier()

    def on_step_end(self, args, state, control, **kwargs):
        current_tier = controller.get_tier()
        if current_tier != self._last_tier:
            print(f"  Rebuilding dataset for tier: {current_tier}")
            new_ds = Dataset.from_list(build_prompt_dataset(200))
            self._trainer.train_dataset = new_ds
            self._last_tier = current_tier
```

Section 8 — GRPOTrainer setup

```python
from trl import GRPOTrainer, GRPOConfig

grpo_config = GRPOConfig(
    output_dir="./grpo_checkpoints",
    num_generations=8,
    learning_rate=3e-6,
    max_steps=400,
    save_steps=20,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_completion_length=192,   # enough for any valid action JSON
    logging_steps=1,
    report_to="wandb",
)

FastLanguageModel.for_training(model)

dataset = Dataset.from_list(build_prompt_dataset(200))

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_fn_outcome, reward_fn_format],
    args=grpo_config,
    train_dataset=dataset,
)
trainer.add_callback(CurriculumDatasetCallback(trainer))
```

Section 9 — Run training

```python
import wandb
wandb.init(project="wildfire-grpo", name="qwen7b-v2")

print(f"Starting GRPO — {grpo_config.max_steps} steps, {grpo_config.num_generations} gen/prompt")
print(f"Reward: 1 model step at step-0, heuristic continuation to episode completion")
print(f"Start tier: {controller.get_tier()}")

trainer.train()
print("Training complete.")

history = controller.get_history()
stats = [{'step': ep, 'tier': t, 'mean_reward': r} for ep, t, r in history]
with open('./training_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Stats saved -> training_stats.json")
```

Section 10 — Evaluate vs baselines

- Load scripts/results.json for heuristic and random baseline scores
- For each tier in [easy, medium, hard], run 15 full episodes (seeds 42–56):
  - FastLanguageModel.for_inference(model)
  - Instantiate a FRESH LLMAgent per episode (it is stateful: _step, _prev_burning, parse counters)
  - Model drives every step until env.done
  - Record total_reward, pop_saved_pct, json_success_rate
- Print comparison table: Trained vs Heuristic vs Random, including vs_heuristic delta
- Print JSON success rate per tier
- assert: for at least 1 tier, trained_mean > heuristic_mean - 1.0

LLMAgent class to implement:
```python
class LLMAgent:
    def __init__(self, model, tokenizer, tier, max_steps):
        self.model = model
        self.tokenizer = tokenizer
        self.tier = tier
        self.max_steps = max_steps
        self._step = 0
        self._prev_burning = 0
        self.json_success = self.regex_fallback = self.safe_idle = 0

    def act(self, obs):
        prompt = serialize_observation(obs, self._step, self.max_steps,
                                       tier=self.tier,
                                       prev_cells_burning=self._prev_burning)
        self._prev_burning = obs.stats.cells_burning
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=128,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        action, status = parse_action(text, obs)
        if status == "json_success":      self.json_success += 1
        elif status == "regex_fallback":  self.regex_fallback += 1
        else:                             self.safe_idle += 1
        self._step += 1
        return action
```

Section 11 — Save and push

- model.save_pretrained("./grpo_final")
- tokenizer.save_pretrained("./grpo_final")
- model.push_to_hub("YOUR_HF_USERNAME/wildfire-grpo-7b")
- !zip -r grpo_final.zip ./grpo_final
- files.download("grpo_final.zip")

IMPLEMENTATION CHECKLIST:
[ ] reward_fn_outcome uses seed from dataset row, NOT random.choice(SEED_POOL)
[ ] reward_fn_outcome resets env with env.reset(task_id=ep_tier, seed=ep_seed) — step-0 only
[ ] reward_fn_outcome runs heuristic until env.done (not a fixed step count)
[ ] reward_fn_format calls check_json_format(), NOT parse_action(text, None)
[ ] build_prompt_dataset has no step offset — always step-0 — and always saves seed in the row
[ ] CurriculumDatasetCallback triggers dataset rebuild on tier change
[ ] LLMAgent instantiated FRESH per episode in the eval section
[ ] FastLanguageModel.for_inference/for_training toggled correctly around eval calls
[ ] WildfireEnv instantiated fresh per completion in reward_fn_outcome (not shared)
[ ] HeuristicAgent instantiated fresh per episode in reward_fn_outcome (it has step_count state)
```

---

## Prompt 4 — Evaluation and comparison script

```
Write a standalone Python script `scripts/eval_trained_model.py` that evaluates a trained HF
adapter model against the heuristic and random baselines on the Wildfire Containment Simulator.

PURPOSE: Source-of-truth comparison table after training is complete.
Saves results to scripts/trained_results.json.

INPUTS (argparse):
- --model-path: HF hub ID or local path to the trained adapter (e.g. "username/wildfire-grpo-7b")
- --base-model: base model (default "unsloth/Qwen2.5-7B-Instruct")
- --num-seeds: evaluation seeds per tier (default 15, uses seeds 200–214 to avoid train overlap)
- --tiers: space-separated list (default "easy medium hard")

EXISTING FILES:
- graders/grader_easy.py, grader_medium.py, grader_hard.py — grade(agent, seed) -> (float, details_dict)
- agents/heuristic_agent.py — HeuristicAgent
- agents/random_agent.py — RandomAgent(seed=N)
- scripts/results.json — existing baselines
- env/wildfire_env.py, env/serialization.py, env/action_parser.py

SYSTEM_PROMPT = (
    'You are an AI Incident Commander managing wildfire containment. '
    'You will receive a situation briefing each step. '
    'Respond with ONLY a valid JSON action object and nothing else. '
    'Example: {"action_type": "idle"}'
)

LLM AGENT CLASS (stateful — MUST be instantiated fresh per episode):
```python
class LLMAgent:
    """
    Wraps the trained model for grader compatibility.
    Must be re-instantiated for every episode — _step and _prev_burning
    are per-episode state and will produce wrong prompts if reused.
    """
    def __init__(self, model, tokenizer, tier, max_steps):
        self.model = model
        self.tokenizer = tokenizer
        self.tier = tier
        self.max_steps = max_steps
        self._step = 0
        self._prev_burning = 0
        self.json_success = self.regex_fallback = self.safe_idle = 0

    def act(self, obs):
        import torch
        prompt = serialize_observation(obs, self._step, self.max_steps,
                                       tier=self.tier,
                                       prev_cells_burning=self._prev_burning)
        self._prev_burning = obs.stats.cells_burning
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(input_ids, max_new_tokens=128,
                                      pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        action, status = parse_action(text, obs)
        if status == "json_success":      self.json_success += 1
        elif status == "regex_fallback":  self.regex_fallback += 1
        else:                             self.safe_idle += 1
        self._step += 1
        return action
```

GRADER WRAPPER (because graders pass agent to grade(), so agent is shared across seeds by default):
For LLMAgent, override this by not using grade() directly. Instead inline the grader logic and
instantiate a fresh LLMAgent(model, tokenizer, tier, max_steps) before EACH episode.

OUTPUT FORMAT:
```
=== Evaluation: Trained Model vs Baselines ===
Model:  username/wildfire-grpo-7b
Seeds:  200-214  (15 per tier)

Tier      Trained    Heuristic   Random     vs Heuristic
-------------------------------------------------------
easy      +7.21±0.3  +7.53±0.1  +6.23±3.1  -0.32
medium    +6.89±1.2  +6.31±2.8  +1.31±3.2  +0.58 ✓
hard      +4.12±2.1  +4.74±3.8  +2.16±3.0  -0.62

JSON success rate:  easy=91.2%  medium=88.4%  hard=85.1%
Pop saved rate:     easy=100%   medium=97%    hard=93%
```

Also save to scripts/trained_results.json in the same format as scripts/results.json, with an
additional "json_success_rate" field per tier.
```

---

