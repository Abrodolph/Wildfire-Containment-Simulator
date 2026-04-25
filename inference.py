"""
Wildfire Containment Simulator — Inference Script
===================================================
Runs an LLM agent (via OpenAI-compatible client) against all three task tiers
and emits structured [START] / [STEP] / [END] logs for automated evaluation.

Required environment variables:
    API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace / API key

Optional:
    TASK_NAME      Run a single task: easy | medium | hard  (default: all three)
"""

import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from env import WildfireEnv, Action, ActionType
from env.models import Observation

# ── Environment variables ──────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

TASKS              = ["easy", "medium", "hard"]
SEED               = 42
SUCCESS_THRESHOLD  = 0.5
TEMPERATURE        = 0.2
MAX_TOKENS         = 120

# ── Structured log helpers ─────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=wildfire-containment-simulator model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation → LLM prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI wildfire incident commander. Each step issue exactly ONE action as JSON.

    Action types and required fields:
      deploy_crew    : {"action_type":"deploy_crew","crew_id":"crew_N","target_row":R,"target_col":C}
      move_crew      : {"action_type":"move_crew","crew_id":"crew_N","direction":"N|S|E|W|NE|NW|SE|SW"}
      drop_retardant : {"action_type":"drop_retardant","tanker_id":"tanker_N","target_row":R,"target_col":C}
      build_firebreak: {"action_type":"build_firebreak","crew_id":"crew_N","direction":"N|S|E|W|NE|NW|SE|SW"}
      recon_flight   : {"action_type":"recon_flight","target_row":R,"target_col":C}
      idle           : {"action_type":"idle","reason":"..."}

    Strategy:
    - DEPLOY undeployed crews first (deploy_crew) before any other crew action.
    - MOVE crews toward fire to suppress it.
    - BUILD firebreaks between fire and populated zones.
    - DROP retardant on high-intensity clusters near populated cells.
    - Output ONLY raw JSON. No explanation, no markdown, no code fences.
""").strip()


def build_user_prompt(obs: Observation, step: int, history: List[str]) -> str:
    stats   = obs.stats
    weather = obs.weather
    res     = obs.resources

    burning = [
        f"({cell.row},{cell.col},{cell.intensity_bin.value})"
        for row in obs.grid for cell in row
        if cell.fire_state.value in ("burning", "ember")
    ][:12]

    populated_safe = [
        f"({cell.row},{cell.col})"
        for row in obs.grid for cell in row
        if cell.is_populated and cell.fire_state.value not in ("burned_out", "burning")
    ][:8]

    crews   = [f"{c.crew_id}@({c.row},{c.col}) deployed={c.is_deployed} active={c.is_active}"
               for c in res.crews]
    tankers = [f"{t.tanker_id} cooldown={t.cooldown_remaining} active={t.is_active}"
               for t in res.tankers]

    history_block = "\n".join(history[-4:]) if history else "none"

    return textwrap.dedent(f"""
        Step {step} / {stats.max_steps}
        Fire: {stats.cells_burning} burning, {stats.cells_burned} burned out
        Population lost: {stats.population_lost} | Containment: {stats.containment_pct:.1f}%
        Weather: {weather.wind_speed_kmh:.0f} km/h @ {weather.wind_direction_deg:.0f}° | humidity {weather.humidity_pct:.0f}% | rain={weather.rain_active}

        Burning cells (row,col,intensity): {burning}
        Safe populated cells: {populated_safe}

        Crews:   {crews}
        Tankers: {tankers}
        Firebreak budget: {res.firebreak_budget} | Recon budget: {res.recon_budget}

        Recent events: {obs.recent_events}
        Last actions:
        {history_block}

        Output your next action as JSON:
    """).strip()


# ── LLM → Action ──────────────────────────────────────────────────────────────

def _compact_action(action: Action) -> str:
    """Short human-readable string for [STEP] log."""
    at = action.action_type.value
    if at == "deploy_crew":
        return f"deploy_crew({action.crew_id},{action.target_row},{action.target_col})"
    if at == "move_crew":
        return f"move_crew({action.crew_id},{action.direction.value})"
    if at == "drop_retardant":
        return f"drop_retardant({action.tanker_id},{action.target_row},{action.target_col})"
    if at == "build_firebreak":
        return f"build_firebreak({action.crew_id},{action.direction.value})"
    if at == "recon_flight":
        return f"recon_flight({action.target_row},{action.target_col})"
    return f"idle({action.reason or ''})"


def get_llm_action(
    client: OpenAI,
    obs: Observation,
    step: int,
    history: List[str],
) -> tuple[Action, str, Optional[str]]:
    """Call LLM, parse JSON action. Falls back to IDLE on any failure."""
    user_prompt = build_user_prompt(obs, step, history)
    error: Optional[str] = None

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        data   = json.loads(raw)
        action = Action(**data)
        return action, _compact_action(action), None

    except Exception as exc:
        error = str(exc)[:80]
        idle  = Action(action_type=ActionType.IDLE, reason="llm_parse_error")
        return idle, "idle(llm_parse_error)", error


# ── Single-task episode ────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str, seed: int) -> float:
    """Run one full episode and return the final score in [0, 1]."""
    env = WildfireEnv()
    obs = env.reset(task_id=task_id, seed=seed)

    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_id, model=MODEL_NAME)

    try:
        step = 0
        while not env.done:
            step += 1
            action, action_str, error = get_llm_action(client, obs, step, history)

            result      = env.step(action)
            obs         = result.observation
            reward      = result.reward
            done        = result.done
            steps_taken = step

            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:.2f}")

        # Score = final composite reward (consistent with graders)
        score   = rewards[-1] if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)[:120]
        print(f"[DEBUG] Episode error: {error_msg}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_override = os.getenv("TASK_NAME")
    tasks         = [task_override] if task_override else TASKS

    results = {}
    for task_id in tasks:
        results[task_id] = run_task(client, task_id, seed=SEED)

    # Final summary line (not part of scored format, helpful for debugging)
    summary = " | ".join(f"{t}={s:.3f}" for t, s in results.items())
    print(f"\n[SUMMARY] {summary}", flush=True)


if __name__ == "__main__":
    main()
