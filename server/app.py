"""
Wildfire Containment Simulator — FastAPI Server (server/app.py)
===============================================================
OpenEnv multi-mode deployment entry point.
Serves the environment over HTTP on port 7860 for HuggingFace Spaces.

New in v2:
  - Serves the interactive frontend at /ui/ (StaticFiles)
  - GET  /state/render  — lightweight canvas-ready snapshot (respects ground-truth)
  - POST /auto_step     — runs N steps with a built-in agent (module-level instance)
  - Module-level _active_agent resets alongside _env on /reset
"""

import os
import sys

# Ensure project root is on the path so `env` and `agents` packages are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from env import WildfireEnv, Action
from agents import HeuristicAgent, RandomAgent

# ── Frontend static directory (relative to this file, not cwd) ──────────────
_FRONTEND_DIR = os.path.join(_PROJECT_ROOT, "frontend")

app = FastAPI(
    title="Wildfire Containment Simulator",
    description=(
        "OpenEnv x Scaler Hackathon | Sponsored by Meta & HuggingFace. "
        "An RL environment where an AI agent dispatches firefighting resources "
        "to contain a wildfire before it reaches populated zones."
    ),
    version="2.0.0",
)

# ── Optional CORS for local development only ─────────────────────────────────
# Set DEV_CORS=1 in your shell when running the server locally with a separate
# dev server (e.g. Live Server on port 5500).  Never set in production.
if os.getenv("DEV_CORS"):
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# ── Module-level singletons ───────────────────────────────────────────────────
_env = WildfireEnv()
_active_agent: Optional[HeuristicAgent | RandomAgent] = None


# ── Frontend static files ─────────────────────────────────────────────────────
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=_FRONTEND_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the interactive frontend."""
    return RedirectResponse(url="/ui/")


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "wildfire-containment-simulator", "version": "2.0.0"}


# ── Core environment endpoints ────────────────────────────────────────────────

@app.post("/reset")
def reset(task_id: str = "easy", seed: int = 42):
    """
    Reset the environment.

    Returns: Observation (directly — not wrapped in StepResult).
    task_id: easy | medium | hard
    """
    global _active_agent
    _active_agent = None  # Clear agent so it is recreated fresh for the new episode
    try:
        obs = _env.reset(task_id=task_id, seed=seed)
        return obs.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(action: Action):
    """
    Execute one simulation step.

    Returns: StepResult { observation, reward, done, info }
    """
    try:
        result = _env.step(action)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def state():
    """Full ground-truth state for grading (bypasses fog-of-war)."""
    return _env.state()


# ── New: lightweight render snapshot ─────────────────────────────────────────

@app.get("/state/render")
def state_render():
    """
    Trimmed ground-truth snapshot for the 'Ground Truth' canvas overlay.

    Only exposes the fields the frontend canvas needs. Bypasses fog-of-war —
    use only for the debug overlay, never as the primary canvas source.
    """
    if _env.grid is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    raw = _env.state()
    grid = raw["grid"]

    return {
        "grid": [
            [
                {
                    "row": cell["row"],
                    "col": cell["col"],
                    "fire_state": cell["fire_state"],
                    "fire_intensity": cell.get("fire_intensity", 0.0),
                    "fuel_type": cell.get("fuel_type", "grass"),
                    "is_populated": cell.get("is_populated", False),
                    "crew_present": cell.get("crew_present", False),
                }
                for cell in row
            ]
            for row in grid
        ],
        "resources": raw.get("resources", {}),
        "weather": raw.get("weather", {}),
        "stats": {
            "current_step": raw.get("current_step", 0),
            "cells_burned": raw.get("cells_burned", 0),
            "population_lost": raw.get("population_lost", 0),
            "total_population": raw.get("total_population", 0),
        },
    }


# ── New: auto-step with built-in agent ───────────────────────────────────────

class StepSnapshot(BaseModel):
    """One step's worth of data returned by /auto_step."""
    observation: dict
    reward: float
    done: bool
    info: dict
    action_taken: dict


@app.post("/auto_step")
def auto_step(n: int = 1, agent: str = "heuristic"):
    """
    Run N simulation steps using a built-in agent.

    The agent instance is kept module-level so its internal step_count and
    state survive across consecutive n=1 calls.  The agent is reset (set to
    None) whenever /reset is called.

    agent: "heuristic" | "random"
    n: number of steps to execute (capped at episode_length to prevent abuse)
    """
    global _active_agent

    if _env._current_obs is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    # Cap n to remaining steps
    max_n = max(1, _env.config.episode_length - _env.current_step)
    n = min(n, max_n, 50)  # hard cap at 50 per request

    # Create agent if needed (preserves state across calls)
    if _active_agent is None:
        if agent == "random":
            _active_agent = RandomAgent()
        else:
            _active_agent = HeuristicAgent()

    snapshots: list[dict] = []
    try:
        for _ in range(n):
            if _env.done:
                break
            obs = _env._current_obs
            action = _active_agent.act(obs)
            result = _env.step(action)
            snapshots.append(StepSnapshot(
                observation=result.observation.model_dump(),
                reward=result.reward,
                done=result.done,
                info=result.info,
                action_taken=action.model_dump(),
            ).model_dump())
            if result.done:
                break
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    done = snapshots[-1]["done"] if snapshots else _env.done
    return {"steps": snapshots, "done": done}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Entry point for [project.scripts] serve command."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
