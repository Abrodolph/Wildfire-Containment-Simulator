"""
Tests for the new server routes: /ui/, root redirect, /state/render, /auto_step.

Run with:  pytest tests/test_server_routes.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Ensure the project root is importable (mirrors server/app.py sys.path setup)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.app import app

client = TestClient(app, follow_redirects=False)


# ── /  redirect ───────────────────────────────────────────────────────────────

def test_root_redirects_to_ui():
    r = client.get("/")
    assert r.status_code in (307, 308), f"Expected redirect, got {r.status_code}"
    assert r.headers.get("location", "").startswith("/ui")


# ── /ui/  static serving ──────────────────────────────────────────────────────

def test_ui_serves_html():
    r = TestClient(app, follow_redirects=True).get("/ui/")
    # If frontend/ dir exists the page should be served; if not, we get 404
    # (acceptable in CI if frontend/ hasn't been built yet)
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert "text/html" in r.headers.get("content-type", "")


# ── /health ───────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


# ── /state/render  before reset ───────────────────────────────────────────────

def test_state_render_before_reset_returns_400():
    # Force uninitialised state
    from server.app import _env
    _env.grid = None
    _env._current_obs = None

    r = client.get("/state/render")
    assert r.status_code == 400


# ── /state/render  after reset ────────────────────────────────────────────────

def test_state_render_after_reset():
    client.post("/reset?task_id=easy&seed=42")
    r = client.get("/state/render")
    assert r.status_code == 200

    data = r.json()
    assert "grid" in data
    assert "weather" in data
    assert "resources" in data

    # Easy tier = 15×15
    assert len(data["grid"]) == 15
    assert len(data["grid"][0]) == 15

    # Each cell has the expected fields
    cell = data["grid"][0][0]
    for field in ("row", "col", "fire_state", "fire_intensity", "fuel_type",
                  "is_populated", "crew_present"):
        assert field in cell, f"Missing field '{field}' in render cell"


# ── /auto_step  without prior reset ──────────────────────────────────────────

def test_auto_step_without_reset_returns_400():
    import sys
    smod = sys.modules["server.app"]
    smod._env._current_obs = None
    smod._active_agent = None

    r = client.post("/auto_step?n=1&agent=heuristic")
    assert r.status_code == 400


# ── /auto_step  heuristic ────────────────────────────────────────────────────

def test_auto_step_heuristic():
    client.post("/reset?task_id=easy&seed=42")
    r = client.post("/auto_step?n=3&agent=heuristic")
    assert r.status_code == 200

    data = r.json()
    assert "steps" in data
    assert "done" in data
    assert len(data["steps"]) <= 3

    for snap in data["steps"]:
        assert "observation" in snap
        assert "reward" in snap
        assert "done" in snap
        assert "info" in snap
        assert "action_taken" in snap


# ── /auto_step  random ───────────────────────────────────────────────────────

def test_auto_step_random():
    client.post("/reset?task_id=easy&seed=0")
    r = client.post("/auto_step?n=1&agent=random")
    assert r.status_code == 200
    data = r.json()
    assert len(data["steps"]) >= 1


# ── /auto_step  agent persists across calls ───────────────────────────────────

def test_auto_step_agent_persists():
    """
    Calling /auto_step n=1 twice should not recreate the agent,
    so the heuristic's internal step_count must increment correctly.
    """
    import sys
    smod = sys.modules["server.app"]

    client.post("/reset?task_id=easy&seed=42")
    assert smod._active_agent is None  # cleared by /reset

    client.post("/auto_step?n=1&agent=heuristic")
    agent_after_first = smod._active_agent
    assert agent_after_first is not None

    client.post("/auto_step?n=1&agent=heuristic")
    agent_after_second = smod._active_agent
    # Same instance (not re-created)
    assert agent_after_first is agent_after_second


# ── /reset  clears active agent ──────────────────────────────────────────────

def test_reset_clears_active_agent():
    import sys
    smod = sys.modules["server.app"]

    client.post("/reset?task_id=easy&seed=42")
    client.post("/auto_step?n=1&agent=heuristic")
    assert smod._active_agent is not None

    client.post("/reset?task_id=easy&seed=42")
    assert smod._active_agent is None


# ── /reset returns Observation shape ─────────────────────────────────────────

def test_reset_returns_observation_not_step_result():
    r = client.post("/reset?task_id=easy&seed=42")
    assert r.status_code == 200
    data = r.json()

    # Must be an Observation: has grid, weather, resources, stats
    for field in ("grid", "weather", "resources", "stats"):
        assert field in data, f"Expected Observation field '{field}' missing"

    # Must NOT be wrapped in StepResult
    assert "observation" not in data
    assert "reward" not in data


# ── /step returns StepResult shape ───────────────────────────────────────────

def test_step_returns_step_result():
    client.post("/reset?task_id=easy&seed=42")
    action = {"action_type": "idle", "reason": "test"}
    r = client.post("/step", json=action)
    assert r.status_code == 200
    data = r.json()

    for field in ("observation", "reward", "done", "info"):
        assert field in data, f"Expected StepResult field '{field}' missing"

    # Observation is nested
    assert "grid" in data["observation"]
