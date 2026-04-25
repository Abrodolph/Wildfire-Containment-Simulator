"""
Robust LLM output → Action parser with 3-layer fallback.

Layer 1: Direct JSON parse
Layer 2: Regex field extraction
Layer 3: Safe IDLE fallback
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Tuple

from .models import Action, ActionType, Direction

if TYPE_CHECKING:
    from .models import Observation

_SAFE_IDLE = Action(action_type=ActionType.IDLE, reason="parse_failure")

_ACTION_TYPES = {a.value for a in ActionType}
_DIRECTIONS = {d.value for d in Direction}


def parse_action(llm_output: str, obs: "Observation") -> Tuple[Action, str]:
    """
    Convert raw LLM text into a validated Action.

    Returns (action, status) where status is one of:
      "json_success", "regex_fallback", "safe_idle"
    """
    grid_rows = len(obs.grid)
    grid_cols = len(obs.grid[0]) if grid_rows > 0 else 0

    # Layer 1 — direct JSON
    action, status = _try_json(llm_output)
    if action is not None:
        action = _bounds_check(action, grid_rows, grid_cols)
        return action, status

    # Layer 2 — regex
    action, status = _try_regex(llm_output)
    if action is not None:
        action = _bounds_check(action, grid_rows, grid_cols)
        return action, status

    # Layer 3 — safe fallback
    return _SAFE_IDLE, "safe_idle"


# ── Layer 1 ──────────────────────────────────────────────────

def _try_json(text: str) -> Tuple[Action | None, str]:
    raw = _extract_json_block(text)
    if raw is None:
        return None, "safe_idle"
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None, "safe_idle"
        # Normalise action_type casing
        if "action_type" in data:
            data["action_type"] = str(data["action_type"]).lower()
        if data.get("action_type") not in _ACTION_TYPES:
            return None, "safe_idle"
        action = Action(**data)
        return action, "json_success"
    except Exception:
        return None, "safe_idle"


def _extract_json_block(text: str) -> str | None:
    """Find first balanced {...} block, stripping ```json fences."""
    # Strip code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ── Layer 2 ──────────────────────────────────────────────────

def _try_regex(text: str) -> Tuple[Action | None, str]:
    # action_type
    at_match = re.search(
        r'action_type["\s:]+["\']?(' + "|".join(_ACTION_TYPES) + r")[\"']?",
        text,
        re.IGNORECASE,
    )
    if not at_match:
        return None, "safe_idle"

    action_type = at_match.group(1).lower()

    def _str(pattern: str) -> str | None:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1) if m else None

    def _int(pattern: str) -> int | None:
        m = re.search(pattern, text, re.IGNORECASE)
        return int(m.group(1)) if m else None

    crew_id = _str(r'crew_id["\s:]+["\']?(crew_\d+)["\']?')
    tanker_id = _str(r'tanker_id["\s:]+["\']?(tanker_\d+)["\']?')
    target_row = _int(r'target_row["\s:]+(\d+)')
    target_col = _int(r'target_col["\s:]+(\d+)')
    direction_raw = _str(
        r'direction["\s:]+["\']?(' + "|".join(_DIRECTIONS) + r")[\"']?"
    )
    direction = direction_raw.upper() if direction_raw else None

    try:
        action = Action(
            action_type=action_type,
            crew_id=crew_id,
            tanker_id=tanker_id,
            target_row=target_row,
            target_col=target_col,
            direction=direction,
        )
        return action, "regex_fallback"
    except Exception:
        return None, "safe_idle"


# ── Bounds check ─────────────────────────────────────────────

def _bounds_check(action: Action, grid_rows: int, grid_cols: int) -> Action:
    """Downgrade to IDLE if target coords are outside the grid."""
    row, col = action.target_row, action.target_col
    if row is None and col is None:
        return action
    if row is None or col is None:
        return _SAFE_IDLE
    if not (0 <= row < grid_rows and 0 <= col < grid_cols):
        return _SAFE_IDLE
    return action
