"""
Random agent baseline for the Wildfire Containment Simulator.

Selects random valid actions each step. Serves as the lower-bound
baseline for score comparison.
"""

from __future__ import annotations

import numpy as np

from env.models import (
    Action, ActionType, Observation, Direction,
    FireState, FuelType, DIRECTION_DELTAS,
)


class RandomAgent:
    """Agent that picks a random valid action each step."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Observation) -> Action:
        """Select a random valid action given the current observation."""
        # Collect available actions
        candidates: list[Action] = []

        # DEPLOY_CREW: deploy undeployed crews to safe cells
        for crew in obs.resources.crews:
            if crew.is_active and not crew.is_deployed:
                safe_cells = self._get_safe_cells(obs)
                if safe_cells:
                    r, c = safe_cells[self.rng.integers(0, len(safe_cells))]
                    candidates.append(Action(
                        action_type=ActionType.DEPLOY_CREW,
                        crew_id=crew.crew_id,
                        target_row=r, target_col=c,
                    ))

        # MOVE_CREW: move deployed crews in random direction
        for crew in obs.resources.crews:
            if crew.is_active and crew.is_deployed:
                valid_dirs = self._get_valid_move_dirs(obs, crew.row, crew.col)
                if valid_dirs:
                    d = valid_dirs[self.rng.integers(0, len(valid_dirs))]
                    candidates.append(Action(
                        action_type=ActionType.MOVE_CREW,
                        crew_id=crew.crew_id,
                        direction=d,
                    ))

        # DROP_RETARDANT: drop on burning area
        for tanker in obs.resources.tankers:
            if tanker.is_active and tanker.cooldown_remaining == 0:
                burning = self._get_burning_cells(obs)
                if burning:
                    r, c = burning[self.rng.integers(0, len(burning))]
                    candidates.append(Action(
                        action_type=ActionType.DROP_RETARDANT,
                        tanker_id=tanker.tanker_id,
                        target_row=r, target_col=c,
                    ))

        # BUILD_FIREBREAK: if crew deployed and budget available
        if obs.resources.firebreak_budget > 0:
            for crew in obs.resources.crews:
                if crew.is_active and crew.is_deployed:
                    dirs = list(Direction)
                    self.rng.shuffle(dirs)
                    for d in dirs:
                        dr, dc = DIRECTION_DELTAS[d]
                        nr, nc = crew.row + dr, crew.col + dc
                        if self._is_valid_firebreak_target(obs, nr, nc):
                            candidates.append(Action(
                                action_type=ActionType.BUILD_FIREBREAK,
                                crew_id=crew.crew_id,
                                direction=d,
                            ))
                            break

        # IDLE: always available
        candidates.append(Action(
            action_type=ActionType.IDLE,
            reason="Random agent waiting",
        ))

        # Pick random candidate
        idx = self.rng.integers(0, len(candidates))
        return candidates[idx]

    def _get_safe_cells(self, obs: Observation) -> list[tuple[int, int]]:
        """Get cells that are safe to deploy a crew to."""
        safe = []
        for row in obs.grid:
            for cell in row:
                if (cell.fire_state in (FireState.UNBURNED, FireState.FIREBREAK, FireState.SUPPRESSED)
                        and cell.fuel_type not in (FuelType.WATER,)
                        and not cell.crew_present):
                    safe.append((cell.row, cell.col))
        # Sample a subset to avoid huge lists
        if len(safe) > 20:
            indices = self.rng.choice(len(safe), 20, replace=False)
            safe = [safe[i] for i in indices]
        return safe

    def _get_valid_move_dirs(self, obs: Observation, row: int, col: int) -> list[Direction]:
        """Get directions a crew can move from (row, col)."""
        valid = []
        rows = len(obs.grid)
        cols = len(obs.grid[0]) if rows > 0 else 0
        for d in Direction:
            dr, dc = DIRECTION_DELTAS[d]
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cell = obs.grid[nr][nc]
                if (cell.fuel_type != FuelType.WATER
                        and cell.fire_state not in (FireState.UNKNOWN,)):
                    valid.append(d)
        return valid

    def _get_burning_cells(self, obs: Observation) -> list[tuple[int, int]]:
        """Get cells that are currently burning."""
        burning = []
        for row in obs.grid:
            for cell in row:
                if cell.fire_state == FireState.BURNING:
                    burning.append((cell.row, cell.col))
        return burning

    def _is_valid_firebreak_target(self, obs: Observation, row: int, col: int) -> bool:
        """Check if a cell is valid for firebreak construction."""
        rows = len(obs.grid)
        cols = len(obs.grid[0]) if rows > 0 else 0
        if not (0 <= row < rows and 0 <= col < cols):
            return False
        cell = obs.grid[row][col]
        return (cell.fire_state == FireState.UNBURNED
                and cell.fuel_type not in (FuelType.WATER, FuelType.URBAN))
