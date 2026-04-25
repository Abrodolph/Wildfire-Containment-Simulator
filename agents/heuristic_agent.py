"""
Heuristic agent for the Wildfire Containment Simulator.

Uses a priority-based decision stack to select the best action each step:
1. EMERGENCY: Evacuate endangered crews
2. PROTECT POPULATION: Firebreak between fire and populated zones
3. AIR SUPPORT: Drop retardant on highest-intensity clusters
4. CONTAIN PERIMETER: Deploy/move crews to fire perimeter downwind
5. RECON: Reveal unknown regions (hard tier)
6. IDLE: Wait for situation to evolve
"""

from __future__ import annotations

import math
from typing import Optional

from env.models import (
    Action, ActionType, Observation, Direction,
    FireState, FuelType, IntensityBin, DIRECTION_DELTAS,
    CellObservation, CrewState, TankerState,
)


class HeuristicAgent:
    """
    Greedy heuristic agent that scores situations and picks the highest-value action.

    This is the primary baseline deliverable for the hackathon submission.
    """

    def __init__(self):
        self.step_count = 0

    def act(self, obs: Observation) -> Action:
        """Select the best action using the priority decision stack."""
        self.step_count += 1

        # ── Priority 0: DEPLOY — get undeployed crews onto the field first ──
        action = self._initial_deployment(obs)
        if action:
            return action

        # ── Priority 1: EMERGENCY — evacuate endangered crews ──
        action = self._check_crew_emergency(obs)
        if action:
            return action

        # ── Priority 2: PROTECT POPULATION — firebreak near populated zones ──
        action = self._protect_population(obs)
        if action:
            return action

        # ── Priority 3: AIR SUPPORT — retardant on hottest cluster ──
        action = self._air_support(obs)
        if action:
            return action

        # ── Priority 4: CONTAIN PERIMETER — deploy/move crews to fire edge ──
        action = self._contain_perimeter(obs)
        if action:
            return action

        # ── Priority 5: RECON — reveal unknown areas ──
        action = self._recon(obs)
        if action:
            return action

        # ── Priority 6: IDLE ──
        return Action(action_type=ActionType.IDLE, reason="No high-value action available")

    # ══════════════════════════════════════════════════
    # PRIORITY 0: INITIAL DEPLOYMENT
    # ══════════════════════════════════════════════════

    def _initial_deployment(self, obs: Observation) -> Optional[Action]:
        """Deploy undeployed crews to gain visibility and start working."""
        undeployed = [c for c in obs.resources.crews if c.is_active and not c.is_deployed]
        if not undeployed:
            return None

        burning = self._get_burning_cells_list(obs)
        tanker_ready = any(
            t.is_active and t.cooldown_remaining == 0
            for t in obs.resources.tankers
        )
        if burning and tanker_ready:
            return None
        if not burning and obs.resources.recon_budget > 0 and self._unknown_cell_count(obs) >= 20:
            return None

        crew = undeployed[0]
        rows = len(obs.grid)
        cols = len(obs.grid[0]) if rows > 0 else 0

        # Strategy: deploy near known fire, or spread across grid for visibility
        if burning:
            # Deploy near fire but not too close
            fr, fc = burning[0]
            deploy_r, deploy_c = self._find_safe_deploy_near(obs, fr, fc)
            if deploy_r is not None:
                return Action(
                    action_type=ActionType.DEPLOY_CREW,
                    crew_id=crew.crew_id,
                    target_row=deploy_r,
                    target_col=deploy_c,
                )

        # No visible fire (fog-of-war) — spread crews across grid quadrants
        crew_idx = 0
        for i, c in enumerate(obs.resources.crews):
            if c.crew_id == crew.crew_id:
                crew_idx = i
                break

        # Place in different quadrants
        quadrants = [
            (rows // 4, cols // 4),
            (rows // 4, 3 * cols // 4),
            (3 * rows // 4, cols // 4),
            (3 * rows // 4, 3 * cols // 4),
            (rows // 2, cols // 2),
            (rows // 2, cols // 4),
        ]
        target_r, target_c = quadrants[crew_idx % len(quadrants)]

        # Find safe cell near target
        deploy_r, deploy_c = self._find_safe_deploy_near(obs, target_r, target_c)
        if deploy_r is not None:
            return Action(
                action_type=ActionType.DEPLOY_CREW,
                crew_id=crew.crew_id,
                target_row=deploy_r,
                target_col=deploy_c,
            )

        # Fallback: deploy at grid center
        return Action(
            action_type=ActionType.DEPLOY_CREW,
            crew_id=crew.crew_id,
            target_row=rows // 2,
            target_col=cols // 2,
        )

    def _get_burning_cells_list(self, obs: Observation) -> list[tuple[int, int]]:
        """Get list of known burning cell coordinates."""
        return [
            (cell.row, cell.col)
            for row in obs.grid for cell in row
            if cell.fire_state in (FireState.BURNING, FireState.EMBER)
        ]

    # ══════════════════════════════════════════════════
    # PRIORITY 1: CREW EMERGENCY
    # ══════════════════════════════════════════════════

    def _check_crew_emergency(self, obs: Observation) -> Optional[Action]:
        """Move any crew that is adjacent to high-intensity fire."""
        for crew in obs.resources.crews:
            if not crew.is_active or not crew.is_deployed:
                continue

            # Check if crew's current cell or neighbors are dangerous
            danger = self._cell_danger(obs, crew.row, crew.col)
            if danger < 0.6:
                continue

            # Find safest adjacent direction to flee
            best_dir = None
            min_danger = danger
            for d in Direction:
                dr, dc = DIRECTION_DELTAS[d]
                nr, nc = crew.row + dr, crew.col + dc
                if not self._in_bounds(obs, nr, nc):
                    continue
                cell = obs.grid[nr][nc]
                if cell.fuel_type == FuelType.WATER:
                    continue
                if cell.fire_state in (FireState.BURNING, FireState.EMBER):
                    continue
                d_val = self._cell_danger(obs, nr, nc)
                if d_val < min_danger:
                    min_danger = d_val
                    best_dir = d

            if best_dir:
                return Action(
                    action_type=ActionType.MOVE_CREW,
                    crew_id=crew.crew_id,
                    direction=best_dir,
                )

        return None

    # ══════════════════════════════════════════════════
    # PRIORITY 2: PROTECT POPULATION
    # ══════════════════════════════════════════════════

    def _protect_population(self, obs: Observation) -> Optional[Action]:
        """Build firebreaks between fire and populated zones."""
        if obs.resources.firebreak_budget <= 0:
            return None

        # Find populated cells threatened by nearby fire
        threatened = []
        for row in obs.grid:
            for cell in row:
                if not cell.is_populated:
                    continue
                if cell.fire_state in (FireState.BURNED_OUT, FireState.BURNING):
                    continue
                # Check if fire is within 3 cells
                fire_dist = self._nearest_fire_distance(obs, cell.row, cell.col)
                if fire_dist is not None and fire_dist <= 5:
                    threatened.append((cell.row, cell.col, fire_dist))

        if not threatened:
            return None

        # Sort by closest fire
        threatened.sort(key=lambda x: x[2])
        target_r, target_c, _ = threatened[0]

        # Find a deployed crew that can build a firebreak toward the fire
        fire_dir = self._direction_toward_fire(obs, target_r, target_c)
        if fire_dir is None:
            return None

        # Find the crew closest to this populated cell
        best_crew = self._find_closest_crew(obs, target_r, target_c, deployed_only=True)

        if best_crew:
            crew = best_crew
            # If crew is adjacent to the target area, build firebreak
            dist_to_target = abs(crew.row - target_r) + abs(crew.col - target_c)
            if dist_to_target <= 2:
                # Try to build firebreak in the direction fire is coming from
                for d in self._prioritized_directions(obs, crew.row, crew.col, fire_dir):
                    dr, dc = DIRECTION_DELTAS[d]
                    nr, nc = crew.row + dr, crew.col + dc
                    if self._is_valid_firebreak(obs, nr, nc):
                        return Action(
                            action_type=ActionType.BUILD_FIREBREAK,
                            crew_id=crew.crew_id,
                            direction=d,
                        )

            # Otherwise move crew toward the threatened area
            move_dir = self._best_direction_toward(crew.row, crew.col, target_r, target_c, obs)
            if move_dir:
                return Action(
                    action_type=ActionType.MOVE_CREW,
                    crew_id=crew.crew_id,
                    direction=move_dir,
                )

        # Deploy an undeployed crew near the threatened population
        undeployed = self._find_closest_crew(obs, target_r, target_c, deployed_only=False, undeployed_only=True)
        if undeployed:
            # Deploy near the threatened cell (between fire and population)
            deploy_r, deploy_c = self._find_safe_deploy_near(obs, target_r, target_c)
            if deploy_r is not None:
                return Action(
                    action_type=ActionType.DEPLOY_CREW,
                    crew_id=undeployed.crew_id,
                    target_row=deploy_r,
                    target_col=deploy_c,
                )

        return None

    # ══════════════════════════════════════════════════
    # PRIORITY 3: AIR SUPPORT
    # ══════════════════════════════════════════════════

    def _air_support(self, obs: Observation) -> Optional[Action]:
        """Drop retardant on the highest-intensity fire cluster."""
        available_tankers = [
            t for t in obs.resources.tankers
            if t.is_active and t.cooldown_remaining == 0
        ]
        if not available_tankers:
            return None

        # Find highest-intensity burning cluster
        best_target = self._find_hottest_cluster(obs)
        if best_target is None:
            return None

        tr, tc = best_target
        # Check smoke density at target
        if obs.grid[tr][tc].smoke_density > 0.8:
            return None

        tanker = available_tankers[0]
        return Action(
            action_type=ActionType.DROP_RETARDANT,
            tanker_id=tanker.tanker_id,
            target_row=tr,
            target_col=tc,
        )

    # ══════════════════════════════════════════════════
    # PRIORITY 4: CONTAIN PERIMETER
    # ══════════════════════════════════════════════════

    def _contain_perimeter(self, obs: Observation) -> Optional[Action]:
        """Deploy or move crews to the fire perimeter, preferring the downwind side."""
        # Find fire perimeter cells (unburned cells adjacent to fire)
        perimeter = self._get_fire_perimeter_cells(obs)
        if not perimeter:
            return None

        # Score perimeter cells: higher score = more valuable to defend
        scored = []
        for r, c in perimeter:
            score = self._perimeter_cell_score(obs, r, c)
            scored.append((r, c, score))
        scored.sort(key=lambda x: -x[2])

        # Get all deployed active crews
        active_crews = [c for c in obs.resources.crews if c.is_active and c.is_deployed]
        if not active_crews:
            return None

        # Cycle through crews round-robin based on step count
        crew = active_crews[self.step_count % len(active_crews)]

        # Find the best perimeter cell for THIS crew
        best_target = None
        best_score = -1
        for target_r, target_c, score in scored[:10]:
            if obs.grid[target_r][target_c].crew_present:
                continue
            # Prefer targets close to this crew
            dist = abs(crew.row - target_r) + abs(crew.col - target_c)
            adjusted_score = score - dist * 0.3  # Penalize distant targets
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_target = (target_r, target_c)

        if best_target is None:
            return None

        target_r, target_c = best_target
        dist = abs(crew.row - target_r) + abs(crew.col - target_c)

        if dist <= 1:
            # Adjacent — build firebreak if possible
            if obs.resources.firebreak_budget > 0:
                for d in Direction:
                    dr, dc = DIRECTION_DELTAS[d]
                    nr, nc = crew.row + dr, crew.col + dc
                    if nr == target_r and nc == target_c and self._is_valid_firebreak(obs, nr, nc):
                        return Action(
                            action_type=ActionType.BUILD_FIREBREAK,
                            crew_id=crew.crew_id,
                            direction=d,
                        )

        # Move toward target
        move_dir = self._best_direction_toward(crew.row, crew.col, target_r, target_c, obs)
        if move_dir:
            return Action(
                action_type=ActionType.MOVE_CREW,
                crew_id=crew.crew_id,
                direction=move_dir,
            )

        return None

    # ══════════════════════════════════════════════════
    # PRIORITY 5: RECON
    # ══════════════════════════════════════════════════

    def _recon(self, obs: Observation) -> Optional[Action]:
        """Send recon flight over unknown areas. Conserve budget, space out usage."""
        if obs.resources.recon_budget <= 0:
            return None

        # Count unknown cells
        unknown_cells = []
        for row in obs.grid:
            for cell in row:
                if cell.fire_state == FireState.UNKNOWN:
                    unknown_cells.append((cell.row, cell.col))

        if len(unknown_cells) < 20:
            return None

        visible_fire = bool(self._get_burning_cells_list(obs))
        early_blind_recon = not visible_fire and self.step_count <= 3
        undeployed = [c for c in obs.resources.crews if c.is_active and not c.is_deployed]

        if not early_blind_recon:
            # Don't recon until all crews are deployed.
            if undeployed:
                return None

            # Only recon every ~30 steps to conserve budget.
            if self.step_count % 30 != 5:
                return None

        # Cluster unknown cells and pick a dense region
        # Simple approach: find the unknown cell farthest from any deployed crew
        crew_positions = [(c.row, c.col) for c in obs.resources.crews if c.is_active and c.is_deployed]
        if not crew_positions:
            rows = len(obs.grid)
            cols = len(obs.grid[0]) if rows > 0 else 0
            if rows >= 35 and cols >= 35:
                target = (rows // 4, cols // 4)
                if obs.resources.recon_budget < 3:
                    target = (rows // 2, 3 * cols // 4)
                return Action(
                    action_type=ActionType.RECON_FLIGHT,
                    target_row=target[0],
                    target_col=target[1],
                )
            best_cell = min(
                unknown_cells,
                key=lambda p: abs(p[0] - rows // 2) + abs(p[1] - cols // 2),
            )
            return Action(
                action_type=ActionType.RECON_FLIGHT,
                target_row=best_cell[0],
                target_col=best_cell[1],
            )

        best_cell = None
        max_min_dist = -1
        for ur, uc in unknown_cells:
            min_dist = min(abs(ur - cr) + abs(uc - cc) for cr, cc in crew_positions)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_cell = (ur, uc)

        if best_cell:
            return Action(
                action_type=ActionType.RECON_FLIGHT,
                target_row=best_cell[0],
                target_col=best_cell[1],
            )

        return None

    # ══════════════════════════════════════════════════
    # HELPER METHODS
    # ══════════════════════════════════════════════════

    def _unknown_cell_count(self, obs: Observation) -> int:
        return sum(
            1
            for row in obs.grid
            for cell in row
            if cell.fire_state == FireState.UNKNOWN
        )

    def _in_bounds(self, obs: Observation, r: int, c: int) -> bool:
        return 0 <= r < len(obs.grid) and 0 <= c < len(obs.grid[0])

    def _cell_danger(self, obs: Observation, r: int, c: int) -> float:
        """Compute danger level of a cell (0=safe, 1=deadly)."""
        if not self._in_bounds(obs, r, c):
            return 0.0

        cell = obs.grid[r][c]
        danger = 0.0

        # Direct fire
        if cell.fire_state == FireState.BURNING:
            intensity_vals = {
                IntensityBin.NONE: 0, IntensityBin.LOW: 0.3,
                IntensityBin.MEDIUM: 0.5, IntensityBin.HIGH: 0.7,
                IntensityBin.EXTREME: 0.95,
            }
            danger = max(danger, intensity_vals.get(cell.intensity_bin, 0.5))

        # Adjacent fire
        for d in Direction:
            dr, dc = DIRECTION_DELTAS[d]
            nr, nc = r + dr, c + dc
            if self._in_bounds(obs, nr, nc):
                n_cell = obs.grid[nr][nc]
                if n_cell.fire_state == FireState.BURNING:
                    intensity_vals = {
                        IntensityBin.NONE: 0, IntensityBin.LOW: 0.15,
                        IntensityBin.MEDIUM: 0.3, IntensityBin.HIGH: 0.5,
                        IntensityBin.EXTREME: 0.7,
                    }
                    danger = max(danger, intensity_vals.get(n_cell.intensity_bin, 0.3))

        return danger

    def _nearest_fire_distance(self, obs: Observation, r: int, c: int) -> Optional[int]:
        """Manhattan distance to nearest burning cell. None if no fire visible."""
        min_dist = None
        for row in obs.grid:
            for cell in row:
                if cell.fire_state in (FireState.BURNING, FireState.EMBER):
                    dist = abs(cell.row - r) + abs(cell.col - c)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
        return min_dist

    def _direction_toward_fire(self, obs: Observation, r: int, c: int) -> Optional[Direction]:
        """Find direction from (r,c) toward nearest fire."""
        closest = None
        min_dist = float("inf")
        for row in obs.grid:
            for cell in row:
                if cell.fire_state in (FireState.BURNING, FireState.EMBER):
                    dist = abs(cell.row - r) + abs(cell.col - c)
                    if dist < min_dist:
                        min_dist = dist
                        closest = (cell.row, cell.col)
        if closest is None:
            return None

        dr = closest[0] - r
        dc = closest[1] - c
        return self._delta_to_direction(dr, dc)

    def _delta_to_direction(self, dr: int, dc: int) -> Direction:
        """Convert row/col deltas to nearest Direction enum."""
        # Normalize to -1/0/1
        nr = 0 if dr == 0 else (1 if dr > 0 else -1)
        nc = 0 if dc == 0 else (1 if dc > 0 else -1)

        delta_map = {v: k for k, v in DIRECTION_DELTAS.items()}
        return delta_map.get((nr, nc), Direction.N)

    def _prioritized_directions(self, obs: Observation, r: int, c: int, primary: Direction) -> list[Direction]:
        """Return directions ordered by priority, starting with primary."""
        dirs = [primary]
        for d in Direction:
            if d != primary:
                dirs.append(d)
        return dirs

    def _find_closest_crew(
        self, obs: Observation, r: int, c: int,
        deployed_only: bool = False, undeployed_only: bool = False,
    ) -> Optional[CrewState]:
        """Find the closest active crew to position (r,c)."""
        best = None
        min_dist = float("inf")
        for crew in obs.resources.crews:
            if not crew.is_active:
                continue
            if deployed_only and not crew.is_deployed:
                continue
            if undeployed_only and crew.is_deployed:
                continue

            if crew.is_deployed:
                dist = abs(crew.row - r) + abs(crew.col - c)
            else:
                dist = 0  # Undeployed crews can deploy anywhere

            if dist < min_dist:
                min_dist = dist
                best = crew

        return best

    def _find_safe_deploy_near(self, obs: Observation, r: int, c: int) -> tuple[Optional[int], Optional[int]]:
        """Find a safe cell near (r,c) for crew deployment."""
        rows = len(obs.grid)
        cols = len(obs.grid[0]) if rows > 0 else 0

        # Search in expanding rings
        for radius in range(0, 6):
            candidates = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) + abs(dc) != radius and radius > 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        cell = obs.grid[nr][nc]
                        if (cell.fire_state in (FireState.UNBURNED, FireState.FIREBREAK, FireState.SUPPRESSED)
                                and cell.fuel_type != FuelType.WATER
                                and not cell.crew_present
                                and self._cell_danger(obs, nr, nc) < 0.5):
                            candidates.append((nr, nc))
            if candidates:
                # Pick the one closest to the fire (to be useful)
                candidates.sort(key=lambda p: self._nearest_fire_distance(obs, p[0], p[1]) or 999)
                return candidates[0]

        return None, None

    def _best_direction_toward(self, fr: int, fc: int, tr: int, tc: int, obs: Observation) -> Optional[Direction]:
        """Find the best safe direction to move from (fr,fc) toward (tr,tc)."""
        best_dir = None
        best_dist = abs(fr - tr) + abs(fc - tc)

        for d in Direction:
            dr, dc = DIRECTION_DELTAS[d]
            nr, nc = fr + dr, fc + dc
            if not self._in_bounds(obs, nr, nc):
                continue
            cell = obs.grid[nr][nc]
            if cell.fuel_type == FuelType.WATER:
                continue
            if cell.fire_state in (FireState.BURNING, FireState.EMBER):
                continue
            if self._cell_danger(obs, nr, nc) >= 0.6:
                continue

            dist = abs(nr - tr) + abs(nc - tc)
            if dist < best_dist:
                best_dist = dist
                best_dir = d

        return best_dir

    def _find_hottest_cluster(self, obs: Observation) -> Optional[tuple[int, int]]:
        """Find the burning cell with highest intensity, preferring clusters."""
        rows = len(obs.grid)
        cols = len(obs.grid[0]) if rows > 0 else 0

        best = None
        best_score = -1.0

        for row in obs.grid:
            for cell in row:
                if cell.fire_state != FireState.BURNING:
                    continue

                # Base score from intensity
                intensity_vals = {
                    IntensityBin.NONE: 0, IntensityBin.LOW: 0.25,
                    IntensityBin.MEDIUM: 0.5, IntensityBin.HIGH: 0.75,
                    IntensityBin.EXTREME: 1.0,
                }
                score = intensity_vals.get(cell.intensity_bin, 0.5)

                # Bonus for burning neighbors (cluster)
                for d in Direction:
                    dr, dc = DIRECTION_DELTAS[d]
                    nr, nc = cell.row + dr, cell.col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if obs.grid[nr][nc].fire_state == FireState.BURNING:
                            score += 0.1

                # Bonus for proximity to populated cells
                for d in Direction:
                    dr, dc = DIRECTION_DELTAS[d]
                    for dist in range(1, 4):
                        nr, nc = cell.row + dr * dist, cell.col + dc * dist
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if obs.grid[nr][nc].is_populated:
                                score += 0.5 / dist

                if score > best_score:
                    best_score = score
                    best = (cell.row, cell.col)

        return best

    def _get_fire_perimeter_cells(self, obs: Observation) -> list[tuple[int, int]]:
        """Get unburned cells adjacent to fire (the containment line)."""
        rows = len(obs.grid)
        cols = len(obs.grid[0]) if rows > 0 else 0
        perimeter = set()

        for row in obs.grid:
            for cell in row:
                if cell.fire_state not in (FireState.BURNING, FireState.EMBER):
                    continue
                for d in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = cell.row + d[0], cell.col + d[1]
                    if 0 <= nr < rows and 0 <= nc < cols:
                        n_cell = obs.grid[nr][nc]
                        if n_cell.fire_state == FireState.UNBURNED:
                            perimeter.add((nr, nc))

        return list(perimeter)

    def _perimeter_cell_score(self, obs: Observation, r: int, c: int) -> float:
        """Score a perimeter cell for defensive priority."""
        score = 1.0

        cell = obs.grid[r][c]

        # Heavily prioritize cells near populated areas
        pop_dist = self._nearest_populated_distance(obs, r, c)
        if pop_dist is not None:
            score += 5.0 / max(1, pop_dist)

        # Prioritize downwind cells (fire will spread toward them)
        wind_dir = obs.weather.wind_direction_deg
        wind_rad = math.radians(wind_dir + 180)  # Direction fire spreads
        fire_dist = self._nearest_fire_distance(obs, r, c)
        if fire_dist is not None and fire_dist <= 2:
            score += 2.0

        # Prioritize cells with high fuel load (will burn intensely if ignited)
        fuel_vals = {FuelType.GRASS: 0.5, FuelType.SHRUB: 0.7, FuelType.TIMBER: 1.0, FuelType.URBAN: 1.5}
        score += fuel_vals.get(cell.fuel_type, 0.3)

        return score

    def _nearest_populated_distance(self, obs: Observation, r: int, c: int) -> Optional[int]:
        """Manhattan distance to nearest populated cell."""
        min_dist = None
        for row in obs.grid:
            for cell in row:
                if cell.is_populated and cell.fire_state != FireState.BURNED_OUT:
                    dist = abs(cell.row - r) + abs(cell.col - c)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
        return min_dist

    def _is_valid_firebreak(self, obs: Observation, r: int, c: int) -> bool:
        """Check if cell is valid for firebreak construction."""
        if not self._in_bounds(obs, r, c):
            return False
        cell = obs.grid[r][c]
        return (cell.fire_state == FireState.UNBURNED
                and cell.fuel_type not in (FuelType.WATER, FuelType.URBAN))
