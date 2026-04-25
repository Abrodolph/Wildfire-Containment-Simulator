"""
Resource management for the Wildfire Containment Simulator.

Manages ground crews, air tankers, and firebreak construction.
Each resource type has distinct mechanics and constraints.
"""

from __future__ import annotations

from .models import (
    CrewState, TankerState, ResourceState, Direction,
    DIRECTION_DELTAS, FireState, FuelType, TierConfig,
)
from .grid import Grid


class ResourceManager:
    """
    Manages all firefighting resources: crews, tankers, firebreak budget.

    Handles deployment, movement, suppression, retardant drops,
    and firebreak construction with full constraint checking.
    """

    def __init__(self, config: TierConfig, grid: Grid):
        self.config = config
        self.grid = grid

        # Initialize crews
        self.crews: list[CrewState] = []
        for i in range(config.num_crews):
            self.crews.append(CrewState(
                crew_id=f"crew_{i}",
                row=0, col=0,
                is_deployed=False,
                is_active=True,
            ))

        # Initialize tankers
        self.tankers: list[TankerState] = []
        for i in range(config.num_tankers):
            self.tankers.append(TankerState(
                tanker_id=f"tanker_{i}",
                cooldown_remaining=0,
                is_active=True,
            ))

        self.firebreak_budget = config.firebreak_budget
        self.recon_budget = config.recon_budget

        # Track revealed cells from recon flights (step -> set of cells)
        self.revealed_cells: set[tuple[int, int]] = set()
        self.reveal_expiry: dict[tuple[int, int], int] = {}  # cell -> step when it expires

        # Stats
        self.total_retardant_drops = 0
        self.total_firebreaks_built = 0
        self.wasted_actions = 0
        self.idle_crew_steps = 0
        self.crew_casualties = False

        # Multi-agent crew state
        self._crew_objectives: dict[str, str] = {}  # crew_id -> objective string
        self._ic_ordered_this_step: set[str] = set()  # crews that got an IC order this step
        self.autonomous_saves: int = 0  # times local policy retreat prevented a casualty

    def reset(self) -> None:
        """Reset all resources to initial state."""
        for crew in self.crews:
            crew.is_deployed = False
            crew.is_active = True
            crew.row = 0
            crew.col = 0
        for tanker in self.tankers:
            tanker.cooldown_remaining = 0
            tanker.is_active = True
        self.firebreak_budget = self.config.firebreak_budget
        self.recon_budget = self.config.recon_budget
        self.revealed_cells.clear()
        self.reveal_expiry.clear()
        self.total_retardant_drops = 0
        self.total_firebreaks_built = 0
        self.wasted_actions = 0
        self.idle_crew_steps = 0
        self.crew_casualties = False
        self._crew_objectives = {}
        self._ic_ordered_this_step = set()
        self.autonomous_saves = 0

    # ─── Crew Operations ─────────────────────────────

    def deploy_crew(self, crew_id: str, row: int, col: int) -> tuple[bool, str]:
        """Deploy a crew to a target cell. Returns (success, message)."""
        crew = self._get_crew(crew_id)
        if crew is None:
            return False, f"Crew {crew_id} not found"
        if not crew.is_active:
            return False, f"Crew {crew_id} is inactive (lost)"
        if crew.is_deployed:
            return False, f"Crew {crew_id} already deployed. Use MOVE_CREW instead."

        if not self.grid._in_bounds(row, col):
            return False, f"Target ({row},{col}) out of bounds"

        static = self.grid.static_grid[row][col]
        dynamic = self.grid.dynamic_grid[row][col]

        if static.fuel_type == FuelType.WATER:
            return False, f"Cannot deploy crew to water cell ({row},{col})"
        if dynamic.fire_intensity > 0.7:
            return False, f"Cell ({row},{col}) too dangerous (intensity {dynamic.fire_intensity:.2f})"

        # Clear old position if any
        if crew.is_deployed:
            self.grid.dynamic_grid[crew.row][crew.col].crew_present = False

        crew.row = row
        crew.col = col
        crew.is_deployed = True
        self.grid.dynamic_grid[row][col].crew_present = True

        return True, f"Crew {crew_id} deployed to ({row},{col})"

    def move_crew(self, crew_id: str, direction: Direction) -> tuple[bool, str]:
        """Move a deployed crew one cell in the given direction."""
        crew = self._get_crew(crew_id)
        if crew is None:
            return False, f"Crew {crew_id} not found"
        if not crew.is_active:
            return False, f"Crew {crew_id} is inactive"
        if not crew.is_deployed:
            return False, f"Crew {crew_id} not deployed. Use DEPLOY_CREW first."

        dr, dc = DIRECTION_DELTAS[direction]
        nr, nc = crew.row + dr, crew.col + dc

        if not self.grid._in_bounds(nr, nc):
            return False, f"Cannot move {crew_id} {direction.value}: out of bounds"

        static = self.grid.static_grid[nr][nc]
        dynamic = self.grid.dynamic_grid[nr][nc]

        if static.fuel_type == FuelType.WATER:
            return False, f"Cannot move to water cell ({nr},{nc})"
        if dynamic.fire_intensity > 0.7:
            return False, f"Cell ({nr},{nc}) too dangerous (intensity {dynamic.fire_intensity:.2f})"

        # Move
        self.grid.dynamic_grid[crew.row][crew.col].crew_present = False
        crew.row = nr
        crew.col = nc
        self.grid.dynamic_grid[nr][nc].crew_present = True

        return True, f"Crew {crew_id} moved {direction.value} to ({nr},{nc})"

    def apply_suppression(self) -> list[str]:
        """
        All deployed, active crews suppress fire at their current cell.
        Called each tick. Returns event messages.
        """
        events = []
        for crew in self.crews:
            if not crew.is_active or not crew.is_deployed:
                if crew.is_active and not crew.is_deployed:
                    self.idle_crew_steps += 1
                continue

            dyn = self.grid.dynamic_grid[crew.row][crew.col]

            # Check for crew casualty (fire intensity spiked around them)
            if dyn.fire_intensity > 0.85:
                crew.is_active = False
                self.crew_casualties = True
                self.grid.dynamic_grid[crew.row][crew.col].crew_present = False
                events.append(f"CREW CASUALTY: {crew.crew_id} trapped at ({crew.row},{crew.col})!")
                continue

            if dyn.fire_state in (FireState.BURNING, FireState.EMBER):
                # Suppress: reduce intensity
                dyn.suppression_level = min(1.0, dyn.suppression_level + 0.15)
                dyn.fire_intensity = max(0.0, dyn.fire_intensity - 0.15)

                if dyn.fire_intensity <= 0.0:
                    dyn.fire_state = FireState.SUPPRESSED
                    events.append(f"Crew {crew.crew_id} suppressed fire at ({crew.row},{crew.col})")

        return events

    # ─── Multi-Agent Crew Local Policy ───────────────

    def set_crew_objective(self, crew_id: str, objective: str) -> tuple[bool, str]:
        """IC sets a high-level objective for a crew. Persists until changed."""
        crew = self._get_crew(crew_id)
        if crew is None:
            return False, f"Crew {crew_id} not found"
        if not crew.is_active:
            return False, f"Crew {crew_id} is inactive"
        self._crew_objectives[crew_id] = objective
        self._ic_ordered_this_step.add(crew_id)
        return True, f"Crew {crew_id} assigned objective: {objective}"

    def clear_ic_orders(self) -> None:
        """Call at the start of each step to reset per-step IC tracking."""
        self._ic_ordered_this_step = set()

    def get_crew_local_obs(self, crew_id: str) -> dict:
        """Return 3x3 neighbourhood view centred on the crew's position."""
        crew = self._get_crew(crew_id)
        if crew is None or not crew.is_deployed:
            return {}
        cells = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = crew.row + dr, crew.col + dc
                if not self.grid._in_bounds(r, c):
                    cells.append({"row": r, "col": c, "fire_state": "out_of_bounds",
                                  "intensity": 0.0, "smoke": 0.0})
                else:
                    dyn = self.grid.dynamic_grid[r][c]
                    cells.append({
                        "row": r, "col": c,
                        "fire_state": dyn.fire_state.value,
                        "intensity": round(dyn.fire_intensity, 3),
                        "smoke": round(dyn.smoke_density, 3),
                    })
        return {
            "crew_id": crew_id,
            "position": (crew.row, crew.col),
            "health": "active" if crew.is_active else "casualty",
            "neighborhood": cells,
            "objective": self._crew_objectives.get(crew_id, "none"),
        }

    def apply_local_policies(self) -> list[str]:
        """
        Run each deployed crew's local policy for crews NOT ordered by IC this step.
        Called after fire spread, before suppression.
        """
        events = []
        for crew in self.crews:
            if not crew.is_active or not crew.is_deployed:
                continue
            if crew.crew_id in self._ic_ordered_this_step:
                continue

            dyn = self.grid.dynamic_grid[crew.row][crew.col]
            objective = self._crew_objectives.get(crew.crew_id, "advance")

            if objective == "hold":
                continue

            if dyn.fire_intensity > 0.8:
                # Retreat: move away from fire centre
                direction = self._retreat_direction(crew)
                if direction is not None:
                    old_intensity = dyn.fire_intensity
                    ok, msg = self.move_crew(crew.crew_id, direction)
                    if ok:
                        new_dyn = self.grid.dynamic_grid[crew.row][crew.col]
                        if new_dyn.fire_intensity < old_intensity:
                            self.autonomous_saves += 1
                            events.append(
                                f"AUTO-RETREAT: {crew.crew_id} retreated {direction.value} "
                                f"(intensity was {old_intensity:.2f})"
                            )
                        else:
                            events.append(f"AUTO-RETREAT: {crew.crew_id} moved {direction.value}")
            else:
                # Advance toward nearest fire in 3x3, biased by objective
                direction = self._advance_direction(crew, objective)
                if direction is not None:
                    ok, msg = self.move_crew(crew.crew_id, direction)
                    if ok:
                        events.append(f"AUTO-ADVANCE: {crew.crew_id} moved {direction.value}")

        return events

    def _retreat_direction(self, crew) -> "Direction | None":
        """Find the safest direction to retreat from current position."""
        best_dir = None
        best_score = -1.0
        for direction, (dr, dc) in DIRECTION_DELTAS.items():
            nr, nc = crew.row + dr, crew.col + dc
            if not self.grid._in_bounds(nr, nc):
                continue
            static = self.grid.static_grid[nr][nc]
            if static.fuel_type == FuelType.WATER:
                continue
            dyn = self.grid.dynamic_grid[nr][nc]
            # Prefer low intensity, not burning
            score = 1.0 - dyn.fire_intensity
            if dyn.fire_state in (FireState.BURNING, FireState.EMBER):
                score -= 0.5
            if score > best_score:
                best_score = score
                best_dir = direction
        return best_dir

    def _advance_direction(self, crew, objective: str) -> "Direction | None":
        """Find direction toward nearest fire, biased by objective."""
        # Look in 3x3 for fire targets
        targets = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                r, c = crew.row + dr, crew.col + dc
                if not self.grid._in_bounds(r, c):
                    continue
                dyn = self.grid.dynamic_grid[r][c]
                if dyn.fire_state in (FireState.BURNING, FireState.EMBER):
                    targets.append((dr, dc))

        if not targets:
            return None

        # Pick target, with direction bias from objective
        bias = {
            "prioritize_north": (-1, 0),
            "prioritize_south": (1, 0),
            "prioritize_east": (0, 1),
            "prioritize_west": (0, -1),
        }.get(objective, (0, 0))

        best = min(targets, key=lambda t: abs(t[0] - bias[0]) + abs(t[1] - bias[1]))
        # Find Direction matching (dr, dc)
        for direction, delta in DIRECTION_DELTAS.items():
            if delta == best:
                nr, nc = crew.row + best[0], crew.col + best[1]
                if self.grid._in_bounds(nr, nc):
                    static = self.grid.static_grid[nr][nc]
                    if static.fuel_type != FuelType.WATER:
                        return direction
        return None

    # ─── Tanker Operations ────────────────────────────

    def drop_retardant(self, tanker_id: str, center_row: int, center_col: int) -> tuple[bool, str]:
        """Drop retardant on a 3x3 area centered on (center_row, center_col)."""
        tanker = self._get_tanker(tanker_id)
        if tanker is None:
            return False, f"Tanker {tanker_id} not found"
        if not tanker.is_active:
            return False, f"Tanker {tanker_id} inactive"
        if tanker.cooldown_remaining > 0:
            return False, f"Tanker {tanker_id} on cooldown ({tanker.cooldown_remaining} steps)"

        if not self.grid._in_bounds(center_row, center_col):
            return False, f"Target ({center_row},{center_col}) out of bounds"

        # Check smoke density at target
        if self.grid.dynamic_grid[center_row][center_col].smoke_density > 0.8:
            return False, f"Smoke too dense at ({center_row},{center_col}) for tanker drop"

        # Apply retardant to 3x3 area
        affected = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = center_row + dr, center_col + dc
                if not self.grid._in_bounds(r, c):
                    continue
                dyn = self.grid.dynamic_grid[r][c]
                dyn.fire_intensity = max(0.0, dyn.fire_intensity - 0.4)
                dyn.moisture = min(1.0, dyn.moisture + 0.2)
                dyn.suppression_level = min(1.0, dyn.suppression_level + 0.3)

                if dyn.fire_state in (FireState.BURNING, FireState.EMBER) and dyn.fire_intensity <= 0.0:
                    dyn.fire_state = FireState.SUPPRESSED
                affected += 1

        tanker.cooldown_remaining = self.config.tanker_cooldown
        self.total_retardant_drops += 1

        return True, f"Tanker {tanker_id} dropped retardant at ({center_row},{center_col}), {affected} cells affected"

    def tick_tanker_cooldowns(self) -> None:
        """Reduce cooldown timers by 1 each step."""
        for tanker in self.tankers:
            if tanker.cooldown_remaining > 0:
                tanker.cooldown_remaining -= 1

    # ─── Firebreak Operations ─────────────────────────

    def build_firebreak(self, crew_id: str, direction: Direction) -> tuple[bool, str]:
        """Build a firebreak in the cell adjacent to the crew in the given direction."""
        crew = self._get_crew(crew_id)
        if crew is None:
            return False, f"Crew {crew_id} not found"
        if not crew.is_active or not crew.is_deployed:
            return False, f"Crew {crew_id} not deployed/active"
        if self.firebreak_budget <= 0:
            return False, "No firebreak budget remaining"

        dr, dc = DIRECTION_DELTAS[direction]
        tr, tc = crew.row + dr, crew.col + dc

        if not self.grid._in_bounds(tr, tc):
            return False, f"Target ({tr},{tc}) out of bounds"

        static = self.grid.static_grid[tr][tc]
        dynamic = self.grid.dynamic_grid[tr][tc]

        if static.fuel_type in (FuelType.WATER, FuelType.URBAN):
            return False, f"Cannot build firebreak on {static.fuel_type.value} cell"
        if dynamic.fire_state != FireState.UNBURNED:
            return False, f"Cell ({tr},{tc}) is not UNBURNED (state: {dynamic.fire_state.value})"

        dynamic.fire_state = FireState.FIREBREAK
        dynamic.fire_intensity = 0.0
        self.firebreak_budget -= 1
        self.total_firebreaks_built += 1

        return True, f"Firebreak built at ({tr},{tc}) by {crew_id}. Budget: {self.firebreak_budget}"

    # ─── Recon Operations ─────────────────────────────

    def recon_flight(self, center_row: int, center_col: int, current_step: int) -> tuple[bool, str]:
        """Execute a reconnaissance flight revealing a 10x10 area for 5 steps."""
        if self.recon_budget <= 0:
            return False, "No recon budget remaining"

        if not self.grid._in_bounds(center_row, center_col):
            return False, f"Target ({center_row},{center_col}) out of bounds"

        # Reveal 10x10 area
        for r in range(max(0, center_row - 5), min(self.grid.rows, center_row + 5)):
            for c in range(max(0, center_col - 5), min(self.grid.cols, center_col + 5)):
                self.revealed_cells.add((r, c))
                self.reveal_expiry[(r, c)] = current_step + 5

        self.recon_budget -= 1
        return True, f"Recon flight over ({center_row},{center_col}). {len(self.revealed_cells)} cells revealed."

    def expire_reveals(self, current_step: int) -> None:
        """Remove expired cell reveals."""
        expired = [cell for cell, step in self.reveal_expiry.items() if current_step >= step]
        for cell in expired:
            self.revealed_cells.discard(cell)
            del self.reveal_expiry[cell]

    # ─── Crew Loss (Hard tier) ────────────────────────

    def apply_crew_loss(self, crew_id: str) -> list[str]:
        """Disable a specific crew (injury event)."""
        crew = self._get_crew(crew_id)
        if crew is None:
            return []
        if not crew.is_active:
            return []

        crew.is_active = False
        if crew.is_deployed:
            self.grid.dynamic_grid[crew.row][crew.col].crew_present = False
            crew.is_deployed = False

        return [f"CREW LOSS: {crew_id} injured and evacuated."]

    # ─── State / Observation ──────────────────────────

    def get_resource_state(self) -> ResourceState:
        """Build resource state for observation."""
        return ResourceState(
            crews=[c.model_copy() for c in self.crews],
            tankers=[t.model_copy() for t in self.tankers],
            firebreak_budget=self.firebreak_budget,
            recon_budget=self.recon_budget,
        )

    def get_crew_positions(self) -> list[tuple[int, int]]:
        """Return positions of all deployed, active crews."""
        return [
            (c.row, c.col) for c in self.crews
            if c.is_active and c.is_deployed
        ]

    def get_total_possible_actions(self, episode_length: int) -> int:
        """Estimate total possible meaningful actions for efficiency scoring."""
        return episode_length * self.config.num_crews

    # ─── Helpers ──────────────────────────────────────

    def _get_crew(self, crew_id: str) -> CrewState | None:
        for c in self.crews:
            if c.crew_id == crew_id:
                return c
        return None

    def _get_tanker(self, tanker_id: str) -> TankerState | None:
        for t in self.tankers:
            if t.tanker_id == tanker_id:
                return t
        return None
