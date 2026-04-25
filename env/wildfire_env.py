"""
Wildfire Containment Simulator — Main Environment.

Implements the OpenEnv API: step(), reset(), state().
Orchestrates grid, fire spread, weather, resources, and reward computation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from pydantic import ValidationError

from .models import (
    Action, ActionType, Observation, StepResult, ClusterStats,
    FireState, FuelType, TierConfig, TIER_EASY, TIER_MEDIUM, TIER_HARD,
)
from .grid import Grid
from .fire_spread import FireSpreadEngine
from .weather import WeatherEngine
from .resources import ResourceManager
from .reward import RewardCalculator
from .briefing import generate_briefing, OperationalBriefing

logger = logging.getLogger(__name__)


class WildfireEnv:
    """
    Wildfire Containment Simulator environment.

    Simulates a grid-based wildfire where an AI agent dispatches
    firefighting resources to contain the fire before it reaches
    populated zones.

    API:
        reset(task_id, seed) -> Observation
        step(action) -> StepResult
        state() -> dict
    """

    TIER_MAP = {
        "easy": TIER_EASY,
        "medium": TIER_MEDIUM,
        "hard": TIER_HARD,
    }

    def __init__(self, config: Optional[TierConfig] = None):
        self.config = config or TIER_EASY
        self.rng = np.random.default_rng(42)
        self.current_step = 0
        self.done = False

        # Components (initialized in reset)
        self.grid: Optional[Grid] = None
        self.fire_engine: Optional[FireSpreadEngine] = None
        self.weather: Optional[WeatherEngine] = None
        self.resources: Optional[ResourceManager] = None
        self.reward_calc: Optional[RewardCalculator] = None

        self.events_log: list[str] = []

        # Episode-level tracking for new reward structure
        self._prev_action: Optional[Action] = None
        self._invalid_action_count: int = 0
        self._crew_casualty_occurred: bool = False
        self._prev_state: Optional[dict] = None
        self.active_briefing: Optional[OperationalBriefing] = None

        # Last observation returned to the agent (agent's view, not ground truth)
        self._current_obs: Optional[Observation] = None

    def reset(self, task_id: str = "easy", seed: int = 42) -> Observation:
        """
        Initialize the environment for a new episode.

        Args:
            task_id: One of "easy", "medium", "hard".
            seed: Random seed for reproducibility.

        Returns:
            Initial observation.
        """
        self.config = self.TIER_MAP.get(task_id, TIER_EASY)
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.done = False
        self.events_log = []
        self._prev_action = None
        self._invalid_action_count = 0
        self._crew_casualty_occurred = False
        self._prev_state = None

        # Initialize components
        self.grid = Grid(self.config, self.rng)
        self.fire_engine = FireSpreadEngine(self.grid, self.rng)
        self.weather = WeatherEngine(self.config, self.rng)
        self.resources = ResourceManager(self.config, self.grid)
        self.reward_calc = RewardCalculator(self.config)
        self.reward_calc.reset()
        self.resources.reset()
        self.weather.reset()

        # Ignite initial fire points
        self._ignite_initial_fires()

        # Generate operational briefing for this episode
        self.active_briefing = generate_briefing(self.config, self.rng, self.grid)

        # Build and return initial observation (with briefing attached)
        obs = self._build_observation()
        obs.briefing = self.active_briefing
        self.events_log.append("Episode started. Fire ignited.")
        self._current_obs = obs
        return obs

    def step(self, action: Action) -> StepResult:
        """
        Execute one simulation step.

        Follows the 11-step tick sequence:
        1. Validate action
        2. Execute action
        3. Spread fire
        4. Update intensities (handled inside spread)
        5. Apply suppression
        6. Evolve weather
        7. Update moisture
        8. Propagate smoke
        9. Compute reward
        10. Check termination
        11. Build observation

        Args:
            action: The agent's chosen action.

        Returns:
            StepResult with observation, reward, done flag, and info dict.
        """
        if self.done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"error": "Episode already finished"},
            )

        step_events: list[str] = []

        # Snapshot state before this step's changes
        prev_state = self._snapshot_state()

        # ── Step 1: Validate action ──
        action_was_redundant = self._is_redundant(action)
        valid, msg = self._validate_action(action)
        if not valid:
            self.reward_calc.record_invalid_action()
            self._invalid_action_count += 1
            self.resources.wasted_actions += 1
            step_events.append(f"Invalid action: {msg}")
            # Skip to reward/termination
        else:
            # ── Step 2: Execute action ──
            exec_events = self._execute_action(action)
            step_events.extend(exec_events)

        self._prev_action = action

        # ── Step 3-4: Spread fire + update intensities ──
        ws = self.weather.state
        spread_events = self.fire_engine.spread_step(ws.wind_speed_kmh, ws.wind_direction_deg)
        step_events.extend(spread_events)

        # ── Step 5: Apply suppression ──
        supp_events = self.resources.apply_suppression()
        step_events.extend(supp_events)

        # ── Step 6: Evolve weather ──
        weather_events = self.weather.step(self.current_step)
        step_events.extend(weather_events)

        # ── Step 7: Update moisture ──
        self.grid.update_moisture(ws.rain_active, ws.humidity_pct)

        # ── Step 8: Propagate smoke ──
        self.grid.propagate_smoke(ws.wind_direction_deg, ws.wind_speed_kmh)

        # ── Tick tanker cooldowns ──
        self.resources.tick_tanker_cooldowns()

        # ── Expire recon reveals ──
        self.resources.expire_reveals(self.current_step)

        # ── Handle staggered ignition (hard tier) ──
        if (self.config.staggered_ignition_step is not None
                and self.current_step == self.config.staggered_ignition_step):
            self._ignite_staggered_fire()
            step_events.append("NEW IGNITION: Additional fire started!")

        # ── Handle crew loss (hard tier) ──
        if (self.config.enable_crew_loss
                and self.config.crew_loss_step == self.current_step
                and self.config.crew_loss_id):
            loss_events = self.resources.apply_crew_loss(self.config.crew_loss_id)
            step_events.extend(loss_events)

        # Track crew casualty
        if self.resources.crew_casualties:
            self._crew_casualty_occurred = True

        self.current_step += 1

        # Log a hold-message when fire is extinguished before min_active_steps so
        # agents (and the LLM) understand the episode must continue for monitoring.
        burning_now = (self.grid.count_by_state(FireState.BURNING)
                       + self.grid.count_by_state(FireState.EMBER))
        if burning_now == 0 and self.current_step < self.config.min_active_steps:
            step_events.append(
                f"All fires contained. Holding perimeter until step "
                f"{self.config.min_active_steps} (min_active_steps)."
            )

        # ── Step 9: Compute reward ──
        legacy_reward = self.reward_calc.compute_reward(self.grid, self.resources, self.current_step)

        current_state = self._snapshot_state()
        step_reward = self.reward_calc.compute_step_reward(
            prev_state, current_state, valid, action_was_redundant
        )

        # ── Step 10: Check termination ──
        self.done = self._check_termination()

        terminal_reward = 0.0
        if self.done:
            terminal_state = dict(current_state)
            terminal_state["crew_casualty_occurred"] = self._crew_casualty_occurred
            terminal_state["invalid_action_count"] = self._invalid_action_count
            if self.active_briefing:
                terminal_state["priority_zones"] = self.active_briefing.priority_populated_zones
                terminal_state["_grid_ref"] = self.grid
            terminal_reward = self.reward_calc.compute_terminal_reward(
                terminal_state, self.current_step, self.config.episode_length
            )

        reward = step_reward + terminal_reward

        # ── Step 11: Build observation ──
        obs = self._build_observation()

        # Keep last 5 events
        self.events_log = (self.events_log + step_events)[-20:]

        info = {
            "step": self.current_step,
            "events": step_events,
            "legacy_reward": round(legacy_reward, 4),
            "reward_breakdown": self.reward_calc.get_component_breakdown(
                self.grid, self.resources, self.current_step
            ),
        }

        result = StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self.done,
            info=info,
        )
        self._current_obs = result.observation
        return result

    def state(self) -> dict:
        """
        Return full ground-truth state for grading/debugging.
        NOT for agent use — contains information hidden from the agent.
        """
        if self.grid is None:
            return {"error": "Environment not initialized. Call reset() first."}

        # Full grid state without any occlusion
        full_grid = []
        for r in range(self.grid.rows):
            row = []
            for c in range(self.grid.cols):
                static = self.grid.static_grid[r][c]
                dynamic = self.grid.dynamic_grid[r][c]
                row.append({
                    "row": r, "col": c,
                    "fuel_type": static.fuel_type.value,
                    "fuel_load": static.fuel_load,
                    "elevation_m": static.elevation_m,
                    "is_populated": static.is_populated,
                    "population": static.population,
                    "fire_state": dynamic.fire_state.value,
                    "fire_intensity": round(dynamic.fire_intensity, 4),
                    "moisture": round(dynamic.moisture, 4),
                    "time_burning": dynamic.time_burning,
                    "suppression_level": round(dynamic.suppression_level, 4),
                    "smoke_density": round(dynamic.smoke_density, 4),
                    "crew_present": dynamic.crew_present,
                })
            full_grid.append(row)

        return {
            "tier": self.config.tier_name,
            "current_step": self.current_step,
            "done": self.done,
            "grid": full_grid,
            "weather": self.weather.get_true_state().model_dump(),
            "resources": self.resources.get_resource_state().model_dump(),
            "reward_breakdown": self.reward_calc.get_component_breakdown(
                self.grid, self.resources, self.current_step
            ),
            "total_population": self.grid.get_total_population(),
            "population_lost": self.grid.get_population_lost(),
            "cells_burned": self.grid.get_burned_count(),
            "total_burnable": self.grid.get_total_burnable(),
        }

    # ══════════════════════════════════════════════════
    # PRIVATE METHODS
    # ══════════════════════════════════════════════════

    def _snapshot_state(self) -> dict:
        """Capture a lightweight state dict for reward delta computation."""
        total, contained = self.grid.get_fire_perimeter()
        containment_pct = contained / total if total > 0 else 1.0
        return {
            "containment_pct": containment_pct,
            "pop_lost": self.grid.get_population_lost(),
            "total_pop": self.grid.get_total_population(),
        }

    def _is_redundant(self, action: Action) -> bool:
        """True if action is a meaningless repeat of the previous action.

        Actions that use target coordinates (DROP_RETARDANT, DEPLOY_CREW, RECON_FLIGHT)
        are redundant when the type + target cell match.  Directional actions (MOVE_CREW,
        BUILD_FIREBREAK) require the same crew_id AND direction to be redundant — two
        consecutive MOVE_CREW steps by different crews, or in different directions, are
        valid patrol behaviour and must not be penalised.
        """
        if self._prev_action is None:
            return False
        prev = self._prev_action
        if action.action_type != prev.action_type:
            return False
        # Coordinate-targeted actions: redundant when same cell is targeted again
        if action.target_row is not None or prev.target_row is not None:
            return (action.target_row == prev.target_row
                    and action.target_col == prev.target_col)
        # Crew directional actions: redundant only when same crew moves same direction
        if action.crew_id is not None:
            return (action.crew_id == prev.crew_id
                    and action.direction == prev.direction)
        return False

    def _ignite_initial_fires(self) -> None:
        """Place initial fire ignition points based on tier config.

        Ignition candidates are shifted away from populated cells to ensure
        a minimum survivable distance, reducing unwinnable-scenario variance.

        Intensity is set high enough (0.65) that a single tanker drop (-0.4)
        leaves residual fire (0.25) so the episode cannot be solved in 1-2
        steps. The fire must spread, be actively managed, and burn for at
        least min_active_steps before the episode can end.
        """
        rows, cols = self.config.grid_rows, self.config.grid_cols

        # Minimum Manhattan distance from any populated cell per tier
        min_pop_dist = {"easy": 4, "medium": 6, "hard": 7}.get(self.config.tier_name, 5)

        if self.config.tier_name == "easy":
            # Two ignition points spread across the grid so crews must split
            r1, c1 = self._find_ignition_candidate(rows // 2, cols // 3, min_pop_dist)
            self.grid.ignite_cell(r1, c1, intensity=0.65)
            r2, c2 = self._find_ignition_candidate(rows // 2, 2 * cols // 3, min_pop_dist)
            self.grid.ignite_cell(r2, c2, intensity=0.65)
        elif self.config.tier_name == "medium":
            # Three ignition points: forces genuine multi-front management
            r1, c1 = self._find_ignition_candidate(rows // 4, cols // 3, min_pop_dist)
            self.grid.ignite_cell(r1, c1, intensity=0.65)
            r2, c2 = self._find_ignition_candidate(2 * rows // 3, 2 * cols // 3, min_pop_dist)
            self.grid.ignite_cell(r2, c2, intensity=0.65)
            r3, c3 = self._find_ignition_candidate(rows // 2, cols // 2, min_pop_dist)
            self.grid.ignite_cell(r3, c3, intensity=0.65)
        else:
            # Two initial points (third comes later via staggered ignition at step 30)
            r1, c1 = self._find_ignition_candidate(rows // 4, cols // 4, min_pop_dist)
            self.grid.ignite_cell(r1, c1, intensity=0.65)
            r2, c2 = self._find_ignition_candidate(rows // 2, 3 * cols // 4, min_pop_dist)
            self.grid.ignite_cell(r2, c2, intensity=0.65)

    def _find_ignition_candidate(self, target_r: int, target_c: int, min_pop_dist: int) -> tuple[int, int]:
        """Return the nearest valid ignition cell to (target_r, target_c) that is at
        least min_pop_dist (Manhattan) from every populated cell.

        Searches in expanding rings; falls back to the original target if no
        compliant cell is found within the grid bounds.
        """
        rows, cols = self.config.grid_rows, self.config.grid_cols

        pop_cells = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if self.grid.static_grid[r][c].is_populated
        ]

        def _min_pop_dist(r: int, c: int) -> int:
            if not pop_cells:
                return 9999
            return min(abs(r - pr) + abs(c - pc) for pr, pc in pop_cells)

        for radius in range(max(rows, cols)):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if radius > 0 and abs(dr) + abs(dc) != radius:
                        continue
                    r, c = target_r + dr, target_c + dc
                    if not self.grid._in_bounds(r, c):
                        continue
                    static = self.grid.static_grid[r][c]
                    if static.fuel_type in (FuelType.WATER, FuelType.ROAD):
                        continue
                    if _min_pop_dist(r, c) >= min_pop_dist:
                        return r, c

        return target_r, target_c

    def _ignite_staggered_fire(self) -> None:
        """Ignite additional fire point(s) for hard tier."""
        rows, cols = self.config.grid_rows, self.config.grid_cols
        # Place in an area likely to cause problems
        target_r = 3 * rows // 4
        target_c = cols // 3
        # Find nearest unburned cell
        for dr in range(5):
            for dc in range(5):
                r, c = target_r + dr, target_c + dc
                if self.grid._in_bounds(r, c):
                    if self.grid.dynamic_grid[r][c].fire_state == FireState.UNBURNED:
                        self.grid.ignite_cell(r, c, intensity=0.7)
                        return

    def _validate_action(self, action: Action) -> tuple[bool, str]:
        """Validate action parameters. Returns (is_valid, error_message)."""
        try:
            # Pydantic validation already ran on construction,
            # but we do semantic validation here
            if action.action_type == ActionType.DEPLOY_CREW:
                if not self.grid._in_bounds(action.target_row, action.target_col):
                    return False, f"Target ({action.target_row},{action.target_col}) out of bounds"

            elif action.action_type == ActionType.DROP_RETARDANT:
                if not self.grid._in_bounds(action.target_row, action.target_col):
                    return False, f"Target ({action.target_row},{action.target_col}) out of bounds"

            elif action.action_type == ActionType.RECON_FLIGHT:
                if not self.grid._in_bounds(action.target_row, action.target_col):
                    return False, f"Target ({action.target_row},{action.target_col}) out of bounds"

            return True, ""

        except Exception as e:
            return False, str(e)

    def _execute_action(self, action: Action) -> list[str]:
        """Execute a validated action. Returns event messages."""
        events = []
        at = action.action_type

        if at == ActionType.DEPLOY_CREW:
            ok, msg = self.resources.deploy_crew(action.crew_id, action.target_row, action.target_col)
            events.append(msg)
            if not ok:
                self.resources.wasted_actions += 1

        elif at == ActionType.MOVE_CREW:
            ok, msg = self.resources.move_crew(action.crew_id, action.direction)
            events.append(msg)
            if not ok:
                self.resources.wasted_actions += 1

        elif at == ActionType.DROP_RETARDANT:
            ok, msg = self.resources.drop_retardant(action.tanker_id, action.target_row, action.target_col)
            events.append(msg)
            if not ok:
                self.resources.wasted_actions += 1

        elif at == ActionType.BUILD_FIREBREAK:
            ok, msg = self.resources.build_firebreak(action.crew_id, action.direction)
            events.append(msg)
            if not ok:
                self.resources.wasted_actions += 1

        elif at == ActionType.RECON_FLIGHT:
            ok, msg = self.resources.recon_flight(action.target_row, action.target_col, self.current_step)
            events.append(msg)
            if not ok:
                self.resources.wasted_actions += 1

        elif at == ActionType.IDLE:
            reason = action.reason or "No action taken"
            events.append(f"IDLE: {reason}")

        return events

    def _check_termination(self) -> bool:
        """Check if the episode should end."""
        # Time limit
        if self.current_step >= self.config.episode_length:
            return True

        # Fire fully contained (no burning cells)
        burning = self.grid.count_by_state(FireState.BURNING)
        ember = self.grid.count_by_state(FireState.EMBER)
        if burning == 0 and ember == 0:
            # Enforce minimum active steps — prevents trivial 1-2 step episodes
            # where a single tanker drop or natural burnout ends the episode
            # before the agent has taken any meaningful sequence of actions.
            if self.current_step < self.config.min_active_steps:
                return False
            # Don't terminate before staggered ignition fires (hard tier)
            if (self.config.staggered_ignition_step
                    and self.current_step < self.config.staggered_ignition_step):
                return False
            return True

        # All populated zones burned (catastrophic failure)
        total_pop = self.grid.get_total_population()
        lost_pop = self.grid.get_population_lost()
        if total_pop > 0 and lost_pop >= total_pop:
            return True

        return False

    def _build_observation(self) -> Observation:
        """Build the agent's observation with appropriate noise/occlusion."""
        # Grid observation with fog/smoke
        crew_positions = self.resources.get_crew_positions()
        grid_obs = self.grid.build_observation(
            enable_fog=self.config.enable_fog_of_war,
            fog_radius=self.config.fog_visibility_radius,
            crew_positions=crew_positions,
            revealed_cells=self.resources.revealed_cells,
        )

        # Weather observation (possibly noisy)
        weather_obs = self.weather.get_observation()

        # Resource state (fully observable)
        resource_state = self.resources.get_resource_state()

        # Stats
        total_burnable = self.grid.get_total_burnable()
        cells_burned = self.grid.get_burned_count()
        total_pop = self.grid.get_total_population()
        pop_lost = self.grid.get_population_lost()

        area_saved_pct = round(
            100.0 * (total_burnable - cells_burned) / total_burnable, 1
        ) if total_burnable > 0 else 100.0

        civilians_saved_pct = round(
            100.0 * (total_pop - pop_lost) / total_pop, 1
        ) if total_pop > 0 else 100.0

        stats = ClusterStats(
            cells_burned=cells_burned,
            cells_burning=self.grid.count_by_state(FireState.BURNING),
            cells_saved=total_burnable - cells_burned - self.grid.count_by_state(FireState.BURNING),
            population_threatened=self._count_threatened_population(),
            population_lost=pop_lost,
            total_population=total_pop,
            containment_pct=self._compute_containment_pct(),
            area_saved_pct=area_saved_pct,
            civilians_saved_pct=civilians_saved_pct,
            current_step=self.current_step,
            max_steps=self.config.episode_length,
            firebreaks_built=self.resources.total_firebreaks_built,
            retardant_drops=self.resources.total_retardant_drops,
        )

        # Recent events (last 5)
        recent = self.events_log[-5:] if self.events_log else []

        return Observation(
            grid=grid_obs,
            weather=weather_obs,
            resources=resource_state,
            stats=stats,
            recent_events=recent,
        )

    def _count_threatened_population(self) -> int:
        """Count population within 3 cells of active fire."""
        threatened = 0
        burning_cells = self.grid.get_burning_cells()
        counted = set()

        for br, bc in burning_cells:
            for r in range(max(0, br - 3), min(self.grid.rows, br + 4)):
                for c in range(max(0, bc - 3), min(self.grid.cols, bc + 4)):
                    if (r, c) not in counted:
                        static = self.grid.static_grid[r][c]
                        if static.is_populated:
                            dynamic = self.grid.dynamic_grid[r][c]
                            if dynamic.fire_state not in (FireState.BURNED_OUT, FireState.BURNING):
                                threatened += static.population
                                counted.add((r, c))
        return threatened

    def _compute_containment_pct(self) -> float:
        """Compute fire containment percentage."""
        total, contained = self.grid.get_fire_perimeter()
        if total == 0:
            return 100.0
        return round(100.0 * contained / total, 1)
