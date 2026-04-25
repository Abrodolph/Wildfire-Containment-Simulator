"""
Reward computation for the Wildfire Containment Simulator.

Computes a weighted composite reward in [0.0, 1.0] from five components:
containment, population safety, resource efficiency, speed, and area saved.
"""

from __future__ import annotations

from .models import TierConfig
from .grid import Grid
from .resources import ResourceManager


class RewardCalculator:
    """
    Computes per-step reward as a weighted composite of five normalized components.

    All components are in [0, 1]. The final reward applies multiplicative penalties
    for catastrophic failures (crew casualties, populated cells burned).
    """

    def __init__(self, config: TierConfig):
        self.config = config
        self.invalid_action_count = 0
        self.steps_with_fire = 0
        self.containment_achieved = False
        self.containment_step: int | None = None

    def reset(self) -> None:
        self.invalid_action_count = 0
        self.steps_with_fire = 0
        self.containment_achieved = False
        self.containment_step = None

    def record_invalid_action(self) -> None:
        self.invalid_action_count += 1

    def compute_reward(
        self,
        grid: Grid,
        resources: ResourceManager,
        current_step: int,
    ) -> float:
        """
        Compute the composite reward for the current state.

        Returns a float in [0.0, 1.0].
        """
        cfg = self.config

        # Track fire presence
        from .models import FireState
        burning_count = grid.count_by_state(FireState.BURNING) + grid.count_by_state(FireState.EMBER)
        if burning_count > 0:
            self.steps_with_fire += 1

        # Check containment
        if burning_count == 0 and current_step > 0 and not self.containment_achieved:
            self.containment_achieved = True
            self.containment_step = current_step

        # ── Component 1: Containment Score ──
        total_perim, contained_perim = grid.get_fire_perimeter()
        if total_perim > 0:
            containment_score = contained_perim / total_perim
        else:
            # No active fire perimeter = either no fire or fully contained
            containment_score = 1.0 if self.containment_achieved else 0.5

        # ── Component 2: Population Safety ──
        total_pop = grid.get_total_population()
        lost_pop = grid.get_population_lost()
        if total_pop > 0:
            population_score = 1.0 - (lost_pop / total_pop)
        else:
            population_score = 1.0

        # ── Component 3: Resource Efficiency ──
        total_possible = resources.get_total_possible_actions(current_step + 1)
        wasted = resources.wasted_actions + resources.idle_crew_steps
        if total_possible > 0:
            efficiency_score = 1.0 - min(1.0, wasted / total_possible)
        else:
            efficiency_score = 1.0

        # ── Component 4: Speed Score ──
        if self.containment_achieved and self.containment_step is not None:
            speed_score = 1.0 - (self.containment_step / cfg.episode_length)
        elif burning_count == 0 and current_step == 0:
            speed_score = 1.0
        else:
            # Fire still active — score based on progress
            speed_score = max(0.0, 0.3 - (current_step / cfg.episode_length) * 0.3)

        # ── Component 5: Area Saved ──
        total_burnable = grid.get_total_burnable()
        burned = grid.get_burned_count()
        if total_burnable > 0:
            area_score = 1.0 - (burned / total_burnable)
        else:
            area_score = 1.0

        # ── Weighted composite ──
        weights = [cfg.w_containment, cfg.w_population, cfg.w_efficiency, cfg.w_speed, cfg.w_area]
        scores = [containment_score, population_score, efficiency_score, speed_score, area_score]
        total_weight = sum(weights)

        reward = sum(w * s for w, s in zip(weights, scores)) / total_weight if total_weight > 0 else 0.0

        # ── Penalty: invalid actions ──
        reward -= 0.02 * self.invalid_action_count

        # ── Penalty: populated cell burned ──
        if lost_pop > 0:
            # Linear penalty per populated cell lost (not exponential)
            pop_cells_lost = sum(
                1 for r in range(grid.rows) for c in range(grid.cols)
                if grid.dynamic_grid[r][c].fire_state == FireState.BURNED_OUT
                and grid.static_grid[r][c].is_populated
            )
            reward *= max(0.15, 1.0 - 0.08 * pop_cells_lost)

        # ── Penalty: crew casualty ──
        if resources.crew_casualties:
            reward = 0.0

        return float(max(0.0, min(1.0, reward)))

    def compute_step_reward(
        self,
        prev_state: dict,
        current_state: dict,
        action_was_valid: bool,
        action_was_redundant: bool,
    ) -> float:
        """Dense per-step reward based on state deltas."""
        total_pop = current_state.get("total_pop", 0)

        delta_containment = current_state["containment_pct"] - prev_state["containment_pct"]

        if total_pop > 0:
            prev_pop_safety = 1.0 - prev_state["pop_lost"] / total_pop
            curr_pop_safety = 1.0 - current_state["pop_lost"] / total_pop
            delta_pop_safety = curr_pop_safety - prev_pop_safety
        else:
            delta_pop_safety = 0.0

        reward = (delta_containment * 0.4) + (delta_pop_safety * 0.4)
        if action_was_redundant:
            reward -= 0.1
        return reward

    def compute_terminal_reward(
        self,
        final_state: dict,
        episode_steps: int,
        max_steps: int,
    ) -> float:
        """Sparse terminal reward applied only on episode end."""
        total_pop = final_state.get("total_pop", 0)
        pop_lost = final_state.get("pop_lost", 0)

        reward = 0.0
        if pop_lost == 0:
            reward += 5.0
            efficiency_bonus = (max_steps - episode_steps) / max_steps * 2.0
            reward += efficiency_bonus
        else:
            reward += -3.0 * (pop_lost / total_pop) if total_pop > 0 else -3.0

        if final_state.get("crew_casualty_occurred", False):
            reward -= 2.0

        invalid_penalty = min(0.2, 0.01 * final_state.get("invalid_action_count", 0))
        reward -= invalid_penalty

        # Briefing adherence bonus: +1.0 if all priority zones survived
        priority_zones = final_state.get("priority_zones", [])
        if priority_zones:
            grid_ref = final_state.get("_grid_ref")
            if grid_ref is not None:
                from .models import FireState
                all_safe = all(
                    grid_ref.dynamic_grid[r][c].fire_state not in (
                        FireState.BURNED_OUT, FireState.BURNING, FireState.EMBER
                    )
                    for r, c in priority_zones
                    if 0 <= r < grid_ref.rows and 0 <= c < grid_ref.cols
                )
                if all_safe:
                    reward += 1.0

        return reward

    def get_component_breakdown(self, grid: Grid, resources: ResourceManager, current_step: int) -> dict:
        """Return individual component scores for debugging/logging."""
        from .models import FireState

        burning_count = grid.count_by_state(FireState.BURNING) + grid.count_by_state(FireState.EMBER)

        total_perim, contained_perim = grid.get_fire_perimeter()
        containment_score = contained_perim / total_perim if total_perim > 0 else (1.0 if self.containment_achieved else 0.5)

        total_pop = grid.get_total_population()
        lost_pop = grid.get_population_lost()
        population_score = 1.0 - (lost_pop / total_pop) if total_pop > 0 else 1.0

        total_possible = resources.get_total_possible_actions(current_step + 1)
        wasted = resources.wasted_actions + resources.idle_crew_steps
        efficiency_score = 1.0 - min(1.0, wasted / total_possible) if total_possible > 0 else 1.0

        if self.containment_achieved and self.containment_step is not None:
            speed_score = 1.0 - (self.containment_step / self.config.episode_length)
        else:
            speed_score = max(0.0, 0.3 - (current_step / self.config.episode_length) * 0.3)

        total_burnable = grid.get_total_burnable()
        burned = grid.get_burned_count()
        area_score = 1.0 - (burned / total_burnable) if total_burnable > 0 else 1.0

        return {
            "containment": round(containment_score, 4),
            "population_safety": round(population_score, 4),
            "efficiency": round(efficiency_score, 4),
            "speed": round(speed_score, 4),
            "area_saved": round(area_score, 4),
            "burning_cells": burning_count,
            "population_lost": lost_pop,
            "invalid_actions": self.invalid_action_count,
            "crew_casualty": resources.crew_casualties,
        }
