"""
Fire spread engine for the Wildfire Containment Simulator.

Implements a Rothermel-inspired cellular automaton where each burning cell
attempts to ignite its 8 neighbors based on fuel, wind, slope, moisture,
and suppression factors.
"""

from __future__ import annotations

import math

import numpy as np

from .models import FireState, FuelType
from .grid import Grid


# Base ignition rates by fuel type (tuned for balanced gameplay)
BASE_RATES: dict[FuelType, float] = {
    FuelType.GRASS: 0.25,
    FuelType.SHRUB: 0.18,
    FuelType.TIMBER: 0.12,
    FuelType.URBAN: 0.12,   # 0.5x effective (applied separately)
    FuelType.WATER: 0.0,
    FuelType.ROAD: 0.0,
}

# Burn duration (steps before burnout) by fuel type
BURN_DURATION: dict[FuelType, int] = {
    FuelType.GRASS: 4,
    FuelType.SHRUB: 6,
    FuelType.TIMBER: 10,
    FuelType.URBAN: 8,
    FuelType.WATER: 0,
    FuelType.ROAD: 0,
}

# Intensity multiplier for urban structures
URBAN_INTENSITY_MULT = 2.0
URBAN_IGNITION_MULT = 0.5

# 8-neighbor offsets (row_delta, col_delta)
NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


class FireSpreadEngine:
    """
    Manages fire propagation across the grid each simulation step.

    The spread model computes per-neighbor ignition probabilities using:
        P(ignite) = base_rate * fuel_factor * wind_factor * slope_factor
                    * (1 - moisture) * (1 - suppression) * tier_scale
    """

    # Tier-based difficulty scaling for spread rate
    # Tuned so that: random agent ~0.2-0.4, heuristic ~0.6-0.8 on easy
    TIER_SPREAD_SCALE = {
        "easy": 1.0,
        "medium": 0.7,
        "hard": 0.55,
    }

    def __init__(self, grid: Grid, rng: np.random.Generator):
        self.grid = grid
        self.rng = rng
        self.cell_size_m = 100.0  # Each cell represents 100m x 100m
        self.tier_scale = self.TIER_SPREAD_SCALE.get(grid.config.tier_name, 0.5)

    def spread_step(self, wind_speed: float, wind_dir_deg: float) -> list[str]:
        """
        Execute one step of fire spread.

        1. For each BURNING cell, attempt to ignite neighbors.
        2. Update intensities (grow/decay).
        3. Transition cells that have exhausted fuel to BURNED_OUT.

        Returns a list of event strings for the observation log.
        """
        events: list[str] = []
        grid = self.grid

        # Collect currently burning cells (snapshot to avoid iteration issues)
        burning_cells = []
        for r in range(grid.rows):
            for c in range(grid.cols):
                if grid.dynamic_grid[r][c].fire_state == FireState.BURNING:
                    burning_cells.append((r, c))

        # Phase 1: Attempt ignition of neighbors
        new_ignitions: list[tuple[int, int, float]] = []

        for r, c in burning_cells:
            source_dyn = grid.dynamic_grid[r][c]
            source_static = grid.static_grid[r][c]

            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not grid._in_bounds(nr, nc):
                    continue

                target_static = grid.static_grid[nr][nc]
                target_dyn = grid.dynamic_grid[nr][nc]

                # Skip non-ignitable cells
                if target_static.fuel_type in (FuelType.WATER, FuelType.ROAD):
                    continue
                if target_dyn.fire_state != FireState.UNBURNED:
                    continue

                # Compute ignition probability
                prob = self._compute_ignition_prob(
                    source_r=r, source_c=c,
                    target_r=nr, target_c=nc,
                    source_intensity=source_dyn.fire_intensity,
                    wind_speed=wind_speed,
                    wind_dir_deg=wind_dir_deg,
                )

                if self.rng.random() < prob:
                    # Initial intensity depends on source and fuel
                    init_intensity = 0.2 + source_dyn.fire_intensity * 0.3
                    new_ignitions.append((nr, nc, init_intensity))

        # Apply new ignitions
        for nr, nc, intensity in new_ignitions:
            if grid.dynamic_grid[nr][nc].fire_state == FireState.UNBURNED:
                grid.ignite_cell(nr, nc, intensity)
                if grid.static_grid[nr][nc].is_populated:
                    pop = grid.static_grid[nr][nc].population
                    events.append(f"FIRE reached populated cell ({nr},{nc}) with {pop} people!")

        # Phase 2: Update intensities and burn timers
        for r in range(grid.rows):
            for c in range(grid.cols):
                dyn = grid.dynamic_grid[r][c]
                static = grid.static_grid[r][c]

                if dyn.fire_state == FireState.BURNING:
                    dyn.time_burning += 1
                    max_dur = BURN_DURATION.get(static.fuel_type, 6)

                    # Intensity curve: ramp up, peak, decay
                    peak_step = max_dur // 3
                    if dyn.time_burning <= peak_step:
                        # Ramp up
                        growth = 0.15 * static.fuel_load
                        if static.fuel_type == FuelType.URBAN:
                            growth *= URBAN_INTENSITY_MULT
                        dyn.fire_intensity = min(1.0, dyn.fire_intensity + growth)
                    elif dyn.time_burning <= 2 * peak_step:
                        # Peak / plateau
                        pass
                    else:
                        # Decay
                        decay = 0.1
                        dyn.fire_intensity = max(0.05, dyn.fire_intensity - decay)

                    # Apply suppression reduction
                    if dyn.suppression_level > 0:
                        dyn.fire_intensity = max(0.0, dyn.fire_intensity - dyn.suppression_level * 0.1)

                    # Check for burnout
                    if dyn.time_burning >= max_dur or dyn.fire_intensity <= 0.0:
                        dyn.fire_state = FireState.BURNED_OUT
                        dyn.fire_intensity = 0.0
                        events.append(f"Cell ({r},{c}) burned out.")

                    # Transition to ember if intensity low
                    elif dyn.fire_intensity < 0.15 and dyn.time_burning > peak_step:
                        dyn.fire_state = FireState.EMBER

                elif dyn.fire_state == FireState.EMBER:
                    dyn.time_burning += 1
                    dyn.fire_intensity = max(0.0, dyn.fire_intensity - 0.05)
                    max_dur = BURN_DURATION.get(static.fuel_type, 6)
                    if dyn.time_burning >= max_dur + 3 or dyn.fire_intensity <= 0.0:
                        dyn.fire_state = FireState.BURNED_OUT
                        dyn.fire_intensity = 0.0

        if new_ignitions:
            events.append(f"{len(new_ignitions)} new cell(s) ignited this step.")

        return events

    def _compute_ignition_prob(
        self,
        source_r: int, source_c: int,
        target_r: int, target_c: int,
        source_intensity: float,
        wind_speed: float,
        wind_dir_deg: float,
    ) -> float:
        """Compute probability of fire spreading from source to target cell."""
        target_static = self.grid.static_grid[target_r][target_c]
        target_dyn = self.grid.dynamic_grid[target_r][target_c]

        # Base rate by fuel type
        base = BASE_RATES.get(target_static.fuel_type, 0.0)
        if base <= 0:
            return 0.0

        # Urban ignition penalty
        if target_static.fuel_type == FuelType.URBAN:
            base *= URBAN_IGNITION_MULT

        # Fuel factor
        fuel_factor = target_static.fuel_load

        # Source intensity factor (hotter fires spread faster)
        intensity_factor = 0.5 + source_intensity * 0.5

        # Wind factor
        wind_factor = self._compute_wind_factor(
            source_r, source_c, target_r, target_c,
            wind_speed, wind_dir_deg
        )

        # Slope factor (fire travels uphill faster)
        slope_factor = self._compute_slope_factor(source_r, source_c, target_r, target_c)

        # Moisture dampening
        moisture_factor = 1.0 - target_dyn.moisture

        # Suppression dampening
        suppression_factor = 1.0 - target_dyn.suppression_level

        prob = (base * fuel_factor * intensity_factor * wind_factor
                * slope_factor * moisture_factor * suppression_factor
                * self.tier_scale)

        return float(np.clip(prob, 0.0, 0.95))  # Cap at 95%

    def _compute_wind_factor(
        self,
        sr: int, sc: int, tr: int, tc: int,
        wind_speed: float, wind_dir_deg: float,
    ) -> float:
        """
        Wind factor: fire spreads faster downwind.
        wind_dir_deg is the direction wind blows FROM (meteorological convention).
        So fire spreads in the opposite direction.
        """
        if wind_speed < 1.0:
            return 1.0

        # Direction from source to target
        dr = tr - sr
        dc = tc - sc
        spread_angle = math.atan2(dc, -dr)  # -dr because row increases downward

        # Wind blows FROM wind_dir, so fire spreads TOWARD wind_dir + 180
        wind_rad = math.radians(wind_dir_deg + 180)

        angle_diff = spread_angle - wind_rad
        cos_diff = math.cos(angle_diff)

        # Scale: 1.0 at crosswind, up to 2.5 downwind, down to 0.3 upwind
        factor = 1.0 + cos_diff * min(wind_speed / 40.0, 1.5)
        return max(0.3, factor)

    def _compute_slope_factor(
        self, sr: int, sc: int, tr: int, tc: int
    ) -> float:
        """Slope factor: fire accelerates uphill."""
        source_elev = self.grid.static_grid[sr][sc].elevation_m
        target_elev = self.grid.static_grid[tr][tc].elevation_m
        elev_diff = target_elev - source_elev

        # Positive diff = uphill = faster spread
        factor = 1.0 + 0.3 * max(0.0, elev_diff / self.cell_size_m)
        # Slight slowdown going downhill
        if elev_diff < 0:
            factor = max(0.7, 1.0 + 0.1 * elev_diff / self.cell_size_m)

        return float(np.clip(factor, 0.5, 2.0))
