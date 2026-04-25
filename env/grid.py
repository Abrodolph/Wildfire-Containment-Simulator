"""
Grid terrain simulation for the Wildfire Containment Simulator.

Manages the NxM grid of cells, including terrain generation, cell state updates,
smoke propagation, and moisture dynamics.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .models import (
    CellStatic, CellDynamic, CellObservation, FireState, FuelType,
    IntensityBin, TierConfig,
)


class Grid:
    """
    NxM grid of terrain cells with static properties and dynamic state.

    Attributes:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        static_grid: 2D list of CellStatic (immutable terrain).
        dynamic_grid: 2D list of CellDynamic (mutable fire/moisture/smoke state).
    """

    def __init__(self, config: TierConfig, rng: np.random.Generator):
        self.rows = config.grid_rows
        self.cols = config.grid_cols
        self.config = config
        self.rng = rng

        # Initialize grids
        self.static_grid: list[list[CellStatic]] = []
        self.dynamic_grid: list[list[CellDynamic]] = []

        self._generate_terrain()

    def _generate_terrain(self) -> None:
        """Generate terrain based on tier configuration."""
        rows, cols = self.rows, self.cols

        # Generate elevation map using simple gradient + noise
        elevation = np.zeros((rows, cols))
        if self.config.tier_name == "easy":
            # Flat terrain
            elevation[:] = 0.0
        elif self.config.tier_name == "medium":
            # Valley: low center, higher edges (canyon terrain)
            for r in range(rows):
                for c in range(cols):
                    dist_from_center = abs(c - cols // 2) / (cols // 2)
                    elevation[r, c] = dist_from_center * 500.0
            elevation += self.rng.normal(0, 20, (rows, cols))
            elevation = np.clip(elevation, 0, 500)
        else:
            # Complex terrain with ridges and valleys
            for r in range(rows):
                for c in range(cols):
                    # Create a ridge running diagonally
                    ridge = math.sin(r / 8.0) * 400 + math.cos(c / 6.0) * 300
                    elevation[r, c] = max(0, ridge + 300)
            elevation += self.rng.normal(0, 40, (rows, cols))
            elevation = np.clip(elevation, 0, 1200)

        # Generate fuel type map
        fuel_map = self._generate_fuel_map()

        # Place water bodies
        water_cells = self._place_water()

        # Place populated zones
        pop_cells = self._place_populations()

        # Build static grid
        self.static_grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                ft = fuel_map[r][c]
                is_water = (r, c) in water_cells
                if is_water:
                    ft = FuelType.WATER

                pop = pop_cells.get((r, c), 0)
                fuel_load = self._fuel_load_for_type(ft)

                cell = CellStatic(
                    row=r, col=c,
                    elevation_m=float(elevation[r, c]),
                    fuel_type=ft,
                    fuel_load=fuel_load,
                    is_populated=pop > 0,
                    population=pop,
                    is_water=is_water,
                )
                row.append(cell)
            self.static_grid.append(row)

        # Build dynamic grid (all unburned, default moisture)
        base_moisture = 0.3 if self.config.humidity_init < 50 else 0.5
        self.dynamic_grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                moisture = base_moisture + self.rng.normal(0, 0.05)
                moisture = float(np.clip(moisture, 0.05, 0.95))
                row.append(CellDynamic(moisture=moisture))
            self.dynamic_grid.append(row)

    def _generate_fuel_map(self) -> list[list[FuelType]]:
        """Generate fuel types based on tier."""
        rows, cols = self.rows, self.cols
        fuel_map = [[FuelType.GRASS for _ in range(cols)] for _ in range(rows)]

        if self.config.tier_name == "easy":
            # All grass, simple
            pass
        elif self.config.tier_name == "medium":
            # Valley floor = grass, hillsides = shrub, ridgeline = timber
            for r in range(rows):
                for c in range(cols):
                    dist = abs(c - cols // 2) / (cols // 2)
                    if dist > 0.7:
                        fuel_map[r][c] = FuelType.TIMBER
                    elif dist > 0.35:
                        fuel_map[r][c] = FuelType.SHRUB
        else:
            # Complex mixed terrain with some roads and urban
            for r in range(rows):
                for c in range(cols):
                    val = self.rng.random()
                    if val < 0.35:
                        fuel_map[r][c] = FuelType.GRASS
                    elif val < 0.60:
                        fuel_map[r][c] = FuelType.SHRUB
                    elif val < 0.85:
                        fuel_map[r][c] = FuelType.TIMBER
                    else:
                        fuel_map[r][c] = FuelType.GRASS  # Will assign urban/road below

            # Place roads (horizontal and vertical corridors)
            road_row = rows // 3
            road_col = cols // 2
            for c in range(cols):
                fuel_map[road_row][c] = FuelType.ROAD
            for r in range(rows):
                fuel_map[r][road_col] = FuelType.ROAD

        return fuel_map

    def _place_water(self) -> set[tuple[int, int]]:
        """Place water bodies on the grid."""
        water = set()
        rows, cols = self.rows, self.cols

        if self.config.tier_name == "easy":
            # 2 small water patches
            water.add((rows // 4, cols // 4))
            water.add((rows // 4, cols // 4 + 1))
            water.add((3 * rows // 4, 3 * cols // 4))
            water.add((3 * rows // 4, 3 * cols // 4 + 1))
        elif self.config.tier_name == "medium":
            # Small lake in valley
            cr, cc = rows // 2, cols // 2
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r, c = cr + dr, cc + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        water.add((r, c))
        else:
            # River running vertically + small lake
            river_col = cols // 4
            for r in range(rows // 3, 2 * rows // 3):
                water.add((r, river_col))
                water.add((r, river_col + 1))
            # Small lake
            lake_r, lake_c = 3 * rows // 4, 3 * cols // 4
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = lake_r + dr, lake_c + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        if abs(dr) + abs(dc) <= 3:
                            water.add((r, c))
        return water

    def _place_populations(self) -> dict[tuple[int, int], int]:
        """Place populated zones. Returns dict of (row, col) -> population."""
        pop = {}
        rows, cols = self.rows, self.cols

        if self.config.tier_name == "easy":
            # 2 small clusters near edges
            for dr in range(2):
                for dc in range(2):
                    pop[(1 + dr, 1 + dc)] = 3
                    pop[(rows - 3 + dr, cols - 3 + dc)] = 2
        elif self.config.tier_name == "medium":
            # 3 settlements in valley floor
            positions = [(rows // 4, cols // 2), (rows // 2, cols // 3), (3 * rows // 4, cols // 2 + 2)]
            pops = [20, 15, 15]
            for (pr, pc), p in zip(positions, pops):
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        r, c = pr + dr, pc + dc
                        if 0 <= r < rows and 0 <= c < cols:
                            pop[(r, c)] = p // 9 + 1
        else:
            # 1 town + 4 rural clusters
            # Town center
            town_r, town_c = 3 * rows // 4, cols // 2
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = town_r + dr, town_c + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        pop[(r, c)] = 8
                        # Mark as urban in fuel map (will be set after static grid build)
            # Rural clusters
            rural_centers = [
                (rows // 5, cols // 5),
                (rows // 5, 4 * cols // 5),
                (2 * rows // 3, cols // 5),
                (rows // 3, 3 * cols // 4),
            ]
            for cr, cc in rural_centers:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        r, c = cr + dr, cc + dc
                        if 0 <= r < rows and 0 <= c < cols:
                            pop[(r, c)] = 4

        return pop

    def _fuel_load_for_type(self, ft: FuelType) -> float:
        """Default fuel load by fuel type."""
        loads = {
            FuelType.GRASS: 0.7,
            FuelType.SHRUB: 0.8,
            FuelType.TIMBER: 0.9,
            FuelType.URBAN: 0.6,
            FuelType.WATER: 0.0,
            FuelType.ROAD: 0.0,
        }
        base = loads.get(ft, 0.5)
        noise = float(self.rng.normal(0, 0.05))
        return float(np.clip(base + noise, 0.0, 1.0))

    # ─── Ignition ─────────────────────────────────────

    def ignite_cell(self, row: int, col: int, intensity: float = 0.3) -> bool:
        """
        Ignite a cell. Returns True if successful.
        Cannot ignite water, road, firebreak, or already-burning cells.
        """
        if not self._in_bounds(row, col):
            return False

        static = self.static_grid[row][col]
        dynamic = self.dynamic_grid[row][col]

        if static.fuel_type in (FuelType.WATER, FuelType.ROAD):
            return False
        if dynamic.fire_state in (FireState.BURNING, FireState.EMBER, FireState.BURNED_OUT,
                                   FireState.FIREBREAK, FireState.SUPPRESSED):
            return False

        dynamic.fire_state = FireState.BURNING
        dynamic.fire_intensity = float(np.clip(intensity, 0.1, 1.0))
        dynamic.time_burning = 0
        return True

    # ─── Smoke Propagation ────────────────────────────

    def propagate_smoke(self, wind_dir_deg: float, wind_speed: float) -> None:
        """
        Propagate smoke downwind from burning cells.
        Smoke density decays with distance and over time.
        """
        if not self.config.enable_smoke_occlusion:
            return

        # Decay existing smoke
        for r in range(self.rows):
            for c in range(self.cols):
                dyn = self.dynamic_grid[r][c]
                if dyn.fire_state not in (FireState.BURNING, FireState.EMBER):
                    dyn.smoke_density = max(0.0, dyn.smoke_density - 0.1)

        # Generate new smoke from burning cells
        wind_rad = math.radians(wind_dir_deg)
        dr_wind = -math.cos(wind_rad)  # N = row decreasing
        dc_wind = math.sin(wind_rad)

        spread_dist = max(2, int(wind_speed / 10))

        for r in range(self.rows):
            for c in range(self.cols):
                dyn = self.dynamic_grid[r][c]
                if dyn.fire_state in (FireState.BURNING, FireState.EMBER):
                    # Smoke at the source
                    dyn.smoke_density = min(0.9, dyn.smoke_density + 0.3)

                    # Propagate downwind
                    for dist in range(1, spread_dist + 1):
                        sr = int(r + dr_wind * dist)
                        sc = int(c + dc_wind * dist)
                        if self._in_bounds(sr, sc):
                            smoke_add = 0.2 / dist
                            self.dynamic_grid[sr][sc].smoke_density = min(
                                0.9, self.dynamic_grid[sr][sc].smoke_density + smoke_add
                            )

    # ─── Moisture Updates ─────────────────────────────

    def update_moisture(self, rain_active: bool, humidity_pct: float) -> None:
        """Update moisture levels based on rain and humidity."""
        for r in range(self.rows):
            for c in range(self.cols):
                dyn = self.dynamic_grid[r][c]
                if rain_active:
                    dyn.moisture = min(1.0, dyn.moisture + 0.05)
                else:
                    # Dry out slowly based on humidity
                    dry_rate = 0.01 * (1.0 - humidity_pct / 100.0)
                    dyn.moisture = max(0.0, dyn.moisture - dry_rate)

    # ─── Observation Builder ──────────────────────────

    def build_observation(
        self,
        enable_fog: bool = False,
        fog_radius: int = 7,
        crew_positions: Optional[list[tuple[int, int]]] = None,
        revealed_cells: Optional[set[tuple[int, int]]] = None,
    ) -> list[list[CellObservation]]:
        """
        Build the agent-visible grid observation.
        Applies smoke occlusion and fog-of-war as configured.
        """
        if crew_positions is None:
            crew_positions = []
        if revealed_cells is None:
            revealed_cells = set()

        # Compute visible cells under fog-of-war
        visible = set()
        if enable_fog:
            for cr, cc in crew_positions:
                for r in range(max(0, cr - fog_radius), min(self.rows, cr + fog_radius + 1)):
                    for c in range(max(0, cc - fog_radius), min(self.cols, cc + fog_radius + 1)):
                        if (r - cr) ** 2 + (c - cc) ** 2 <= fog_radius ** 2:
                            visible.add((r, c))
            visible |= revealed_cells
        else:
            # All cells visible
            for r in range(self.rows):
                for c in range(self.cols):
                    visible.add((r, c))

        obs_grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                static = self.static_grid[r][c]
                dynamic = self.dynamic_grid[r][c]

                if (r, c) not in visible:
                    # Fog of war — completely unknown
                    row.append(CellObservation(
                        row=r, col=c,
                        fire_state=FireState.UNKNOWN,
                    ))
                    continue

                # Check smoke occlusion
                fire_state = dynamic.fire_state
                if self.config.enable_smoke_occlusion and dynamic.smoke_density > 0.6:
                    if fire_state in (FireState.BURNING, FireState.EMBER, FireState.UNBURNED):
                        fire_state = FireState.UNKNOWN

                # Quantize intensity
                intensity_bin = self._quantize_intensity(dynamic.fire_intensity)

                row.append(CellObservation(
                    row=r, col=c,
                    fire_state=fire_state,
                    intensity_bin=intensity_bin,
                    smoke_density=round(dynamic.smoke_density, 2),
                    is_populated=static.is_populated,
                    crew_present=dynamic.crew_present,
                    fuel_type=static.fuel_type,
                    elevation_m=static.elevation_m,
                ))
            obs_grid.append(row)

        return obs_grid

    # ─── Helpers ──────────────────────────────────────

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    @staticmethod
    def _quantize_intensity(intensity: float) -> IntensityBin:
        if intensity <= 0.0:
            return IntensityBin.NONE
        elif intensity <= 0.25:
            return IntensityBin.LOW
        elif intensity <= 0.5:
            return IntensityBin.MEDIUM
        elif intensity <= 0.75:
            return IntensityBin.HIGH
        else:
            return IntensityBin.EXTREME

    def get_burning_cells(self) -> list[tuple[int, int]]:
        """Return coordinates of all currently burning cells."""
        burning = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.dynamic_grid[r][c].fire_state in (FireState.BURNING, FireState.EMBER):
                    burning.append((r, c))
        return burning

    def get_total_population(self) -> int:
        """Total population across all cells."""
        total = 0
        for r in range(self.rows):
            for c in range(self.cols):
                total += self.static_grid[r][c].population
        return total

    def get_population_lost(self) -> int:
        """Population in burned cells."""
        lost = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.dynamic_grid[r][c].fire_state == FireState.BURNED_OUT:
                    lost += self.static_grid[r][c].population
        return lost

    def get_total_burnable(self) -> int:
        """Count of cells that can burn (not water/road)."""
        count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.static_grid[r][c].fuel_type not in (FuelType.WATER, FuelType.ROAD):
                    count += 1
        return count

    def get_burned_count(self) -> int:
        """Count of cells that have burned out."""
        count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.dynamic_grid[r][c].fire_state == FireState.BURNED_OUT:
                    count += 1
        return count

    def count_by_state(self, state: FireState) -> int:
        """Count cells in a given fire state."""
        count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.dynamic_grid[r][c].fire_state == state:
                    count += 1
        return count

    def get_fire_perimeter(self) -> tuple[int, int]:
        """
        Returns (total_perimeter_edges, contained_edges).
        A perimeter edge is an edge of a burning/ember cell adjacent to a non-burning cell.
        A contained edge borders water, firebreak, burned_out, or grid boundary.
        """
        total = 0
        contained = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.dynamic_grid[r][c].fire_state not in (FireState.BURNING, FireState.EMBER):
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not self._in_bounds(nr, nc):
                        # Grid boundary = contained
                        total += 1
                        contained += 1
                        continue
                    neighbor_state = self.dynamic_grid[nr][nc].fire_state
                    neighbor_fuel = self.static_grid[nr][nc].fuel_type
                    if neighbor_state not in (FireState.BURNING, FireState.EMBER):
                        total += 1
                        if neighbor_state in (FireState.FIREBREAK, FireState.BURNED_OUT, FireState.SUPPRESSED):
                            contained += 1
                        elif neighbor_fuel in (FuelType.WATER, FuelType.ROAD):
                            contained += 1
        return total, contained
