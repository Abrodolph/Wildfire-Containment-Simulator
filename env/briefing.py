"""
Operational briefing system — generates a structured incident briefing on reset().
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Optional, Tuple

from pydantic import BaseModel

if TYPE_CHECKING:
    import numpy as np
    from .grid import Grid
    from .models import TierConfig

_IGNITION_CAUSES = [
    "Lightning strike",
    "Downed power line",
    "Unattended campfire",
    "Equipment spark",
    "Arson (under investigation)",
    "Vehicle exhaust",
]

_INFRA_LABELS = ["North Road", "East Road", "Supply Route", "Evacuation Corridor"]

_WIND_SHIFT_FORECASTS = [
    "Wind shift southwest expected by step 60.",
    "Forecast: wind backing to northwest by step 70, speed increasing.",
    "Weather service warns of sudden direction change near step 65.",
    "Dry front approaching — wind shift likely between steps 55 and 80.",
]

_GENERIC_FORECASTS = [
    "Humidity expected to drop below 20% by mid-episode.",
    "Elevated fire weather conditions through entire operational period.",
    "No precipitation expected. Fire behavior will remain extreme.",
    "Overnight humidity recovery may assist suppression after step 100.",
]


class OperationalBriefing(BaseModel):
    incident_id: str
    ignition_cause: str
    priority_populated_zones: List[Tuple[int, int]]
    priority_infrastructure: List[Tuple[int, int]]
    forecast_events: List[str]
    declared_time: str


def generate_briefing(
    tier_config: "TierConfig",
    rng: "np.random.Generator",
    grid: "Grid",
) -> OperationalBriefing:
    py_rng = random.Random(int(rng.integers(0, 2**31)))

    # Pick top 2 largest populated clusters by population count
    pop_cells: List[Tuple[int, int, int]] = []  # (row, col, population)
    for r in range(grid.rows):
        for c in range(grid.cols):
            static = grid.static_grid[r][c]
            if static.is_populated and static.population > 0:
                pop_cells.append((r, c, static.population))

    pop_cells.sort(key=lambda x: x[2], reverse=True)
    priority_zones = [(r, c) for r, c, _ in pop_cells[:2]]

    # Road cells as infrastructure (up to 2)
    from .models import FuelType
    road_cells = [
        (r, c)
        for r in range(grid.rows)
        for c in range(grid.cols)
        if grid.static_grid[r][c].fuel_type == FuelType.ROAD
    ]
    # Pick a sample spread across the grid
    step = max(1, len(road_cells) // 2)
    infra = road_cells[::step][:2]

    # Forecast events
    forecasts = []
    if tier_config.enable_wind_shifts:
        forecasts.append(py_rng.choice(_WIND_SHIFT_FORECASTS))
    forecasts.append(py_rng.choice(_GENERIC_FORECASTS))

    # Incident ID
    hour = py_rng.randint(0, 23)
    minute = py_rng.choice([0, 15, 30, 45])
    incident_id = f"WF-{tier_config.tier_name.upper()[:3]}-{py_rng.randint(1000, 9999)}"
    declared_time = f"{hour:02d}:{minute:02d}"

    return OperationalBriefing(
        incident_id=incident_id,
        ignition_cause=py_rng.choice(_IGNITION_CAUSES),
        priority_populated_zones=priority_zones,
        priority_infrastructure=infra,
        forecast_events=forecasts,
        declared_time=declared_time,
    )


def briefing_to_text(briefing: OperationalBriefing) -> str:
    zone_list = ", ".join(f"({r},{c})" for r, c in briefing.priority_populated_zones) or "none identified"
    infra_list = ", ".join(f"({r},{c})" for r, c in briefing.priority_infrastructure) or "none identified"
    forecast_lines = "\n".join(f"- {f}" for f in briefing.forecast_events)

    return (
        f"=== OPERATIONAL BRIEFING ===\n"
        f"Incident {briefing.incident_id} declared at {briefing.declared_time}.\n"
        f"Cause: {briefing.ignition_cause}.\n"
        f"\n"
        f"PRIORITY 1: Protect populated zones at {zone_list}.\n"
        f"PRIORITY 2: Maintain routes at {infra_list} open where possible.\n"
        f"\n"
        f"FORECAST:\n"
        f"{forecast_lines}\n"
        f"\n"
        f"Commander's intent: Contain fire with zero civilian casualties. "
        f"Preserve crew safety."
    )
