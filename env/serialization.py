"""
Converts an Observation into a structured text prompt for LLM agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Observation

from .models import FireState, IntensityBin


def serialize_observation(
    obs: "Observation",
    step_num: int,
    max_steps: int,
    tier: str = "",
    prev_cells_burning: int = 0,
) -> str:
    situation = _format_situation(obs, prev_cells_burning)
    grid_summary = _summarize_grid_regions(obs.grid)
    resources = _format_resources(obs.resources)
    events = _format_events(obs.recent_events)

    parts = []

    # Prepend briefing on first observation (step 0)
    if obs.briefing is not None:
        from .briefing import briefing_to_text
        parts.append(briefing_to_text(obs.briefing))
        parts.append("")
    elif step_num > 0 and hasattr(obs, "_briefing_reminder") and obs._briefing_reminder:
        parts.append(obs._briefing_reminder)
        parts.append("")

    tier_str = f" [{tier.upper()}]" if tier else ""
    parts += [
        f"=== WILDFIRE INCIDENT COMMAND{tier_str} — STEP {step_num}/{max_steps} ===",
        "",
        "SITUATION:",
        situation,
        "",
        "GRID SUMMARY (smoke-obscured cells marked [?]):",
        grid_summary,
        "",
        "RESOURCES:",
        resources,
        "",
        "RECENT EVENTS:",
        events,
        "",
        "Available actions: deploy_crew, move_crew, drop_retardant, build_firebreak, recon_flight, idle",
        'Produce your action as JSON: {"action_type": "...", ...}',
    ]
    return "\n".join(parts)


# ── Situation block ──────────────────────────────────────────

def _format_situation(obs: "Observation", prev_cells_burning: int = 0) -> str:
    stats = obs.stats
    w = obs.weather

    burning = stats.cells_burning
    land_saved = round(stats.area_saved_pct, 1)
    civ_safe = round(stats.civilians_saved_pct, 1)
    cells_burned = stats.cells_burned
    pop_at_risk = stats.population_threatened

    wind_dir = _deg_to_compass(w.wind_direction_deg)
    rain = "active" if w.rain_active else "inactive"

    last_event = obs.recent_events[-1] if obs.recent_events else "None"

    # Spread delta — positive means fire is growing, negative means shrinking
    delta = burning - prev_cells_burning
    if delta > 0:
        spread_str = f" (+{delta} spreading)"
    elif delta < 0:
        spread_str = f" ({delta} shrinking)"
    else:
        spread_str = " (stable)"

    lines = [
        f"- Fire active on {burning} cells{spread_str}. Land saved: {land_saved}% of burnable area "
        f"({cells_burned} cells burned out). Civilians safe: {civ_safe}%. "
        f"Population at risk: {pop_at_risk} zones.",
        f"- Wind: {w.wind_speed_kmh:.0f} km/h {wind_dir} (±5 km/h noise). Humidity: {w.humidity_pct:.0f}%. Rain: {rain}.",
        f"- Last event: {last_event}",
    ]
    return "\n".join(lines)


def _deg_to_compass(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(deg / 45.0) % 8
    return dirs[idx]


# ── Grid summary ─────────────────────────────────────────────

def _summarize_grid_regions(grid: list) -> str:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    fire_cells: list[tuple[int, int]] = []
    pop_cells: list[tuple[int, int]] = []
    firebreak_cells: list[tuple[int, int]] = []
    fog_count = 0

    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell.fire_state == FireState.UNKNOWN:
                fog_count += 1
            elif cell.fire_state in (FireState.BURNING, FireState.EMBER):
                fire_cells.append((r, c))
            elif cell.fire_state == FireState.FIREBREAK:
                firebreak_cells.append((r, c))
            if cell.is_populated:
                pop_cells.append((r, c))

    lines: list[str] = []

    fire_regions = _cluster_to_bboxes(fire_cells, max_regions=5)
    for bbox in fire_regions:
        lines.append(f"  FIRE — {bbox}")

    pop_regions = _cluster_to_bboxes(pop_cells, max_regions=5)
    for bbox in pop_regions:
        lines.append(f"  POPULATED — {bbox}")

    fb_regions = _cluster_to_bboxes(firebreak_cells, max_regions=5)
    for bbox in fb_regions:
        lines.append(f"  FIREBREAK — {bbox}")

    if fog_count > 0:
        lines.append(f"  [?] {fog_count} cells obscured by smoke or fog-of-war")

    if not lines:
        lines.append("  No active fire detected.")

    return "\n".join(lines)


def _cluster_to_bboxes(cells: list[tuple[int, int]], max_regions: int) -> list[str]:
    """Group cells into rectangular bounding boxes using a greedy sweep."""
    if not cells:
        return []

    cell_set = set(cells)
    visited: set[tuple[int, int]] = set()
    regions: list[tuple[int, int, int, int, int]] = []  # (size, rmin, rmax, cmin, cmax)

    for seed in cells:
        if seed in visited:
            continue
        r0, c0 = seed
        rmin = rmax = r0
        cmin = cmax = c0
        stack = [seed]
        region_cells: list[tuple[int, int]] = []

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            region_cells.append((r, c))
            rmin, rmax = min(rmin, r), max(rmax, r)
            cmin, cmax = min(cmin, c), max(cmax, c)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in visited:
                    stack.append(nb)

        regions.append((len(region_cells), rmin, rmax, cmin, cmax))

    regions.sort(key=lambda x: -x[0])
    result = []
    for size, rmin, rmax, cmin, cmax in regions[:max_regions]:
        if rmin == rmax and cmin == cmax:
            result.append(f"Row {rmin}, Col {cmin} ({size} cell)")
        else:
            result.append(f"Row {rmin}-{rmax}, Col {cmin}-{cmax} ({size} cells)")
    return result


# ── Resources block ──────────────────────────────────────────

def _format_resources(resources) -> str:
    lines: list[str] = []

    for crew in resources.crews:
        if not crew.is_active:
            status = "CASUALTY"
        elif crew.is_deployed:
            status = f"deployed at ({crew.row},{crew.col}), active"
        else:
            status = "undeployed, available"
        lines.append(f"  {crew.crew_id}: {status}")

    for tanker in resources.tankers:
        if not tanker.is_active:
            t_status = "inactive"
        elif tanker.cooldown_remaining > 0:
            t_status = f"cooldown {tanker.cooldown_remaining} steps remaining"
        else:
            t_status = "ready"
        lines.append(f"  {tanker.tanker_id}: {t_status}")

    lines.append(f"  Firebreaks remaining: {resources.firebreak_budget}. Recon flights remaining: {resources.recon_budget}")
    return "\n".join(lines)


# ── Events block ─────────────────────────────────────────────

def _format_events(events: list[str]) -> str:
    if not events:
        return "  None"
    recent = events[-3:]
    return "\n".join(f"  - {e}" for e in recent)
