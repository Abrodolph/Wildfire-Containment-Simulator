"""
Frame rendering helpers for episode replay GIFs.
"""

from __future__ import annotations

from typing import List

import numpy as np


def render_frame(state: dict, step: int, stats: dict | None = None) -> np.ndarray:
    """
    Render a ground-truth state dict into an RGB uint8 array (H_px, W_px, 3).

    The figure is 8x8 inches at 100 dpi = 800x800 px.
    Main panel (top 85%): grid. Bottom strip: stats bar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrow
    import io

    grid = state["grid"]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 1

    fig = plt.figure(figsize=(8, 8), dpi=100)
    # Main panel
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.83])
    # Stats strip
    ax_bar = fig.add_axes([0.02, 0.01, 0.96, 0.12])
    ax_bar.axis("off")

    # ── Build colour grid ──
    rgb = np.ones((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            fs = cell["fire_state"]
            intensity = cell.get("fire_intensity", 0.0)
            if fs == "burning":
                sat = 0.4 + 0.6 * intensity
                rgb[r, c] = [1.0, 1.0 - sat * 0.8, 0.0]
            elif fs == "ember":
                rgb[r, c] = [0.9, 0.4, 0.0]
            elif fs == "burned_out":
                rgb[r, c] = [0.25, 0.22, 0.20]
            elif fs == "firebreak":
                rgb[r, c] = [0.55, 0.35, 0.15]
            elif fs == "suppressed":
                rgb[r, c] = [0.6, 0.8, 0.6]
            else:
                # Unburned: shade by fuel
                fuel = cell.get("fuel_type", "grass")
                if fuel == "water":
                    rgb[r, c] = [0.3, 0.5, 0.9]
                elif fuel == "road":
                    rgb[r, c] = [0.7, 0.7, 0.7]
                elif fuel == "timber":
                    rgb[r, c] = [0.1, 0.45, 0.1]
                elif fuel == "shrub":
                    rgb[r, c] = [0.5, 0.7, 0.2]
                elif fuel == "urban":
                    rgb[r, c] = [0.8, 0.75, 0.7]
                else:
                    rgb[r, c] = [0.7, 0.85, 0.4]

    ax.imshow(rgb, origin="upper", aspect="auto", interpolation="nearest")

    # ── Populated cell outlines ──
    for r in range(rows):
        for c in range(cols):
            if grid[r][c].get("is_populated"):
                rect = mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    linewidth=1.5, edgecolor="blue", facecolor="none"
                )
                ax.add_patch(rect)

    # ── Crew markers ──
    resources = state.get("resources", {})
    for crew in resources.get("crews", []):
        if not crew.get("is_deployed") or not crew.get("is_active", True):
            continue
        cr, cc = crew["row"], crew["col"]
        ax.plot(cc, cr, "o", color="lime", markersize=7, markeredgecolor="black", markeredgewidth=0.8)
        ax.text(cc, cr - 0.6, crew["crew_id"].replace("crew_", "c"),
                ha="center", va="bottom", fontsize=5, color="white",
                fontweight="bold")

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Step {step}", fontsize=9, pad=2)

    # ── Stats strip ──
    weather = state.get("weather", {})
    wind_spd = weather.get("wind_speed_kmh", 0)
    wind_dir = weather.get("wind_direction_deg", 0)
    cells_burning = state.get("cells_burning", 0) if stats is None else stats.get("cells_burning", 0)
    containment = state.get("containment_pct", 0) if stats is None else stats.get("containment_pct", 0)
    pop_lost = state.get("population_lost", 0) if stats is None else stats.get("population_lost", 0)

    # Fallback: compute from grid if not in state root
    if cells_burning == 0:
        cells_burning = sum(1 for r in grid for c in r if c["fire_state"] == "burning")

    strip_text = (
        f"Step {step}  |  Burning: {cells_burning}  |  Containment: {containment:.1f}%  |  "
        f"Pop lost: {pop_lost}  |  Wind: {wind_spd:.0f} km/h"
    )
    ax_bar.text(0.5, 0.5, strip_text, ha="center", va="center",
                fontsize=8, transform=ax_bar.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray"))

    # Wind arrow
    import math
    rad = math.radians(wind_dir)
    dx, dy = math.sin(rad) * 0.08, -math.cos(rad) * 0.08
    ax_bar.annotate("", xy=(0.92 + dx, 0.5 + dy), xytext=(0.92 - dx, 0.5 - dy),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))

    # Convert figure to RGB array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    import imageio.v3 as iio
    img = iio.imread(buf, extension=".png")
    return img[:, :, :3].astype(np.uint8)


def render_episode_gif(frames: List[np.ndarray], output_path: str, fps: int = 5) -> None:
    """Stitch RGB frames into an animated GIF at the given fps."""
    import imageio.v3 as iio
    iio.imwrite(output_path, frames, extension=".gif", loop=0,
                duration=int(1000 / fps))
