"""
Pydantic data models for the Wildfire Containment Simulator.

This module defines the complete type contract between all environment components.
Every action, observation, cell state, and result is typed and validated here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ══════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════

class FuelType(str, Enum):
    """Terrain fuel classification. Determines burn rate and ignition probability."""
    GRASS = "grass"
    SHRUB = "shrub"
    TIMBER = "timber"
    URBAN = "urban"
    WATER = "water"
    ROAD = "road"


class FireState(str, Enum):
    """Current fire status of a grid cell."""
    UNBURNED = "unburned"
    BURNING = "burning"
    EMBER = "ember"          # Low intensity, dying down
    BURNED_OUT = "burned_out"
    FIREBREAK = "firebreak"  # Manually constructed, non-flammable
    SUPPRESSED = "suppressed"  # Was burning, now extinguished by crew
    UNKNOWN = "unknown"      # Hidden by smoke/fog-of-war


class Priority(str, Enum):
    """Job/event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Direction(str, Enum):
    """8-directional movement for crews."""
    N = "N"
    S = "S"
    E = "E"
    W = "W"
    NE = "NE"
    NW = "NW"
    SE = "SE"
    SW = "SW"


class ActionType(str, Enum):
    """All possible agent actions."""
    DEPLOY_CREW = "deploy_crew"
    MOVE_CREW = "move_crew"
    ORDER_CREW_OBJECTIVE = "order_crew_objective"
    DROP_RETARDANT = "drop_retardant"
    BUILD_FIREBREAK = "build_firebreak"
    RECON_FLIGHT = "recon_flight"
    IDLE = "idle"


class CrewObjective(str, Enum):
    """Objective directive for ORDER_CREW_OBJECTIVE."""
    HOLD = "hold"
    ADVANCE = "advance"
    RETREAT = "retreat"
    PRIORITIZE_NORTH = "prioritize_north"
    PRIORITIZE_SOUTH = "prioritize_south"
    PRIORITIZE_EAST = "prioritize_east"
    PRIORITIZE_WEST = "prioritize_west"


class IntensityBin(str, Enum):
    """Quantized fire intensity as seen by the agent."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# ══════════════════════════════════════════════════════
# DIRECTION HELPERS
# ══════════════════════════════════════════════════════

DIRECTION_DELTAS: dict[Direction, tuple[int, int]] = {
    Direction.N:  (-1, 0),
    Direction.S:  (1, 0),
    Direction.E:  (0, 1),
    Direction.W:  (0, -1),
    Direction.NE: (-1, 1),
    Direction.NW: (-1, -1),
    Direction.SE: (1, 1),
    Direction.SW: (1, -1),
}


# ══════════════════════════════════════════════════════
# CELL MODELS
# ══════════════════════════════════════════════════════

class CellStatic(BaseModel):
    """Immutable terrain properties of a grid cell."""
    row: int
    col: int
    elevation_m: float = Field(ge=0, le=2000, description="Height in meters")
    fuel_type: FuelType
    fuel_load: float = Field(ge=0.0, le=1.0, description="Density of burnable material")
    is_populated: bool = False
    population: int = Field(ge=0, default=0)
    is_water: bool = False

    @model_validator(mode="after")
    def water_consistency(self) -> "CellStatic":
        if self.fuel_type == FuelType.WATER:
            self.is_water = True
            self.fuel_load = 0.0
        if self.fuel_type == FuelType.ROAD:
            self.fuel_load = 0.0
        return self


class CellDynamic(BaseModel):
    """Mutable runtime state of a grid cell. Updated each step."""
    fire_state: FireState = FireState.UNBURNED
    fire_intensity: float = Field(ge=0.0, le=1.0, default=0.0)
    moisture: float = Field(ge=0.0, le=1.0, default=0.3)
    time_burning: int = Field(ge=0, default=0)
    suppression_level: float = Field(ge=0.0, le=1.0, default=0.0)
    smoke_density: float = Field(ge=0.0, le=1.0, default=0.0)
    crew_present: bool = False


class CellObservation(BaseModel):
    """What the agent sees for a single cell (may be degraded by smoke/fog)."""
    row: int
    col: int
    fire_state: FireState
    intensity_bin: IntensityBin = IntensityBin.NONE
    smoke_density: float = 0.0
    is_populated: bool = False
    crew_present: bool = False
    fuel_type: FuelType = FuelType.GRASS
    elevation_m: float = 0.0


# ══════════════════════════════════════════════════════
# WEATHER MODELS
# ══════════════════════════════════════════════════════

class WeatherState(BaseModel):
    """Full ground-truth weather (used internally)."""
    wind_speed_kmh: float = Field(ge=0, le=60, default=10.0)
    wind_direction_deg: float = Field(ge=0, lt=360, default=0.0)
    humidity_pct: float = Field(ge=0, le=100, default=40.0)
    rain_active: bool = False
    rain_steps_remaining: int = 0


class WeatherObservation(BaseModel):
    """Noisy weather readings visible to the agent."""
    wind_speed_kmh: float  # +/- 5 km/h noise
    wind_direction_deg: float  # +/- 20 deg noise
    humidity_pct: float  # Exact
    rain_active: bool  # Observable


# ══════════════════════════════════════════════════════
# RESOURCE MODELS
# ══════════════════════════════════════════════════════

class CrewState(BaseModel):
    """State of a single ground crew."""
    crew_id: str
    row: int
    col: int
    is_deployed: bool = False
    is_active: bool = True  # False if crew lost (injury)


class TankerState(BaseModel):
    """State of a single air tanker."""
    tanker_id: str
    cooldown_remaining: int = 0  # 0 = ready to drop
    is_active: bool = True


class ResourceState(BaseModel):
    """Complete resource state visible to the agent."""
    crews: list[CrewState]
    tankers: list[TankerState]
    firebreak_budget: int = Field(ge=0, description="Remaining firebreak cells")
    recon_budget: int = Field(ge=0, default=0, description="Remaining recon flights")


# ══════════════════════════════════════════════════════
# ACTION MODEL
# ══════════════════════════════════════════════════════

class Action(BaseModel):
    """
    Agent action. One action per step.

    Validation catches invalid actions at the type level.
    Semantic validation (VRAM-like feasibility checks) happens in the environment.
    """
    action_type: ActionType

    # DEPLOY_CREW / DROP_RETARDANT / RECON_FLIGHT params
    target_row: Optional[int] = None
    target_col: Optional[int] = None

    # DEPLOY_CREW / MOVE_CREW / BUILD_FIREBREAK params
    crew_id: Optional[str] = None

    # MOVE_CREW / BUILD_FIREBREAK params
    direction: Optional[Direction] = None

    # DROP_RETARDANT params
    tanker_id: Optional[str] = None

    # ORDER_CREW_OBJECTIVE params
    objective: Optional[CrewObjective] = None

    # IDLE params
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_params(self) -> "Action":
        """Ensure required parameters are present for each action type."""
        t = self.action_type

        if t == ActionType.DEPLOY_CREW:
            if self.crew_id is None:
                raise ValueError("DEPLOY_CREW requires crew_id")
            if self.target_row is None or self.target_col is None:
                raise ValueError("DEPLOY_CREW requires target_row and target_col")

        elif t == ActionType.MOVE_CREW:
            if self.crew_id is None:
                raise ValueError("MOVE_CREW requires crew_id")
            if self.direction is None:
                raise ValueError("MOVE_CREW requires direction")

        elif t == ActionType.ORDER_CREW_OBJECTIVE:
            if self.crew_id is None:
                raise ValueError("ORDER_CREW_OBJECTIVE requires crew_id")
            if self.objective is None:
                raise ValueError("ORDER_CREW_OBJECTIVE requires objective")

        elif t == ActionType.DROP_RETARDANT:
            if self.tanker_id is None:
                raise ValueError("DROP_RETARDANT requires tanker_id")
            if self.target_row is None or self.target_col is None:
                raise ValueError("DROP_RETARDANT requires target_row and target_col")

        elif t == ActionType.BUILD_FIREBREAK:
            if self.crew_id is None:
                raise ValueError("BUILD_FIREBREAK requires crew_id")
            if self.direction is None:
                raise ValueError("BUILD_FIREBREAK requires direction")

        elif t == ActionType.RECON_FLIGHT:
            if self.target_row is None or self.target_col is None:
                raise ValueError("RECON_FLIGHT requires target_row and target_col")

        return self


# ══════════════════════════════════════════════════════
# OBSERVATION MODEL
# ══════════════════════════════════════════════════════

class ClusterStats(BaseModel):
    """Running statistics about the episode."""
    cells_burned: int = 0
    cells_burning: int = 0
    cells_saved: int = 0
    population_threatened: int = 0
    population_lost: int = 0
    total_population: int = Field(ge=0, default=0, description="Initial population (for UI % civ safe)")
    containment_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    # Meaningful progress metrics shown to agent and display
    area_saved_pct: float = Field(ge=0.0, le=100.0, default=100.0,
        description="Percentage of burnable land not yet burned")
    civilians_saved_pct: float = Field(ge=0.0, le=100.0, default=100.0,
        description="Percentage of civilians in unburned zones")
    current_step: int = 0
    max_steps: int = 100
    firebreaks_built: int = 0
    retardant_drops: int = 0


class Observation(BaseModel):
    """Complete observation returned to the agent each step."""
    grid: list[list[CellObservation]]
    weather: WeatherObservation
    resources: ResourceState
    stats: ClusterStats
    recent_events: list[str] = Field(default_factory=list, max_length=5)
    briefing: Optional[Any] = None  # OperationalBriefing on first obs, None thereafter


# ══════════════════════════════════════════════════════
# STEP RESULT
# ══════════════════════════════════════════════════════

class StepResult(BaseModel):
    """Returned by env.step(). Contains everything the agent needs."""
    observation: Observation
    reward: float
    done: bool = False
    info: dict = Field(default_factory=dict)


# ══════════════════════════════════════════════════════
# TIER CONFIGURATION
# ══════════════════════════════════════════════════════

class TierConfig(BaseModel):
    """Configuration for a difficulty tier."""
    tier_name: str
    grid_rows: int
    grid_cols: int
    num_crews: int
    num_tankers: int
    firebreak_budget: int
    recon_budget: int = 0
    episode_length: int
    num_ignition_points: int = 1
    staggered_ignition_step: Optional[int] = None  # Step at which extra ignition(s) start
    enable_smoke_occlusion: bool = False
    enable_sensor_noise: bool = False
    enable_fog_of_war: bool = False
    fog_visibility_radius: int = 7
    enable_wind_shifts: bool = False
    enable_crew_loss: bool = False
    crew_loss_step: Optional[int] = None
    crew_loss_id: Optional[str] = None
    tanker_cooldown: int = 5
    min_active_steps: int = 5   # episode cannot end via fire-out before this step
    wind_speed_init: float = 10.0
    wind_dir_init: float = 0.0
    humidity_init: float = 40.0

    # Reward weights
    w_containment: float = 0.30
    w_population: float = 0.35
    w_efficiency: float = 0.10
    w_speed: float = 0.15
    w_area: float = 0.10


# ══════════════════════════════════════════════════════
# PRESET TIER CONFIGS
# ══════════════════════════════════════════════════════

TIER_EASY = TierConfig(
    tier_name="easy",
    grid_rows=15,
    grid_cols=15,
    num_crews=4,
    num_tankers=1,
    firebreak_budget=15,
    recon_budget=0,
    episode_length=80,
    num_ignition_points=2,
    enable_smoke_occlusion=False,
    enable_sensor_noise=False,
    enable_fog_of_war=False,
    enable_wind_shifts=False,
    min_active_steps=25,
    wind_speed_init=10.0,
    wind_dir_init=0.0,
    humidity_init=40.0,
    w_containment=0.30,
    w_population=0.35,
    w_efficiency=0.10,
    w_speed=0.15,
    w_area=0.10,
)

TIER_MEDIUM = TierConfig(
    tier_name="medium",
    grid_rows=25,
    grid_cols=25,
    num_crews=5,
    num_tankers=2,
    firebreak_budget=20,
    recon_budget=1,
    episode_length=150,
    num_ignition_points=3,
    enable_smoke_occlusion=True,
    enable_sensor_noise=True,
    enable_fog_of_war=False,
    enable_wind_shifts=True,
    min_active_steps=45,
    wind_speed_init=15.0,
    wind_dir_init=45.0,
    humidity_init=35.0,
    w_containment=0.25,
    w_population=0.35,
    w_efficiency=0.15,
    w_speed=0.10,
    w_area=0.15,
)

TIER_HARD = TierConfig(
    tier_name="hard",
    grid_rows=40,
    grid_cols=40,
    num_crews=6,
    num_tankers=3,
    firebreak_budget=30,
    recon_budget=3,
    episode_length=300,
    num_ignition_points=3,
    staggered_ignition_step=30,
    min_active_steps=80,
    enable_smoke_occlusion=True,
    enable_sensor_noise=True,
    enable_fog_of_war=True,
    fog_visibility_radius=7,
    enable_wind_shifts=True,
    enable_crew_loss=True,
    crew_loss_step=40,
    crew_loss_id="crew_5",
    wind_speed_init=20.0,
    wind_dir_init=90.0,
    humidity_init=30.0,
    w_containment=0.20,
    w_population=0.40,
    w_efficiency=0.15,
    w_speed=0.10,
    w_area=0.15,
)
