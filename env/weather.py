"""
Stochastic weather engine for the Wildfire Containment Simulator.

Models wind (random walk + shift events), humidity (sinusoidal daily cycle),
and rain (Poisson events with fixed duration).
"""

from __future__ import annotations

import numpy as np

from .models import WeatherState, WeatherObservation, TierConfig


class WeatherEngine:
    """
    Evolves weather state each simulation step.

    Wind: random walk with configurable drift and occasional shift events.
    Humidity: sinusoidal daily cycle with perturbation.
    Rain: Poisson-triggered events that last 5-15 steps.
    """

    def __init__(self, config: TierConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.steps_since_shift = 0

        self.state = WeatherState(
            wind_speed_kmh=config.wind_speed_init,
            wind_direction_deg=config.wind_dir_init,
            humidity_pct=config.humidity_init,
            rain_active=False,
            rain_steps_remaining=0,
        )

    def reset(self) -> None:
        """Reset weather to initial conditions."""
        self.state = WeatherState(
            wind_speed_kmh=self.config.wind_speed_init,
            wind_direction_deg=self.config.wind_dir_init,
            humidity_pct=self.config.humidity_init,
            rain_active=False,
            rain_steps_remaining=0,
        )
        self.steps_since_shift = 0

    def step(self, current_step: int) -> list[str]:
        """
        Advance weather by one step. Returns list of event strings.
        """
        events: list[str] = []
        s = self.state

        # ── Wind speed: random walk ──
        if self.config.tier_name != "easy":
            speed_delta = float(self.rng.normal(0, 2.0))
            s.wind_speed_kmh = float(np.clip(s.wind_speed_kmh + speed_delta, 0, 60))

            # Wind direction: slow drift
            dir_delta = float(self.rng.normal(0, 8.0))
            s.wind_direction_deg = (s.wind_direction_deg + dir_delta) % 360

        # ── Wind shift events ──
        if self.config.enable_wind_shifts:
            self.steps_since_shift += 1
            if self.steps_since_shift >= 50:
                if self.rng.random() < 0.10:
                    shift = self.rng.choice([-90, 90])
                    s.wind_direction_deg = (s.wind_direction_deg + shift) % 360
                    s.wind_speed_kmh = min(60, s.wind_speed_kmh + 10)
                    self.steps_since_shift = 0
                    events.append(
                        f"WIND SHIFT: direction jumped to {s.wind_direction_deg:.0f} deg, "
                        f"speed now {s.wind_speed_kmh:.0f} km/h"
                    )

        # ── Humidity: sinusoidal daily cycle ──
        # Assume 1 step = ~15 min, so 96 steps = 1 day
        day_phase = (current_step % 96) / 96.0  # 0-1 over the day
        base_humidity = self.config.humidity_init
        # Lower at midday (phase ~0.5), higher at dawn/dusk
        import math
        cycle = base_humidity + 15 * math.cos(2 * math.pi * (day_phase - 0.5))
        perturbation = float(self.rng.normal(0, 2.0))
        s.humidity_pct = float(np.clip(cycle + perturbation, 10, 95))

        # ── Rain events ──
        if s.rain_active:
            s.rain_steps_remaining -= 1
            if s.rain_steps_remaining <= 0:
                s.rain_active = False
                events.append("Rain stopped.")
        else:
            # Small chance of rain each step
            rain_prob = 0.005 if self.config.tier_name == "easy" else 0.01
            if self.rng.random() < rain_prob:
                s.rain_active = True
                s.rain_steps_remaining = int(self.rng.integers(5, 16))
                events.append(f"Rain started! Expected duration: {s.rain_steps_remaining} steps.")

        return events

    def get_observation(self) -> WeatherObservation:
        """Return noisy weather observation for the agent."""
        s = self.state

        if self.config.enable_sensor_noise:
            noisy_speed = s.wind_speed_kmh + float(self.rng.normal(0, 5.0))
            noisy_speed = float(np.clip(noisy_speed, 0, 80))

            noisy_dir = s.wind_direction_deg + float(self.rng.normal(0, 20.0))
            noisy_dir = noisy_dir % 360
        else:
            noisy_speed = s.wind_speed_kmh
            noisy_dir = s.wind_direction_deg

        return WeatherObservation(
            wind_speed_kmh=round(noisy_speed, 1),
            wind_direction_deg=round(noisy_dir, 1),
            humidity_pct=round(s.humidity_pct, 1),
            rain_active=s.rain_active,
        )

    def get_true_state(self) -> WeatherState:
        """Return ground-truth weather (for graders/state())."""
        return self.state.model_copy()
