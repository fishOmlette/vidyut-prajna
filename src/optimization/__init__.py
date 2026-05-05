"""Optimization module for Vidyut Prajna.

Contains EV charging schedule optimization with tariff, solar, and constraint awareness.
"""

from .optimizer import (
    optimize_charging_schedule,
    stress_label,
    derated_capacity,
    GRID_CO2_INTENSITY,
    SOLAR_CO2_INTENSITY,
    BASE_TARIFF_INR_KWH,
)
from .siting import recommend_station_locations

__all__ = [
    "optimize_charging_schedule",
    "stress_label",
    "derated_capacity",
    "GRID_CO2_INTENSITY",
    "SOLAR_CO2_INTENSITY",
    "BASE_TARIFF_INR_KWH",
    "recommend_station_locations",
]
