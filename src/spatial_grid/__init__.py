"""Spatial Grid module for Vidyut Prajna.

Contains H3 hexagonal grid generation, data simulation, and visualization.
"""

from .simulation import (
    CityConfig,
    generate_synthetic_data,
    build_h3_grid,
    build_h3_adjacency,
    NEIGHBORHOOD_ANCHORS,
    DEMO_SCENARIOS,
    ZONE_PARAMS,
)
from .generator import VidyutPrajnaGrid

__all__ = [
    "CityConfig",
    "generate_synthetic_data",
    "build_h3_grid",
    "build_h3_adjacency",
    "VidyutPrajnaGrid",
    "NEIGHBORHOOD_ANCHORS",
    "DEMO_SCENARIOS",
    "ZONE_PARAMS",
]
