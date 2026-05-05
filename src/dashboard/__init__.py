"""Dashboard module for Vidyut Prajna.

Contains the Plotly Dash web application for visualization.
"""

from .app import app, run_server
from .utils import build_geojson, aggregate_load_timeseries, format_kw, format_pct

__all__ = [
    "app",
    "run_server",
    "build_geojson",
    "aggregate_load_timeseries",
    "format_kw",
    "format_pct",
]
