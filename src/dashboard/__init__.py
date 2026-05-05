"""Dashboard module for Vidyut Prajna.

Contains the Plotly Dash web application for visualization.
"""

from .utils import build_geojson, aggregate_load_timeseries, format_inr, format_kw, format_pct


def __getattr__(name: str):
    if name in {"app", "run_server"}:
        from .app import app, run_server

        return {"app": app, "run_server": run_server}[name]
    raise AttributeError(name)


__all__ = [
    "app",
    "run_server",
    "build_geojson",
    "aggregate_load_timeseries",
    "format_inr",
    "format_kw",
    "format_pct",
]
