"""Utility functions for the Vidyut Prajna dashboard."""

from __future__ import annotations

from typing import Dict, List

import h3
import numpy as np
import pandas as pd


def h3_cell_to_polygon(cell: str) -> List[List[float]]:
    """Return GeoJSON polygon coordinates [lon, lat] for one H3 cell."""
    boundary = h3.cell_to_boundary(cell)
    coords = [[float(lon), float(lat)] for lat, lon in boundary]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def build_geojson(grid_df: pd.DataFrame) -> Dict:
    """Build GeoJSON FeatureCollection from grid DataFrame."""
    features = []
    for _, row in grid_df.iterrows():
        cell = row["h3_cell"]
        features.append({
            "type": "Feature",
            "id": cell,
            "properties": {
                "h3_cell": cell,
                "zone_name": row.get("zone_name", "Unknown"),
                "zone_type": row.get("zone_type", "Unknown"),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [h3_cell_to_polygon(cell)]
            },
        })
    return {"type": "FeatureCollection", "features": features}


def aggregate_load_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate load data by timestamp across all cells."""
    agg_dict = {
        "baseline_total_load_kw": ("baseline_total_load_kw", "sum"),
        "optimized_total_load_kw": ("optimized_total_load_kw", "sum"),
        "baseline_ev_load_kw": ("baseline_ev_load_kw", "sum"),
        "optimized_ev_load_kw": ("optimized_ev_load_kw", "sum"),
        "safe_capacity_kw": ("transformer_capacity_kw", lambda s: float(0.95 * s.sum())),
    }
    
    if "solar_generation_kw" in df.columns:
        agg_dict["solar_generation_kw"] = ("solar_generation_kw", "sum")
    
    if "prediction_std_kw" in df.columns:
        agg_dict["prediction_std_kw"] = ("prediction_std_kw", lambda s: float(np.sqrt((s ** 2).sum())))
    
    return (
        df.groupby("timestamp", as_index=False)
        .agg(**agg_dict)
        .sort_values("timestamp")
    )


def format_kw(v: float) -> str:
    """Format kilowatts for display."""
    return f"{v:,.0f} kW"


def format_pct(v: float) -> str:
    """Format percentage for display."""
    return f"{v:.1f}%"
