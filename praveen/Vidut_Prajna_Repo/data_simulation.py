"""Synthetic data generator for Vidyut Prajna.

Creates a Bengaluru digital twin at H3-cell level with realistic daily structure:
residential/commercial/logistics/IT-corridor demand profiles, weekday/weekend
differentiation, monsoon seasonality, multi-class EV fleet composition,
BESCOM time-of-use tariff signals, and localised transformer capacity limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

import h3
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CityConfig:
    h3_resolution: int = 8
    max_cells: int = 72
    num_days: int = 3
    freq: str = "15min"
    seed: int = 42
    start: str = "2026-01-01"


# ---------------------------------------------------------------------------
# Bengaluru neighbourhood anchors (~20 for broader coverage)
# ---------------------------------------------------------------------------

NEIGHBORHOOD_ANCHORS = [
    {"name": "MG Road",          "lat": 12.9757, "lon": 77.6050, "zone_type": "commercial"},
    {"name": "Indiranagar",      "lat": 12.9719, "lon": 77.6412, "zone_type": "mixed"},
    {"name": "Whitefield",       "lat": 12.9698, "lon": 77.7500, "zone_type": "it_corridor"},
    {"name": "HSR Layout",       "lat": 12.9121, "lon": 77.6446, "zone_type": "residential"},
    {"name": "Electronic City",  "lat": 12.8399, "lon": 77.6770, "zone_type": "it_corridor"},
    {"name": "Peenya",           "lat": 13.0280, "lon": 77.5197, "zone_type": "logistics"},
    {"name": "Hebbal",           "lat": 13.0358, "lon": 77.5970, "zone_type": "mixed"},
    {"name": "Yelahanka",        "lat": 13.1007, "lon": 77.5963, "zone_type": "residential"},
    {"name": "Jayanagar",        "lat": 12.9250, "lon": 77.5938, "zone_type": "residential"},
    {"name": "Koramangala",      "lat": 12.9352, "lon": 77.6245, "zone_type": "mixed"},
    {"name": "Rajajinagar",      "lat": 12.9915, "lon": 77.5546, "zone_type": "residential"},
    {"name": "Outer Ring Road",  "lat": 12.9360, "lon": 77.6900, "zone_type": "it_corridor"},
    {"name": "Marathahalli",     "lat": 12.9591, "lon": 77.7009, "zone_type": "it_corridor"},
    {"name": "BTM Layout",       "lat": 12.9166, "lon": 77.6101, "zone_type": "residential"},
    {"name": "Sarjapur Road",    "lat": 12.9100, "lon": 77.6860, "zone_type": "mixed"},
    {"name": "Bannerghatta Rd",  "lat": 12.8878, "lon": 77.5966, "zone_type": "residential"},
    {"name": "KR Puram",         "lat": 13.0050, "lon": 77.6960, "zone_type": "logistics"},
    {"name": "Majestic",         "lat": 12.9770, "lon": 77.5713, "zone_type": "commercial"},
    {"name": "JP Nagar",         "lat": 12.9063, "lon": 77.5857, "zone_type": "residential"},
    {"name": "Banashankari",     "lat": 12.9255, "lon": 77.5468, "zone_type": "residential"},
]

ZONE_PARAMS = {
    "residential": {
        "capacity_range": (380, 720),
        "station_range": (2, 7),
        "priority_share": (0.12, 0.25),
        "deadline_steps": (16, 28),
    },
    "commercial": {
        "capacity_range": (550, 950),
        "station_range": (4, 12),
        "priority_share": (0.15, 0.30),
        "deadline_steps": (10, 20),
    },
    "logistics": {
        "capacity_range": (500, 900),
        "station_range": (5, 14),
        "priority_share": (0.25, 0.45),
        "deadline_steps": (4, 12),
    },
    "mixed": {
        "capacity_range": (450, 850),
        "station_range": (3, 10),
        "priority_share": (0.18, 0.35),
        "deadline_steps": (8, 18),
    },
    "it_corridor": {
        "capacity_range": (520, 920),
        "station_range": (6, 15),
        "priority_share": (0.14, 0.28),
        "deadline_steps": (10, 22),
    },
}

# EV fleet power draws (kW) and share weights per zone
FLEET_MIX = {
    "residential":  {"2W": (3.3, 0.40), "3W": (7.4, 0.10), "4W": (22.0, 0.45), "bus": (60.0, 0.05)},
    "commercial":   {"2W": (3.3, 0.15), "3W": (7.4, 0.15), "4W": (22.0, 0.55), "bus": (60.0, 0.15)},
    "logistics":    {"2W": (3.3, 0.05), "3W": (7.4, 0.30), "4W": (22.0, 0.25), "bus": (60.0, 0.40)},
    "mixed":        {"2W": (3.3, 0.25), "3W": (7.4, 0.20), "4W": (22.0, 0.40), "bus": (60.0, 0.15)},
    "it_corridor":  {"2W": (3.3, 0.30), "3W": (7.4, 0.05), "4W": (22.0, 0.55), "bus": (60.0, 0.10)},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_hour(hour: float, center: float, width: float) -> float:
    diff = min(abs(hour - center), 24.0 - abs(hour - center))
    return math.exp(-0.5 * (diff / width) ** 2)


def hour_float(ts: pd.Timestamp) -> float:
    return ts.hour + ts.minute / 60.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def nearest_anchor(lat: float, lon: float) -> Dict[str, object]:
    distances = [haversine_km(lat, lon, a["lat"], a["lon"]) for a in NEIGHBORHOOD_ANCHORS]
    return NEIGHBORHOOD_ANCHORS[int(np.argmin(distances))]


def bescom_tariff_multiplier(hour: float) -> float:
    """BESCOM-style ToU tariff multiplier (dimensionless)."""
    if 22.0 <= hour or hour < 6.0:
        return 0.70   # off-peak
    if (6.0 <= hour < 10.0) or (14.0 <= hour < 18.0):
        return 1.00   # mid-peak
    return 1.35       # on-peak (10-14, 18-22)


def solar_generation_kw(hour: float, capacity_kw: float, cloud_factor: float = 1.0) -> float:
    """Simple bell-curve solar generation peaking at noon."""
    if hour < 6.0 or hour > 18.5:
        return 0.0
    solar = capacity_kw * max(0.0, gaussian_hour(hour, 12.5, 3.0)) * cloud_factor
    return float(solar)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_bengaluru_h3_grid(config: CityConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    cells: set[str] = set()
    for anchor in NEIGHBORHOOD_ANCHORS:
        center = h3.latlng_to_cell(anchor["lat"], anchor["lon"], config.h3_resolution)
        cells.update(h3.grid_disk(center, 1))
    ordered_cells = sorted(cells)
    if len(ordered_cells) > config.max_cells:
        ordered_cells = sorted(rng.choice(ordered_cells, size=config.max_cells, replace=False).tolist())

    records = []
    for idx, cell in enumerate(ordered_cells):
        lat, lon = h3.cell_to_latlng(cell)
        anchor = nearest_anchor(lat, lon)
        zone_type = str(anchor["zone_type"])
        params = ZONE_PARAMS[zone_type]
        capacity_kw = float(rng.uniform(*params["capacity_range"]))
        station_count = int(rng.integers(params["station_range"][0], params["station_range"][1] + 1))
        priority_share = float(rng.uniform(*params["priority_share"]))
        deadline_steps = int(rng.integers(params["deadline_steps"][0], params["deadline_steps"][1] + 1))
        ev_adoption = float(rng.uniform(0.35, 1.15))
        charger_density = station_count / max(haversine_km(lat, lon, anchor["lat"], anchor["lon"]), 0.8)
        solar_capacity = float(rng.uniform(15, 80))  # rooftop solar kW at this cell
        records.append({
            "h3_cell": cell, "lat": lat, "lon": lon,
            "zone_name": str(anchor["name"]), "zone_type": zone_type,
            "transformer_capacity_kw": capacity_kw,
            "station_count": station_count, "priority_share": priority_share,
            "deadline_steps": deadline_steps, "ev_adoption_index": ev_adoption,
            "charger_density_index": charger_density,
            "solar_capacity_kw": solar_capacity,
            "cell_rank": idx,
        })
    return pd.DataFrame.from_records(records)


def build_h3_adjacency(cells: Iterable[str]) -> Dict[str, List[str]]:
    cell_set = set(cells)
    adjacency: Dict[str, List[str]] = {}
    for cell in cell_set:
        nbrs = [n for n in h3.grid_disk(cell, 1) if n in cell_set and n != cell]
        adjacency[cell] = sorted(nbrs)
    return adjacency


# ---------------------------------------------------------------------------
# Weather, traffic, demand profiles
# ---------------------------------------------------------------------------

def _weather_for_timestamps(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for day, day_ts in pd.Series(timestamps).groupby(pd.Series(timestamps).dt.date):
        rain_center = float(rng.choice([16.5, 18.0, 20.0]))
        rain_strength = float(rng.choice([0.0, 0.0, 0.8, 1.5, 2.5]))
        # Monsoon boost for Jun-Sep
        month = pd.Timestamp(day).month
        if 6 <= month <= 9:
            rain_strength *= 2.2
        cloud_factor = max(0.3, 1.0 - 0.25 * rain_strength)
        for ts in day_ts:
            h = hour_float(pd.Timestamp(ts))
            temp = 25.5 + 5.5 * math.sin(2 * math.pi * (h - 8) / 24) + rng.normal(0, 0.45)
            if 3 <= month <= 5:  # pre-monsoon heat
                temp += 3.5
            rainfall = max(0.0, rain_strength * gaussian_hour(h, rain_center, 1.8) + rng.normal(0, 0.03))
            rows.append({
                "timestamp": pd.Timestamp(ts), "temperature_c": temp,
                "rainfall_mm": rainfall, "cloud_factor": cloud_factor,
            })
    return pd.DataFrame(rows)


def _traffic_intensity(hour: float, zone_type: str, rainfall_mm: float,
                       is_weekend: bool, rng: np.random.Generator) -> float:
    commute = 0.42 * gaussian_hour(hour, 8.5, 1.25) + 0.52 * gaussian_hour(hour, 18.5, 1.55)
    midday = 0.18 * gaussian_hour(hour, 13.0, 2.5)
    zone_bonus = {"residential": 0.03, "commercial": 0.11, "logistics": 0.16,
                  "mixed": 0.09, "it_corridor": 0.14}[zone_type]
    rain_bonus = min(0.12, 0.045 * rainfall_mm)
    weekend_factor = 0.65 if is_weekend and zone_type in ("commercial", "it_corridor", "logistics") else 1.0
    if is_weekend and zone_type == "residential":
        weekend_factor = 1.15
    value = (0.18 + commute + midday + zone_bonus + rain_bonus + rng.normal(0, 0.035)) * weekend_factor
    return float(np.clip(value, 0.05, 1.0))


def _ev_profile(hour: float, zone_type: str, is_weekend: bool) -> float:
    """Dimensionless EV charging demand shape by land-use archetype."""
    weekend_res_boost = 1.25 if is_weekend else 1.0
    weekend_work_damp = 0.55 if is_weekend else 1.0

    if zone_type == "residential":
        return weekend_res_boost * (0.20 + 1.25 * gaussian_hour(hour, 20.0, 2.2) + 0.28 * gaussian_hour(hour, 7.5, 1.3))
    if zone_type == "commercial":
        return weekend_work_damp * (0.22 + 0.92 * gaussian_hour(hour, 12.8, 3.0) + 0.35 * gaussian_hour(hour, 17.5, 2.0))
    if zone_type == "logistics":
        return weekend_work_damp * (0.30 + 0.88 * gaussian_hour(hour, 11.0, 1.8) + 1.05 * gaussian_hour(hour, 18.0, 1.9) + 0.42 * gaussian_hour(hour, 23.0, 1.8))
    if zone_type == "it_corridor":
        # IT parks: strong 10AM-8PM plateau, drops on weekends
        return weekend_work_damp * (0.18 + 1.10 * gaussian_hour(hour, 13.0, 3.5) + 0.65 * gaussian_hour(hour, 18.5, 2.0))
    # mixed
    return 0.24 + 0.55 * gaussian_hour(hour, 13.0, 2.8) + 0.98 * gaussian_hour(hour, 19.0, 2.3)


def _base_grid_profile(hour: float, zone_type: str) -> float:
    """Transformer non-EV loading as a fraction of capacity."""
    if zone_type == "residential":
        return 0.34 + 0.32 * gaussian_hour(hour, 20.5, 2.8) + 0.08 * gaussian_hour(hour, 7.5, 1.5)
    if zone_type == "commercial":
        return 0.38 + 0.31 * gaussian_hour(hour, 13.0, 4.0) + 0.10 * gaussian_hour(hour, 18.0, 1.8)
    if zone_type == "logistics":
        return 0.35 + 0.24 * gaussian_hour(hour, 11.5, 2.7) + 0.22 * gaussian_hour(hour, 18.0, 2.3)
    if zone_type == "it_corridor":
        return 0.40 + 0.28 * gaussian_hour(hour, 13.5, 3.5) + 0.15 * gaussian_hour(hour, 18.0, 2.0)
    return 0.36 + 0.22 * gaussian_hour(hour, 12.5, 3.8) + 0.24 * gaussian_hour(hour, 19.0, 2.4)


def _weighted_fleet_power(zone_type: str, rng: np.random.Generator) -> float:
    """Compute effective per-session kW draw from fleet mix."""
    mix = FLEET_MIX.get(zone_type, FLEET_MIX["mixed"])
    total = 0.0
    for _class, (kw, share) in mix.items():
        total += kw * share
    return total * (1.0 + rng.normal(0, 0.08))


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_city_data(config: CityConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """Generate synthetic time-series data, H3 grid metadata, and graph adjacency."""
    config = config or CityConfig()
    rng = np.random.default_rng(config.seed)
    grid_df = build_bengaluru_h3_grid(config)
    adjacency = build_h3_adjacency(grid_df["h3_cell"].tolist())

    periods = int(pd.Timedelta(days=config.num_days) / pd.Timedelta(config.freq))
    timestamps = pd.date_range(pd.Timestamp(config.start), periods=periods, freq=config.freq)
    weather_df = _weather_for_timestamps(timestamps, rng)

    rows: List[Dict[str, object]] = []
    for _, cell in grid_df.iterrows():
        cell_multiplier = float(rng.lognormal(mean=0.0, sigma=0.18))
        capacity_kw = float(cell["transformer_capacity_kw"])
        station_count = int(cell["station_count"])
        zone_type = str(cell["zone_type"])
        ev_adoption = float(cell["ev_adoption_index"])
        solar_cap = float(cell["solar_capacity_kw"])
        fleet_power = _weighted_fleet_power(zone_type, rng)

        for _, w in weather_df.iterrows():
            ts = pd.Timestamp(w["timestamp"])
            h = hour_float(ts)
            rainfall = float(w["rainfall_mm"])
            temperature = float(w["temperature_c"])
            cloud = float(w["cloud_factor"])
            is_weekend = ts.dayofweek >= 5
            traffic = _traffic_intensity(h, zone_type, rainfall, is_weekend, rng)
            tariff = bescom_tariff_multiplier(h)
            solar_gen = solar_generation_kw(h, solar_cap, cloud)

            weather_impact = 1.0 + 0.055 * rainfall + 0.012 * max(0.0, temperature - 30.0)
            traffic_impact = 1.0 + 0.33 * traffic
            demand_shape = _ev_profile(h, zone_type, is_weekend)

            # Scale demand using fleet-weighted power and station count
            demand_kw = (fleet_power * 0.35 + 8.5 * station_count) * demand_shape * weather_impact * traffic_impact * ev_adoption * cell_multiplier
            demand_kw += rng.normal(0, 4.0 + 0.05 * demand_kw)
            demand_kw = float(max(3.0, demand_kw))

            base_grid_fraction = _base_grid_profile(h, zone_type) + 0.035 * rainfall + rng.normal(0, 0.015)
            grid_base_load_kw = float(capacity_kw * np.clip(base_grid_fraction, 0.20, 0.90))
            total_unmanaged_kw = grid_base_load_kw + demand_kw

            rows.append({
                "timestamp": ts, "h3_cell": cell["h3_cell"],
                "lat": float(cell["lat"]), "lon": float(cell["lon"]),
                "zone_name": cell["zone_name"], "zone_type": zone_type,
                "demand_kw": demand_kw, "traffic_intensity": traffic,
                "temperature_c": temperature, "rainfall_mm": rainfall,
                "grid_base_load_kw": grid_base_load_kw,
                "unmanaged_total_load_kw": total_unmanaged_kw,
                "transformer_capacity_kw": capacity_kw,
                "station_count": station_count,
                "priority_share": float(cell["priority_share"]),
                "deadline_steps": int(cell["deadline_steps"]),
                "ev_adoption_index": ev_adoption,
                "charger_density_index": float(cell["charger_density_index"]),
                "tariff_multiplier": tariff,
                "solar_generation_kw": solar_gen,
                "is_weekend": int(is_weekend),
            })

    data_df = pd.DataFrame(rows).sort_values(["timestamp", "h3_cell"]).reset_index(drop=True)
    return data_df, grid_df, adjacency


if __name__ == "__main__":
    df, grid, adj = generate_synthetic_city_data(CityConfig(max_cells=24, num_days=1))
    print(df.head())
    print(f"Generated {len(df):,} rows, {len(grid)} H3 cells, {len(adj)} graph nodes")
