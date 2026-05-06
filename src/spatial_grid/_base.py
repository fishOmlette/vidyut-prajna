"""Bengaluru city configuration and data simulation.

Inspired by Praveen's data_simulation.py but adapted to work with
the existing spatial_grid module structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import h3
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CityConfig:
    """Configuration for Bengaluru simulation."""
    h3_resolution: int = 8
    max_cells: int = 54
    num_days: int = 3
    freq: str = "30min"
    seed: int = 42
    start: str = "2026-05-01"
    scenario: str = "orr_whitefield"


# Bengaluru neighborhood anchors with zone types
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

DEMO_SCENARIOS = {
    "orr_whitefield": {
        "label": "ORR-Whitefield IT charging corridor",
        "route": ["HSR Layout", "Sarjapur Road", "Outer Ring Road", "Marathahalli", "Whitefield"],
    },
    "south_residential": {
        "label": "South Bengaluru residential charging belt",
        "route": ["Banashankari", "JP Nagar", "Jayanagar", "BTM Layout", "HSR Layout"],
    },
    "central_commercial": {
        "label": "Central commercial-mixed demand cluster",
        "route": ["Majestic", "MG Road", "Indiranagar", "Koramangala"],
    },
}

# Zone-specific parameters
ZONE_PARAMS = {
    "residential": {
        "capacity_range": (380, 720),       # Transformer capacity kVA
        "station_range": (2, 7),             # EV charging stations
        "priority_share": (0.12, 0.25),      # Non-shiftable charging fraction
        "deadline_steps": (16, 28),          # Charging deadline window
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


def gaussian_hour(hour: float, center: float, width: float) -> float:
    """Gaussian time-of-day profile."""
    diff = min(abs(hour - center), 24.0 - abs(hour - center))
    return math.exp(-0.5 * (diff / width) ** 2)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in km."""
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def nearest_anchor(lat: float, lon: float) -> Dict:
    """Find nearest neighborhood anchor."""
    distances = [haversine_km(lat, lon, a["lat"], a["lon"]) for a in NEIGHBORHOOD_ANCHORS]
    return NEIGHBORHOOD_ANCHORS[int(np.argmin(distances))]


def _anchor_by_name(name: str) -> Dict:
    for anchor in NEIGHBORHOOD_ANCHORS:
        if anchor["name"] == name:
            return anchor
    raise KeyError(f"Unknown Bengaluru anchor: {name}")


def bescom_tariff_multiplier(hour: float) -> float:
    """BESCOM-style ToU tariff multiplier."""
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


def _grid_path_cells(start_cell: str, end_cell: str) -> List[str]:
    """Return a contiguous H3 path, with interpolation fallback for rare path failures."""
    try:
        return list(h3.grid_path_cells(start_cell, end_cell))
    except Exception:
        start_lat, start_lon = h3.cell_to_latlng(start_cell)
        end_lat, end_lon = h3.cell_to_latlng(end_cell)
        distance_km = haversine_km(start_lat, start_lon, end_lat, end_lon)
        samples = max(2, int(distance_km / 0.75))
        cells: List[str] = []
        for i in range(samples + 1):
            frac = i / samples
            lat = start_lat + frac * (end_lat - start_lat)
            lon = start_lon + frac * (end_lon - start_lon)
            cell = h3.latlng_to_cell(lat, lon, h3.get_resolution(start_cell))
            if not cells or cells[-1] != cell:
                cells.append(cell)
        return cells


def _scenario_route_cells(config: CityConfig) -> Tuple[List[str], str]:
    """Build an ordered, contiguous corridor for a hackathon-scale demo."""
    scenario = DEMO_SCENARIOS.get(config.scenario, DEMO_SCENARIOS["orr_whitefield"])
    route = [_anchor_by_name(name) for name in scenario["route"]]
    anchor_cells = [
        h3.latlng_to_cell(anchor["lat"], anchor["lon"], config.h3_resolution)
        for anchor in route
    ]

    ordered: List[str] = []
    seen: set[str] = set()
    for start_cell, end_cell in zip(anchor_cells, anchor_cells[1:]):
        for cell in _grid_path_cells(start_cell, end_cell):
            if cell not in seen:
                ordered.append(cell)
                seen.add(cell)

    if not ordered:
        ordered = [anchor_cells[0]]
        seen.add(anchor_cells[0])

    queue = list(ordered)
    for cell in queue:
        if len(ordered) >= config.max_cells:
            break
        for nbr in sorted(h3.grid_disk(cell, 1)):
            if nbr not in seen:
                ordered.append(nbr)
                seen.add(nbr)
                queue.append(nbr)
            if len(ordered) >= config.max_cells:
                break

    return ordered[: config.max_cells], str(scenario["label"])


def build_h3_grid(config: CityConfig) -> pd.DataFrame:
    """Build a contiguous H3 grid around the configured Bengaluru scenario."""
    rng = np.random.default_rng(config.seed)
    ordered_cells, scenario_label = _scenario_route_cells(config)

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
        solar_capacity = float(rng.uniform(15, 80))
        growth_bonus = {"residential": 0.08, "commercial": 0.06, "logistics": 0.10,
                        "mixed": 0.11, "it_corridor": 0.16}[zone_type]
        demand_growth_index = float(np.clip(rng.normal(0.18 + growth_bonus, 0.045), 0.08, 0.45))
        
        records.append({
            "h3_cell": cell,
            "lat": lat,
            "lon": lon,
            "zone_name": str(anchor["name"]),
            "zone_type": zone_type,
            "scenario": config.scenario,
            "corridor_name": scenario_label,
            "transformer_capacity_kw": capacity_kw,
            "station_count": station_count,
            "priority_share": priority_share,
            "deadline_steps": deadline_steps,
            "ev_adoption_index": ev_adoption,
            "charger_density_index": charger_density,
            "demand_growth_index": demand_growth_index,
            "solar_capacity_kw": solar_capacity,
            "cell_rank": idx,
        })
    
    return pd.DataFrame.from_records(records)


def build_h3_adjacency(cells: Iterable[str]) -> Dict[str, List[str]]:
    """Build adjacency dictionary for H3 cells."""
    cell_set = set(cells)
    adjacency: Dict[str, List[str]] = {}
    for cell in cell_set:
        nbrs = [n for n in h3.grid_disk(cell, 1) if n in cell_set and n != cell]
        adjacency[cell] = sorted(nbrs)
    return adjacency


def _ev_profile(hour: float, zone_type: str, is_weekend: bool) -> float:
    """EV charging demand profile by zone type and time."""
    weekend_res_boost = 1.25 if is_weekend else 1.0
    weekend_work_damp = 0.55 if is_weekend else 1.0

    if zone_type == "residential":
        return weekend_res_boost * (0.20 + 1.25 * gaussian_hour(hour, 20.0, 2.2) + 0.28 * gaussian_hour(hour, 7.5, 1.3))
    if zone_type == "commercial":
        return weekend_work_damp * (0.22 + 0.92 * gaussian_hour(hour, 12.8, 3.0) + 0.35 * gaussian_hour(hour, 17.5, 2.0))
    if zone_type == "logistics":
        return weekend_work_damp * (0.30 + 0.88 * gaussian_hour(hour, 11.0, 1.8) + 1.05 * gaussian_hour(hour, 18.0, 1.9))
    if zone_type == "it_corridor":
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


def _traffic_intensity(hour: float, zone_type: str, is_weekend: bool, rng: np.random.Generator) -> float:
    """Traffic intensity profile."""
    commute = 0.42 * gaussian_hour(hour, 8.5, 1.25) + 0.52 * gaussian_hour(hour, 18.5, 1.55)
    midday = 0.18 * gaussian_hour(hour, 13.0, 2.5)
    zone_bonus = {"residential": 0.03, "commercial": 0.11, "logistics": 0.16,
                  "mixed": 0.09, "it_corridor": 0.14}[zone_type]
    weekend_factor = 0.65 if is_weekend and zone_type in ("commercial", "it_corridor", "logistics") else 1.0
    if is_weekend and zone_type == "residential":
        weekend_factor = 1.15
    value = (0.18 + commute + midday + zone_bonus + rng.normal(0, 0.035)) * weekend_factor
    return float(np.clip(value, 0.05, 1.0))


def _weighted_fleet_power(zone_type: str, rng: np.random.Generator) -> float:
    """Compute effective per-session kW draw from fleet mix."""
    mix = FLEET_MIX.get(zone_type, FLEET_MIX["mixed"])
    total = 0.0
    for _class, (kw, share) in mix.items():
        total += kw * share
    return total * (1.0 + rng.normal(0, 0.08))


def generate_synthetic_data(config: CityConfig = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """Generate synthetic time-series data, grid metadata, and adjacency.
    
    Returns:
        raw_df: Time-series data with demand and features
        grid_df: H3 cell metadata
        adjacency: Dict mapping each cell to its neighbors
    """
    config = config or CityConfig()
    rng = np.random.default_rng(config.seed)
    
    grid_df = build_h3_grid(config)
    adjacency = build_h3_adjacency(grid_df["h3_cell"].tolist())
    
    periods = int(pd.Timedelta(days=config.num_days) / pd.Timedelta(config.freq))
    timestamps = pd.date_range(pd.Timestamp(config.start), periods=periods, freq=config.freq)
    
    rows: List[Dict] = []
    
    for _, cell in grid_df.iterrows():
        cell_multiplier = float(rng.lognormal(mean=0.0, sigma=0.18))
        capacity_kw = float(cell["transformer_capacity_kw"])
        zone_type = str(cell["zone_type"])
        ev_adoption = float(cell["ev_adoption_index"])
        solar_cap = float(cell["solar_capacity_kw"])
        fleet_power = _weighted_fleet_power(zone_type, rng)
        
        for ts in timestamps:
            hour = ts.hour + ts.minute / 60.0
            is_weekend = ts.dayofweek >= 5
            
            # Weather
            temp = 25.5 + 5.5 * math.sin(2 * math.pi * (hour - 8) / 24) + rng.normal(0, 0.45)
            rainfall = max(0.0, rng.normal(0.1, 0.3)) if rng.random() < 0.2 else 0.0
            cloud_factor = max(0.3, 1.0 - 0.25 * rainfall)
            
            # Base grid load (non-EV)
            base_load_frac = _base_grid_profile(hour, zone_type)
            base_load_kw = capacity_kw * base_load_frac * (1 + rng.normal(0, 0.05))
            
            # EV demand
            ev_profile = _ev_profile(hour, zone_type, is_weekend)
            ev_demand_kw = ev_adoption * ev_profile * fleet_power * cell_multiplier * cell["station_count"]
            ev_demand_kw *= (1 + rng.normal(0, 0.08))
            
            # Traffic
            traffic = _traffic_intensity(hour, zone_type, is_weekend, rng)
            
            # Solar
            solar_kw = solar_generation_kw(hour, solar_cap, cloud_factor)
            
            # Tariff
            tariff = bescom_tariff_multiplier(hour)
            
            total_demand = base_load_kw + ev_demand_kw
            
            rows.append({
                "h3_cell": cell["h3_cell"],
                "timestamp": ts,
                "zone_name": cell["zone_name"],
                "zone_type": zone_type,
                "lat": float(cell["lat"]),
                "lon": float(cell["lon"]),
                "scenario": cell["scenario"],
                "corridor_name": cell["corridor_name"],
                "demand_kw": round(ev_demand_kw, 2),  # EV demand (what we forecast/optimize)
                "grid_base_load_kw": round(base_load_kw, 2),
                "total_load_kw": round(total_demand, 2),
                "transformer_capacity_kw": capacity_kw,
                "station_count": int(cell["station_count"]),
                "temperature_c": round(temp, 1),
                "rainfall_mm": round(rainfall, 2),
                "traffic_intensity": round(traffic, 3),
                "solar_generation_kw": round(solar_kw, 2),
                "tariff_multiplier": tariff,
                "ev_adoption_index": ev_adoption,
                "charger_density_index": cell["charger_density_index"],
                "demand_growth_index": cell["demand_growth_index"],
                "priority_share": cell["priority_share"],
                "deadline_steps": cell["deadline_steps"],
                "is_weekend": is_weekend,
            })
    
    raw_df = pd.DataFrame(rows)
    return raw_df, grid_df, adjacency


if __name__ == "__main__":
    print("Generating synthetic Bengaluru data...")
    config = CityConfig(max_cells=20, num_days=3)
    data, grid, adj = generate_synthetic_data(config)
    
    print(f"\nGrid: {len(grid)} H3 cells")
    print(f"Zone types: {grid['zone_type'].value_counts().to_dict()}")
    print(f"\nTime series: {len(data)} rows")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"\nSample data:")
    print(data[["timestamp", "h3_cell", "zone_name", "demand_kw", "grid_base_load_kw"]].head(10))
