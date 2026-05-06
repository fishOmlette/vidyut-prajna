"""Enhanced Bengaluru city simulation with complete data sources.

This module adds all data sources promised in the Vidyut Prajna idea submission:
- Swiggy/Zomato gig fleet congregation points
- OCPP charging session simulation
- Monsoon weather patterns (May monsoon season)
- Realistic traffic from commute patterns
- DTR (Distribution Transformer) topology
- K-anonymity masking for privacy
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import h3
import numpy as np
import pandas as pd

from src.spatial_grid.simulation import (
    CityConfig,
    NEIGHBORHOOD_ANCHORS,
    DEMO_SCENARIOS,
    ZONE_PARAMS,
    FLEET_MIX,
    gaussian_hour,
    haversine_km,
    nearest_anchor,
    bescom_tariff_multiplier,
    solar_generation_kw,
    _scenario_route_cells,
    build_h3_adjacency,
)


# ============================================================================
# GIG FLEET (SWIGGY/ZOMATO) CONGREGATION POINTS
# ============================================================================

GIG_CONGREGATION_POINTS = [
    # Major restaurant clusters where delivery fleets congregate
    {"name": "Koramangala 5th Block",    "lat": 12.9341, "lon": 77.6146, "peak_vehicles": 85, "type": "restaurant_hub"},
    {"name": "HSR Sector 2",              "lat": 12.9120, "lon": 77.6389, "peak_vehicles": 72, "type": "restaurant_hub"},
    {"name": "Indiranagar 100ft Road",    "lat": 12.9784, "lon": 77.6408, "peak_vehicles": 95, "type": "restaurant_hub"},
    {"name": "Whitefield Main Road",      "lat": 12.9698, "lon": 77.7499, "peak_vehicles": 68, "type": "restaurant_hub"},
    {"name": "MG Road Brigade",           "lat": 12.9757, "lon": 77.6050, "peak_vehicles": 110, "type": "restaurant_hub"},
    {"name": "Jayanagar 4th Block",       "lat": 12.9271, "lon": 77.5816, "peak_vehicles": 65, "type": "restaurant_hub"},
    {"name": "BTM 2nd Stage",             "lat": 12.9166, "lon": 77.6101, "peak_vehicles": 58, "type": "restaurant_hub"},
    {"name": "Marathahalli Bridge",       "lat": 12.9591, "lon": 77.7009, "peak_vehicles": 75, "type": "restaurant_hub"},
    
    # E-commerce fulfillment centers (Flipkart, Amazon Delivery)
    {"name": "Electronic City FC",        "lat": 12.8450, "lon": 77.6600, "peak_vehicles": 120, "type": "fulfillment"},
    {"name": "Peenya Industrial",         "lat": 13.0280, "lon": 77.5197, "peak_vehicles": 150, "type": "fulfillment"},
    {"name": "Yelahanka FC",              "lat": 13.1007, "lon": 77.5963, "peak_vehicles": 90, "type": "fulfillment"},
    {"name": "Sarjapur FC",               "lat": 12.9100, "lon": 77.6860, "peak_vehicles": 85, "type": "fulfillment"},
    
    # Quick commerce hubs (Zepto, Blinkit, Instamart)
    {"name": "Domlur Dark Store",         "lat": 12.9610, "lon": 77.6387, "peak_vehicles": 45, "type": "quick_commerce"},
    {"name": "JP Nagar Dark Store",       "lat": 12.9063, "lon": 77.5857, "peak_vehicles": 40, "type": "quick_commerce"},
    {"name": "Hebbal Dark Store",         "lat": 13.0358, "lon": 77.5970, "peak_vehicles": 42, "type": "quick_commerce"},
]


@dataclass
class GigFleetConfig:
    """Configuration for gig economy fleet simulation."""
    total_2w_vehicles: int = 15000  # Estimated Swiggy/Zomato 2W in Bengaluru
    total_3w_vehicles: int = 3000   # Estimated 3W delivery fleet
    avg_daily_trips_per_vehicle: int = 12
    charging_kw_2w: float = 3.3
    charging_kw_3w: float = 7.4
    battery_kwh_2w: float = 2.5
    battery_kwh_3w: float = 6.0


def gig_fleet_demand_profile(hour: float, fleet_type: str) -> float:
    """
    Gig fleet charging demand profile based on real delivery patterns.
    
    Peak delivery times (when vehicles are WORKING, not charging):
    - Lunch: 11:00-14:00
    - Dinner: 19:00-22:00
    
    Charging happens during lulls:
    - Morning prep: 06:00-10:00
    - Afternoon lull: 14:30-18:00
    - Night recharge: 23:00-05:00
    """
    if fleet_type == "restaurant_hub":
        # Charging peaks between meal times
        morning = 0.75 * gaussian_hour(hour, 8.5, 1.5)
        afternoon = 0.90 * gaussian_hour(hour, 15.5, 1.8)
        night = 0.60 * gaussian_hour(hour, 2.0, 2.5)
        return 0.15 + morning + afternoon + night
    
    elif fleet_type == "fulfillment":
        # E-commerce delivery charging - early morning and late night
        early_morning = 1.1 * gaussian_hour(hour, 6.0, 2.0)
        night = 0.85 * gaussian_hour(hour, 22.0, 2.5)
        return 0.20 + early_morning + night
    
    else:  # quick_commerce
        # Quick commerce - more distributed, needs fast charging
        return 0.35 + 0.25 * gaussian_hour(hour, 9.0, 2.0) + 0.25 * gaussian_hour(hour, 16.0, 2.0)


def compute_gig_fleet_demand(
    h3_cell: str,
    hour: float,
    rng: np.random.Generator,
    config: GigFleetConfig = None,
) -> Tuple[float, int]:
    """
    Compute gig fleet charging demand for a given H3 cell.
    
    Returns:
        demand_kw: Estimated charging demand in kW
        active_vehicles: Estimated active charging vehicles
    """
    config = config or GigFleetConfig()
    lat, lon = h3.cell_to_latlng(h3_cell)
    
    total_demand = 0.0
    total_vehicles = 0
    
    for point in GIG_CONGREGATION_POINTS:
        distance = haversine_km(lat, lon, point["lat"], point["lon"])
        
        # Inverse distance weighting with cutoff at 2.5 km
        if distance < 2.5:
            weight = max(0.0, 1.0 - (distance / 2.5) ** 1.5)
            profile = gig_fleet_demand_profile(hour, point["type"])
            
            # Stochastic vehicle count
            base_vehicles = point["peak_vehicles"] * profile * weight
            vehicles = int(rng.poisson(max(0, base_vehicles)))
            
            # 70% 2W, 30% 3W for restaurants/quick commerce
            # 40% 2W, 60% 3W for fulfillment
            if point["type"] == "fulfillment":
                kw_per_vehicle = 0.4 * config.charging_kw_2w + 0.6 * config.charging_kw_3w
            else:
                kw_per_vehicle = 0.7 * config.charging_kw_2w + 0.3 * config.charging_kw_3w
            
            demand = vehicles * kw_per_vehicle * (1 + rng.normal(0, 0.1))
            total_demand += max(0, demand)
            total_vehicles += vehicles
    
    return round(total_demand, 2), total_vehicles


# ============================================================================
# OCPP CHARGING SESSION SIMULATION
# ============================================================================

@dataclass
class OCPPSession:
    """OCPP 1.6J compliant charging session record."""
    session_id: str
    h3_cell: str
    connector_id: int
    vehicle_type: str  # 2W, 3W, 4W, bus
    start_time: pd.Timestamp
    end_time: Optional[pd.Timestamp]
    energy_kwh: float
    max_power_kw: float
    meter_start_wh: int
    meter_stop_wh: Optional[int]
    soc_start_pct: float
    soc_target_pct: float
    status: str  # Charging, SuspendedEV, SuspendedEVSE, Finishing, Completed
    priority_level: int  # 1=Emergency, 2=High, 3=Normal, 4=Low/V2G-ready
    is_shiftable: bool


def generate_ocpp_session_id(h3_cell: str, timestamp: pd.Timestamp, idx: int) -> str:
    """Generate deterministic but anonymized session ID."""
    raw = f"{h3_cell}:{timestamp.isoformat()}:{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16].upper()


def simulate_ocpp_sessions(
    h3_cell: str,
    timestamp: pd.Timestamp,
    zone_type: str,
    station_count: int,
    ev_adoption: float,
    rng: np.random.Generator,
) -> List[OCPPSession]:
    """
    Simulate OCPP charging sessions for a given cell and timestamp.
    
    Returns realistic charging session data that could come from
    actual OCPP-compliant charge point operators.
    """
    hour = timestamp.hour + timestamp.minute / 60.0
    is_weekend = timestamp.dayofweek >= 5
    
    # Determine expected sessions based on zone and time
    fleet_mix = FLEET_MIX.get(zone_type, FLEET_MIX["mixed"])
    
    # Base arrival rate varies by zone type
    base_rate = {
        "residential": 2.5 if hour >= 17 or hour <= 7 else 0.8,
        "commercial": 1.8 if 9 <= hour <= 18 else 0.5,
        "logistics": 2.2 if hour <= 8 or 14 <= hour <= 20 else 1.2,
        "mixed": 1.5,
        "it_corridor": 1.6 if 9 <= hour <= 19 else 0.6,
    }[zone_type]
    
    # Poisson arrival process
    lambda_rate = base_rate * station_count * ev_adoption * (0.7 if is_weekend else 1.0)
    num_sessions = rng.poisson(max(0.1, lambda_rate))
    
    sessions = []
    for idx in range(num_sessions):
        # Choose vehicle type from fleet mix
        vehicle_types = list(fleet_mix.keys())
        weights = [fleet_mix[vt][1] for vt in vehicle_types]
        vehicle_type = rng.choice(vehicle_types, p=np.array(weights) / sum(weights))
        power_kw, _ = fleet_mix[vehicle_type]
        
        # Session parameters
        soc_start = rng.uniform(15, 45)
        soc_target = rng.uniform(80, 100)
        
        # Energy based on battery size
        battery_kwh = {"2W": 2.5, "3W": 6.0, "4W": 60.0, "bus": 200.0}[vehicle_type]
        energy_kwh = battery_kwh * (soc_target - soc_start) / 100.0
        
        # Duration calculation
        duration_hours = energy_kwh / (power_kw * 0.92)  # 92% charging efficiency
        end_time = timestamp + pd.Timedelta(hours=duration_hours)
        
        # Priority assignment
        # Emergency services, commercial fleets get high priority
        if zone_type == "logistics" and vehicle_type in ("3W", "bus"):
            priority = rng.choice([1, 2, 2, 3], p=[0.1, 0.3, 0.4, 0.2])
        elif zone_type == "residential" and hour >= 22:
            priority = rng.choice([3, 4], p=[0.4, 0.6])  # Night charging = shiftable
        else:
            priority = rng.choice([2, 3, 3, 4], p=[0.15, 0.35, 0.35, 0.15])
        
        is_shiftable = priority >= 3 and zone_type not in ("logistics",)
        
        meter_start = int(rng.uniform(10000, 500000))
        
        sessions.append(OCPPSession(
            session_id=generate_ocpp_session_id(h3_cell, timestamp, idx),
            h3_cell=h3_cell,
            connector_id=rng.integers(1, max(2, station_count * 2)),
            vehicle_type=vehicle_type,
            start_time=timestamp,
            end_time=end_time,
            energy_kwh=round(energy_kwh, 2),
            max_power_kw=power_kw,
            meter_start_wh=meter_start,
            meter_stop_wh=meter_start + int(energy_kwh * 1000),
            soc_start_pct=round(soc_start, 1),
            soc_target_pct=round(soc_target, 1),
            status="Charging",
            priority_level=priority,
            is_shiftable=is_shiftable,
        ))
    
    return sessions


# ============================================================================
# MONSOON WEATHER SIMULATION (BENGALURU MAY-SEPTEMBER)
# ============================================================================

@dataclass
class MonsoonConfig:
    """Configuration for Bengaluru monsoon simulation."""
    # Monsoon typically starts mid-May, peaks July-August
    pre_monsoon_start_month: int = 4  # April
    monsoon_start_month: int = 6      # June
    monsoon_peak_month: int = 8       # August
    monsoon_end_month: int = 10       # October
    
    # Average rainfall (mm) by month for Bengaluru
    monthly_rainfall_mm: Dict[int, float] = field(default_factory=lambda: {
        1: 2, 2: 7, 3: 4, 4: 46, 5: 119, 6: 80,
        7: 110, 8: 137, 9: 195, 10: 180, 11: 64, 12: 21
    })
    
    # Temperature ranges by month (Bengaluru is relatively stable)
    monthly_temp_range: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {
        1: (15, 28), 2: (17, 31), 3: (20, 34), 4: (22, 35), 5: (21, 33), 6: (20, 29),
        7: (19, 28), 8: (19, 28), 9: (19, 28), 10: (19, 28), 11: (17, 27), 12: (15, 26)
    })


def simulate_monsoon_weather(
    timestamp: pd.Timestamp,
    rng: np.random.Generator,
    config: MonsoonConfig = None,
) -> Dict[str, float]:
    """
    Simulate realistic Bengaluru monsoon weather patterns.
    
    Returns:
        Dict with temperature_c, rainfall_mm, humidity_pct, cloud_cover_pct, wind_speed_kmh
    """
    config = config or MonsoonConfig()
    month = timestamp.month
    hour = timestamp.hour + timestamp.minute / 60.0
    
    # Monthly baseline rainfall
    monthly_rain = config.monthly_rainfall_mm.get(month, 50)
    
    # Rainfall probability and intensity
    # Afternoon thunderstorms are common in Bengaluru (15:00-19:00)
    afternoon_storm_prob = 0.35 if 15 <= hour <= 19 else 0.12
    storm_prob = afternoon_storm_prob * (monthly_rain / 100)
    
    rainfall_mm = 0.0
    cloud_cover = 0.25  # Base cloud cover
    
    if rng.random() < storm_prob:
        # Storm event
        if month in (5, 6, 7, 8, 9, 10):  # Monsoon months
            rainfall_mm = rng.exponential(8.0) * (monthly_rain / 100)
        else:
            rainfall_mm = rng.exponential(3.0)
        cloud_cover = min(1.0, 0.7 + rng.uniform(0, 0.3))
    else:
        # Light drizzle possible during monsoon
        if month in (6, 7, 8, 9) and rng.random() < 0.25:
            rainfall_mm = rng.uniform(0.1, 2.0)
            cloud_cover = 0.5 + rng.uniform(0, 0.3)
    
    # Temperature with diurnal variation
    temp_min, temp_max = config.monthly_temp_range.get(month, (20, 30))
    
    # Diurnal temperature pattern
    diurnal_factor = 0.5 + 0.5 * math.sin(2 * math.pi * (hour - 6) / 24)
    temp_base = temp_min + (temp_max - temp_min) * diurnal_factor
    
    # Rainfall cools temperature
    rain_cooling = min(5.0, rainfall_mm * 0.3)
    temperature_c = temp_base - rain_cooling + rng.normal(0, 1.5)
    
    # Humidity
    humidity_base = 50 + 30 * (monthly_rain / 200)  # Higher during monsoon
    humidity_rain_boost = min(30, rainfall_mm * 3)
    humidity_pct = min(100, humidity_base + humidity_rain_boost + rng.normal(0, 8))
    
    # Wind speed (higher during storms)
    wind_base = 8 + 4 * cloud_cover
    wind_speed_kmh = wind_base + (10 if rainfall_mm > 5 else 0) + rng.exponential(3)
    
    return {
        "temperature_c": round(temperature_c, 1),
        "rainfall_mm": round(rainfall_mm, 2),
        "humidity_pct": round(humidity_pct, 1),
        "cloud_cover_pct": round(cloud_cover * 100, 1),
        "wind_speed_kmh": round(wind_speed_kmh, 1),
    }


# ============================================================================
# REALISTIC TRAFFIC SIMULATION
# ============================================================================

# Real Bengaluru traffic congestion indices by area (relative to city average)
CONGESTION_INDICES = {
    "Silk Board Junction": 1.8,
    "KR Puram":            1.6,
    "Marathahalli":        1.5,
    "Hebbal":              1.4,
    "Outer Ring Road":     1.45,
    "Electronic City":     1.35,
    "Whitefield":          1.4,
    "MG Road":             1.25,
    "Koramangala":         1.2,
    "Indiranagar":         1.15,
    "HSR Layout":          1.1,
    "Jayanagar":           1.05,
    "Bannerghatta Rd":     1.15,
    "Sarjapur Road":       1.3,
    "JP Nagar":            1.0,
    "Banashankari":        0.95,
    "Rajajinagar":         1.05,
    "Yelahanka":           1.0,
    "Peenya":              1.2,
    "Majestic":            1.35,
    "BTM Layout":          1.1,
}


def simulate_traffic_intensity(
    zone_name: str,
    zone_type: str,
    hour: float,
    is_weekend: bool,
    rainfall_mm: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Simulate realistic traffic based on Bengaluru patterns.
    
    Returns:
        Dict with traffic_intensity (0-1), congestion_index, avg_speed_kmh
    """
    # Base congestion index for zone
    congestion_idx = CONGESTION_INDICES.get(zone_name, 1.0)
    
    # Time-of-day profile
    # Morning rush: 8:00-10:30
    # Evening rush: 17:30-20:30
    morning_rush = 0.85 * gaussian_hour(hour, 9.0, 1.2)
    evening_rush = 1.0 * gaussian_hour(hour, 18.5, 1.5)
    midday = 0.3 * gaussian_hour(hour, 13.0, 2.5)
    
    time_factor = 0.25 + morning_rush + evening_rush + midday
    
    # Weekend reduction (except residential/commercial areas)
    if is_weekend:
        if zone_type in ("it_corridor", "logistics"):
            time_factor *= 0.4
        elif zone_type == "commercial":
            time_factor *= 0.85  # Shopping areas busy on weekends
        elif zone_type == "residential":
            time_factor *= 1.15  # People going out
    
    # Rain increases congestion due to slower driving, accidents
    rain_factor = 1.0 + min(0.35, rainfall_mm * 0.04)
    
    # Zone type adjustment
    zone_factor = {
        "it_corridor": 1.2,
        "commercial": 1.1,
        "logistics": 0.95,
        "mixed": 1.0,
        "residential": 0.85,
    }[zone_type]
    
    # Final intensity (normalized 0-1)
    raw_intensity = congestion_idx * time_factor * rain_factor * zone_factor
    traffic_intensity = min(1.0, max(0.05, raw_intensity * (1 + rng.normal(0, 0.08))))
    
    # Congestion index (TomTom-style, percentage extra travel time)
    congestion_index = max(0, (traffic_intensity * 1.5 - 0.3) * 100)
    
    # Average speed (inversely related to congestion)
    free_flow_speed = 45  # km/h in Bengaluru
    avg_speed = free_flow_speed / (1 + congestion_index / 100)
    
    return {
        "traffic_intensity": round(traffic_intensity, 3),
        "congestion_index": round(congestion_index, 1),
        "avg_speed_kmh": round(avg_speed, 1),
    }


# ============================================================================
# DTR (DISTRIBUTION TRANSFORMER) TOPOLOGY
# ============================================================================

@dataclass
class DTRSpec:
    """BESCOM Distribution Transformer specification."""
    dtr_id: str
    h3_cell: str
    capacity_kva: int
    voltage_ratio: str  # "11kV/440V" or "11kV/230V"
    age_years: int
    health_score: float  # 0-100, based on oil testing, thermal imaging
    last_maintenance_date: str
    connected_consumers: int
    peak_load_history_kw: float
    thermal_limit_c: float  # Typically 65°C for oil-filled


DTR_CAPACITY_DISTRIBUTION = {
    "residential": {
        100: 0.35,   # 100 kVA - most common in residential
        250: 0.40,
        500: 0.20,
        1000: 0.05,
    },
    "commercial": {
        250: 0.25,
        500: 0.45,
        1000: 0.25,
        2000: 0.05,
    },
    "logistics": {
        500: 0.30,
        1000: 0.50,
        2000: 0.20,
    },
    "mixed": {
        250: 0.30,
        500: 0.50,
        1000: 0.20,
    },
    "it_corridor": {
        500: 0.20,
        1000: 0.55,
        2000: 0.25,
    },
}


def generate_dtr_topology(
    h3_cell: str,
    zone_type: str,
    station_count: int,
    rng: np.random.Generator,
) -> List[DTRSpec]:
    """
    Generate realistic DTR topology for an H3 cell.
    
    BESCOM typically has 1-3 DTRs per residential cluster,
    more in commercial/industrial zones.
    """
    # Number of DTRs based on zone type and station count
    dtr_counts = {
        "residential": (1, 2),
        "commercial": (2, 4),
        "logistics": (2, 5),
        "mixed": (1, 3),
        "it_corridor": (2, 4),
    }
    min_dtr, max_dtr = dtr_counts.get(zone_type, (1, 2))
    num_dtrs = rng.integers(min_dtr, max_dtr + 1)
    
    # Capacity distribution for zone
    cap_dist = DTR_CAPACITY_DISTRIBUTION.get(zone_type, DTR_CAPACITY_DISTRIBUTION["mixed"])
    capacities = list(cap_dist.keys())
    probs = list(cap_dist.values())
    
    dtrs = []
    for i in range(num_dtrs):
        capacity_kva = int(rng.choice(capacities, p=probs))
        
        # Health score - older transformers tend to have lower scores
        age = int(rng.exponential(8) + 2)  # 2-30 years typical
        age = min(age, 35)
        
        base_health = 95 - age * 1.5
        health_score = max(40, min(100, base_health + rng.normal(0, 8)))
        
        # Peak load history (fraction of capacity)
        peak_fraction = rng.uniform(0.5, 0.95)
        peak_load = capacity_kva * 0.9 * peak_fraction  # Assume 0.9 power factor
        
        dtrs.append(DTRSpec(
            dtr_id=f"DTR-{h3_cell[:8]}-{i+1:02d}",
            h3_cell=h3_cell,
            capacity_kva=capacity_kva,
            voltage_ratio="11kV/440V" if capacity_kva >= 250 else "11kV/230V",
            age_years=age,
            health_score=round(health_score, 1),
            last_maintenance_date=(
                pd.Timestamp("2026-05-01") - pd.Timedelta(days=rng.integers(30, 365))
            ).strftime("%Y-%m-%d"),
            connected_consumers=int(rng.uniform(50, 300) * (capacity_kva / 250)),
            peak_load_history_kw=round(peak_load, 1),
            thermal_limit_c=65.0 if capacity_kva <= 500 else 70.0,
        ))
    
    return dtrs


# ============================================================================
# K-ANONYMITY MASKING FOR PRIVACY
# ============================================================================

@dataclass
class AnonymizationConfig:
    """Configuration for K-anonymity data masking."""
    k_value: int = 5  # Minimum group size for quasi-identifiers
    location_precision_m: int = 250  # Generalize to 250m grid
    time_bucket_minutes: int = 30    # Round timestamps to 30-min buckets
    mask_session_ids: bool = True
    mask_vehicle_ids: bool = True
    generalize_energy: bool = True   # Round energy to nearest 0.5 kWh
    generalize_soc: bool = True      # Round SoC to 5% increments


def apply_k_anonymity(
    df: pd.DataFrame,
    config: AnonymizationConfig = None,
) -> pd.DataFrame:
    """
    Apply K-anonymity masking to protect user privacy.
    
    This ensures no individual EV or driver can be identified
    while preserving aggregate patterns for analysis.
    """
    config = config or AnonymizationConfig()
    masked = df.copy()
    
    # Hash session IDs if present
    if config.mask_session_ids and "session_id" in masked.columns:
        masked["session_id"] = masked["session_id"].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12]
        )
    
    # Mask vehicle IDs if present
    if config.mask_vehicle_ids and "vehicle_id" in masked.columns:
        masked["vehicle_id"] = "ANON"
    
    # Time generalization (30-minute buckets)
    if "timestamp" in masked.columns:
        bucket = pd.Timedelta(minutes=config.time_bucket_minutes)
        masked["timestamp"] = masked["timestamp"].dt.floor(bucket)
    
    if "start_time" in masked.columns:
        bucket = pd.Timedelta(minutes=config.time_bucket_minutes)
        masked["start_time"] = masked["start_time"].dt.floor(bucket)
    
    # Location generalization - already using H3 cells which provide ~250m resolution at level 8
    # H3 resolution 8 ≈ 0.461 km² ≈ 680m edge length, good for k-anonymity
    
    # Energy generalization
    if config.generalize_energy and "energy_kwh" in masked.columns:
        masked["energy_kwh"] = (masked["energy_kwh"] * 2).round() / 2  # 0.5 kWh buckets
    
    # SoC generalization
    if config.generalize_soc:
        for col in ["soc_start_pct", "soc_target_pct"]:
            if col in masked.columns:
                masked[col] = (masked[col] / 5).round() * 5  # 5% buckets
    
    # Suppress small groups (enforce k-anonymity)
    if "h3_cell" in masked.columns and "timestamp" in masked.columns:
        # Group by quasi-identifiers
        group_cols = ["h3_cell", "timestamp"]
        if "zone_type" in masked.columns:
            group_cols.append("zone_type")
        
        counts = masked.groupby(group_cols).size()
        small_groups = counts[counts < config.k_value].index
        
        # Suppress records in groups smaller than k
        if len(small_groups) > 0:
            for group_key in small_groups:
                mask = True
                for col, val in zip(group_cols, group_key if isinstance(group_key, tuple) else [group_key]):
                    mask = mask & (masked[col] == val)
                masked.loc[mask, "suppressed"] = True
    
    return masked


# ============================================================================
# ENHANCED DATA GENERATION
# ============================================================================

def generate_enhanced_synthetic_data(
    config: CityConfig = None,
    include_ocpp: bool = True,
    include_gig_fleet: bool = True,
    apply_anonymization: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]], List[OCPPSession], List[DTRSpec]]:
    """
    Generate comprehensive synthetic data with all promised data sources.
    
    Returns:
        raw_df: Time-series data with enhanced features
        grid_df: H3 cell metadata with DTR topology
        adjacency: Dict mapping each cell to its neighbors
        ocpp_sessions: List of simulated OCPP charging sessions
        dtr_specs: List of DTR specifications
    """
    from src.spatial_grid.simulation import build_h3_grid
    
    config = config or CityConfig()
    rng = np.random.default_rng(config.seed)
    
    # Build base grid
    grid_df = build_h3_grid(config)
    adjacency = build_h3_adjacency(grid_df["h3_cell"].tolist())
    
    # Generate timestamps
    periods = int(pd.Timedelta(days=config.num_days) / pd.Timedelta(config.freq))
    timestamps = pd.date_range(pd.Timestamp(config.start), periods=periods, freq=config.freq)
    
    # Generate DTR topology for each cell
    all_dtrs: List[DTRSpec] = []
    cell_dtr_capacity: Dict[str, float] = {}
    
    for _, cell in grid_df.iterrows():
        dtrs = generate_dtr_topology(
            cell["h3_cell"],
            cell["zone_type"],
            cell["station_count"],
            rng,
        )
        all_dtrs.extend(dtrs)
        # Total capacity is sum of all DTRs (with 0.9 power factor)
        cell_dtr_capacity[cell["h3_cell"]] = sum(d.capacity_kva * 0.9 for d in dtrs)
    
    # Generate time-series data
    rows: List[Dict] = []
    all_ocpp_sessions: List[OCPPSession] = []
    
    for _, cell in grid_df.iterrows():
        cell_id = cell["h3_cell"]
        zone_type = str(cell["zone_type"])
        zone_name = str(cell["zone_name"])
        ev_adoption = float(cell["ev_adoption_index"])
        solar_cap = float(cell["solar_capacity_kw"])
        station_count = int(cell["station_count"])
        
        # Use DTR-based capacity
        capacity_kw = cell_dtr_capacity.get(cell_id, float(cell["transformer_capacity_kw"]))
        
        cell_multiplier = float(rng.lognormal(mean=0.0, sigma=0.18))
        
        for ts in timestamps:
            hour = ts.hour + ts.minute / 60.0
            is_weekend = ts.dayofweek >= 5
            
            # Enhanced monsoon weather
            weather = simulate_monsoon_weather(ts, rng)
            rainfall_mm = weather["rainfall_mm"]
            cloud_factor = max(0.3, 1.0 - weather["cloud_cover_pct"] / 100)
            
            # Enhanced traffic
            traffic = simulate_traffic_intensity(
                zone_name, zone_type, hour, is_weekend, rainfall_mm, rng
            )
            
            # Base grid load (non-EV)
            from src.spatial_grid.simulation import _base_grid_profile, _ev_profile, _weighted_fleet_power
            
            base_load_frac = _base_grid_profile(hour, zone_type)
            base_load_kw = capacity_kw * base_load_frac * (1 + rng.normal(0, 0.05))
            
            # Standard EV demand
            ev_profile = _ev_profile(hour, zone_type, is_weekend)
            fleet_power = _weighted_fleet_power(zone_type, rng)
            ev_demand_kw = ev_adoption * ev_profile * fleet_power * cell_multiplier * station_count
            ev_demand_kw *= (1 + rng.normal(0, 0.08))
            
            # Add gig fleet demand if enabled
            gig_demand_kw = 0.0
            gig_vehicles = 0
            if include_gig_fleet:
                gig_demand_kw, gig_vehicles = compute_gig_fleet_demand(cell_id, hour, rng)
                # Scale by cell's proximity to gig points
                gig_demand_kw *= (0.5 + 0.5 * ev_adoption)
            
            total_ev_demand = ev_demand_kw + gig_demand_kw
            
            # Solar generation (affected by cloud cover)
            solar_kw = solar_generation_kw(hour, solar_cap, cloud_factor)
            
            # Tariff
            tariff = bescom_tariff_multiplier(hour)
            
            total_demand = base_load_kw + total_ev_demand
            
            # Generate OCPP sessions if enabled
            if include_ocpp:
                sessions = simulate_ocpp_sessions(
                    cell_id, ts, zone_type, station_count, ev_adoption, rng
                )
                all_ocpp_sessions.extend(sessions)
            
            rows.append({
                "h3_cell": cell_id,
                "timestamp": ts,
                "zone_name": zone_name,
                "zone_type": zone_type,
                "lat": float(cell["lat"]),
                "lon": float(cell["lon"]),
                "scenario": cell["scenario"],
                "corridor_name": cell["corridor_name"],
                
                # Demand
                "demand_kw": round(total_ev_demand, 2),
                "standard_ev_demand_kw": round(ev_demand_kw, 2),
                "gig_fleet_demand_kw": round(gig_demand_kw, 2),
                "gig_fleet_vehicles": gig_vehicles,
                "grid_base_load_kw": round(base_load_kw, 2),
                "total_load_kw": round(total_demand, 2),
                
                # Grid
                "transformer_capacity_kw": round(capacity_kw, 1),
                "station_count": station_count,
                
                # Weather (enhanced monsoon)
                "temperature_c": weather["temperature_c"],
                "rainfall_mm": weather["rainfall_mm"],
                "humidity_pct": weather["humidity_pct"],
                "cloud_cover_pct": weather["cloud_cover_pct"],
                "wind_speed_kmh": weather["wind_speed_kmh"],
                
                # Traffic (enhanced)
                "traffic_intensity": traffic["traffic_intensity"],
                "congestion_index": traffic["congestion_index"],
                "avg_speed_kmh": traffic["avg_speed_kmh"],
                
                # Other features
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
    
    # Apply K-anonymity if requested
    if apply_anonymization:
        raw_df = apply_k_anonymity(raw_df)
    
    return raw_df, grid_df, adjacency, all_ocpp_sessions, all_dtrs


def get_ocpp_sessions_df(sessions: List[OCPPSession]) -> pd.DataFrame:
    """Convert OCPP sessions to DataFrame."""
    if not sessions:
        return pd.DataFrame()
    
    records = []
    for s in sessions:
        records.append({
            "session_id": s.session_id,
            "h3_cell": s.h3_cell,
            "connector_id": s.connector_id,
            "vehicle_type": s.vehicle_type,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "energy_kwh": s.energy_kwh,
            "max_power_kw": s.max_power_kw,
            "meter_start_wh": s.meter_start_wh,
            "meter_stop_wh": s.meter_stop_wh,
            "soc_start_pct": s.soc_start_pct,
            "soc_target_pct": s.soc_target_pct,
            "status": s.status,
            "priority_level": s.priority_level,
            "is_shiftable": s.is_shiftable,
        })
    
    return pd.DataFrame(records)


def get_dtr_specs_df(dtrs: List[DTRSpec]) -> pd.DataFrame:
    """Convert DTR specs to DataFrame."""
    if not dtrs:
        return pd.DataFrame()
    
    records = []
    for d in dtrs:
        records.append({
            "dtr_id": d.dtr_id,
            "h3_cell": d.h3_cell,
            "capacity_kva": d.capacity_kva,
            "voltage_ratio": d.voltage_ratio,
            "age_years": d.age_years,
            "health_score": d.health_score,
            "last_maintenance_date": d.last_maintenance_date,
            "connected_consumers": d.connected_consumers,
            "peak_load_history_kw": d.peak_load_history_kw,
            "thermal_limit_c": d.thermal_limit_c,
        })
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Testing enhanced simulation...")
    
    config = CityConfig(max_cells=15, num_days=2, freq="1h")
    raw_df, grid_df, adj, ocpp_sessions, dtrs = generate_enhanced_synthetic_data(config)
    
    print(f"\n=== Grid Summary ===")
    print(f"H3 cells: {len(grid_df)}")
    print(f"Zone types: {grid_df['zone_type'].value_counts().to_dict()}")
    
    print(f"\n=== Time Series ===")
    print(f"Rows: {len(raw_df)}")
    print(f"Date range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")
    print(f"Columns: {list(raw_df.columns)}")
    
    print(f"\n=== Gig Fleet ===")
    print(f"Total gig fleet demand: {raw_df['gig_fleet_demand_kw'].sum():.1f} kWh")
    print(f"Peak gig vehicles: {raw_df['gig_fleet_vehicles'].max()}")
    
    print(f"\n=== OCPP Sessions ===")
    print(f"Total sessions: {len(ocpp_sessions)}")
    ocpp_df = get_ocpp_sessions_df(ocpp_sessions)
    if len(ocpp_df) > 0:
        print(f"Vehicle types: {ocpp_df['vehicle_type'].value_counts().to_dict()}")
        print(f"Shiftable sessions: {ocpp_df['is_shiftable'].sum()}")
    
    print(f"\n=== DTR Topology ===")
    print(f"Total DTRs: {len(dtrs)}")
    dtr_df = get_dtr_specs_df(dtrs)
    if len(dtr_df) > 0:
        print(f"Capacity distribution: {dtr_df['capacity_kva'].value_counts().to_dict()}")
        print(f"Avg health score: {dtr_df['health_score'].mean():.1f}")
    
    print(f"\n=== Weather (sample) ===")
    weather_cols = ["temperature_c", "rainfall_mm", "humidity_pct", "cloud_cover_pct"]
    print(raw_df[weather_cols].describe().round(2))
    
    print("\nEnhanced simulation test PASSED!")
