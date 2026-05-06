"""Kernel Density Estimation (KDE) for EV arrival prediction.

Implements the KDE-based arrival estimation mentioned in the Vidyut Prajna proposal:
"H3 hexagons have equal distances between all neighbours, which is mathematically
vital for accurate Kernel Density Estimation (KDE) of vehicle arrivals."

This module provides:
- Spatio-temporal KDE for arrival prediction
- Arrival rate estimation per H3 cell
- Peak arrival time prediction
- Demand propagation based on neighbor influence
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h3
import numpy as np
import pandas as pd


@dataclass
class KDEConfig:
    """Configuration for KDE arrival estimation."""
    temporal_bandwidth_hours: float = 1.5  # Kernel bandwidth for time
    spatial_bandwidth_km: float = 0.8      # Kernel bandwidth for space
    min_samples: int = 10                   # Minimum samples for valid KDE
    grid_resolution_minutes: int = 15       # Prediction resolution


def gaussian_kernel(distance: float, bandwidth: float) -> float:
    """Standard Gaussian kernel."""
    return math.exp(-0.5 * (distance / bandwidth) ** 2) / (bandwidth * math.sqrt(2 * math.pi))


def epanechnikov_kernel(distance: float, bandwidth: float) -> float:
    """Epanechnikov kernel - optimal for MSE in density estimation."""
    u = distance / bandwidth
    if abs(u) > 1:
        return 0.0
    return 0.75 * (1 - u ** 2) / bandwidth


class SpatioTemporalKDE:
    """
    Spatio-temporal Kernel Density Estimation for EV arrivals.
    
    Uses H3 hexagonal grid for spatial component, leveraging the isotropic
    property of hexagons (equal distances to all neighbors).
    """
    
    def __init__(
        self,
        config: KDEConfig = None,
        kernel: str = "gaussian",  # "gaussian" or "epanechnikov"
    ):
        self.config = config or KDEConfig()
        self.kernel_func = gaussian_kernel if kernel == "gaussian" else epanechnikov_kernel
        
        # Learned parameters
        self.arrival_data: Optional[pd.DataFrame] = None
        self.h3_centroids: Dict[str, Tuple[float, float]] = {}
        self.fitted = False
    
    def _h3_distance_km(self, cell1: str, cell2: str) -> float:
        """Compute distance between H3 cell centers in km."""
        if cell1 not in self.h3_centroids or cell2 not in self.h3_centroids:
            lat1, lon1 = h3.cell_to_latlng(cell1)
            lat2, lon2 = h3.cell_to_latlng(cell2)
            self.h3_centroids[cell1] = (lat1, lon1)
            self.h3_centroids[cell2] = (lat2, lon2)
        else:
            lat1, lon1 = self.h3_centroids[cell1]
            lat2, lon2 = self.h3_centroids[cell2]
        
        # Haversine distance
        radius_km = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * radius_km * math.asin(math.sqrt(a))
    
    def fit(
        self,
        historical_df: pd.DataFrame,
        session_col: str = "session_id",
        time_col: str = "timestamp",
        cell_col: str = "h3_cell",
    ) -> "SpatioTemporalKDE":
        """
        Fit KDE on historical arrival data.
        
        Args:
            historical_df: DataFrame with historical charging sessions
            session_col: Column identifying unique sessions
            time_col: Column with arrival timestamps
            cell_col: Column with H3 cell IDs
        """
        df = historical_df.copy()
        
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Extract arrival events
        arrival_data = []
        for cell in df[cell_col].unique():
            cell_df = df[df[cell_col] == cell]
            for _, row in cell_df.iterrows():
                ts = row[time_col]
                arrival_data.append({
                    "h3_cell": cell,
                    "timestamp": ts,
                    "hour": ts.hour + ts.minute / 60.0,
                    "day_of_week": ts.dayofweek,
                    "is_weekend": ts.dayofweek >= 5,
                })
        
        self.arrival_data = pd.DataFrame(arrival_data)
        
        # Cache cell centroids
        for cell in self.arrival_data["h3_cell"].unique():
            lat, lon = h3.cell_to_latlng(cell)
            self.h3_centroids[cell] = (lat, lon)
        
        self.fitted = True
        return self
    
    def fit_from_demand(
        self,
        demand_df: pd.DataFrame,
        demand_col: str = "demand_kw",
        time_col: str = "timestamp",
        cell_col: str = "h3_cell",
        power_per_session_kw: float = 7.0,
    ) -> "SpatioTemporalKDE":
        """
        Fit KDE from aggregate demand data (when individual sessions unavailable).
        
        Converts demand to estimated arrival count using average power.
        """
        df = demand_df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Estimate arrivals from demand
        arrival_data = []
        for _, row in df.iterrows():
            cell = row[cell_col]
            ts = row[time_col]
            demand = float(row[demand_col])
            
            # Estimate number of sessions
            n_sessions = max(1, int(demand / power_per_session_kw))
            
            for _ in range(n_sessions):
                # Add jitter to timestamps for KDE smoothing
                jitter_minutes = np.random.normal(0, 10)
                jittered_ts = ts + pd.Timedelta(minutes=jitter_minutes)
                
                arrival_data.append({
                    "h3_cell": cell,
                    "timestamp": jittered_ts,
                    "hour": jittered_ts.hour + jittered_ts.minute / 60.0,
                    "day_of_week": jittered_ts.dayofweek,
                    "is_weekend": jittered_ts.dayofweek >= 5,
                })
        
        self.arrival_data = pd.DataFrame(arrival_data)
        
        for cell in self.arrival_data["h3_cell"].unique():
            lat, lon = h3.cell_to_latlng(cell)
            self.h3_centroids[cell] = (lat, lon)
        
        self.fitted = True
        return self
    
    def estimate_arrival_rate(
        self,
        target_cell: str,
        target_hour: float,
        is_weekend: bool = False,
    ) -> float:
        """
        Estimate arrival rate for a specific cell and time.
        
        Returns arrivals per hour using KDE.
        """
        if not self.fitted or self.arrival_data is None:
            raise RuntimeError("KDE not fitted. Call fit() first.")
        
        # Filter by day type
        relevant = self.arrival_data[self.arrival_data["is_weekend"] == is_weekend]
        
        if len(relevant) < self.config.min_samples:
            # Fall back to all data
            relevant = self.arrival_data
        
        if len(relevant) < self.config.min_samples:
            return 0.0
        
        # Compute kernel density
        density = 0.0
        
        for _, arrival in relevant.iterrows():
            # Temporal component
            hour_diff = abs(arrival["hour"] - target_hour)
            # Handle wraparound (e.g., 23:00 to 01:00)
            hour_diff = min(hour_diff, 24 - hour_diff)
            temporal_weight = self.kernel_func(hour_diff, self.config.temporal_bandwidth_hours)
            
            # Spatial component
            if arrival["h3_cell"] == target_cell:
                spatial_weight = self.kernel_func(0, self.config.spatial_bandwidth_km)
            else:
                distance = self._h3_distance_km(arrival["h3_cell"], target_cell)
                spatial_weight = self.kernel_func(distance, self.config.spatial_bandwidth_km)
            
            density += temporal_weight * spatial_weight
        
        # Normalize to arrivals per hour
        # Scale factor based on historical data density
        time_span_hours = (
            (self.arrival_data["timestamp"].max() - self.arrival_data["timestamp"].min())
            .total_seconds() / 3600
        )
        if time_span_hours > 0:
            density *= len(relevant) / time_span_hours
        
        return density
    
    def predict_arrivals(
        self,
        target_cells: List[str],
        start_time: pd.Timestamp,
        hours_ahead: int = 24,
    ) -> pd.DataFrame:
        """
        Predict arrival rates for multiple cells over a time horizon.
        
        Returns DataFrame with predicted arrival rates.
        """
        if not self.fitted:
            raise RuntimeError("KDE not fitted. Call fit() first.")
        
        predictions = []
        resolution = self.config.grid_resolution_minutes
        steps = int(hours_ahead * 60 / resolution)
        
        for step in range(steps):
            timestamp = start_time + pd.Timedelta(minutes=step * resolution)
            hour = timestamp.hour + timestamp.minute / 60.0
            is_weekend = timestamp.dayofweek >= 5
            
            for cell in target_cells:
                rate = self.estimate_arrival_rate(cell, hour, is_weekend)
                
                predictions.append({
                    "h3_cell": cell,
                    "timestamp": timestamp,
                    "hour": hour,
                    "is_weekend": is_weekend,
                    "predicted_arrival_rate": rate,
                })
        
        return pd.DataFrame(predictions)
    
    def get_peak_arrival_times(
        self,
        cell: str,
        is_weekend: bool = False,
        n_peaks: int = 3,
    ) -> List[Tuple[float, float]]:
        """
        Find peak arrival times for a cell.
        
        Returns list of (hour, rate) tuples sorted by rate descending.
        """
        if not self.fitted:
            raise RuntimeError("KDE not fitted. Call fit() first.")
        
        # Sample arrival rate across the day
        hours = np.linspace(0, 23.75, 96)  # 15-minute resolution
        rates = [self.estimate_arrival_rate(cell, h, is_weekend) for h in hours]
        
        # Find local maxima
        peaks = []
        for i in range(1, len(rates) - 1):
            if rates[i] > rates[i-1] and rates[i] > rates[i+1]:
                peaks.append((hours[i], rates[i]))
        
        # Sort by rate and return top n
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:n_peaks]


class NeighborDemandPropagation:
    """
    Model demand propagation between neighboring H3 cells.
    
    When one cell has high demand, neighboring cells may see spillover
    (e.g., EVs driving to adjacent areas to find available chargers).
    """
    
    def __init__(
        self,
        propagation_factor: float = 0.15,  # Fraction of excess demand that spills over
        max_spillover_rings: int = 2,       # How far spillover reaches
    ):
        self.propagation_factor = propagation_factor
        self.max_spillover_rings = max_spillover_rings
    
    def compute_neighbor_pressure(
        self,
        df: pd.DataFrame,
        adjacency: Dict[str, List[str]],
        cell_col: str = "h3_cell",
        demand_col: str = "demand_kw",
        capacity_col: str = "transformer_capacity_kw",
    ) -> pd.DataFrame:
        """
        Compute neighbor demand pressure for each cell.
        
        High utilization in neighbors may push demand to this cell.
        """
        df = df.copy()
        
        # Compute utilization per cell-time
        if "utilization" not in df.columns:
            df["utilization"] = df[demand_col] / df[capacity_col].clip(lower=1)
        
        # Group by timestamp
        neighbor_pressure = []
        
        for ts in df["timestamp"].unique():
            ts_df = df[df["timestamp"] == ts]
            util_by_cell = dict(zip(ts_df[cell_col], ts_df["utilization"]))
            
            for cell in ts_df[cell_col].unique():
                neighbors = adjacency.get(cell, [])
                
                # Compute weighted neighbor pressure
                pressure = 0.0
                total_weight = 0.0
                
                for ring in range(1, self.max_spillover_rings + 1):
                    ring_neighbors = h3.grid_disk(cell, ring)
                    ring_weight = 1.0 / (ring ** 2)  # Decay with distance
                    
                    for nbr in ring_neighbors:
                        if nbr != cell and nbr in util_by_cell:
                            nbr_util = util_by_cell[nbr]
                            # Pressure from neighbors above 70% utilization
                            if nbr_util > 0.7:
                                excess = nbr_util - 0.7
                                pressure += excess * ring_weight
                                total_weight += ring_weight
                
                if total_weight > 0:
                    pressure /= total_weight
                
                neighbor_pressure.append({
                    "h3_cell": cell,
                    "timestamp": ts,
                    "neighbor_pressure": pressure,
                    "spillover_demand_kw": pressure * self.propagation_factor * ts_df[
                        ts_df[cell_col] == cell
                    ][demand_col].iloc[0] if len(ts_df[ts_df[cell_col] == cell]) > 0 else 0,
                })
        
        pressure_df = pd.DataFrame(neighbor_pressure)
        
        # Merge back to original
        df = df.merge(pressure_df, on=[cell_col, "timestamp"], how="left")
        df["neighbor_pressure"] = df["neighbor_pressure"].fillna(0)
        df["spillover_demand_kw"] = df["spillover_demand_kw"].fillna(0)
        
        return df


def add_kde_features(
    df: pd.DataFrame,
    adjacency: Dict[str, List[str]],
    historical_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add KDE-based arrival features to a DataFrame.
    
    This enriches the input data with:
    - predicted_arrival_rate: KDE-estimated arrival rate
    - neighbor_demand_kw: Weighted demand from neighbors
    - neighbor_pressure: Spillover pressure from congested neighbors
    """
    df = df.copy()
    
    # Fit KDE on historical or same data
    kde = SpatioTemporalKDE()
    fit_df = historical_df if historical_df is not None else df
    kde.fit_from_demand(fit_df)
    
    # Add arrival rate predictions
    arrival_rates = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        cell = row["h3_cell"]
        hour = ts.hour + ts.minute / 60.0
        is_weekend = ts.dayofweek >= 5
        
        rate = kde.estimate_arrival_rate(cell, hour, is_weekend)
        arrival_rates.append(rate)
    
    df["predicted_arrival_rate"] = arrival_rates
    
    # Add neighbor demand
    neighbor_demand = []
    for _, row in df.iterrows():
        cell = row["h3_cell"]
        ts = row["timestamp"]
        
        neighbors = adjacency.get(cell, [])
        nbr_demand = 0.0
        
        for nbr in neighbors:
            nbr_row = df[(df["h3_cell"] == nbr) & (df["timestamp"] == ts)]
            if len(nbr_row) > 0:
                nbr_demand += nbr_row["demand_kw"].iloc[0]
        
        if neighbors:
            nbr_demand /= len(neighbors)
        
        neighbor_demand.append(nbr_demand)
    
    df["neighbor_demand_kw"] = neighbor_demand
    
    # Add neighbor pressure
    propagation = NeighborDemandPropagation()
    df = propagation.compute_neighbor_pressure(df, adjacency)
    
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("src", 1)[0])
    
    from src.spatial_grid.simulation import CityConfig, generate_synthetic_data, build_h3_adjacency
    
    print("Testing KDE arrival estimation...")
    
    config = CityConfig(max_cells=15, num_days=3, freq="1h")
    data, grid, adj = generate_synthetic_data(config)
    
    # Test KDE fitting from demand
    kde = SpatioTemporalKDE()
    kde.fit_from_demand(data)
    
    print(f"\n=== KDE Fitted ===")
    print(f"  Arrival samples: {len(kde.arrival_data)}")
    print(f"  Unique cells: {kde.arrival_data['h3_cell'].nunique()}")
    
    # Test arrival rate estimation
    test_cell = data["h3_cell"].iloc[0]
    for hour in [8.0, 12.0, 18.0, 22.0]:
        rate = kde.estimate_arrival_rate(test_cell, hour, is_weekend=False)
        print(f"  Arrival rate at {hour:.0f}:00 (weekday): {rate:.2f}/hr")
    
    # Test peak finding
    peaks = kde.get_peak_arrival_times(test_cell, is_weekend=False)
    print(f"\n=== Peak Arrival Times ===")
    for hour, rate in peaks:
        print(f"  {hour:.1f}:00 - Rate: {rate:.2f}/hr")
    
    # Test feature enrichment
    enriched = add_kde_features(data, adj)
    print(f"\n=== Enriched Features ===")
    print(f"  New columns: {[c for c in enriched.columns if c not in data.columns]}")
    print(f"  Avg arrival rate: {enriched['predicted_arrival_rate'].mean():.2f}")
    print(f"  Avg neighbor demand: {enriched['neighbor_demand_kw'].mean():.2f} kW")
    print(f"  Avg neighbor pressure: {enriched['neighbor_pressure'].mean():.4f}")
    
    print("\nKDE arrival estimation test PASSED!")
