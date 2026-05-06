"""Legacy grid generator for backward compatibility."""

import h3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Seed for reproducibility
np.random.seed(42)

class VidyutPrajnaGrid:
    def __init__(self, center_lat=12.9716, center_lng=77.5946, resolution=9):
        """
        Initializes the Bengaluru H3 grid.
        Default: Bangalore City Center (MG Road area).
        """
        self.center = (center_lat, center_lng)
        self.res = resolution # Level 9 as per proposal

    def generate_ward_hexagons(self, radius_km=3):
        """Creates a hexagonal mesh around a target ward."""
        base_hex = h3.latlng_to_cell(self.center[0], self.center[1], self.res)
        # grid_disk defines the 'living organism' spread of the grid[cite: 1]
        hexagons = h3.grid_disk(base_hex, radius_km)
        return list(hexagons)

    def simulate_grid_telemetry(self, hex_ids, days=7):
        """
        Generates 'One-Way Data Ingest' telemetry[cite: 1].
        Simulates the Coincident Peak: Residential + Unmanaged EV load[cite: 1].
        Now includes weekend/weekday variation.
        """
        records = []
        start_time = datetime(2026, 5, 1, 0, 0)  # Friday
        
        for h_id in hex_ids:
            for hour_offset in range(24 * days):
                current_time = start_time + timedelta(hours=hour_offset)
                hour = current_time.hour
                day_of_week = current_time.weekday()  # 0=Monday, 5=Saturday, 6=Sunday
                is_weekend = day_of_week >= 5
                
                # 1. Base Residential Load (Evening Spike)
                # Peaks around 19:00 - 21:00 (HSR Layout style)[cite: 1]
                # Weekends: Higher daytime load, slightly lower evening peak
                if is_weekend:
                    res_base = 100 * (1 + 0.3 * np.sin((hour - 14) * np.pi / 12))
                    res_base *= 1.1  # 10% higher overall on weekends (people at home)
                else:
                    res_base = 100 * (1 + 0.5 * np.sin((hour - 15) * np.pi / 12))
                
                # 2. Unmanaged EV Load (The Risk Factor)
                # Aligning with residential peak creates the 'instability' valley[cite: 1]
                # Weekends: Charging spread throughout the day
                ev_unmanaged = 0
                if is_weekend:
                    if 10 <= hour <= 22:  # Wider charging window on weekends
                        ev_unmanaged = np.random.uniform(30, 60)
                else:
                    if 18 <= hour <= 22:  # Typical unmanaged charging window[cite: 1]
                        ev_unmanaged = np.random.uniform(40, 80)
                
                total_demand = res_base + ev_unmanaged
                capacity_limit = 150.0  # Simulated DTR limit[cite: 1]
                
                records.append({
                    "hex_id": h_id,
                    "timestamp": current_time,
                    "day_type": "weekend" if is_weekend else "weekday",
                    "residential_kw": round(res_base, 2),
                    "ev_unmanaged_kw": round(ev_unmanaged, 2),
                    "total_demand_kw": round(total_demand, 2),
                    "capacity_limit_kw": capacity_limit,
                    "capacity_breach": total_demand > capacity_limit
                })
        
        return pd.DataFrame(records)

if __name__ == "__main__":
    # Get repo root path (two levels up from this file)
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(REPO_ROOT, "data", "raw")
    
    # Ensure data directory exists
    os.makedirs(DATA_PATH, exist_ok=True)
    
    grid_engine = VidyutPrajnaGrid()
    whitefield_hexes = grid_engine.generate_ward_hexagons(radius_km=2)
    telemetry_df = grid_engine.simulate_grid_telemetry(whitefield_hexes, days=7)
    
    # Save for Phase 2: Intelligence Plane
    output_file = os.path.join(DATA_PATH, "synthetic_telemetry.csv")
    telemetry_df.to_csv(output_file, index=False)
    
    # Summary stats
    breach_count = telemetry_df['capacity_breach'].sum()
    total_hours = len(telemetry_df)
    print(f"Generated {total_hours} rows for {len(whitefield_hexes)} hex cells over 7 days.")
    print(f"Capacity breaches: {breach_count} ({100*breach_count/total_hours:.1f}% of all hex-hours)")