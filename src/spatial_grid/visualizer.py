import pandas as pd
import matplotlib.pyplot as plt
import os

# Get repo root path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(REPO_ROOT, "data", "raw", "synthetic_telemetry.csv")


def plot_aggregate_load(df):
    """Original view: Total load across all hexagons."""
    hourly_summary = df.groupby('timestamp').agg({
        'residential_kw': 'sum',
        'ev_unmanaged_kw': 'sum',
        'capacity_limit_kw': 'sum'
    }).reset_index()

    plt.figure(figsize=(14, 6))
    plt.stackplot(hourly_summary['timestamp'], 
                  hourly_summary['residential_kw'], 
                  hourly_summary['ev_unmanaged_kw'], 
                  labels=['Existing Residential Demand', 'Unmanaged EV Load'],
                  colors=['#FFA500', '#800080'], alpha=0.7)

    plt.axhline(y=hourly_summary['capacity_limit_kw'].iloc[0], color='r', linestyle='--', 
                label='Aggregate Grid Capacity')

    plt.title("Vidyut Prajna: Aggregate Load Profile (7 Days)")
    plt.xlabel("Time")
    plt.ylabel("Total Power Demand (kW)")
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_per_hex_breaches(df):
    """Per-hex capacity breach analysis - the real problem."""
    # Count breaches per hour across all hexagons
    breach_by_time = df.groupby('timestamp')['capacity_breach'].sum().reset_index()
    breach_by_time.columns = ['timestamp', 'hexes_in_breach']
    
    # Add day type for coloring
    breach_by_time['day_type'] = df.groupby('timestamp')['day_type'].first().values
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Number of hexagons breaching capacity per hour
    colors = ['#E74C3C' if d == 'weekday' else '#3498DB' for d in breach_by_time['day_type']]
    axes[0].bar(breach_by_time['timestamp'], breach_by_time['hexes_in_breach'], 
                color=colors, alpha=0.7, width=0.03)
    axes[0].set_title("Per-Hex Capacity Breaches Over Time")
    axes[0].set_ylabel("# Hexagons Exceeding Capacity")
    axes[0].set_xlabel("Time")
    
    # Add legend for day type
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E74C3C', alpha=0.7, label='Weekday'),
                       Patch(facecolor='#3498DB', alpha=0.7, label='Weekend')]
    axes[0].legend(handles=legend_elements, loc='upper right')
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    
    # Plot 2: Average load profile by hour (weekday vs weekend)
    df['hour'] = df['timestamp'].dt.hour
    hourly_profile = df.groupby(['hour', 'day_type']).agg({
        'total_demand_kw': 'mean',
        'capacity_limit_kw': 'first'
    }).reset_index()
    
    weekday_data = hourly_profile[hourly_profile['day_type'] == 'weekday']
    weekend_data = hourly_profile[hourly_profile['day_type'] == 'weekend']
    
    axes[1].plot(weekday_data['hour'], weekday_data['total_demand_kw'], 
                 'o-', color='#E74C3C', label='Weekday Avg', linewidth=2, markersize=6)
    axes[1].plot(weekend_data['hour'], weekend_data['total_demand_kw'], 
                 's-', color='#3498DB', label='Weekend Avg', linewidth=2, markersize=6)
    axes[1].axhline(y=150, color='r', linestyle='--', label='Capacity Limit (150 kW)', alpha=0.7)
    
    axes[1].set_title("Average Per-Hex Load Profile: Weekday vs Weekend")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Avg Demand per Hex (kW)")
    axes[1].set_xticks(range(0, 24))
    axes[1].legend(loc='upper left')
    axes[1].grid(axis='both', linestyle=':', alpha=0.6)
    axes[1].fill_between(weekday_data['hour'], weekday_data['total_demand_kw'], 150, 
                         where=weekday_data['total_demand_kw'] > 150, 
                         color='#E74C3C', alpha=0.3, label='_Breach Zone')
    
    plt.tight_layout()


def print_breach_summary(df):
    """Print summary statistics about capacity breaches."""
    total_records = len(df)
    breach_records = df['capacity_breach'].sum()
    unique_hexes = df['hex_id'].nunique()
    hexes_with_breaches = df[df['capacity_breach']]['hex_id'].nunique()
    
    print("\n" + "="*50)
    print("CAPACITY BREACH SUMMARY")
    print("="*50)
    print(f"Total hex-hours analyzed: {total_records:,}")
    print(f"Hex-hours with breach:    {breach_records:,} ({100*breach_records/total_records:.1f}%)")
    print(f"Unique hexagons:          {unique_hexes}")
    print(f"Hexagons with breaches:   {hexes_with_breaches} ({100*hexes_with_breaches/unique_hexes:.1f}%)")
    
    # Breakdown by day type
    print("\nBy Day Type:")
    for day_type in ['weekday', 'weekend']:
        subset = df[df['day_type'] == day_type]
        breaches = subset['capacity_breach'].sum()
        total = len(subset)
        print(f"  {day_type.capitalize()}: {breaches:,} breaches ({100*breaches/total:.1f}%)")
    
    # Peak breach hours
    df['hour'] = df['timestamp'].dt.hour
    breach_by_hour = df[df['capacity_breach']].groupby('hour').size()
    if len(breach_by_hour) > 0:
        peak_hour = breach_by_hour.idxmax()
        print(f"\nPeak breach hour: {peak_hour}:00 ({breach_by_hour[peak_hour]} breaches)")
    print("="*50 + "\n")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print_breach_summary(df)
    plot_aggregate_load(df)
    plot_per_hex_breaches(df)
    plt.show()