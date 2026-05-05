"""Vidyut Prajna - AI-Driven EV Charging Optimization for Bengaluru.

Entry point for the application.

Usage:
    python main.py              # Run the dashboard
    python main.py --test       # Run tests
    python main.py --simulate   # Generate simulation data only
"""

import argparse
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_dashboard():
    """Run the Dash dashboard."""
    from src.dashboard.app import run_server
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    print(f"\nStarting Vidyut Prajna dashboard on http://127.0.0.1:{port}")
    run_server(debug=debug, port=port)


def run_simulation():
    """Generate and display simulation data."""
    from src.spatial_grid.simulation import CityConfig, generate_synthetic_data
    
    print("Generating synthetic Bengaluru data...")
    config = CityConfig(max_cells=20, num_days=3)
    data, grid, adj = generate_synthetic_data(config)
    
    print(f"\nGrid: {len(grid)} H3 cells")
    print(f"Zone types: {grid['zone_type'].value_counts().to_dict()}")
    print(f"\nTime series: {len(data)} rows")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Save to data folder
    os.makedirs("data/processed", exist_ok=True)
    data.to_csv("data/processed/simulation_data.csv", index=False)
    grid.to_csv("data/processed/grid_metadata.csv", index=False)
    print("\nData saved to data/processed/")


def run_tests():
    """Run the test suite."""
    print("Running Vidyut Prajna tests...\n")
    
    # Test spatial grid
    print("=" * 50)
    print("Testing Spatial Grid Module")
    print("=" * 50)
    from src.spatial_grid.simulation import CityConfig, generate_synthetic_data
    config = CityConfig(max_cells=10, num_days=2, freq="1h")
    data, grid, adj = generate_synthetic_data(config)
    assert len(grid) > 0, "Grid generation failed"
    assert len(data) > 0, "Data generation failed"
    assert len(adj) > 0, "Adjacency generation failed"
    print(f"Generated {len(grid)} cells, {len(data)} rows")
    print("PASSED\n")
    
    # Test intelligence module
    print("=" * 50)
    print("Testing Intelligence Module")
    print("=" * 50)
    from src.intelligence.forecaster import STGCNForecaster
    
    times = sorted(data["timestamp"].unique())
    train_split = int(len(times) * 0.7)
    train = data[data["timestamp"].isin(times[:train_split])]
    future = data[data["timestamp"].isin(times[train_split:train_split + 6])]
    
    forecaster = STGCNForecaster(seq_len=4, epochs=2, num_blocks=1)
    forecaster.fit(train, adj)
    assert forecaster.training_info is not None, "Training failed"
    print(f"Model trained: {forecaster.training_info.epochs} epochs")
    
    pred = forecaster.forecast(train, future, adj, horizon_steps=3)
    assert len(pred) > 0, "Forecasting failed"
    assert "predicted_demand_kw" in pred.columns, "Missing predictions"
    print(f"Forecast generated: {len(pred)} rows")
    print("PASSED\n")
    
    # Test optimization module
    print("=" * 50)
    print("Testing Optimization Module")
    print("=" * 50)
    from src.optimization.optimizer import optimize_charging_schedule
    
    optimized, metrics = optimize_charging_schedule(pred)
    assert "optimized_ev_load_kw" in optimized.columns, "Optimization failed"
    assert "peak_reduction_pct" in metrics, "Metrics missing"
    print(f"Peak reduction: {metrics['peak_reduction_pct']:.1f}%")
    print(f"Cost savings: ₹{metrics['estimated_cost_savings_inr']:.0f}")
    print("PASSED\n")
    
    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Vidyut Prajna - AI-Driven EV Charging Optimization"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run the test suite"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Generate simulation data only"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.simulate:
        run_simulation()
    else:
        run_dashboard()


if __name__ == "__main__":
    main()
