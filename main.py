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

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_: object, **__: object) -> bool:
        return False


load_dotenv()

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
    config = CityConfig(
        h3_resolution=int(os.getenv("H3_RESOLUTION", "8")),
        max_cells=int(os.getenv("MAX_CELLS", "54")),
        num_days=int(os.getenv("NUM_DAYS", "7")),
        freq=os.getenv("FREQ", "1h"),
        scenario=os.getenv("SCENARIO", "orr_whitefield"),
    )
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
    config = CityConfig(max_cells=18, num_days=4, freq="1h", scenario="south_residential")
    data, grid, adj = generate_synthetic_data(config)
    assert len(grid) > 0, "Grid generation failed"
    assert len(data) > 0, "Data generation failed"
    assert len(adj) > 0, "Adjacency generation failed"
    visited = set()
    stack = [grid["h3_cell"].iloc[0]]
    while stack:
        cell = stack.pop()
        if cell in visited:
            continue
        visited.add(cell)
        stack.extend([n for n in adj.get(cell, []) if n not in visited])
    assert len(visited) == len(grid), "Generated H3 cells are not contiguous"
    assert {"station_count", "demand_growth_index", "corridor_name"}.issubset(data.columns), "Planning features missing"
    print(f"Generated {len(grid)} adjacent cells, {len(data)} rows")
    print("PASSED\n")
    
    # Test intelligence module
    print("=" * 50)
    print("Testing Intelligence Module")
    print("=" * 50)
    import torch
    from src.intelligence.forecaster import FUTURE_EXOG_COLS, SEQUENCE_FEATURE_COLS, STGCNForecaster
    from src.intelligence.graph_utils import get_adjacency_matrix
    from src.intelligence.model import VidyutPrajnaForecaster
    
    times = sorted(data["timestamp"].unique())
    train_split = len(times) - 12
    train = data[data["timestamp"].isin(times[:train_split])]
    future = data[data["timestamp"].isin(times[train_split:train_split + 12])]

    adj_tensor = get_adjacency_matrix(sorted(grid["h3_cell"].tolist()))
    shape_model = VidyutPrajnaForecaster(
        adj_matrix=adj_tensor,
        in_channels=len(SEQUENCE_FEATURE_COLS),
        future_channels=len(FUTURE_EXOG_COLS),
        hidden_channels=12,
        num_blocks=1,
    )
    mock_hist = torch.randn(2, 4, len(grid), len(SEQUENCE_FEATURE_COLS))
    mock_future = torch.randn(2, len(grid), len(FUTURE_EXOG_COLS))
    with torch.no_grad():
        shape_pred = shape_model(mock_hist, mock_future)
    assert shape_pred.shape == (2, len(grid), 1), "Future-conditioned STGCN shape mismatch"
    
    forecaster = STGCNForecaster(seq_len=8, epochs=1, num_blocks=1, hidden_size=16)
    forecaster.fit(train, adj)
    assert forecaster.training_info is not None, "Training failed"
    print(f"Model trained: {forecaster.training_info.epochs} epochs")
    
    pred = forecaster.forecast(train, future, adj, horizon_steps=12)
    assert len(pred) > 0, "Forecasting failed"
    assert "predicted_demand_kw" in pred.columns, "Missing predictions"
    assert {"stgcn_predicted_demand_kw", "seasonal_baseline_kw", "forecast_method"}.issubset(pred.columns), "Forecast diagnostics missing"
    agg_pred = pred.groupby("timestamp")["predicted_demand_kw"].sum()
    agg_actual = pred.groupby("timestamp")["actual_demand_kw"].sum()
    assert agg_pred.std() > max(1.0, float(agg_actual.std() or 0) * 0.25), "Aggregate forecast is still effectively flat"
    flat = pred.copy()
    flat["stgcn_predicted_demand_kw"] = float(flat["actual_demand_kw"].mean())
    method, info = forecaster._choose_forecast_method(flat)
    assert method in {"seasonal_baseline", "stgcn_seasonal_blend"}, "Flatline guardrail did not activate"
    print(f"Forecast generated: {len(pred)} rows")
    print(f"Forecast method: {forecaster.forecast_info.get('forecast_method')}")
    print("PASSED\n")
    
    # Test optimization module
    print("=" * 50)
    print("Testing Optimization Module")
    print("=" * 50)
    from src.optimization.optimizer import optimize_charging_schedule
    
    optimized, metrics = optimize_charging_schedule(pred)
    assert "optimized_ev_load_kw" in optimized.columns, "Optimization failed"
    assert "peak_reduction_pct" in metrics, "Metrics missing"
    assert abs(metrics["energy_preservation_error_pct"]) < 0.5, "EV energy was not preserved"
    assert "overload_events_before" in metrics and "overload_events_after" in metrics, "Overload metrics missing"
    print(f"Peak reduction: {metrics['peak_reduction_pct']:.1f}%")
    print(f"Cost savings: ₹{metrics['estimated_cost_savings_inr']:.0f}")
    print("PASSED\n")

    # Test infrastructure siting
    print("=" * 50)
    print("Testing Infrastructure Siting Module")
    print("=" * 50)
    from src.optimization.siting import recommend_station_locations

    recommendations, siting_summary = recommend_station_locations(optimized, adj, top_n=4)
    assert len(recommendations) == 4, "Siting recommendations missing"
    assert recommendations["siting_score"].is_monotonic_decreasing, "Siting ranks are not sorted"
    assert "capture_improvement_pct" in siting_summary, "Uniform baseline comparison missing"
    print(f"Top site: {recommendations.iloc[0]['zone_name']} ({recommendations.iloc[0]['siting_score']:.1f})")
    print("PASSED\n")

    # Test LLM fallback without hosted API
    print("=" * 50)
    print("Testing Grounded Explanation Fallback")
    print("=" * 50)
    from src.dashboard.llm_interface import VidyutLLM
    from src.dashboard.utils import build_llm_context

    context = build_llm_context(metrics, optimized, 0, recommendations, siting_summary)
    answer = VidyutLLM(api_key="").answer("Which zone is highest risk?", context)
    assert "Local grounded explanation" in answer, "LLM fallback did not run"
    print("LLM fallback summary generated")
    print("PASSED\n")

    # Test dashboard bootstrap/import on a deliberately small demo.
    print("=" * 50)
    print("Testing Dashboard Bootstrap")
    print("=" * 50)
    os.environ.update({
        "MAX_CELLS": "8",
        "NUM_DAYS": "2",
        "FREQ": "1h",
        "EPOCHS": "1",
        "SEQ_LEN": "4",
        "HIDDEN_SIZE": "12",
        "STGCN_BLOCKS": "1",
        "STATION_BUDGET": "3",
    })
    from src.dashboard.app import app as dash_app

    assert dash_app.title == "Vidyut Prajna", "Dashboard app did not bootstrap"
    print("Dashboard app imported successfully")
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
