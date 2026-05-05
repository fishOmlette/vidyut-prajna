"""Smoke tests for the Vidyut Prajna intelligence layer."""

from __future__ import annotations

import torch

from src.intelligence.forecaster import FUTURE_EXOG_COLS, SEQUENCE_FEATURE_COLS, STGCNForecaster
from src.intelligence.graph_utils import get_adjacency_matrix
from src.intelligence.model import STGCNBlock, VidyutPrajnaForecaster
from src.spatial_grid.simulation import CityConfig, generate_synthetic_data


def test_intelligence_suite():
    data, grid, adj = generate_synthetic_data(
        CityConfig(max_cells=18, num_days=4, freq="1h", scenario="orr_whitefield")
    )
    cells = sorted(grid["h3_cell"].tolist())
    adj_matrix = get_adjacency_matrix(cells)
    assert adj_matrix.shape == (len(cells), len(cells))
    assert torch.all(torch.diag(adj_matrix) > 0)

    batch_size = 2
    time_steps = 8
    stgcn_block = STGCNBlock(len(SEQUENCE_FEATURE_COLS), 16, adj_matrix)
    mock_hist = torch.randn(batch_size, time_steps, len(cells), len(SEQUENCE_FEATURE_COLS))
    with torch.no_grad():
        block_out = stgcn_block(mock_hist)
    assert block_out.shape == (batch_size, time_steps, len(cells), 16)

    model = VidyutPrajnaForecaster(
        adj_matrix=adj_matrix,
        in_channels=len(SEQUENCE_FEATURE_COLS),
        future_channels=len(FUTURE_EXOG_COLS),
        hidden_channels=16,
        out_channels=1,
        num_blocks=1,
    )
    mock_future = torch.randn(batch_size, len(cells), len(FUTURE_EXOG_COLS))
    with torch.no_grad():
        pred = model(mock_hist, mock_future)
    assert pred.shape == (batch_size, len(cells), 1)

    times = sorted(data["timestamp"].unique())
    train_times = times[:-12]
    future_times = times[-12:]
    train = data[data["timestamp"].isin(train_times)]
    future = data[data["timestamp"].isin(future_times)]

    forecaster = STGCNForecaster(seq_len=8, epochs=1, hidden_size=16, num_blocks=1)
    forecaster.fit(train, adj)
    forecast = forecaster.forecast(train, future, adj, horizon_steps=12)
    agg_pred = forecast.groupby("timestamp")["predicted_demand_kw"].sum()
    agg_actual = forecast.groupby("timestamp")["actual_demand_kw"].sum()
    assert agg_pred.std() > max(1.0, float(agg_actual.std() or 0.0) * 0.25)
    assert "forecast_method" in forecast.columns
    assert forecaster.forecast_info.get("forecast_method") is not None

    flat = forecast.copy()
    flat["stgcn_predicted_demand_kw"] = float(flat["actual_demand_kw"].mean())
    method, _ = forecaster._choose_forecast_method(flat)
    assert method in {"seasonal_baseline", "stgcn_seasonal_blend"}

    print("All Intelligence Plane tests PASSED.")


if __name__ == "__main__":
    test_intelligence_suite()
