import torch
import pandas as pd
import numpy as np
import os
from graph_utils import get_adjacency_matrix
from model import STGCNBlock, VidyutPrajnaForecaster

def test_intelligence_suite():
    # 1. Setup Data Paths
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(REPO_ROOT, "data", "raw", "synthetic_telemetry.csv")
    
    if not os.path.exists(DATA_PATH):
        print("Error: Run generator.py first to create synthetic data.")
        return

    # 2. Test Adjacency Matrix Logic
    print("--- Testing Graph Utilities ---")
    df = pd.read_csv(DATA_PATH)
    unique_hexes = df['hex_id'].unique().tolist()
    num_nodes = len(unique_hexes)
    
    adj = get_adjacency_matrix(unique_hexes)
    
    # Assertions for Graph Sanity
    assert adj.shape == (num_nodes, num_nodes), f"Expected ({num_nodes}, {num_nodes}), got {adj.shape}"
    assert torch.all(torch.diag(adj) > 0), "Self-loops (diagonal) missing in adjacency matrix."
    print(f"Adjacency Matrix Verified: {num_nodes}x{num_nodes} nodes.")

    # 3. Test STGCN Block
    print("\n--- Testing STGCN Block ---")
    batch_size = 8
    time_steps = 12
    in_channels = 2
    out_channels = 16
    
    mock_input = torch.randn(batch_size, time_steps, num_nodes, in_channels)
    
    stgcn_block = STGCNBlock(in_channels, out_channels, adj)
    
    with torch.no_grad():
        output = stgcn_block(mock_input)
    
    expected_shape = (batch_size, time_steps, num_nodes, out_channels)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"STGCN Block Forward Pass Verified: Output shape is {output.shape}")

    # 4. Test Full Forecaster Model
    print("\n--- Testing VidyutPrajnaForecaster ---")
    forecaster = VidyutPrajnaForecaster(
        adj_matrix=adj,
        in_channels=2,  # [residential_kw, ev_unmanaged_kw]
        hidden_channels=32,
        out_channels=1,  # Predict total_demand_kw
        num_blocks=2
    )
    
    with torch.no_grad():
        prediction = forecaster(mock_input)
    
    # Output: (batch, num_nodes, 1) - next hour prediction per hex
    expected_pred_shape = (batch_size, num_nodes, 1)
    assert prediction.shape == expected_pred_shape, f"Expected {expected_pred_shape}, got {prediction.shape}"
    print(f"Forecaster Verified: Predicts {prediction.shape} (batch, nodes, features)")
    
    # 5. Test model parameter count
    total_params = sum(p.numel() for p in forecaster.parameters())
    trainable_params = sum(p.numel() for p in forecaster.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total")

    print("\nAll Intelligence Plane unit tests PASSED.")

if __name__ == "__main__":
    test_intelligence_suite()