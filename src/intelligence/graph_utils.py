"""Graph adjacency utilities for STGCN spatial modeling."""

import h3
import torch
import numpy as np

def get_adjacency_matrix(hex_ids):
    """
    Creates a normalized adjacency matrix for the STGCN.
    Nodes = H3 Cells, Edges = Physical Adjacency.
    """
    n = len(hex_ids)
    adj = np.zeros((n, n))
    hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
    
    for hex_id in hex_ids:
        idx_i = hex_to_idx[hex_id]
        # Get immediate neighbors (grid_disk with k=1 includes self)
        neighbors = h3.grid_disk(hex_id, 1)
        for neighbor in neighbors:
            if neighbor in hex_to_idx:
                idx_j = hex_to_idx[neighbor]
                adj[idx_i, idx_j] = 1
    
    # Note: grid_disk(k=1) already includes self-loops (center cell)
    # No need to add np.eye(n) again
    
    # Degree-normalization for GCN stability (symmetric normalization)
    d = np.array(adj.sum(1))
    d_inv = np.power(d, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.diag(d_inv)
    
    norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
    return torch.from_numpy(norm_adj).float()