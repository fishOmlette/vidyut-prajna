import torch
import torch.nn as nn
import torch.nn.functional as F


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Block.
    Combines Graph Convolution (spatial) with LSTM (temporal).
    """
    def __init__(self, in_channels, out_channels, adj_matrix, dropout=0.1):
        super(STGCNBlock, self).__init__()
        # Register adjacency as buffer (not a parameter, but moves with model)
        self.register_buffer('adj', adj_matrix)
        
        # Spatial: Graph Convolution
        self.gcn = nn.Linear(in_channels, out_channels)
        self.bn_spatial = nn.BatchNorm1d(out_channels)
        
        # Temporal: LSTM
        self.temporal = nn.LSTM(out_channels, out_channels, batch_first=True)
        self.bn_temporal = nn.BatchNorm1d(out_channels)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, time_steps, num_nodes, features)
        b, t, n, f = x.shape
        
        # 1. Spatial Processing (Graph Convolution)
        # Multiply adjacency (n, n) with features along node dimension
        # adj @ x: for each (b, t) slice, multiply (n, n) @ (n, f) -> (n, f)
        x = torch.einsum('ij,btjf->btif', self.adj, x)
        x = self.gcn(x)  # Linear transform: (b, t, n, f) -> (b, t, n, out)
        
        # BatchNorm expects (N, C, L) or (N, C), reshape accordingly
        x = x.permute(0, 3, 1, 2).reshape(b * x.shape[3], -1, t * n)
        x = x.reshape(b, -1, t * n)  # (b, out_channels, t*n)
        x = self.bn_spatial(x)
        x = x.reshape(b, -1, t, n).permute(0, 2, 3, 1)  # Back to (b, t, n, out)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 2. Temporal Processing (LSTM)
        # Reshape to treat each node's timeline as a sequence
        out_channels = x.shape[-1]
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, out_channels)  # (b*n, t, out)
        x, _ = self.temporal(x)
        
        # Reshape back to original structure
        x = x.reshape(b, n, t, -1).permute(0, 2, 1, 3)  # (b, t, n, out)
        
        # Apply temporal batch norm
        x_flat = x.permute(0, 3, 1, 2).reshape(b, -1, t * n)
        x_flat = self.bn_temporal(x_flat)
        x = x_flat.reshape(b, -1, t, n).permute(0, 2, 3, 1)
        
        return x


class VidyutPrajnaForecaster(nn.Module):
    """
    Full forecasting model with stacked STGCN blocks.
    Predicts next-hour load for each hex cell.
    """
    def __init__(self, adj_matrix, in_channels=2, hidden_channels=32, 
                 out_channels=1, num_blocks=2, dropout=0.1):
        super(VidyutPrajnaForecaster, self).__init__()
        
        self.blocks = nn.ModuleList()
        
        # First block: in_channels -> hidden
        self.blocks.append(STGCNBlock(in_channels, hidden_channels, adj_matrix, dropout))
        
        # Additional blocks: hidden -> hidden
        for _ in range(num_blocks - 1):
            self.blocks.append(STGCNBlock(hidden_channels, hidden_channels, adj_matrix, dropout))
        
        # Output projection: predict next timestep's load
        self.output_layer = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        # x: (batch, time_steps, num_nodes, in_channels)
        for block in self.blocks:
            x = block(x)
        
        # Take last timestep for prediction
        x = x[:, -1, :, :]  # (batch, num_nodes, hidden)
        x = self.output_layer(x)  # (batch, num_nodes, out_channels)
        return x