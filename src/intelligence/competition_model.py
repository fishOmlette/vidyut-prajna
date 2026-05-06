"""Competition-grade probabilistic graph temporal forecaster.

The model is deliberately dependency-light: it uses only PyTorch, but borrows
the strongest practical ideas from TFT-style forecasting, graph attention, and
probabilistic demand prediction.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileLoss(nn.Module):
    """Pinball loss for multi-horizon probabilistic forecasts."""

    def __init__(self, quantiles: Iterable[float] = (0.1, 0.5, 0.9)):
        super().__init__()
        q = torch.tensor(list(quantiles), dtype=torch.float32)
        if q.ndim != 1 or len(q) < 1:
            raise ValueError("quantiles must be a non-empty 1D iterable")
        self.register_buffer("quantiles", q)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # prediction: (batch, horizon, nodes, quantiles)
        # target:     (batch, horizon, nodes)
        error = target.unsqueeze(-1) - prediction
        q = self.quantiles.view(*([1] * (prediction.ndim - 1)), -1)
        loss = torch.maximum(q * error, (q - 1.0) * error)
        return loss.mean()


class GatedResidualNetwork(nn.Module):
    """TFT-style gated residual block with optional dimensional projection."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gated = h * torch.sigmoid(self.gate(h))
        return self.norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """Learns feature-wise relevance weights and embeds scalars to hidden space."""

    def __init__(self, num_features: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.feature_projections = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in range(num_features)]
        )
        self.weight_grn = GatedResidualNetwork(num_features, hidden_dim, num_features, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (..., features)
        weights = torch.softmax(self.weight_grn(x), dim=-1)
        encoded = []
        for idx, projection in enumerate(self.feature_projections):
            encoded.append(F.elu(projection(x[..., idx:idx + 1])))
        encoded_tensor = torch.stack(encoded, dim=-2)  # (..., features, hidden)
        selected = (weights.unsqueeze(-1) * encoded_tensor).sum(dim=-2)
        return selected, weights


class DynamicGraphAttention(nn.Module):
    """Multi-head node attention biased by H3/topology adjacency.

    H3 neighbors receive a strong prior, but the learned edge bias allows
    long-range corridor correlations when the data proves they matter.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        adj_matrix: torch.Tensor,
        dropout: float = 0.1,
        topology_strength: float = 0.85,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("adj_matrix must be square")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_nodes = int(adj_matrix.shape[0])

        adj = adj_matrix.detach().clone().float()
        adj = (adj > 0).float()
        adj.fill_diagonal_(1.0)
        prior = torch.log((1.0 - topology_strength) + topology_strength * adj)
        self.register_buffer("topology_prior", prior)
        self.edge_bias = nn.Parameter(torch.zeros(num_heads, self.num_nodes, self.num_nodes))

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, time, nodes, hidden)
        b, t, n, d = x.shape
        if n != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} graph nodes, got {n}")

        q = self.q_proj(x).view(b, t, n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, n, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, n, self.num_heads, self.head_dim)

        logits = torch.einsum("btnhd,btmhd->bthnm", q, k) / math.sqrt(self.head_dim)
        logits = logits + self.topology_prior.view(1, 1, 1, n, n)
        logits = logits + self.edge_bias.view(1, 1, self.num_heads, n, n)
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)

        h = torch.einsum("bthnm,btmhd->btnhd", attention, v).reshape(b, t, n, d)
        h = self.out_proj(h)
        out = self.norm(x + self.dropout(h))
        if return_attention:
            return out, attention
        return out


class GraphTemporalFusionTransformer(nn.Module):
    """Direct multi-horizon EV demand forecaster.

    Args:
        adj_matrix: normalized or binary H3 adjacency matrix, shape (nodes, nodes).
        history_features: number of historical feature channels.
        future_features: number of known-future exogenous feature channels.
        hidden_dim: latent width.
        horizon: number of forecast steps emitted in one forward pass.
        quantiles: output quantiles, defaults to p10/p50/p90.
    """

    def __init__(
        self,
        adj_matrix: torch.Tensor,
        history_features: int,
        future_features: int,
        hidden_dim: int = 64,
        horizon: int = 24,
        quantiles: Iterable[float] = (0.1, 0.5, 0.9),
        num_heads: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 256,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.num_quantiles = len(self.quantiles)
        self.hidden_dim = hidden_dim

        self.history_vsn = VariableSelectionNetwork(history_features, hidden_dim, dropout)
        self.future_vsn = VariableSelectionNetwork(future_features, hidden_dim, dropout)
        self.history_graph = DynamicGraphAttention(hidden_dim, num_heads, adj_matrix, dropout)
        self.future_graph = DynamicGraphAttention(hidden_dim, num_heads, adj_matrix, dropout)

        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.decoder_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.position = nn.Parameter(torch.zeros(1, max_sequence_length, hidden_dim))
        nn.init.normal_(self.position, mean=0.0, std=0.02)

        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(3, self.num_quantiles)),
        )

    def forward(
        self,
        history_x: torch.Tensor,
        future_x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # history_x: (batch, seq_len, nodes, history_features)
        # future_x:  (batch, horizon, nodes, future_features)
        b, seq_len, n, _ = history_x.shape
        horizon = future_x.shape[1]
        if horizon > self.horizon:
            raise ValueError(f"Model was initialized for horizon {self.horizon}, got {horizon}")
        if seq_len + horizon > self.position.shape[1]:
            raise ValueError("Increase max_sequence_length for this sequence+horizon")

        hist_h, hist_weights = self.history_vsn(history_x)
        hist_h = self.history_graph(hist_h)
        hist_seq = hist_h.permute(0, 2, 1, 3).reshape(b * n, seq_len, self.hidden_dim)
        encoded_hist, _ = self.encoder(hist_seq)
        encoded_hist = encoded_hist.reshape(b, n, seq_len, self.hidden_dim).permute(0, 2, 1, 3)

        future_h, future_weights = self.future_vsn(future_x)
        tokens = torch.cat([encoded_hist, future_h], dim=1)
        tokens = tokens + self.position[:, : seq_len + horizon, :].view(1, seq_len + horizon, 1, self.hidden_dim)

        temporal_tokens = tokens.permute(0, 2, 1, 3).reshape(b * n, seq_len + horizon, self.hidden_dim)
        attended, temporal_attention = self.temporal_attention(
            temporal_tokens,
            temporal_tokens,
            temporal_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        temporal_tokens = self.temporal_norm(temporal_tokens + self.dropout(attended))
        decoded = temporal_tokens[:, -horizon:, :].reshape(b, n, horizon, self.hidden_dim).permute(0, 2, 1, 3)
        decoded = self.future_graph(self.decoder_grn(decoded))

        raw = self.quantile_head(decoded)
        if self.num_quantiles == 3:
            lower_delta = F.softplus(raw[..., 0])
            median = raw[..., 1]
            upper_delta = F.softplus(raw[..., 2])
            prediction = torch.stack([median - lower_delta, median, median + upper_delta], dim=-1)
        else:
            prediction = torch.sort(raw[..., : self.num_quantiles], dim=-1).values

        if not return_attention:
            return prediction
        diagnostics = {
            "history_feature_weights": hist_weights,
            "future_feature_weights": future_weights,
        }
        if isinstance(temporal_attention, torch.Tensor):
            diagnostics["temporal_attention"] = temporal_attention
        return prediction, diagnostics
