"""Intelligence module for Vidyut Prajna.

Contains STGCN baselines and competition-grade probabilistic graph forecasting
models for EV charging demand prediction.
"""

from .model import STGCNBlock, VidyutPrajnaForecaster
from .graph_utils import get_adjacency_matrix
from .forecaster import (
    FEATURE_COLS,
    FUTURE_EXOG_COLS,
    SEQUENCE_FEATURE_COLS,
    STGCNForecaster,
    TrainingInfo,
)
from .competition_forecaster import (
    ADVANCED_FUTURE_EXOG_COLS,
    ADVANCED_SEQUENCE_FEATURE_COLS,
    CompetitionForecaster,
    CompetitionTrainingInfo,
)
from .competition_model import GraphTemporalFusionTransformer, QuantileLoss

__all__ = [
    "STGCNBlock",
    "VidyutPrajnaForecaster",
    "STGCNForecaster",
    "TrainingInfo",
    "get_adjacency_matrix",
    "FEATURE_COLS",
    "SEQUENCE_FEATURE_COLS",
    "FUTURE_EXOG_COLS",
    "ADVANCED_SEQUENCE_FEATURE_COLS",
    "ADVANCED_FUTURE_EXOG_COLS",
    "CompetitionForecaster",
    "CompetitionTrainingInfo",
    "GraphTemporalFusionTransformer",
    "QuantileLoss",
]
