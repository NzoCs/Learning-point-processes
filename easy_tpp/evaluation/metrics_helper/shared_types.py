"""
Shared types and enums for the metrics helper package.
"""

from dataclasses import dataclass
from enum import Enum

import torch


class EvaluationMode(Enum):
    """Evaluation modes for metrics computation."""

    SIMULATION = "simulation"
    PREDICTION = "prediction"


class PredictionMetrics(Enum):
    """Available prediction metrics."""

    TIME_RMSE = "time_rmse"
    TIME_MAE = "time_mae"
    TYPE_ACCURACY = "type_accuracy"
    MACRO_F1SCORE = "macro_f1score"
    RECALL = "recall"
    PRECISION = "precision"
    CROSS_ENTROPY = "cross_entropy"
    CONFUSION_MATRIX = "confusion_matrix"


class SimulationMetrics(Enum):
    """Available simulation metrics."""

    WASSERSTEIN_1D = "wasserstein_1d"
    MMD_RBF_PADDED = "mmd_rbf_padded"
    MMD_WASSERSTEIN = "mmd_wasserstein"


@dataclass
class MaskedValues:
    """Container for masked prediction and ground truth values."""

    true_times: torch.Tensor
    true_types: torch.Tensor
    pred_times: torch.Tensor
    pred_types: torch.Tensor


@dataclass
class TimeValues:
    """Container for time-related values only."""

    true_times: torch.Tensor
    pred_times: torch.Tensor


@dataclass
class TypeValues:
    """Container for type-related values only."""

    true_types: torch.Tensor
    pred_types: torch.Tensor


@dataclass
class SimulationTimeValues:
    """Container for simulation time-related values."""

    true_time_seqs: torch.Tensor
    true_time_delta_seqs: torch.Tensor
    sim_time_seqs: torch.Tensor
    sim_time_delta_seqs: torch.Tensor
    sim_mask: torch.Tensor


@dataclass
class SimulationTypeValues:
    """Container for simulation type-related values."""

    true_type_seqs: torch.Tensor
    sim_type_seqs: torch.Tensor
    sim_mask: torch.Tensor


@dataclass
class SimulationValues:
    """Container for all simulation values - legacy compatibility."""

    true_time_seqs: torch.Tensor
    true_type_seqs: torch.Tensor
    true_time_delta_seqs: torch.Tensor
    sim_time_seqs: torch.Tensor
    sim_type_seqs: torch.Tensor
    sim_mask: torch.Tensor
