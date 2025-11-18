from enum import StrEnum
from typing import TypedDict

import torch


class TimeValues(TypedDict):
    """Container for time-related values only."""

    true_times: torch.Tensor
    pred_times: torch.Tensor


class TypeValues(TypedDict):
    """Container for type-related values only."""

    true_types: torch.Tensor
    pred_types: torch.Tensor


class PredMetrics(StrEnum):
    """Available prediction metrics."""

    TIME_RMSE = "time_rmse"
    TIME_MAE = "time_mae"
    TYPE_ACCURACY = "type_accuracy"
    MACRO_F1SCORE = "macro_f1score"
    RECALL = "recall"
    PRECISION = "precision"
    CROSS_ENTROPY = "cross_entropy"
    CONFUSION_MATRIX = "confusion_matrix"
