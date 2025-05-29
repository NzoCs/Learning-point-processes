"""
Shared types and enums for the metrics helper package.
"""

import torch
from dataclasses import dataclass
from enum import Enum


class EvaluationMode(Enum):
    """Evaluation modes for metrics computation."""
    SIMULATION = "simulation"
    PREDICTION = "prediction"


@dataclass
class MaskedValues:
    """Container for masked prediction and ground truth values."""
    true_times: torch.Tensor
    true_types: torch.Tensor
    pred_times: torch.Tensor
    pred_types: torch.Tensor
