from enum import StrEnum
from typing import TypedDict

import torch


class SimTimeValues(TypedDict):
    """Container for simulation time-related values, including masks for vectorized metrics."""

    true_time_seqs: torch.Tensor  # (B, N)
    true_time_delta_seqs: torch.Tensor  # (B, N)
    sim_time_seqs: torch.Tensor  # (B, N)
    sim_time_delta_seqs: torch.Tensor  # (B, N)
    true_mask: torch.Tensor  # (B, N)  -> 1=valid, 0=padding
    sim_mask: torch.Tensor  # (B, N)  -> 1=valid, 0=padding


class SimTypeValues(TypedDict):
    """Container for simulation type-related values, including masks."""

    true_type_seqs: torch.Tensor  # (B, N)
    sim_type_seqs: torch.Tensor  # (B, N)
    true_mask: torch.Tensor  # (B, N)  -> 1=valid, 0=padding
    sim_mask: torch.Tensor  # (B, N)  -> 1=valid, 0=padding


class SimMetrics(StrEnum):
    """Available simulation metrics."""

    WASSERSTEIN_1D = "wasserstein_1d"
    MMD_RBF_PADDED = "mmd_rbf_padded"
    MMD_WASSERSTEIN = "mmd_wasserstein"
