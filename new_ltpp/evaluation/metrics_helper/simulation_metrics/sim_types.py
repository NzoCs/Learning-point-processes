import torch
from enum import StrEnum
from typing import TypedDict


class SimTimeValues(TypedDict):
    """Container for simulation time-related values."""

    true_time_seqs: torch.Tensor
    true_time_delta_seqs: torch.Tensor
    sim_time_seqs: torch.Tensor
    sim_time_delta_seqs: torch.Tensor


class SimTypeValues(TypedDict):
    """Container for simulation type-related values."""

    true_type_seqs: torch.Tensor
    sim_type_seqs: torch.Tensor



class SimMetrics(StrEnum):
    """Available simulation metrics."""

    WASSERSTEIN_1D = "wasserstein_1d"
    MMD_RBF_PADDED = "mmd_rbf_padded"
    MMD_WASSERSTEIN = "mmd_wasserstein"