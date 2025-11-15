from dataclasses import dataclass
import torch
from enum import StrEnum

@dataclass
class SimTimeValues:
    """Container for simulation time-related values."""

    true_time_seqs: torch.Tensor
    true_time_delta_seqs: torch.Tensor
    sim_time_seqs: torch.Tensor
    sim_time_delta_seqs: torch.Tensor


@dataclass
class SimTypeValues:
    """Container for simulation type-related values."""

    true_type_seqs: torch.Tensor
    sim_type_seqs: torch.Tensor



class SimMetrics(StrEnum):
    """Available simulation metrics."""

    WASSERSTEIN_1D = "wasserstein_1d"
    MMD_RBF_PADDED = "mmd_rbf_padded"
    MMD_WASSERSTEIN = "mmd_wasserstein"
