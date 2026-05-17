"""Protocol definitions for Temporal Point Process models.

ISimulableModel  — minimal interface required by Simulator and ModelVisualizer.
ITPPModel        — full model interface (Lightning steps + core methods).
INeuralTPPModel  — extension for neural architectures.
"""

from pathlib import Path
from typing import Protocol, Union, Any, TYPE_CHECKING

import torch
import torch.optim as optim
from pytorch_lightning.utilities.types import STEP_OUTPUT

from new_ltpp.configs.model_config import ModelConfig
from new_ltpp.configs.runner_config import SimulationConfig
from new_ltpp.models.event_sampler import EventSampler
from new_ltpp.shared_types import Batch, DataInfo, OneStepPred, SimulationResult

if TYPE_CHECKING:
    from new_ltpp.models.simulation.simulator import Simulator
    from new_ltpp.models.simulation.tpp_io import SimulationIOManager


# ---------------------------------------------------------------------------
# ISimulableModel — what Simulator and ModelVisualizer need from the model
# ---------------------------------------------------------------------------


class ISimulableModel(Protocol):
    """Minimal interface required by Simulator and ModelVisualizer.

    Any object satisfying this protocol can be passed to Simulator or
    ModelVisualizer without inheriting from Model.
    """

    num_event_types: int
    pad_token_id: int
    dtime_max: float
    simulation_config: SimulationConfig
    _simulator: "Simulator"  # Injecté par PredictionStatsCallback
    _io_manager: "SimulationIOManager"  # Injecté par PredictionStatsCallback

    @property
    def device(self) -> torch.device: ...

    def get_event_sampler(self) -> EventSampler: ...

    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
        **kwargs,
    ) -> torch.Tensor: ...

    def simulate_from_scratch(
        self,
        num_sequences: int,
        initial_buffer_size: int = 100,
        max_events: int = 10_000,
    ) -> SimulationResult: ...

    def simulate(
        self,
        batch: Batch,
        max_events: int = 10_000,
    ) -> SimulationResult: ...


# ---------------------------------------------------------------------------
# ITPPModel — full model interface
# ---------------------------------------------------------------------------


class ITPPModel(Protocol):
    """Protocol defining the interface for Temporal Point Process models.

    All models should inherit from base_model.Model which implements this protocol.
    """

    # Required attributes
    num_event_types: int
    pad_token_id: int
    output_dir: Path

    def __init__(
        self,
        *,
        model_config: "ModelConfig",
        data_info: "DataInfo",
        output_dir: Path | str,
        **kwargs,
    ) -> None: ...

    # Core model methods

    def loglike_loss(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]: ...

    def compute_intensities_at_sample_dtimes(
        self,
        *,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        compute_last_step_only: bool = False,
    ) -> torch.Tensor: ...

    # Prediction

    def predict_one_step_at_every_event(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
    ) -> OneStepPred: ...

    # PyTorch Lightning

    @property
    def device(self) -> torch.device: ...

    def configure_optimizers(self) -> Union[dict[str, Any], optim.Optimizer]: ...

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT: ...

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT: ...

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT: ...

    def predict_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT: ...


# ---------------------------------------------------------------------------
# INeuralTPPModel — neural architecture extension
# ---------------------------------------------------------------------------


class INeuralTPPModel(ITPPModel, Protocol):
    """Extended protocol for neural network-based TPP models."""

    hidden_size: int
    dropout: float
