"""
Unified model configuration with comprehensive validation and type safety.

This module provides a refactored ModelConfig implementation that follows
best practices for configuration management with proper validation,
error handling, and maintainable architecture.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .base_config import (
    Config,
    ConfigValidationError
    )
from easy_tpp.utils import logger


def get_available_gpu() -> int:
    """Get the available GPU device ID or -1 for CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return 0  # Return first available GPU
        return -1
    except ImportError:
        logger.warning("PyTorch not available, defaulting to CPU")
        return -1


@dataclass
class SchedulerConfig(Config):
    """ Configuration for the learning rate scheduler and training hyperparameters.
    Args:
        lr (float): Learning rate.
        lr_scheduler (bool): Whether to use a learning rate scheduler.
        max_epochs (int): Maximum number of training epochs.
    """

    def __init__(
        self,
        max_epochs: int,
        lr_scheduler: bool,
        lr: float,
        **kwargs
    ):
        self.lr = lr or 1e-3
        self.lr_scheduler = lr_scheduler or True
        self.max_epochs = max_epochs
        super().__init__(**kwargs)
    
    def get_yaml_config(self):
        
        return {
            "lr": self.lr,
            "lr_scheduler": self.lr_scheduler,
            "max_epochs": self.max_epochs,
        }
    
    def get_required_fields(self) -> List[str]:
        return ["max_epochs"]
    
@dataclass
class ThinningConfig(Config):
    """Configuration for thinning process in temporal point processes."""

    num_sample: int = 10
    num_exp: int = 200
    use_mc_samples: bool = True
    num_steps: int = 10
    over_sample_rate: float = 1.5
    num_samples_boundary: int = 5
    dtime_max: float = 5.0

    def get_required_fields(self) -> List[str]:
        """Return list of required field names."""
        return []  # All fields have defaults

    def get_yaml_config(self) -> Dict[str, Any]:
        """Get configuration as YAML-compatible dictionary."""
        return {
            "num_sample": self.num_sample,
            "num_exp": self.num_exp,
            "num_steps": self.num_steps,
            "over_sample_rate": self.over_sample_rate,
            "num_samples_boundary": self.num_samples_boundary,
            "dtime_max": self.dtime_max,
        }

    def validate(self) -> None:
        """Validate thinning configuration."""
        super().validate()

        if self.num_sample <= 0:
            raise ConfigValidationError("num_sample must be positive", "num_sample")

        if self.num_exp <= 0:
            raise ConfigValidationError("num_exp must be positive", "num_exp")

        if self.over_sample_rate <= 1.0:
            raise ConfigValidationError(
                "over_sample_rate must be greater than 1.0", "over_sample_rate"
            )

        if self.dtime_max <= 0:
            raise ConfigValidationError("dtime_max must be positive", "dtime_max")



@dataclass
class SimulationConfig(Config):
    """Configuration for event sequence simulation."""

    start_time: float = 100.0
    end_time: float = 200.0
    batch_size: int = 32
    max_sim_events: int = 10000
    seed: int = 42

    def get_required_fields(self) -> List[str]:
        """Return list of required field names."""
        return []  # All fields have defaults

    def get_yaml_config(self) -> Dict[str, Any]:
        """Get configuration as YAML-compatible dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "batch_size": self.batch_size,
            "max_sim_events": self.max_sim_events,
            "seed": self.seed,
        }

    def validate(self) -> None:
        """Validate simulation configuration."""
        super().validate()

        if self.start_time < 0:
            raise ConfigValidationError("start_time must be non-negative", "start_time")

        if self.end_time <= self.start_time:
            raise ConfigValidationError(
                "end_time must be greater than start_time", "end_time"
            )

        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be positive", "batch_size")

        if self.max_sim_events <= 0:
            raise ConfigValidationError(
                "max_sim_events must be positive", "max_sim_events"
            )


@dataclass
class ModelSpecsConfig(Config):
    """Configuration for model-specific parameters.
    Args:
        hidden_size (int): Size of the hidden layers.
        rnn_type (str): Type of RNN to use (e.g., LSTM, GRU).
        time_emb_size (int): Size of the time embedding.
        num_layers (int): Number of layers in the model.
        num_heads (int): Number of attention heads.
        sharing_param_layer (bool): Whether to share parameters across layers.
        use_ln (bool): Whether to use layer normalization.
        loss_integral_num_sample_per_step (int): Number of samples for loss integral approximation.
        max_seq_len (int): Maximum sequence length for input data.

        #  IntensityFree model specific parameters
        num_mix_components (int): Number of mixture components for intensity-free models.
        mean_log_inter_time (float): Mean of the log inter-event time distribution.
        std_log_inter_time (float): Standard deviation of the log inter-event time distribution.

        #  Ode model specific parameters
        num_mlp_layers (int): Number of MLP layers in the model.
        ode_num_sample_per_step (int): Number of samples per step for ODE solvers.
        proper_marked_intensities (bool): Whether to use proper marked intensities.

        # Hawkes process parameters (optional)
        mu (Optional[float]): Base intensity for Hawkes processes.
        alpha (Optional[float]): Excitation parameter for Hawkes processes.
        beta (Optional[float]): Decay parameter for Hawkes processes.

    This configuration class encapsulates all model-specific parameters
    """

    # Core model parameters
    hidden_size: int = 64
    rnn_type: str = "LSTM"
    time_emb_size: int = 32
    num_layers: int = 2
    num_heads: int = 8
    sharing_param_layer: bool = False
    use_ln: bool = True
    loss_integral_num_sample_per_step: int = 100
    max_seq_len: int = 100

    # IntensityFree model specific parameters
    num_mix_components: int = 1
    mean_log_inter_time: float = 0.0
    std_log_inter_time: float = 1.0

    # Ode model specific parameters
    num_mlp_layers: int = 2
    ode_num_sample_per_step: int = 20
    proper_marked_intensities: bool = False

    # Hawkes model parameters (optional, can be None)
    mu: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    def get_required_fields(self) -> List[str]:
        """Return list of required field names."""
        return []  # All fields have defaults

    def get_yaml_config(self) -> Dict[str, Any]:
        """Get configuration as YAML-compatible dictionary."""
        return {
            "hidden_size": self.hidden_size,
            "rnn_type": self.rnn_type,
            "time_emb_size": self.time_emb_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "sharing_param_layer": self.sharing_param_layer,
            "use_ln": self.use_ln,
            "loss_integral_num_sample_per_step": self.loss_integral_num_sample_per_step,
            "max_seq_len": self.max_seq_len,
            "num_mix_components": self.num_mix_components,
            "mean_log_inter_time": self.mean_log_inter_time,
            "std_log_inter_time": self.std_log_inter_time,
            "num_mlp_layers": self.num_mlp_layers,
            "ode_num_sample_per_step": self.ode_num_sample_per_step,
            "proper_marked_intensities": self.proper_marked_intensities,
            "mu": self.mu,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def validate(self) -> None:
        """Validate model specifications."""
        super().validate()

        if self.hidden_size <= 0:
            raise ConfigValidationError("hidden_size must be positive", "hidden_size")

        if self.time_emb_size <= 0:
            raise ConfigValidationError(
                "time_emb_size must be positive", "time_emb_size"
            )

        if self.num_layers <= 0:
            raise ConfigValidationError("num_layers must be positive", "num_layers")

        if self.num_heads <= 0:
            raise ConfigValidationError("num_heads must be positive", "num_heads")

        if self.num_mix_components <= 0:
            raise ConfigValidationError(
                "num_mix_components must be positive", "num_mix_components"
            )

        if self.num_mlp_layers <= 0:
            raise ConfigValidationError(
                "num_mlp_layers must be positive", "num_mlp_layers"
            )

        if self.ode_num_sample_per_step <= 0:
            raise ConfigValidationError(
                "ode_num_sample_per_step must be positive", "ode_num_sample_per_step"
            )

        valid_rnn_types = ["LSTM", "GRU", "RNN"]
        if self.rnn_type not in valid_rnn_types:
            raise ConfigValidationError(
                f"rnn_type must be one of {valid_rnn_types}", "rnn_type"
            )



@dataclass
class ModelConfig(Config):
    """
    Configuration for the model architecture and specifications.
    Cette classe prend des dictionnaires pour les sous-configs et instancie les classes intermédiaires.
    Args:
        model_id (str): Identifier for the model type (e.g., 'NHP', 'RMTPP').
        num_event_types (int): Number of event types in the model.
        num_event_types_pad (Optional[int]): Padded number of event types, defaults to num_event_types + 1.
        pad_token_id (Optional[int]): ID for the padding token, defaults to num_event_types.
        device (str): Device to run the model on ('cpu', 'cuda', or 'auto').
        gpu (int): GPU device ID, -1 for CPU.
        is_training (bool): Whether the model is in training mode.
        compute_simulation (bool): Whether to compute simulations during training.
        use_mc_samples (bool): Whether to use Monte Carlo samples for training.
        pretrain_model_path (Optional[str]): Path to a pre-trained model, if applicable.
        base_config (dict): Dictionnaire de config pour TrainingConfig.
        specs (dict): Dictionnaire de config pour ModelSpecsConfig.
        thinning (dict): Dictionnaire de config pour ThinningConfig.
        simulation_config (dict): Dictionnaire de config pour SimulationConfig.
    """

    def __init__(
        self,
        device: str = "auto",
        gpu: int = None,
        is_training: bool = False,
        compute_simulation: bool = False,
        pretrain_model_path: Optional[str] = None,
        specs: Union[dict, ModelSpecsConfig] = None,
        thinning_config: Union[dict, ThinningConfig] = None,
        simulation_config: Union[dict, SimulationConfig] = None,
        scheduler_config: Optional[Union[dict, SchedulerConfig]] = None,
        **kwargs
    ):
        self.device = device
        self.gpu = gpu if gpu is not None else get_available_gpu()
        self.is_training = is_training
        self.compute_simulation = compute_simulation
        self.pretrain_model_path = pretrain_model_path

        # Instancie les sous-configs à partir des dicts
        self.specs = specs if isinstance(specs, ModelSpecsConfig) else ModelSpecsConfig(**(specs or {}))
        self.thinning_config = thinning_config if isinstance(thinning_config, ThinningConfig) else ThinningConfig(**(thinning_config or {}))
        self.simulation_config = simulation_config if isinstance(simulation_config, SimulationConfig) else SimulationConfig(**(simulation_config or {}))

        if scheduler_config is not None:
            self.scheduler_config = scheduler_config if isinstance(scheduler_config, SchedulerConfig) else SchedulerConfig(**(scheduler_config))
        else:
            self.scheduler_config = None

        # Set device if auto
        if self.device == "auto":
            self.device = "cuda" if self.gpu >= 0 else "cpu"

        super().__init__(**kwargs)

    def get_required_fields(self) -> List[str]:
        return []

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "gpu": self.gpu,
            "is_training": self.is_training,
            "compute_simulation": self.compute_simulation,
            "use_mc_samples": self.use_mc_samples,
            "pretrain_model_path": self.pretrain_model_path,
            "specs": self.specs.get_yaml_config(),
            "thinning": self.thinning.get_yaml_config(),
            "simulation_config": self.simulation_config.get_yaml_config(),
        }

    def validate(self) -> None:
        super().validate()
        valid_devices = ["cpu", "cuda", "auto"]
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ConfigValidationError(
                f"device must be one of {valid_devices} or 'cuda:N'", "device"
            )
        self.specs.validate()
        self.thinning_config.validate()
        self.simulation_config.validate()
