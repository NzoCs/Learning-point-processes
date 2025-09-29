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
    loss_integral_num_sample_per_step: int = 10
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
            "use_mc_samples": self.use_mc_samples,
            "loss_integral_num_sample_per_step": self.loss_integral_num_sample_per_step,
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

    start_time: float
    end_time: float
    batch_size: int
    max_sim_events: int
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


# ModelSpecsConfig is intentionally a thin alias for a plain dict.
# Each model should define its own defaults and explicit parameters.
ModelSpecsConfig = dict



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
        thinning_config: Union[dict, ThinningConfig] = None,
        simulation_config: Union[dict, SimulationConfig] = None,
        specs: ModelSpecsConfig = None,
        scheduler_config: Optional[Union[dict, SchedulerConfig]] = None,
        **kwargs
    ):
        self.device = device
        self.gpu = gpu if gpu is not None else get_available_gpu()
        self.is_training = is_training
        self.compute_simulation = compute_simulation
        self.pretrain_model_path = pretrain_model_path

        # Instancie les sous-configs à partir des dicts
        self.specs = specs or {}
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
            "pretrain_model_path": self.pretrain_model_path,
            "thinning": self.thinning_config.get_yaml_config(),
            "simulation_config": self.simulation_config.get_yaml_config(),
        }

    def validate(self) -> None:
        super().validate()
        valid_devices = ["cpu", "cuda", "auto"]
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ConfigValidationError(
                f"device must be one of {valid_devices} or 'cuda:N'", "device"
            )
        self.thinning_config.validate()
        self.simulation_config.validate()
