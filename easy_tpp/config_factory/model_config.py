"""
Unified model configuration with comprehensive validation and type safety.

This module provides a refactored ModelConfig implementation that follows
best practices for configuration management with proper validation,
error handling, and maintainable architecture.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from easy_tpp.config_factory.base import (
    BaseConfig,
    ConfigValidationError,
    config_factory,
    config_class,
)
from easy_tpp.utils.const import Backend

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of supported model types."""

    NHP = "NHP"
    RMTPP = "RMTPP"
    THP = "THP"
    SAHP = "SAHP"
    AttNHP = "AttNHP"
    IntensityFree = "IntensityFree"
    Hawkes = "Hawkes"
    FullyNN = "FullyNN"
    ANHN = "ANHN"
    ODE_TPP = "ODE_TPP"
    SelfCorrecting = "SelfCorrecting"


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


@config_class("thinning_config")
@dataclass
class ThinningConfig(BaseConfig):
    """Configuration for thinning process in temporal point processes."""

    num_sample: int = 10
    num_exp: int = 200
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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ThinningConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

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


@config_class("simulation_config")
@dataclass
class SimulationConfig(BaseConfig):
    """Configuration for event sequence simulation."""

    start_time: float = 0.0
    end_time: float = 100.0
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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

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


@config_class("training_config")
@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for model training parameters.
    Args:
        lr (float): Learning rate for the optimizer.
        lr_scheduler (bool): Whether to use a learning rate scheduler.
        max_epochs (int): Maximum number of training epochs.
        dropout (float): Dropout rate for regularization.
        stage (str): Training stage, e.g., 'train', 'eval', 'test'.
        backend (Backend): Backend framework to use (e.g., Torch, TensorFlow).
        dataset_id (Optional[str]): Identifier for the dataset.
        base_dir (Optional[str]): Base directory for saving outputs.
    """

    lr: float = 0.001
    lr_scheduler: bool = True
    max_epochs: int = 1000
    dropout: float = 0.0
    stage: str = "train"
    backend: Backend = Backend.Torch
    dataset_id: Optional[str] = None
    base_dir: Optional[str] = None

    def get_required_fields(self) -> List[str]:
        """Return list of required field names."""
        return []  # All fields have defaults

    def get_yaml_config(self) -> Dict[str, Any]:
        """Get configuration as YAML-compatible dictionary."""
        return {
            "lr": self.lr,
            "lr_scheduler": self.lr_scheduler,
            "max_epochs": self.max_epochs,
            "dropout": self.dropout,
            "stage": self.stage,
            "backend": str(self.backend),
            "dataset_id": self.dataset_id,
            "base_dir": self.base_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        # Map legacy 'dropout_rate' to 'dropout' if present
        if "dropout_rate" in config_dict and "dropout" not in config_dict:
            config_dict["dropout"] = config_dict["dropout_rate"]
        # Handle backend conversion
        backend = config_dict.get("backend", Backend.Torch)
        if isinstance(backend, str):
            if backend.lower() in ["torch", "pytorch"]:
                backend = Backend.Torch
            elif backend.lower() in ["tf", "tensorflow"]:
                backend = Backend.TF
            else:
                raise ConfigValidationError(f"Unknown backend: {backend}", "backend")
        config_dict["backend"] = backend
        return cls(**config_dict)

    def validate(self) -> None:
        """Validate training configuration."""
        super().validate()

        if self.lr <= 0:
            raise ConfigValidationError("Learning rate must be positive", "lr")

        if self.max_epochs <= 0:
            raise ConfigValidationError("max_epochs must be positive", "max_epochs")

        if not (0.0 <= self.dropout <= 1.0):
            raise ConfigValidationError(
                "dropout must be between 0.0 and 1.0", "dropout"
            )

        valid_stages = ["train", "eval", "test"]
        if self.stage not in valid_stages:
            raise ConfigValidationError(f"stage must be one of {valid_stages}", "stage")


@config_class("model_specs_config")
@dataclass
class ModelSpecsConfig(BaseConfig):
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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelSpecsConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

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


@config_class("model_config")
@dataclass
class ModelConfig(BaseConfig):
    """
    Configuration for the model architecture and specifications.
    This class encapsulates all necessary parameters for defining a model,
    including training settings, model specifications, thinning parameters,
    and simulation configurations.
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

        # Sub-configurations:
        base_config (TrainingConfig): Base training configuration.
        specs (Union[ModelSpecsConfig, HawkesSpecsConfig]): Model specifications, can be either general or Hawkes-specific.
        thinning (ThinningConfig): Configuration for thinning process.
        simulation_config (SimulationConfig): Configuration for event sequence simulation.
    """

    model_id: str
    num_event_types: int
    num_event_types_pad: Optional[int] = None
    pad_token_id: Optional[int] = None
    device: str = "auto"
    gpu: int = field(default_factory=get_available_gpu)
    is_training: bool = False
    compute_simulation: bool = False
    use_mc_samples: bool = True
    pretrain_model_path: Optional[str] = None

    # Sub-configurations
    base_config: TrainingConfig = field(default_factory=TrainingConfig)
    specs: ModelSpecsConfig = field(default_factory=ModelSpecsConfig)
    thinning: ThinningConfig = field(default_factory=ThinningConfig)
    simulation_config: SimulationConfig = field(default_factory=SimulationConfig)

    def __post_init__(self):
        """Post-initialization processing."""
        # Set defaults based on other fields
        if self.num_event_types_pad is None:
            self.num_event_types_pad = self.num_event_types + 1

        if self.pad_token_id is None:
            self.pad_token_id = self.num_event_types

        # Set device
        if self.device == "auto":
            self.device = "cuda" if self.gpu >= 0 else "cpu"

        super().__post_init__()

    def get_required_fields(self) -> List[str]:
        """Return list of required field names."""
        return ["model_id", "num_event_types"]

    def get_yaml_config(self) -> Dict[str, Any]:
        """Get configuration as YAML-compatible dictionary."""
        return {
            "model_id": self.model_id,
            "num_event_types": self.num_event_types,
            "num_event_types_pad": self.num_event_types_pad,
            "pad_token_id": self.pad_token_id,
            "device": self.device,
            "gpu": self.gpu,
            "is_training": self.is_training,
            "compute_simulation": self.compute_simulation,
            "use_mc_samples": self.use_mc_samples,
            "pretrain_model_path": self.pretrain_model_path,
            "base_config": self.base_config.get_yaml_config(),
            "specs": self.specs.get_yaml_config(),
            "thinning": self.thinning.get_yaml_config(),
            "simulation_config": self.simulation_config.get_yaml_config(),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        # Extract sub-configurations
        base_config_dict = config_dict.pop("base_config", {})
        specs_dict = config_dict.pop("specs", {})
        thinning_dict = config_dict.pop("thinning", {})
        simulation_dict = config_dict.pop("simulation_config", {})

        specs = ModelSpecsConfig.from_dict(specs_dict)
        base_config = TrainingConfig.from_dict(base_config_dict)
        thinning = ThinningConfig.from_dict(thinning_dict)
        simulation_config = SimulationConfig.from_dict(simulation_dict)

        return cls(
            base_config=base_config,
            specs=specs,
            thinning=thinning,
            simulation_config=simulation_config,
            **config_dict,
        )

    @staticmethod
    def parse_from_yaml_config(yaml_config: Dict[str, Any], **kwargs) -> "ModelConfig":
        """Parse from YAML configuration."""
        config_dict = dict(yaml_config)
        config_dict.update(kwargs)
        return ModelConfig.from_dict(config_dict)

    def copy(self) -> "ModelConfig":
        """Create a copy of the configuration."""
        return ModelConfig.from_dict(self.get_yaml_config())

    def validate(self) -> None:
        """Validate the model configuration."""
        super().validate()

        # Validate model_id
        try:
            ModelType(self.model_id)
        except ValueError:
            valid_models = [model.value for model in ModelType]
            logger.warning(
                f"model_id '{self.model_id}' not in known models {valid_models}"
            )

        # Validate num_event_types
        if self.num_event_types <= 0:
            raise ConfigValidationError(
                "num_event_types must be positive", "num_event_types"
            )

        # Validate device configuration
        valid_devices = ["cpu", "cuda", "auto"]
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ConfigValidationError(
                f"device must be one of {valid_devices} or 'cuda:N'", "device"
            )

        # Validate sub-configurations
        self.base_config.validate()
        self.specs.validate()
        self.thinning.validate()
        self.simulation_config.validate()
