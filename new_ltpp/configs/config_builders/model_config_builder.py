from typing import Any, Dict, Optional, Union, Self, TypedDict, List, cast

from new_ltpp.configs.model_config import ModelConfig
from new_ltpp.configs.config_factory import ConfigType
from new_ltpp.utils import logger

from .base_config_builder import ConfigBuilder


class ModelConfigDict(TypedDict):
    general_specs: Dict[str, Any] | None
    model_specs: Dict[str, Any] | None
    simulation_config: Dict[str, Any] | None
    thinning_config: Dict[str, Any] | None
    scheduler_config: Dict[str, Any] | None
    device: str | None
    gpu: int | None
    is_training: bool | None
    compute_simulation: bool | None
    pretrain_model_path: str | None
    num_mc_samples: int | None

class ModelConfigBuilder(ConfigBuilder):
    """
    Builder for ModelConfig - configures neural point process model architecture and algorithms.

    This builder handles model specifications, simulation settings, thinning algorithm parameters,
    and learning rate scheduler configuration.

    Required Parameters (must be set):
    ===================================
    Core Model Settings:
    - set_general_specs(dict): General model architecture specifications (hidden_size, dropout)
      Example: {"hidden_size": 64, "dropout": 0.1}
    - set_model_specs(dict): Model-specific specifications (num_layers, num_heads, etc.)
      Example: {"num_layers": 2, "num_heads": 4}

    Algorithm Configurations (choose based on use case):
    - set_simulation_config(): For event simulation during training/inference
      Required if compute_simulation=True
    - set_thinning_config(): For thinning algorithm parameters
      Required for most point process models
    - set_scheduler_config(): Learning rate scheduling
      Required for training with LR scheduling

    Optional Parameters:
    ===================
    Model Runtime Settings:
    - set_device(str): Device for computation ('cpu', 'cuda', 'auto')
    - set_gpu(int): GPU device ID (-1 for CPU)
    - set_is_training(bool): Training vs evaluation mode
    - set_compute_simulation(bool): Whether to compute simulations
    - set_pretrain_model_path(str): Path to pre-trained model checkpoint

    Configuration Details:
    =====================

    Simulation Config (set_simulation_config):
    - start_time (float): Simulation start time
    - end_time (float): Simulation end time
    - batch_size (int): Simulation batch size
    - max_sim_events (int): Maximum simulated events
    - seed (int): Random seed for reproducibility

    Thinning Config (set_thinning_config):
    - num_sample (int): Number of thinning samples
    - num_exp (int): Number of experiments
    - loss_integral_num_sample_per_step (int): Samples per integration step
    - over_sample_rate (float): Over-sampling rate
    - num_samples_boundary (int): Boundary samples
    - dtime_max (float): Maximum time delta

    Scheduler Config (set_scheduler_config):
    - max_epochs (int): Maximum training epochs
    - lr_scheduler (bool): Enable LR scheduler
    - lr (float): Learning rate

    Usage Example:
    ==============

    .. code-block:: python

        builder = ModelConfigBuilder()
        builder.set_general_specs({"hidden_size": 64, "dropout": 0.1})
            .set_model_specs({"num_layers": 2, "num_heads": 4})
            .set_simulation_config(
                start_time=0, end_time=50, batch_size=16,
                max_sim_events=1000, seed=42
            )
            .set_thinning_config(
                num_sample=20, num_exp=100
            )
            .set_scheduler_config(
                max_epochs=100, lr_scheduler=True, lr=1e-3
            )
        model_config = builder.build()
    """

    _config_dict: ModelConfigDict

    def __init__(self):
        self._config_dict = {
            "general_specs": None,
            "model_specs": None,
            "simulation_config": None,
            "thinning_config": None,
            "scheduler_config": None,
            "device": None,
            "gpu": None,
            "is_training": None,
            "compute_simulation": None,
            "pretrain_model_path": None,
            "num_mc_samples": None,
        }

    @property
    def config_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self._config_dict)

    @property
    def config_type(self) -> ConfigType:
        return ConfigType.MODEL
    
    @property
    def required_fields(self) -> List[str]:
        return [
            "general_specs", 
            "model_specs", 
            "scheduler_config", 
            "simulation_config", 
            "thinning_config",
            "num_mc_samples",
            ]

    def build(self, **kwargs) -> ModelConfig:
        """Build and return a ModelConfig instance."""

        if len(self.get_unset_required_fields()) > 0:
            raise ValueError(
                f"Cannot build ModelConfig, required fields not set: {self.get_unset_required_fields()}"
            )
        
        logger.info("Building ModelConfig with:", self.config_dict)
        logger.info("Unset fields will be assigned default values:", self.get_unset_fields())

        config_dict_copy = self.get_clean_dict()

        return ModelConfig(**config_dict_copy, **kwargs)

    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load config from a dictionary matching ModelConfigDict structure."""
        self._config_dict = cast(ModelConfigDict, config_dict)

    # ModelConfig explicit parameters
    def set_device(self, device: str) -> Self:
        """Set the device for training.

        Args:
            device: Device to use ('cpu', 'cuda', 'cuda:0', 'auto', etc.)
        """
        self._config_dict["device"] = device
        return self

    def set_gpu(self, gpu: int) -> Self:
        """Set GPU device ID.

        Args:
            gpu: GPU device ID, -1 for CPU
        """
        self._config_dict["gpu"] = gpu
        return self

    def set_is_training(self, is_training: bool) -> Self:
        """Set whether the model is in training mode.

        Args:
            is_training: True for training mode, False for evaluation
        """
        self._config_dict["is_training"] = is_training
        return self

    def set_compute_simulation(self, compute_simulation: bool) -> Self:
        """Set whether to compute simulations during training.

        Args:
            compute_simulation: True to compute simulations
        """
        self._config_dict["compute_simulation"] = compute_simulation
        return self

    def set_pretrain_model_path(self, pretrain_model_path: Optional[str]) -> Self:
        """Set path to pre-trained model.

        Args:
            pretrain_model_path: Path to pre-trained model checkpoint
        """
        self._config_dict["pretrain_model_path"] = pretrain_model_path
        return self

    def set_num_mc_samples(self, num_mc_samples: int) -> Self:
        """Set number of Monte Carlo samples for model estimation.

        Args:
            num_mc_samples: Number of MC samples to use
        """
        self._config_dict["num_mc_samples"] = num_mc_samples
        return self

    def set_general_specs(self, general_specs: Union[Dict[str, Any], Any]) -> Self:
        """Set general model specifications (hidden_size, dropout).

        Args:
            general_specs: Dictionary containing general model specs
        """
        self._config_dict["general_specs"] = general_specs
        return self

    def set_model_specs(self, model_specs: Union[Dict[str, Any], Any]) -> Self:
        """Set model-specific specifications (num_layers, num_heads, etc.).

        Args:
            model_specs: Dictionary containing model-specific specs
        """
        self._config_dict["model_specs"] = model_specs
        return self

    # SimulationConfig explicit parameters
    def set_simulation_config(
        self,
        time_window: float,
        batch_size: int,
        max_sim_events: int,
        seed: int = 42,
    ) -> Self:
        """Set simulation configuration.

        Args:
            start_time: Start time for simulation
            end_time: End time for simulation
            batch_size: Batch size for simulation
            max_sim_events: Maximum number of simulated events
            seed: Random seed for reproducibility
        """
        self._config_dict["simulation_config"] = {
            "time_window": time_window,
            "batch_size": batch_size,
            "initial_buffer_size": max_sim_events,
            "seed": seed,
        }
        return self

    # ThinningConfig explicit parameters
    def set_thinning_config(
        self,
        num_sample: int,
        num_exp: int,
        over_sample_rate: float,
    ) -> Self:
        """Set thinning configuration for sampling algorithm.

        Args:
            num_sample: Number of samples for thinning
            num_exp: Number of experiments to run
            loss_integral_num_sample_per_step: Number of samples per step for loss integral
            over_sample_rate: Over-sampling rate
            num_samples_boundary: Number of samples at boundaries
            dtime_max: Maximum time delta
        """
        self._config_dict["thinning_config"] = {
            "num_sample": num_sample,
            "num_exp": num_exp,
            "over_sample_rate": over_sample_rate,
            "num_samples_boundary": num_sample,
        }
        return self

    # SchedulerConfig explicit parameters
    def set_scheduler_config(
        self, lr_scheduler: bool, lr: float, max_epochs: int
    ) -> Self:
        """Set learning rate scheduler configuration.

        Args:
            lr_scheduler: Whether to use learning rate scheduler
            lr: Learning rate
            max_epochs: Maximum number of training epochs.
        """
        scheduler_config = {
            "lr_scheduler": lr_scheduler,
            "lr": lr,
            "max_epochs": max_epochs,
        }
        self._config_dict["scheduler_config"] = scheduler_config
        return self
