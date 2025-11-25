from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from new_ltpp.configs.config_factory import ConfigType

from .base_config_builder import ConfigBuilder


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
    - use_mc_samples (bool): Use Monte Carlo sampling
    - loss_integral_num_sample_per_step (int): Samples per integration step
    - num_steps (int): Number of thinning steps
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
        builder.set_model_specs({"num_layers": 2, "num_heads": 4})
        builder.set_simulation_config(
            start_time=0, end_time=50, batch_size=16,
            max_sim_events=1000, seed=42
        )
        builder.set_thinning_config(
            num_sample=20, num_exp=100, use_mc_samples=True
        )
        builder.set_scheduler_config(
            max_epochs=100, lr_scheduler=True, lr=1e-3
        )
        model_config = builder.build()
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigType.MODEL, config_dict)

    def from_dict(
        self,
        data: Dict[str, Any],
        model_config_path: Optional[str] = None,
        simulation_config_path: Optional[str] = None,
        thinning_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        general_specs_path: Optional[str] = None,
        model_specs_path: Optional[str] = None,
    ):
        # Load model_cfg only if model_config_path is provided (backward compatibility)
        model_cfg = {}
        if model_config_path:
            model_cfg = self._get_nested_value(data, model_config_path)
        
        simulation_cfg = (
            self._get_nested_value(data, simulation_config_path)
            if simulation_config_path
            else {}
        )
        scheduler_cfg = (
            self._get_nested_value(data, scheduler_config_path)
            if scheduler_config_path
            else {}
        )
        thinning_cfg = (
            self._get_nested_value(data, thinning_config_path)
            if thinning_config_path
            else {}
        )
        
        # Load general_specs and model_specs from specific paths or from model_cfg
        if general_specs_path:
            general_specs = self._get_nested_value(data, general_specs_path)
        else:
            general_specs = model_cfg.get("general_specs", {})
            
        if model_specs_path:
            model_specs = self._get_nested_value(data, model_specs_path)
        else:
            model_specs = model_cfg.get("model_specs", {})
        
        self.config_dict["general_specs"] = general_specs
        self.config_dict["model_specs"] = model_specs
        self.config_dict["simulation_config"] = simulation_cfg
        self.config_dict["thinning_config"] = thinning_cfg
        self.config_dict["scheduler_config"] = scheduler_cfg

        return self.get_missing_fields()

    def load_from_yaml(
        self,
        yaml_path: Union[str, Path],
        model_config_path: Optional[str] = None,
        simulation_config_path: Optional[str] = None,
        thinning_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        general_specs_path: Optional[str] = None,
        model_specs_path: Optional[str] = None,
    ):
        """Load model config from YAML file.

        Args:
            yaml_path: Path to YAML file
            model_config_path: Path to model config (e.g., 'model_configs.neural_small') - optional, for backward compatibility
            simulation_config_path: Path to simulation config
            thinning_config_path: Path to thinning config
            scheduler_config_path: Path to scheduler config
            general_specs_path: Path to general specs (e.g., 'general_specs.default')
            model_specs_path: Path to model specs (e.g., 'model_specs.nhp')
        """
        data = self._load_yaml(yaml_path)
        return self.from_dict(
            data,
            model_config_path,
            simulation_config_path,
            thinning_config_path,
            scheduler_config_path,
            general_specs_path,
            model_specs_path,
        )

    # ModelConfig explicit parameters
    def set_device(self, device: str):
        """Set the device for training.

        Args:
            device: Device to use ('cpu', 'cuda', 'cuda:0', 'auto', etc.)
        """
        return self.set_field("device", device)

    def set_gpu(self, gpu: int):
        """Set GPU device ID.

        Args:
            gpu: GPU device ID, -1 for CPU
        """
        return self.set_field("gpu", gpu)

    def set_is_training(self, is_training: bool):
        """Set whether the model is in training mode.

        Args:
            is_training: True for training mode, False for evaluation
        """
        return self.set_field("is_training", is_training)

    def set_compute_simulation(self, compute_simulation: bool):
        """Set whether to compute simulations during training.

        Args:
            compute_simulation: True to compute simulations
        """
        return self.set_field("compute_simulation", compute_simulation)

    def set_pretrain_model_path(self, pretrain_model_path: Optional[str]):
        """Set path to pre-trained model.

        Args:
            pretrain_model_path: Path to pre-trained model checkpoint
        """
        return self.set_field("pretrain_model_path", pretrain_model_path)

    def set_general_specs(self, general_specs: Union[Dict[str, Any], Any]):
        """Set general model specifications (hidden_size, dropout).

        Args:
            general_specs: Dictionary containing general model specs
        """
        return self.set_field("general_specs", general_specs)

    def set_model_specs(self, model_specs: Union[Dict[str, Any], Any]):
        """Set model-specific specifications (num_layers, num_heads, etc.).

        Args:
            model_specs: Dictionary containing model-specific specs
        """
        return self.set_field("model_specs", model_specs)

    # SimulationConfig explicit parameters
    def set_simulation_config(
        self,
        start_time: float,
        end_time: float,
        batch_size: int,
        max_sim_events: int,
        seed: int = 42,
    ):
        """Set simulation configuration.

        Args:
            start_time: Start time for simulation
            end_time: End time for simulation
            batch_size: Batch size for simulation
            max_sim_events: Maximum number of simulated events
            seed: Random seed for reproducibility
        """
        self.config_dict["simulation_config"] = {
            "start_time": start_time,
            "end_time": end_time,
            "batch_size": batch_size,
            "max_sim_events": max_sim_events,
            "seed": seed,
        }
        return self.get_missing_fields()

    # ThinningConfig explicit parameters
    def set_thinning_config(
        self,
        num_sample: int = 10,
        num_exp: int = 200,
        use_mc_samples: bool = True,
        loss_integral_num_sample_per_step: int = 10,
        num_steps: int = 10,
        over_sample_rate: float = 1.5,
        num_samples_boundary: int = 5,
        dtime_max: float = 5.0,
    ):
        """Set thinning configuration for sampling algorithm.

        Args:
            num_sample: Number of samples for thinning
            num_exp: Number of experiments to run
            use_mc_samples: Whether to use Monte Carlo samples
            loss_integral_num_sample_per_step: Number of samples per step for loss integral
            num_steps: Number of thinning steps
            over_sample_rate: Over-sampling rate
            num_samples_boundary: Number of samples at boundaries
            dtime_max: Maximum time delta
        """
        self.config_dict["thinning_config"] = {
            "num_sample": num_sample,
            "num_exp": num_exp,
            "use_mc_samples": use_mc_samples,
            "loss_integral_num_sample_per_step": loss_integral_num_sample_per_step,
            "num_steps": num_steps,
            "over_sample_rate": over_sample_rate,
            "num_samples_boundary": num_samples_boundary,
            "dtime_max": dtime_max,
        }
        return self.get_missing_fields()

    # SchedulerConfig explicit parameters
    def set_scheduler_config(
        self, lr_scheduler: bool, lr: float, max_epochs: Optional[int] = None
    ):
        """Set learning rate scheduler configuration.

        Args:
            lr_scheduler: Whether to use learning rate scheduler
            lr: Learning rate
            max_epochs: Maximum number of training epochs. If not provided, will be taken from training_config.
        """
        scheduler_config = {
            "lr_scheduler": lr_scheduler,
            "lr": lr,
        }
        if max_epochs is not None:
            scheduler_config["max_epochs"] = max_epochs
        self.config_dict["scheduler_config"] = scheduler_config
        return self.get_missing_fields()

    def get_missing_fields(self) -> List[str]:
        required = []
        return [f for f in required if f not in self.config_dict]
