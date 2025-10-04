from abc import ABC, abstractmethod
from typing import Dict, Union, Any, List, Optional
from pathlib import Path
import yaml

from .config_factory import config_factory, ConfigType
from .data_config import DataConfig
from .runner_config import RunnerConfig

class ConfigBuilder(ABC):
    """Interface for a specific config builder."""

    def __init__(self, config_type: ConfigType, config_dict: Dict[str, Any] = None):
        self.config_type = config_type
        self.config_dict = config_dict or {}

    def set_field(self, field: str, value: Any):
        self.config_dict[field] = value
        return self.get_missing_fields()

    def get_missing_fields(self) -> List[str]:
        # Default no constraints; subclasses should override
        return []

    def get_config_dict(self) -> Dict[str, Any]:
        return self.config_dict

    def build(self, **kwargs) :
        """
        Build a Config instance from the current dict via the factory.
        Args:
            **kwargs: passed to factory.create_config/create_config_by_name
        """
        
        return config_factory.create_config(self.config_type, self.get_config_dict(), **kwargs)

    def _load_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML file with encoding fallback."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as f:
                return yaml.safe_load(f)

    @abstractmethod
    def load_from_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML file with encoding fallback."""
        pass
    
    @abstractmethod
    def from_dict(self, data: Dict[str, Any], *args, **kwargs):
        pass

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get a value via a dotted path (e.g., 'section.key')."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"Path '{path}' not found in YAML")
            current = current[key]
        
        return current


class RunnerConfigBuilder(ConfigBuilder):

    def __init__(self, config_dict: Dict[str, Any] = None):
        super().__init__(ConfigType.RUNNER, config_dict)
        self.model_builder = ModelConfigBuilder(self.config_dict.get("model_config", {}))
        self.data_builder = DataConfigBuilder(self.config_dict.get("data_config", {}))

    def from_dict(self, data: Dict[str, Any],
                  training_config_path: str,
                  model_config_path: str,
                  data_config_path: str,
                  data_loading_config_path: str = None,
                  data_specs_path: str = None,
                  simulation_config_path: str = None,
                  thinning_config_path: str = None,
                  logger_config_path: str = None) -> List[str]:
        
        # Use TrainingConfigBuilder to load training config
        training_cfg = self._get_nested_value(data, training_config_path)
        self.config_dict["training_config"] = training_cfg

        self.model_builder.from_dict(
            data,
            model_config_path,
            simulation_config_path,
            thinning_config_path
        )

        model_cfg = self.model_builder.get_config_dict()
        if "max_epochs" not in model_cfg["scheduler_config"]:
            model_cfg["scheduler_config"] = {
                "lr_scheduler": training_cfg.get("lr_scheduler"),
                "lr": training_cfg.get("lr"),
                "max_epochs": training_cfg.get("max_epochs")
            }

        self.data_builder.from_dict(
            data,
            data_config_path,
            data_loading_config_path,
            data_specs_path
        )

        data_cfg = self.data_builder.get_config_dict()
        self.config_dict["training_config"] = training_cfg
        self.config_dict["model_config"] = model_cfg
        self.config_dict["data_config"] = data_cfg
        
        if logger_config_path:
            logger_cfg = self._get_nested_value(data, logger_config_path)
            self.config_dict["logger_config"] = logger_cfg

        return self.get_missing_fields()

    def build(self, model_id: str, **kwargs) -> RunnerConfig:
        return super().build(model_id=model_id, **kwargs)
    
    def load_from_yaml(self, 
                      yaml_file_path: Union[str, Path], 
                      training_config_path: str,
                      model_config_path: str, 
                      data_config_path: str,
                      data_loading_config_path: str = None,
                      data_specs_path: str = None,
                      simulation_config_path: str = None,
                      thinning_config_path: str = None,
                      logger_config_path: str = None,
                      ) -> List[str]:
        """
        Load complete config from YAML using other builders.
        
        Args:
            yaml_file_path: Path to YAML file
            training_config_path: Path to training config (e.g., 'trainer_configs.quick_test')
            model_config_path: Path to model config (e.g., 'model_configs.neural_small')
            data_config_path: Path to data config (e.g., 'data_configs.test')
            data_loading_config_path: Path to data_loading_config (e.g., 'data_loading_configs.default')
            data_specs_path: Path to tokenizer_specs (e.g., 'tokenizer_specs.standard')
            simulation_config_path: Path to simulation config (e.g., 'simulation_configs.simulation_fast')
            thinning_config_path: Path to thinning config (e.g., 'thinning_configs.thinning_fast')
            logger_config_path: Path to logger config (e.g., 'logger_configs.csv')

        Returns: 
            List of missing fields after loading
        """
        data = self._load_yaml(yaml_file_path)
        return self.from_dict(
            data,
            training_config_path,
            model_config_path,
            data_config_path,
            data_loading_config_path,
            data_specs_path,
            simulation_config_path,
            thinning_config_path,
            logger_config_path
        )

    def set_trainer_config(self, trainer_cfg: Union[Dict[str, Any], Any]):
        self.config_dict["training_config"] = trainer_cfg
        return self.get_missing_fields()

    def set_model_config(self, model_cfg: Union[Dict[str, Any], Any]):
        self.config_dict["model_config"] = model_cfg
        return self.get_missing_fields()

    def set_data_config(self, data_cfg: Union[Dict[str, Any], Any]):
        self.config_dict["data_config"] = data_cfg
        return self.get_missing_fields()
    
    # RunnerConfig explicit parameters
    def set_save_dir(self, save_dir: str):
        """Set save directory for results.
        
        Args:
            save_dir: Path to save directory
        """
        return self.set_field("save_dir", save_dir)
    
    def set_logger_config(self, logger_cfg: Union[Dict[str, Any], Any]):
        """Set logger configuration.
        
        Args:
            logger_cfg: Logger configuration (tensorboard, wandb, csv, etc.)
        """
        self.config_dict["logger_config"] = logger_cfg
        return self.get_missing_fields()
    
    # TrainingConfig convenience methods
    def set_max_epochs(self, max_epochs: int):
        """Set maximum number of training epochs in training_config.
        
        Args:
            max_epochs: Maximum number of epochs
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["max_epochs"] = max_epochs
        return self.get_missing_fields()
    
    def set_batch_size(self, batch_size: int):
        """Set batch size in training_config.
        
        Args:
            batch_size: Number of samples per batch
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["batch_size"] = batch_size
        return self.get_missing_fields()
    
    def set_lr(self, lr: float):
        """Set learning rate in training_config.
        
        Args:
            lr: Learning rate value
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["lr"] = lr
        return self.get_missing_fields()
    
    def set_lr_scheduler(self, lr_scheduler: bool):
        """Set whether to use learning rate scheduler in training_config.
        
        Args:
            lr_scheduler: True to use LR scheduler
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["lr_scheduler"] = lr_scheduler
        return self.get_missing_fields()
    
    def set_dropout(self, dropout: float):
        """Set dropout rate in training_config.
        
        Args:
            dropout: Dropout rate (between 0 and 1)
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["dropout"] = dropout
        return self.get_missing_fields()
    
    def set_val_freq(self, val_freq: int):
        """Set validation frequency in training_config.
        
        Args:
            val_freq: Validation frequency in number of epochs
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["val_freq"] = val_freq
        return self.get_missing_fields()
    
    def set_patience(self, patience: int):
        """Set patience for early stopping in training_config.
        
        Args:
            patience: Number of epochs without improvement before stopping
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["patience"] = patience
        return self.get_missing_fields()
    
    def set_log_freq(self, log_freq: int):
        """Set logging frequency in training_config.
        
        Args:
            log_freq: Logging frequency in number of epochs
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["log_freq"] = log_freq
        return self.get_missing_fields()
    
    def set_checkpoints_freq(self, checkpoints_freq: int):
        """Set checkpoint saving frequency in training_config.
        
        Args:
            checkpoints_freq: Checkpoint frequency in number of epochs
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["checkpoints_freq"] = checkpoints_freq
        return self.get_missing_fields()
    
    def set_accumulate_grad_batches(self, accumulate_grad_batches: int):
        """Set number of batches for gradient accumulation in training_config.
        
        Args:
            accumulate_grad_batches: Number of batches to accumulate before weight update
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["accumulate_grad_batches"] = accumulate_grad_batches
        return self.get_missing_fields()
    
    def set_use_precision_16(self, use_precision_16: bool):
        """Set whether to use 16-bit precision in training_config.
        
        Args:
            use_precision_16: True to use 16-bit precision
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["use_precision_16"] = use_precision_16
        return self.get_missing_fields()
    
    def set_devices(self, devices: int):
        """Set number of devices to use in training_config.
        
        Args:
            devices: Number of devices (1 for single GPU, -1 for all available GPUs)
        """
        if "training_config" not in self.config_dict:
            self.config_dict["training_config"] = {}
        self.config_dict["training_config"]["devices"] = devices
        return self.get_missing_fields()

    def get_missing_fields(self) -> List[str]:
        required = ["training_config", "model_config", "data_config"]
        return [f for f in required if f not in self.config_dict]


class ModelConfigBuilder(ConfigBuilder):
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        super().__init__(ConfigType.MODEL, config_dict)
        
    def from_dict(self, data: Dict[str, Any],
                  model_config_path: str,
                  simulation_config_path: Optional[str] = None,
                  thinning_config_path: Optional[str] = None,
                  scheduler_config_path: Optional[str] = None):
        model_cfg = self._get_nested_value(data, model_config_path)
        simulation_cfg = self._get_nested_value(data, simulation_config_path) if simulation_config_path else {}
        scheduler_cfg = self._get_nested_value(data, scheduler_config_path) if scheduler_config_path else {}
        thinning_cfg = self._get_nested_value(data, thinning_config_path) if thinning_config_path else {}
        self.config_dict = model_cfg
        self.config_dict["simulation_config"] = simulation_cfg
        self.config_dict["thinning_config"] = thinning_cfg
        self.config_dict["scheduler_config"] = scheduler_cfg

        return self.get_missing_fields()

    def load_from_yaml(
            self, 
            yaml_path: Union[str, Path], 
            model_config_path: str, 
            simulation_config_path: Optional[str] = None,
            thinning_config_path: Optional[str] = None,
            scheduler_config_path: Optional[str] = None):
        """Load model config from YAML file.
        
        Args:
            yaml_path: Path to YAML file
            model_config_path: Path to model config (e.g., 'model_configs.neural_small')
            simulation_config_path: Path to simulation config
            thinning_config_path: Path to thinning config
            scheduler_config_path: Path to scheduler config
        """
        data = self._load_yaml(yaml_path)
        return self.from_dict(
            data,
            model_config_path,
            simulation_config_path,
            thinning_config_path,
            scheduler_config_path
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
    
    def set_specs(self, specs: Union[Dict[str, Any], Any]):
        """Set complete model specifications dictionary.
        
        Args:
            specs: Dictionary containing all model specifications
        """
        return self.set_field("specs", specs)

    # SimulationConfig explicit parameters
    def set_simulation_config(self, 
                              start_time: float,
                              end_time: float,
                              batch_size: int,
                              max_sim_events: int,
                              seed: int = 42):
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
            "seed": seed
        }
        return self.get_missing_fields()
    
    # ThinningConfig explicit parameters
    def set_thinning_config(self,
                           num_sample: int = 10,
                           num_exp: int = 200,
                           use_mc_samples: bool = True,
                           loss_integral_num_sample_per_step: int = 10,
                           num_steps: int = 10,
                           over_sample_rate: float = 1.5,
                           num_samples_boundary: int = 5,
                           dtime_max: float = 5.0):
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
            "dtime_max": dtime_max
        }
        return self.get_missing_fields()

    # SchedulerConfig explicit parameters
    def set_scheduler_config(self, 
                            max_epochs: int,
                            lr_scheduler: bool,
                            lr: float):
        """Set learning rate scheduler configuration.
        
        Args:
            max_epochs: Maximum number of training epochs
            lr_scheduler: Whether to use learning rate scheduler
            lr: Learning rate
        """
        self.config_dict["scheduler_config"] = {
            "max_epochs": max_epochs,
            "lr_scheduler": lr_scheduler,
            "lr": lr
        }
        return self.get_missing_fields()

    def get_missing_fields(self) -> List[str]:
        required = []
        return [f for f in required if f not in self.config_dict]




class DataConfigBuilder(ConfigBuilder):

    def __init__(self, config_dict: Dict[str, Any] = None):
        super().__init__(ConfigType.DATA, config_dict)

    def from_dict(self, data: Dict[str, Any],
                  data_config_path: str,
                  data_loading_config_path: str = None,
                  tokenizer_specs_path: str = None):
        data_cfg = self._get_nested_value(data, data_config_path)
        
        # Ensure dataset_id exists
        if isinstance(data_cfg, dict) and "dataset_id" not in data_cfg:
            dataset_id = data_config_path.split('.')[-1]
            data_cfg["dataset_id"] = dataset_id

        # Use DataLoadingSpecsBuilder if requested
        if data_loading_config_path:
            dl_cfg = self._get_nested_value(data, data_loading_config_path)
            data_cfg.setdefault("data_loading_specs", dl_cfg)

        # Merge tokenizer_specs if requested
        if tokenizer_specs_path:
            specs_cfg = self._get_nested_value(data, tokenizer_specs_path)
            data_cfg.setdefault("tokenizer_specs", specs_cfg)
        
        # Edge case: if src_dir is present and any required dir is missing, set all to src_dir
        required_dirs = ["train_dir", "valid_dir", "test_dir"]
        if "src_dir" in data_cfg:
            for d in required_dirs:
                if d not in data_cfg:
                    data_cfg[d] = data_cfg["src_dir"]
            data_cfg.pop("src_dir", None)
        
        self.config_dict = data_cfg
        return self.get_missing_fields()

    def load_from_yaml(self, yaml_path: Union[str, Path], 
                      data_config_path: str,
                      data_loading_config_path: str = None,
                      data_specs_path: str = None):
        """Load data config from YAML file.
        
        Args:
            yaml_path: Path to YAML file
            data_config_path: Path to data config (e.g., 'data_configs.test')
            data_loading_config_path: Optional path to data_loading_config
            data_specs_path: Optional path to tokenizer_specs
        """
        data = self._load_yaml(yaml_path)
        return self.from_dict(
            data,
            data_config_path,
            data_loading_config_path,
            data_specs_path
        )

    # DataConfig explicit parameters
    def set_num_event_types(self, num_event_types: int):
        """Set number of event types.
        
        Args:
            num_event_types: Number of different event types in the dataset
        """
        return self.set_field("num_event_types", num_event_types)
    
    def set_train_dir(self, train_dir: str):
        """Set training data directory.
        
        Args:
            train_dir: Path to training data directory
        """
        return self.set_field("train_dir", train_dir)

    def set_valid_dir(self, valid_dir: str):
        """Set validation data directory.
        
        Args:
            valid_dir: Path to validation data directory
        """
        return self.set_field("valid_dir", valid_dir)

    def set_test_dir(self, test_dir: str):
        """Set test data directory.
        
        Args:
            test_dir: Path to test data directory
        """
        return self.set_field("test_dir", test_dir)
    
    def set_dataset_id(self, dataset_id: str):
        """Set dataset identifier.
        
        Args:
            dataset_id: Unique identifier for the dataset
        """
        return self.set_field("dataset_id", dataset_id)
    
    def set_data_format(self, data_format: str):
        """Set data file format.
        
        Args:
            data_format: Format of dataset files (e.g., 'csv', 'json')
        """
        return self.set_field("data_format", data_format)

    def set_src_dir(self, path: str):
        """Helper to set train/valid/test directories at once from a single source directory.
        
        This mirrors the YAML fallback behavior where a `src_dir` may be provided.
        It sets `train_dir`, `valid_dir` and `test_dir` to `path`.
        
        Args:
            path: Source directory path to use for all data splits
        """
        self.config_dict["train_dir"] = path
        self.config_dict["valid_dir"] = path
        self.config_dict["test_dir"] = path
        return self.get_missing_fields()

    def set_data_loading_specs(self, specs: Union[Dict[str, Any], Any]):
        """Set data loading specifications.
        
        Args:
            specs: Data loading specifications dictionary or DataLoadingSpecsConfig
        """
        return self.set_field("data_loading_specs", specs)

    def set_tokenizer_specs(self, specs: Union[Dict[str, Any], Any]):
        """Set tokenizer specifications.
        
        Args:
            specs: Tokenizer specifications dictionary or TokenizerConfig
        """
        return self.set_field("tokenizer_specs", specs)

    # DataLoadingSpecsConfig convenience methods
    def set_batch_size(self, batch_size: int):
        """Set batch size in data_loading_specs.
        
        Args:
            batch_size: Number of samples per batch
        """
        if "data_loading_specs" not in self.config_dict:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["batch_size"] = batch_size
        return self.get_missing_fields()
    
    def set_num_workers(self, num_workers: int):
        """Set number of workers in data_loading_specs.
        
        Args:
            num_workers: Number of subprocesses for data loading
        """
        if "data_loading_specs" not in self.config_dict:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["num_workers"] = num_workers
        return self.get_missing_fields()
    
    def set_shuffle(self, shuffle: bool):
        """Set shuffle parameter in data_loading_specs.
        
        Args:
            shuffle: Whether to shuffle the dataset
        """
        if "data_loading_specs" not in self.config_dict:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["shuffle"] = shuffle
        return self.get_missing_fields()
    
    # TokenizerConfig convenience methods
    def set_max_len(self, max_len: int):
        """Set maximum sequence length in tokenizer_specs.
        
        Args:
            max_len: Maximum length of sequences after padding/truncation
        """
        if "tokenizer_specs" not in self.config_dict:
            self.config_dict["tokenizer_specs"] = {}
        self.config_dict["tokenizer_specs"]["max_len"] = max_len
        return self.get_missing_fields()
    
    def set_padding_side(self, side: str):
        """Set padding side in tokenizer_specs.
        
        Args:
            side: Side for padding ('left' or 'right')
        """
        if "tokenizer_specs" not in self.config_dict:
            self.config_dict["tokenizer_specs"] = {}
        self.config_dict["tokenizer_specs"]["padding_side"] = side
        return self.get_missing_fields()

    def get_missing_fields(self) -> List[str]:
        required = ["train_dir", "valid_dir", "test_dir", "dataset_id", "num_event_types"]
        # If src_dir is present, directories are considered present
        if "src_dir" in self.config_dict:
            return [f for f in required if f not in self.config_dict and f not in ["train_dir", "valid_dir", "test_dir"]]
        return [f for f in required if f not in self.config_dict]