from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
from easy_tpp.configs.base import (
    BaseConfig,
    ConfigValidationError,
    config_factory,
    config_class,
)
from easy_tpp.utils.log_utils import default_logger


@config_class("tokenizer_config")
@dataclass
class TokenizerConfig(BaseConfig):
    """Configuration for event tokenizer.
    Args:
        num_event_types (int): Number of event types in the dataset.
        pad_token_id (Optional[int]): ID of the padding token. Defaults to num_event_types.
        padding_side (str): Side for padding ('left' or 'right'). Defaults to 'left'.
        truncation_side (str): Side for truncation ('left' or 'right'). Defaults to 'left'.
        padding_strategy (str): Strategy for padding ('longest', 'max_length', etc.). Defaults to 'longest'.
        max_len (Optional[int]): Maximum length of sequences after padding/truncation.
        truncation_strategy (Optional[str]): Strategy for truncation.
        num_event_types_pad (Optional[int]): Number of event types including padding.
        model_input_names (Optional[Any]): Names of model inputs, if applicable.
    """

    num_event_types: int = 0
    pad_token_id: Optional[int] = None
    padding_side: str = "left"
    truncation_side: str = "left"
    padding_strategy: str = "longest"
    max_len: Optional[int] = None
    truncation_strategy: Optional[str] = None
    num_event_types_pad: Optional[int] = None
    model_input_names: Optional[Any] = None

    def __post_init__(self):
        if self.pad_token_id is None:
            self.pad_token_id = self.num_event_types
        if self.num_event_types_pad is None and self.num_event_types is not None:
            self.num_event_types_pad = self.num_event_types + 1
        if self.padding_side not in ["right", "left"]:
            raise ConfigValidationError(
                f"Padding side should be 'right' or 'left', got: {self.padding_side}",
                "padding_side",
            )
        if self.truncation_side not in ["right", "left"]:
            raise ConfigValidationError(
                f"Truncation side should be 'right' or 'left', got: {self.truncation_side}",
                "truncation_side",
            )
        super().__post_init__()

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "num_event_types": self.num_event_types,
            "pad_token_id": self.pad_token_id,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
            "padding_strategy": self.padding_strategy,
            "truncation_strategy": self.truncation_strategy,
            "max_len": self.max_len,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TokenizerConfig":
        from easy_tpp.configs.config_utils import ConfigValidator
        
        # 1. Validate the dictionary
        ConfigValidator.validate_required_fields(
            config_dict, cls._get_required_fields_list(), "TokenizerConfig"
        )
        filtered_dict = ConfigValidator.filter_invalid_fields(config_dict, cls)
        
        # 2. Create the instance
        return cls(**filtered_dict)
    
    @classmethod
    def _get_required_fields_list(cls) -> List[str]:
        """Get required fields as a list for validation."""
        return []

    def get_required_fields(self):
        return []

    def pop(self, key: str, default=None):
        """
        Pop method to make TokenizerConfig compatible with EventTokenizer.
        Returns the attribute value and removes it from the object, or returns default if not found.
        """
        if hasattr(self, key):
            value = getattr(self, key)
            # For special keys that the tokenizer expects to pop, return the value but don't actually remove
            # the attribute since this is a dataclass configuration object
            return value
        return default


@config_class("data_loading_specs_config")
@dataclass
class DataLoadingSpecsConfig(BaseConfig):
    """
    Configuration for data loading specifications.
    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        shuffle (Optional[bool]): Whether to shuffle the dataset.
        padding (Optional[bool]): Whether to apply padding to sequences.
        truncation (Optional[bool]): Whether to truncate sequences to a maximum length.
        tensor_type (str): Type of tensor to return ('pt' for PyTorch, 'tf' for TensorFlow, etc.).
        max_len (Optional[int]): Maximum length of sequences after padding/truncation.
    """

    batch_size: int = 32
    num_workers: int = 1
    shuffle: Optional[bool] = None
    padding: Optional[bool] = None
    truncation: Optional[bool] = None
    tensor_type: str = "pt"
    max_len: Optional[int] = None

    @property
    def max_length(self) -> Optional[int]:
        """Compatibility property for max_length (maps to max_len)."""
        return self.max_len

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "tensor_type": self.tensor_type,
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "padding": self.padding,
            "truncation": self.truncation,
            "max_len": self.max_len,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataLoadingSpecsConfig":
        from easy_tpp.configs.config_utils import ConfigValidator
        
        # 1. Validate the dictionary
        ConfigValidator.validate_required_fields(
            config_dict, cls._get_required_fields_list(), "DataLoadingSpecsConfig"
        )
        filtered_dict = ConfigValidator.filter_invalid_fields(config_dict, cls)
        
        # 2. Create the instance
        return cls(**filtered_dict)
    
    @classmethod
    def _get_required_fields_list(cls) -> List[str]:
        """Get required fields as a list for validation."""
        return []

    def get_required_fields(self):
        return []


@config_class("data_config")
@dataclass
class DataConfig(BaseConfig):
    """
    Configuration for dataset and data processing.
    Args:
        train_dir (str): Path to training data directory.
        valid_dir (str): Path to validation data directory.
        test_dir (str): Path to test data directory.
        data_format (Optional[str]): Format of the dataset files (e.g., 'csv', 'json').
        dataset_id (Optional[str]): Identifier for the dataset.
        data_loading_specs (DataLoadingSpecsConfig): Specifications for loading the data.
        data_specs (TokenizerConfig): Specifications for tokenization and event types.
    """

    train_dir: str
    valid_dir: str
    test_dir: str
    data_format: Optional[str] = None
    dataset_id: Optional[str] = None
    data_loading_specs: DataLoadingSpecsConfig = field(
        default_factory=DataLoadingSpecsConfig
    )
    data_specs: TokenizerConfig = field(default_factory=TokenizerConfig)

    def __post_init__(self):
        if self.data_format is None:
            # Use train_dir to determine format
            self.data_format = self.train_dir.split(".")[-1]
        super().__post_init__()

    def get_yaml_config(self) -> Dict[str, Any]:
        config = {
            "train_dir": self.train_dir,
            "valid_dir": self.valid_dir,
            "test_dir": self.test_dir,
            "data_format": self.data_format,
            "dataset_id": self.dataset_id,
            "data_loading_specs": (
                self.data_loading_specs.get_yaml_config()
                if hasattr(self.data_loading_specs, "get_yaml_config")
                else self.data_loading_specs
            ),
            "data_specs": (
                self.data_specs.get_yaml_config()
                if hasattr(self.data_specs, "get_yaml_config")
                else self.data_specs
            ),
        }

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        from easy_tpp.configs.config_utils import ConfigValidator
        
        # 1. Validate the dictionary
        ConfigValidator.validate_required_fields(
            config_dict, cls._get_required_fields_list(), "DataConfig"
        )
        filtered_dict = ConfigValidator.filter_invalid_fields(config_dict, cls)
        
        # 2. Create sub-configuration instances if needed
        if "data_loading_specs" in filtered_dict and isinstance(filtered_dict["data_loading_specs"], dict):
            filtered_dict["data_loading_specs"] = DataLoadingSpecsConfig.from_dict(
                filtered_dict["data_loading_specs"]
            )
            
        if "data_specs" in filtered_dict and isinstance(filtered_dict["data_specs"], dict):
            filtered_dict["data_specs"] = TokenizerConfig.from_dict(filtered_dict["data_specs"])
        
        # 3. Create the instance
        return cls(**filtered_dict)
    
    @classmethod
    def _get_required_fields_list(cls) -> List[str]:
        """Get required fields as a list for validation."""
        return ["train_dir", "valid_dir", "test_dir"]

    def get_data_dir(self, split: str) -> str:
        """Get directory path for a specific split or raise error if split is invalid."""
        if split in ["train", "dev", "valid", "test"]:
            split = split.lower()
            if split == "train":
                return self.train_dir
            elif split in ["dev", "valid"]:
                return self.valid_dir
            elif split == "test":
                return self.test_dir

        raise ValueError(f"Unknown split: {split}. Valid splits are: train, valid, test.")

    def get_required_fields(self):
        return []
