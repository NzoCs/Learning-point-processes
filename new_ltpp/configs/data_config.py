from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from new_ltpp.configs.base_config import Config, ConfigValidationError
from new_ltpp.utils.log_utils import default_logger


@dataclass
class TokenizerConfig(Config):
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

    num_event_types: int
    padding_side: str = "left"
    truncation_side: str = "left"
    padding_strategy: str = "longest"
    max_len: Optional[int] = None
    truncation_strategy: Optional[str] = None
    num_event_types_pad: Optional[int] = None
    model_input_names: Optional[Any] = None

    def __post_init__(self):

        self.pad_token_id = self.num_event_types
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


@dataclass
class DataLoadingSpecsConfig(Config):
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

    batch_size: int
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

    def get_required_fields(self):
        return []


@dataclass
class DataConfig(Config):
    """
    Configuration for dataset and data processing.
    Args:
        train_dir (str): Path to training data directory.
        valid_dir (str): Path to validation data directory.
        test_dir (str): Path to test data directory.
        data_format (Optional[str]): Format of the dataset files (e.g., 'csv', 'json').
        dataset_id (Optional[str]): Identifier for the dataset.
        data_loading_specs (Union[DataLoadingSpecsConfig, dict]): Specifications for loading the data.
        tokenizer_specs (Union[TokenizerConfig, dict]): Specifications for tokenization and event types.
    """

    def __init__(
        self,
        num_event_types: int,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        dataset_id: str,
        data_format: Optional[str] = None,
        data_loading_specs: Union[DataLoadingSpecsConfig, dict] = None,
        tokenizer_specs: Union[TokenizerConfig, dict] = None,
        **kwargs,
    ):
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.data_format = (
            data_format if data_format is not None else train_dir.split(".")[-1]
        )
        self.dataset_id = dataset_id
        # Instancie si dict, sinon laisse tel quel
        if isinstance(data_loading_specs, dict):
            self.data_loading_specs = DataLoadingSpecsConfig(**data_loading_specs)
        else:
            self.data_loading_specs = (
                data_loading_specs
                if data_loading_specs is not None
                else DataLoadingSpecsConfig()
            )
        if isinstance(tokenizer_specs, dict):
            self.tokenizer_specs = TokenizerConfig(
                **tokenizer_specs, num_event_types=num_event_types
            )
        else:
            self.tokenizer_specs = (
                tokenizer_specs
                if tokenizer_specs is not None
                else TokenizerConfig(num_event_types=num_event_types)
            )
        super().__init__(**kwargs)

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
            "tokenizer_specs": (
                self.tokenizer_specs.get_yaml_config()
                if hasattr(self.tokenizer_specs, "get_yaml_config")
                else self.tokenizer_specs
            ),
        }
        return config

    def get_data_dir(self, split: str) -> str:
        if split in ["train", "dev", "valid", "test"]:
            split = split.lower()
            if split == "train":
                return self.train_dir
            elif split in ["dev", "valid"]:
                return self.valid_dir
            elif split == "test":
                return self.test_dir
        raise ValueError(
            f"Unknown split: {split}. Valid splits are: train, valid, test."
        )

    def get_required_fields(self):
        return []
