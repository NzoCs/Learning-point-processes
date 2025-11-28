from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union, cast

from new_ltpp.configs.base_config import Config, ConfigValidationError
from new_ltpp.utils.const import PaddingStrategy, TruncationStrategy
from new_ltpp.utils.log_utils import default_logger


@dataclass
class TokenizerConfig(Config):
    """Configuration for event tokenizer with strategy-based processing.

    This config enforces a clean separation: either use padding OR truncation, never both.
    Sequences are processed dynamically without fixed max_length.
    The strategy is determined from padding_strategy and truncation_strategy parameters,
    with defaults applied automatically.

    Args:
        num_event_types: Number of event types in the dataset
        padding_strategy: Padding strategy to use. Defaults to LONGEST (dynamic padding).
                         Options: 'longest', 'do_not_pad'
        truncation_strategy: Truncation strategy to use. Defaults to None (no truncation).
                            Options: 'longest_first', 'do_not_truncate', None
        padding_side: Side for padding ('left' or 'right'). Defaults to 'left'.
        truncation_side: Side for truncation ('left' or 'right'). Defaults to 'left'.
        pad_token_id: ID of the padding token (auto-set to num_event_types)
        num_event_types_pad: Number of event types including padding (auto-set)
        model_input_names: Names of model inputs, if applicable

    The final `strategy` attribute is created automatically:
        - If truncation_strategy is provided and not DO_NOT_TRUNCATE, use it
        - Otherwise, use padding_strategy
        - Cannot specify both padding and truncation simultaneously

    Example:
        # Dynamic padding (default)
        config = TokenizerConfig(num_event_types=10)
        # -> strategy = PaddingStrategy.LONGEST

        # Explicit padding
        config = TokenizerConfig(
            num_event_types=10,
            padding_strategy='longest',
            padding_side='right'
        )

        # Truncation (no padding)
        config = TokenizerConfig(
            num_event_types=10,
            truncation_strategy='longest_first',
            truncation_side='left'
        )
    """

    num_event_types: int
    pad_token_id: int | None = None
    padding_strategy: Literal["longest", "do_not_pad"] = "longest"
    truncation_strategy: Literal["longest_first", "do_not_truncate"] = "do_not_truncate"
    padding_side: Literal["left", "right"] = "left"
    truncation_side: Literal["left", "right"] = "left"
    num_event_types_pad: int | None = None
    model_input_names: List[str] | None = None

    # This will be set in __post_init__
    strategy: Union[PaddingStrategy, TruncationStrategy] = field(
        default=PaddingStrategy.LONGEST, init=False
    )

    def __post_init__(self):
        """Validate and normalize configuration."""

        # Set pad_token_id if not provided
        if self.pad_token_id is None:
            self.pad_token_id = self.num_event_types

        if self.num_event_types_pad is None:
            self.num_event_types_pad = self.num_event_types + 1

        # Determine the final strategy: truncation takes precedence if specified
        has_truncation = (
            self.truncation_strategy is not None
            and self.truncation_strategy != "do_not_truncate"
        )
        has_padding = (
            self.padding_strategy is not None and self.padding_strategy != "do_not_pad"
        )

        # Validate: cannot have both active strategies
        if has_truncation and has_padding:
            raise ConfigValidationError(
                "Cannot specify both an active padding strategy and an active truncation strategy. "
                "Use either padding OR truncation, not both.",
                "strategy",
            )

        # Convert to enum and set the final strategy
        if has_truncation:
            self.strategy = TruncationStrategy(self.truncation_strategy)
        elif has_padding:
            self.strategy = PaddingStrategy(self.padding_strategy)
        else:
            # Default to dynamic padding if nothing specified
            self.strategy = PaddingStrategy.LONGEST

        super().__post_init__()

    def get_yaml_config(self) -> Dict[str, Any]:
        """Export configuration to YAML-compatible dict."""
        config_dict = {
            "num_event_types": self.num_event_types,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
            "pad_token_id": self.pad_token_id,
        }

        # Add strategy info (these are already strings/literals)
        if self.padding_strategy is not None:
            config_dict["padding_strategy"] = self.padding_strategy
        if self.truncation_strategy is not None:
            config_dict["truncation_strategy"] = self.truncation_strategy

        return config_dict

    def get_required_fields(self):
        return ["num_event_types"]

    @property
    def is_padding_strategy(self) -> bool:
        """Check if this config uses a padding strategy."""
        return isinstance(self.strategy, PaddingStrategy)

    @property
    def is_truncation_strategy(self) -> bool:
        """Check if this config uses a truncation strategy."""
        return isinstance(self.strategy, TruncationStrategy)


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

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
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
        data_loading_specs: Union[DataLoadingSpecsConfig, dict],
        data_format: Literal["json", "pkl", "hf"],
        tokenizer_specs: Optional[Union[TokenizerConfig, dict]] = None,
        **kwargs,
    ):
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.num_event_types = num_event_types
        self.data_format: Literal["json", "pkl", "hf"] = data_format

        self.dataset_id = dataset_id
        # Instancie si dict, sinon laisse tel quel
        if isinstance(data_loading_specs, dict):
            self.data_loading_specs = DataLoadingSpecsConfig(
                batch_size=data_loading_specs.pop("batch_size"), **data_loading_specs
            )
        else:
            self.data_loading_specs = data_loading_specs
        if isinstance(tokenizer_specs, dict):
            self.tokenizer_specs = TokenizerConfig(
                num_event_types=num_event_types, **tokenizer_specs
            )
        elif tokenizer_specs is None:
            # Default to LONGEST padding strategy (will be set in __post_init__)
            self.tokenizer_specs = TokenizerConfig(num_event_types=num_event_types)
        else:
            self.tokenizer_specs = tokenizer_specs

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
