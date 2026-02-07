from typing import Any, Dict, List, Self, TypedDict, cast, Literal, Optional

from new_ltpp.configs.data_config import DataConfig
from new_ltpp.configs.config_factory import ConfigType
from new_ltpp.utils import logger

from .base_config_builder import ConfigBuilder


class DataConfigDict(TypedDict):
    dataset_id: str | None
    num_event_types: int | None
    train_dir: str | None
    valid_dir: str | None
    test_dir: str | None
    data_format: str | None
    data_loading_specs: Dict[str, Any] | None
    tokenizer_specs: Dict[str, Any] | None


class DataConfigBuilder(ConfigBuilder):
    """
    Builder for DataConfig - configures dataset paths, data loading, and tokenization settings.

    This builder handles dataset directory configuration, data loading specifications,
    and tokenizer settings for sequence processing.

    Required Parameters (must be set):
    ===================================
    Dataset Identification:
    - set_dataset_id(str): Unique identifier for the dataset
    - set_num_event_types(int): Number of different event types in dataset

    Data Directory Configuration (choose one approach):

    Approach 1 - Single source directory:
    - set_src_dir(str): Sets train/valid/test dirs to same path

    Approach 2 - Separate directories:
    - set_train_dir(str): Training data directory
    - set_valid_dir(str): Validation data directory
    - set_test_dir(str): Test data directory

    Optional Parameters:
    ===================
    Data Format:
    - set_data_format(str): File format ('csv', 'json', etc.)

    Data Loading Specifications:
    - set_data_loading_specs(batch_size, num_workers, shuffle, padding, truncation, max_len):
      Configure complete data loading behavior including batch size, workers, shuffling,
      padding/truncation strategies, and maximum sequence length

    Tokenization Settings:
    - set_tokenizer_specs(pad_token_id, padding_strategy, truncation_strategy,
                          padding_side, truncation_side, model_input_names):
      Configure tokenization behavior including padding/truncation strategies and sides

    Configuration Hierarchy:
    =======================
    The builder supports nested configuration via:
    - data_loading_specs: Controls DataLoader behavior (batch_size, num_workers, shuffle,
                          padding, truncation, max_len)
    - tokenizer_specs: Controls sequence tokenization (pad_token_id, padding_strategy,
                       truncation_strategy, padding_side, truncation_side, model_input_names)

    Usage Examples (method chaining style):
    =======================================

    .. code-block:: python

        # Example 1: Simple setup with single source directory
        builder = (
            DataConfigBuilder()
            .set_dataset_id("financial_events")
            .set_num_event_types(5)
            .set_src_dir("/path/to/data")
            .set_data_loading_specs(
                batch_size=32, num_workers=2, shuffle=True, max_len=128
            )
        )
        data_config = builder.build()

        # Example 2: Separate train/valid/test directories with chaining
        builder = (
            DataConfigBuilder()
            .set_dataset_id("financial_events")
            .set_num_event_types(5)
            .set_train_dir("/path/to/train")
            .set_valid_dir("/path/to/valid")
            .set_test_dir("/path/to/test")
            .set_data_format("csv")
            .set_data_loading_specs(
                batch_size=64, num_workers=4, shuffle=True, padding=True, truncation=False, max_len=256
            )
            .set_tokenizer_specs(
                padding_side="right", truncation_side="left", padding_strategy="longest",
                truncation_strategy="do_not_truncate"
            )
        )
        data_config = builder.build()

        # Example 3: Minimal configuration using chaining
        builder = (
            DataConfigBuilder()
            .set_dataset_id("events")
            .set_num_event_types(3)
            .set_src_dir("NzoCs/test_dataset")
            .set_data_loading_specs(batch_size=16)
        )
        data_config = builder.build()
    """

    _config_dict: DataConfigDict

    def __init__(self):
        self._config_dict = {
            "dataset_id": None,
            "num_event_types": None,
            "train_dir": None,
            "valid_dir": None,
            "test_dir": None,
            "data_format": None,
            "data_loading_specs": None,
            "tokenizer_specs": None,
        }
        self.set_tokenizer_specs()

    @property
    def config_dict(self) -> Dict[str, Any]:
        return cast(dict[str, Any], self._config_dict)

    @property
    def config_type(self) -> ConfigType:
        return ConfigType.DATA

    @property
    def required_fields(self) -> List[str]:
        return [
            "train_dir",
            "valid_dir",
            "test_dir",
            "dataset_id",
            "num_event_types",
            "data_loading_specs",
            "data_format",
            "tokenizer_specs",
        ]

    def build(self, **kwargs) -> DataConfig:
        if len(self.get_unset_required_fields()) > 0:
            raise ValueError(
                f"Cannot build DataConfig, required fields not set: {self.get_unset_required_fields()}"
            )

        logger.info("Building DataConfig with: %s", self.config_dict)

        if len(self.get_unset_fields()) > 0:
            logger.info(
                "Unset fields will be assigned default values: %s",
                self.get_unset_fields(),
            )

        config_dict_copy = self.get_clean_dict()

        return DataConfig(**config_dict_copy, **kwargs)

    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load config from a dictionary matching DataConfigDict structure."""
        self._config_dict = cast(DataConfigDict, config_dict)

    # DataConfig explicit parameters
    def set_num_event_types(self, num_event_types: int) -> Self:
        """Set number of event types.

        Args:
            num_event_types: Number of different event types in the dataset
        """
        self.config_dict["num_event_types"] = num_event_types
        return self

    def set_train_dir(self, train_dir: str) -> Self:
        """Set training data directory.

        Args:
            train_dir: Path to training data directory
        """
        self.config_dict["train_dir"] = train_dir
        return self

    def set_valid_dir(self, valid_dir: str) -> Self:
        """Set validation data directory.

        Args:
            valid_dir: Path to validation data directory
        """
        self.config_dict["valid_dir"] = valid_dir
        return self

    def set_test_dir(self, test_dir: str) -> Self:
        """Set test data directory.

        Args:
            test_dir: Path to test data directory
        """
        self.config_dict["test_dir"] = test_dir
        return self

    def set_dataset_id(self, dataset_id: str) -> Self:
        """Set dataset identifier.

        Args:
            dataset_id: Unique identifier for the dataset
        """
        self.config_dict["dataset_id"] = dataset_id
        return self

    def set_data_format(self, data_format: str) -> Self:
        """Set data file format.

        Args:
            data_format: Format of dataset files (e.g., 'csv', 'json')
        """
        self.config_dict["data_format"] = data_format
        return self

    def set_src_dir(self, path: str) -> Self:
        """Helper to set train/valid/test directories at once from a single source directory.

        This mirrors the YAML fallback behavior where a `src_dir` may be provided.
        It sets `train_dir`, `valid_dir` and `test_dir` to `path`.

        Args:
            path: Source directory path to use for all data splits
        """
        self.config_dict["train_dir"] = path
        self.config_dict["valid_dir"] = path
        self.config_dict["test_dir"] = path
        return self

    def set_data_loading_specs(
        self,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = False,
        padding: bool = True,
        truncation: bool = False,
        max_len: Optional[int] = None,
    ) -> Self:
        """Set data loading specifications.

        Args:
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
            shuffle: Whether to shuffle the dataset
            padding: Whether to apply padding
            truncation: Whether to apply truncation
            max_len: Maximum sequence length
        """
        specs: Dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": shuffle,
            "padding": padding,
            "truncation": truncation,
            "max_len": max_len,
        }
        self.config_dict["data_loading_specs"] = specs
        self.batch_size = batch_size
        return self

    def set_tokenizer_specs(
        self,
        pad_token_id: Optional[int] = None,
        padding_strategy: Literal["longest", "do_not_pad"] = "longest",
        truncation_strategy: Literal[
            "longest_first", "do_not_truncate"
        ] = "do_not_truncate",
        padding_side: Literal["left", "right"] = "left",
        truncation_side: Literal["left", "right"] = "left",
        model_input_names: Optional[List[str]] = None,
    ) -> Self:
        """Set tokenizer specifications.

        Args:
            padding_side: Side for padding ('left' or 'right')
            truncation_side: Side for truncation ('left' or 'right')
        """
        tokenizer_specs: Dict[str, Any] = {
            "padding_side": padding_side,
            "truncation_side": truncation_side,
            "pad_token_id": pad_token_id,
            "padding_strategy": padding_strategy,
            "truncation_strategy": truncation_strategy,
            "model_input_names": model_input_names,
        }
        self.config_dict["tokenizer_specs"] = tokenizer_specs
        return self
