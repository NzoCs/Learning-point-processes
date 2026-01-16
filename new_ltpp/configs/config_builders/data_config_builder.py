from typing import Any, Dict, List, Union, Self, TypedDict, cast

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
    - set_batch_size(int): Batch size for data loading
    - set_num_workers(int): Number of data loading workers
    - set_shuffle(bool): Whether to shuffle data
    - set_data_loading_specs(dict): Complete data loading config

    Tokenization Settings:
    - set_max_len(int): Maximum sequence length after padding/truncation
    - set_padding_side(str): Padding side ('left' or 'right')
    - set_tokenizer_specs(dict): Complete tokenizer configuration

    Configuration Hierarchy:
    =======================
    The builder supports nested configuration via:
    - data_loading_specs: Controls DataLoader behavior
    - tokenizer_specs: Controls sequence tokenization

    These can be set entirely via set_data_loading_specs()/set_tokenizer_specs(),
    or partially via convenience methods like set_batch_size(), set_max_len(), etc.

    Usage Examples:
    ===============

    .. code-block:: python

        # Example 1: Simple setup with single source directory
        (builder = DataConfigBuilder()
            .set_dataset_id("financial_events")
            .set_num_event_types(5)
            .set_src_dir("/path/to/data")
            .set_batch_size(32)
            .set_max_len(128)
        )
        data_config = builder.build()

        # Example 2: Separate train/valid/test directories
        (builder = DataConfigBuilder()
            .set_dataset_id("financial_events")
            .set_num_event_types(5)
            .set_train_dir("/path/to/train")
            .set_valid_dir("/path/to/valid")
            .set_test_dir("/path/to/test")
            .set_data_format("csv")
            .set_batch_size(64)
            .set_num_workers(4)
            .set_shuffle(True)
            .set_max_len(256)
            .set_padding_side("right")
        )
        data_config = builder.build()

        # Example 3: Using complete specs dictionaries
        (builder = DataConfigBuilder()
            .set_dataset_id("events")
            .set_num_event_types(3)
            .set_src_dir("/data")
            .set_data_loading_specs({
                "batch_size": 128,
                "num_workers": 8,
                "shuffle": True,
                "pin_memory": True
            })
            .set_tokenizer_specs({
                "max_len": 512,
                "padding_side": "left",
                "truncation": True
            })
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
            "data_loading_specs.batch_size",
            "data_format",
        ]

    def build(self, **kwargs) -> DataConfig:

        if len(self.get_unset_required_fields()) > 0:
            raise ValueError(
                f"Cannot build DataConfig, required fields not set: {self.get_unset_required_fields()}"
            )
        
        logger.info("Building DataConfig with:", self.config_dict)
        logger.info("Unset fields will be assigned default values:", self.get_unset_fields())

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

    def set_data_loading_specs(self, specs: Union[Dict[str, Any], Any]) -> Self:
        """Set data loading specifications.

        Args:
            specs: Data loading specifications dictionary or DataLoadingSpecsConfig
        """
        self.config_dict["data_loading_specs"] = specs
        return self

    def set_tokenizer_specs(self, specs: Union[Dict[str, Any], Any]) -> Self:
        """Set tokenizer specifications.

        Args:
            specs: Tokenizer specifications dictionary or TokenizerConfig
        """
        self.config_dict["tokenizer_specs"] = specs
        return self

    # DataLoadingSpecsConfig convenience methods
    def set_batch_size(self, batch_size: int) -> Self:
        """Set batch size in data_loading_specs.

        Args:
            batch_size: Number of samples per batch
        """
        if self.config_dict["data_loading_specs"] is None:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["batch_size"] = batch_size
        return self

    def set_num_workers(self, num_workers: int) -> Self:
        """Set number of workers in data_loading_specs.

        Args:
            num_workers: Number of subprocesses for data loading
        """
        if self.config_dict["data_loading_specs"] is None:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["num_workers"] = num_workers
        return self

    def set_shuffle(self, shuffle: bool) -> Self:
        """Set shuffle parameter in data_loading_specs.

        Args:
            shuffle: Whether to shuffle the dataset
        """
        if self.config_dict["data_loading_specs"] is None:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["shuffle"] = shuffle
        return self

    # TokenizerConfig convenience methods
    def set_max_len(self, max_len: int) -> Self:
        """Set maximum sequence length in tokenizer_specs.

        Args:
            max_len: Maximum length of sequences after padding/truncation
        """
        if self.config_dict["tokenizer_specs"] is None:
            self.config_dict["tokenizer_specs"] = {}
        self.config_dict["tokenizer_specs"]["max_len"] = max_len
        return self

    def set_padding_side(self, side: str) -> Self:
        """Set padding side in tokenizer_specs.

        Args:
            side: Side for padding ('left' or 'right')
        """
        if self.config_dict["tokenizer_specs"] is None:
            self.config_dict["tokenizer_specs"] = {}
        self.config_dict["tokenizer_specs"]["padding_side"] = side
        return self
