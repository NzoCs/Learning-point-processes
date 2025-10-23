from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, cast

from new_ltpp.configs.base_config import Config
from new_ltpp.configs.config_factory import ConfigType
from new_ltpp.configs.data_config import DataConfig

from .base_config_builder import ConfigBuilder


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
        builder = DataConfigBuilder()
        builder.set_dataset_id("financial_events")
        builder.set_num_event_types(5)
        builder.set_src_dir("/path/to/data")
        builder.set_batch_size(32)
        builder.set_max_len(128)
        data_config = builder.build()

        # Example 2: Separate train/valid/test directories
        builder = DataConfigBuilder()
        builder.set_dataset_id("financial_events")
        builder.set_num_event_types(5)
        builder.set_train_dir("/path/to/train")
        builder.set_valid_dir("/path/to/valid")
        builder.set_test_dir("/path/to/test")
        builder.set_data_format("csv")
        builder.set_batch_size(64)
        builder.set_num_workers(4)
        builder.set_shuffle(True)
        builder.set_max_len(256)
        builder.set_padding_side("right")
        data_config = builder.build()

        # Example 3: Using complete specs dictionaries
        builder = DataConfigBuilder()
        builder.set_dataset_id("events")
        builder.set_num_event_types(3)
        builder.set_src_dir("/data")
        builder.set_data_loading_specs({
            "batch_size": 128,
            "num_workers": 8,
            "shuffle": True,
            "pin_memory": True
        })
        builder.set_tokenizer_specs({
            "max_len": 512,
            "padding_side": "left",
            "truncation": True
        })
        data_config = builder.build()
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigType.DATA, config_dict)

    def build(self, **kwargs) -> DataConfig:
        return cast(DataConfig, super().build(**kwargs))

    def from_dict(
        self,
        data: Dict[str, Any],
        data_config_path: str,
        data_loading_config_path: Optional[str] = None,
        tokenizer_specs_path: Optional[str] = None,
    ) -> List[str]:
        data_cfg = self._get_nested_value(data, data_config_path)

        # Ensure dataset_id exists
        if isinstance(data_cfg, dict) and "dataset_id" not in data_cfg:
            dataset_id = data_config_path.split(".")[-1]
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

    def load_from_yaml(
        self,
        yaml_path: Union[str, Path],
        data_config_path: str,
        data_loading_config_path: Optional[str] = None,
        data_specs_path: Optional[str] = None,
    ) -> List[str]:
        """Load data config from YAML file.

        Args:
            yaml_path: Path to YAML file
            data_config_path: Path to data config (e.g., 'data_configs.test')
            data_loading_config_path: Optional path to data_loading_config
            data_specs_path: Optional path to tokenizer_specs
        """
        data = self._load_yaml(yaml_path)
        return self.from_dict(
            data, data_config_path, data_loading_config_path, data_specs_path
        )

    # DataConfig explicit parameters
    def set_num_event_types(self, num_event_types: int) -> List[str]:
        """Set number of event types.

        Args:
            num_event_types: Number of different event types in the dataset
        """
        return self.set_field("num_event_types", num_event_types)

    def set_train_dir(self, train_dir: str) -> List[str]:
        """Set training data directory.

        Args:
            train_dir: Path to training data directory
        """
        return self.set_field("train_dir", train_dir)

    def set_valid_dir(self, valid_dir: str) -> List[str]:
        """Set validation data directory.

        Args:
            valid_dir: Path to validation data directory
        """
        return self.set_field("valid_dir", valid_dir)

    def set_test_dir(self, test_dir: str) -> List[str]:
        """Set test data directory.

        Args:
            test_dir: Path to test data directory
        """
        return self.set_field("test_dir", test_dir)

    def set_dataset_id(self, dataset_id: str) -> List[str]:
        """Set dataset identifier.

        Args:
            dataset_id: Unique identifier for the dataset
        """
        return self.set_field("dataset_id", dataset_id)

    def set_data_format(self, data_format: str) -> List[str]:
        """Set data file format.

        Args:
            data_format: Format of dataset files (e.g., 'csv', 'json')
        """
        return self.set_field("data_format", data_format)

    def set_src_dir(self, path: str) -> List[str]:
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

    def set_data_loading_specs(self, specs: Union[Dict[str, Any], Any]) -> List[str]:
        """Set data loading specifications.

        Args:
            specs: Data loading specifications dictionary or DataLoadingSpecsConfig
        """
        return self.set_field("data_loading_specs", specs)

    def set_tokenizer_specs(self, specs: Union[Dict[str, Any], Any]) -> List[str]:
        """Set tokenizer specifications.

        Args:
            specs: Tokenizer specifications dictionary or TokenizerConfig
        """
        return self.set_field("tokenizer_specs", specs)

    # DataLoadingSpecsConfig convenience methods
    def set_batch_size(self, batch_size: int) -> List[str]:
        """Set batch size in data_loading_specs.

        Args:
            batch_size: Number of samples per batch
        """
        if "data_loading_specs" not in self.config_dict:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["batch_size"] = batch_size
        return self.get_missing_fields()

    def set_num_workers(self, num_workers: int) -> List[str]:
        """Set number of workers in data_loading_specs.

        Args:
            num_workers: Number of subprocesses for data loading
        """
        if "data_loading_specs" not in self.config_dict:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["num_workers"] = num_workers
        return self.get_missing_fields()

    def set_shuffle(self, shuffle: bool) -> List[str]:
        """Set shuffle parameter in data_loading_specs.

        Args:
            shuffle: Whether to shuffle the dataset
        """
        if "data_loading_specs" not in self.config_dict:
            self.config_dict["data_loading_specs"] = {}
        self.config_dict["data_loading_specs"]["shuffle"] = shuffle
        return self.get_missing_fields()

    # TokenizerConfig convenience methods
    def set_max_len(self, max_len: int) -> List[str]:
        """Set maximum sequence length in tokenizer_specs.

        Args:
            max_len: Maximum length of sequences after padding/truncation
        """
        if "tokenizer_specs" not in self.config_dict:
            self.config_dict["tokenizer_specs"] = {}
        self.config_dict["tokenizer_specs"]["max_len"] = max_len
        return self.get_missing_fields()

    def set_padding_side(self, side: str) -> List[str]:
        """Set padding side in tokenizer_specs.

        Args:
            side: Side for padding ('left' or 'right')
        """
        if "tokenizer_specs" not in self.config_dict:
            self.config_dict["tokenizer_specs"] = {}
        self.config_dict["tokenizer_specs"]["padding_side"] = side
        return self.get_missing_fields()

    def get_missing_fields(self) -> List[str]:
        required = [
            "train_dir",
            "valid_dir",
            "test_dir",
            "dataset_id",
            "num_event_types",
        ]
        # If src_dir is present, directories are considered present
        if "src_dir" in self.config_dict:
            return [
                f
                for f in required
                if f not in self.config_dict
                and f not in ["train_dir", "valid_dir", "test_dir"]
            ]
        return [f for f in required if f not in self.config_dict]
