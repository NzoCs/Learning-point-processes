from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
from easy_tpp.config_factory.base import BaseConfig, ConfigValidationError, config_factory, config_class


@dataclass
class SplitDirectories:
    """Lightweight dataclass containing the three required split directories."""
    train_dir: str
    valid_dir: str
    test_dir: str


@config_class('tokenizer_config')
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
    padding_side: str = 'left'
    truncation_side: str = 'left'
    padding_strategy: str = 'longest'
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
                f"Padding side should be 'right' or 'left', got: {self.padding_side}", "padding_side")
        if self.truncation_side not in ["right", "left"]:
            raise ConfigValidationError(
                f"Truncation side should be 'right' or 'left', got: {self.truncation_side}", "truncation_side")
        super().__post_init__()

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            'num_event_types': self.num_event_types,
            'pad_token_id': self.pad_token_id,
            'padding_side': self.padding_side,
            'truncation_side': self.truncation_side,
            'padding_strategy': self.padding_strategy,
            'truncation_strategy': self.truncation_strategy,
            'max_len': self.max_len
        }    
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TokenizerConfig':
        return cls(**config_dict)

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


@config_class('data_loading_specs_config')
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
    tensor_type: str = 'pt'
    max_len: Optional[int] = None

    @property
    def max_length(self) -> Optional[int]:
        """Compatibility property for max_length (maps to max_len)."""
        return self.max_len

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            'batch_size': self.batch_size,
            'tensor_type': self.tensor_type,
            'num_workers': self.num_workers,
            'shuffle': self.shuffle,
            'padding': self.padding,
            'truncation': self.truncation,
            'max_len': self.max_len
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataLoadingSpecsConfig':
        return cls(**config_dict)

    def get_required_fields(self):
        return []


@config_class('data_config')
@dataclass
class DataConfig(BaseConfig):
    """
    Configuration for dataset and data processing.
    Args:
        data_dirs (Union[SplitDirectories, str]): Either split directories or source directory.
        data_format (Optional[str]): Format of the dataset files (e.g., 'csv', 'json').
        dataset_id (Optional[str]): Identifier for the dataset.
        data_loading_specs (DataLoadingSpecsConfig): Specifications for loading the data.
        data_specs (TokenizerConfig): Specifications for tokenization and event types.
    """

    data_dirs: Union[SplitDirectories, str]
    data_format: Optional[str] = None
    dataset_id: Optional[str] = None
    data_loading_specs: DataLoadingSpecsConfig = field(default_factory=DataLoadingSpecsConfig)
    data_specs: TokenizerConfig = field(default_factory=TokenizerConfig)
    

    def __post_init__(self):
        if self.data_format is None:
            if isinstance(self.data_dirs, SplitDirectories):
                self.data_format = self.data_dirs.train_dir.split('.')[-1]
            elif isinstance(self.data_dirs, str):
                self.data_format = self.data_dirs.split('.')[-1]
        super().__post_init__()

    @property
    def train_dir(self) -> Optional[str]:
        """Access to train directory."""
        return self.data_dirs.train_dir if isinstance(self.data_dirs, SplitDirectories) else None

    @property
    def valid_dir(self) -> Optional[str]:
        """Access to validation directory."""
        return self.data_dirs.valid_dir if isinstance(self.data_dirs, SplitDirectories) else None

    @property
    def test_dir(self) -> Optional[str]:
        """Access to test directory."""
        return self.data_dirs.test_dir if isinstance(self.data_dirs, SplitDirectories) else None

    @property
    def source_dir(self) -> Optional[str]:
        """Access to source directory."""
        return self.data_dirs if isinstance(self.data_dirs, str) else None

    def get_yaml_config(self) -> Dict[str, Any]:
        config = {
            'data_format': self.data_format,
            'dataset_id': self.dataset_id,
            'data_loading_specs': self.data_loading_specs.get_yaml_config() if hasattr(self.data_loading_specs, 'get_yaml_config') else self.data_loading_specs,
            'data_specs': self.data_specs.get_yaml_config() if hasattr(self.data_specs, 'get_yaml_config') else self.data_specs,
        }
        
        if isinstance(self.data_dirs, SplitDirectories):
            config.update({
                'train_dir': self.data_dirs.train_dir,
                'valid_dir': self.data_dirs.valid_dir,
                'test_dir': self.data_dirs.test_dir
            })
        else:
            config['source_dir'] = self.data_dirs
            
        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        dls = config_dict.get('data_loading_specs', {})
        ds = config_dict.get('data_specs', {})
        config_dict = dict(config_dict)
        
        # Handle backward compatibility: convert old format to new format
        train_dir = config_dict.pop('train_dir', None)
        valid_dir = config_dict.pop('valid_dir', None) 
        test_dir = config_dict.pop('test_dir', None)
        source_dir = config_dict.pop('source_dir', None)
        
        # Determine data_dirs based on provided parameters
        if train_dir or valid_dir or test_dir:
            if not all([train_dir, valid_dir, test_dir]):
                raise ConfigValidationError(
                    "When providing split directories, all three (train_dir, valid_dir, test_dir) must be specified.",
                    "split_directories"
                )
            config_dict['data_dirs'] = SplitDirectories(
                train_dir=train_dir,
                valid_dir=valid_dir, 
                test_dir=test_dir
            )
        elif source_dir:
            config_dict['data_dirs'] = source_dir
        else:
            raise ConfigValidationError(
                "Either provide split directories (train_dir, valid_dir, test_dir) or a source_dir.",
                "data_directories"
            )
        
        config_dict['data_loading_specs'] = DataLoadingSpecsConfig.from_dict(dls) if not isinstance(dls, DataLoadingSpecsConfig) else dls
        config_dict['data_specs'] = TokenizerConfig.from_dict(ds) if not isinstance(ds, TokenizerConfig) else ds
        return cls(**config_dict)

    def get_data_dir(self, split=None) -> Optional[str]:
        if split in ['train', 'dev', 'valid', 'test']:
            split = split.lower()
            if isinstance(self.data_dirs, SplitDirectories):
                if split == 'train':
                    return self.data_dirs.train_dir
                elif split in ['dev', 'valid']:
                    return self.data_dirs.valid_dir
                else:  # test
                    return self.data_dirs.test_dir
            else:
                raise ValueError(f"Cannot get split '{split}' from source directory. Use split=None for source directory.")
        
        if split is None:
            if isinstance(self.data_dirs, str):
                return self.data_dirs
            else:
                raise ValueError("The dataset has splits, please provide a split name (train, valid, test).")
        
        raise ValueError(f"Unknown split: {split}. Please provide a valid split name.")

    def get_required_fields(self):
        return []


