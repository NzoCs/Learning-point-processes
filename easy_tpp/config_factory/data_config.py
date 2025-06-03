from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from easy_tpp.config_factory.base import BaseConfig, ConfigValidationError, config_factory, config_class


@config_class('tokenizer_config')
@dataclass
class TokenizerConfig(BaseConfig):
    """Configuration for event tokenizer."""
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
    train_dir: Optional[str] = None
    valid_dir: Optional[str] = None
    test_dir: Optional[str] = None
    source_dir: Optional[str] = None
    data_format: Optional[str] = None
    dataset_id: Optional[str] = None
    data_loading_specs: DataLoadingSpecsConfig = field(default_factory=DataLoadingSpecsConfig)
    data_specs: TokenizerConfig = field(default_factory=TokenizerConfig)
    

    def __post_init__(self):
        if self.data_format is None:
            if self.train_dir is not None:
                self.data_format = self.train_dir.split('.')[-1]
            elif self.source_dir is not None:
                self.data_format = self.source_dir.split('.')[-1]
        super().__post_init__()

    def get_yaml_config(self) -> Dict[str, Any]:
        return {
            'train_dir': self.train_dir,
            'valid_dir': self.valid_dir,
            'test_dir': self.test_dir,
            'source_dir': self.source_dir,
            'data_format': self.data_format,
            'dataset_id': self.dataset_id,
            'data_loading_specs': self.data_loading_specs.get_yaml_config() if hasattr(self.data_loading_specs, 'get_yaml_config') else self.data_loading_specs,
            'data_specs': self.data_specs.get_yaml_config() if hasattr(self.data_specs, 'get_yaml_config') else self.data_specs,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        dls = config_dict.get('data_loading_specs', {})
        ds = config_dict.get('data_specs', {})
        config_dict = dict(config_dict)
        config_dict['data_loading_specs'] = DataLoadingSpecsConfig.from_dict(dls) if not isinstance(dls, DataLoadingSpecsConfig) else dls
        config_dict['data_specs'] = TokenizerConfig.from_dict(ds) if not isinstance(ds, TokenizerConfig) else ds
        return cls(**config_dict)

    def get_data_dir(self, split=None) -> Optional[str]:
        if split in ['train', 'dev', 'valid', 'test']:
            split = split.lower()
            if split == 'train':
                return self.train_dir
            elif split in ['dev', 'valid']:
                return self.valid_dir
            else:
                return self.test_dir
        if split is None:
            if self.source_dir is None:
                raise ValueError("The dataset does not have a split, please provide the source dir.")
            return self.source_dir
        raise ValueError(f"Unknown split: {split}. Please provide a valid split name.")

    def get_required_fields(self):
        return []


