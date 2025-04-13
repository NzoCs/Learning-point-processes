from easy_tpp.config_factory import Config
import multiprocessing


class TokenizerConfig(Config):
    def __init__(self, **kwargs):
        """Initialize the Config class.
        """
        
        self.num_event_types = kwargs.get('num_event_types')
        self.pad_token_id = kwargs.get('pad_token_id')
        self.padding_side = kwargs.get('padding_side', 'left')
        self.truncation_side = kwargs.get('truncation_side', 'left')
        self.padding_strategy = kwargs.get('padding_strategy', 'longest')
        self.max_len = kwargs.get('max_len')
        self.truncation_strategy = kwargs.get('truncation_strategy')
        try:
            self.num_event_types_pad = self.num_event_types + 1
        except:
            self.num_event_types_pad = None
        self.model_input_names = kwargs.get('model_input_names')

        if self.padding_side is not None and self.padding_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )

        if self.truncation_side is not None and self.truncation_side not in ["right", "left"]:
            raise ValueError(
                f"Truncation side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )
    
    def set_attribute(self, name, value):
        """Set the attribute of the config.

        Args:
            name (str): name of the attribute.
            value (any): value of the attribute.
        """
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise AttributeError(f"Attribute {name} not found in config.")

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the data specs in dict format.
        """
        return {
            'num_event_types': self.num_event_types,
            'pad_token_id': self.pad_token_id,
            'padding_side': self.padding_side,
            'truncation_side': self.truncation_side,
            'padding_strategy': self.padding_strategy,
            'truncation_strategy': self.truncation_strategy,
            'max_len': self.max_len
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            TokenizerConfig: Config class for tokenizer specs.
        """
        return TokenizerConfig(**yaml_config)

    def copy(self):
        """Copy the config.

        Returns:
            TokenizerConfig: A copy of current config.
        """
        return TokenizerConfig(
            num_event_types=self.num_event_types,
            pad_token_id=self.pad_token_id,
            padding_side=self.padding_side,
            truncation_side=self.truncation_side,
            padding_strategy=self.padding_strategy,
            truncation_strategy=self.truncation_strategy,
            max_len=self.max_len,
            model_input_names=self.model_input_names
        )


class DataLoadingSpecsConfig(Config):
    """Configuration class for data specifications.
    
    This class manages parameters related to data loading and processing.
    """
    
    def __init__(self, **kwargs):
        """Initialize the DataLoadingSpecsConfig class.
        
        Args:
            batch_size (int, optional): Size of each batch. Defaults to 1.
            num_workers (int, optional): Number of workers for data loading. If None, uses CPU count.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to None.
            padding (bool, optional): Whether to pad sequences. Defaults to None.
            truncation (bool, optional): Whether to truncate sequences. Defaults to None.
            tensor_type (str, optional): Type of tensor to return ('pt', 'tf', etc.). Defaults to 'pt'.
            max_len (int, optional): Maximum sequence length. Defaults to None.
        """
        
        
        self.batch_size = kwargs.get('batch_size', 1)
        
        # Automatically set num_workers based on CPU count if not explicitly provided
        if 'num_workers' in kwargs:
            self.num_workers = kwargs['num_workers']
        else:
            try:
                # Use CPU count with a small margin to avoid overloading
                cpu_count = multiprocessing.cpu_count()
                # Common practice is to use CPU count - 1 to leave one core for other processes
                self.num_workers = max(1, cpu_count - 1)
            except:
                # Fallback to a safe default if CPU count cannot be determined
                self.num_workers = 0
        
        self.shuffle = kwargs.get('shuffle', None)
        self.padding = kwargs.get('padding', None)
        self.truncation = kwargs.get('truncation', None)
        self.tensor_type = kwargs.get('tensor_type', 'pt')
        self.max_length = kwargs.get('max_len', None)

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the data specs in dict format.
        """
        return {
            'batch_size': self.batch_size,
            'tensor_type': self.tensor_type,
            'num_workers': self.num_workers,
            'shuffle': self.shuffle,
            'padding': self.padding,
            'truncation': self.truncation,
            'max_len': self.max_length
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            DataLoadingSpecsConfig: Config class for data specs.
        """
        return DataLoadingSpecsConfig(**yaml_config)

    def copy(self):
        """Copy the config.

        Returns:
            DataLoadingSpecsConfig: a copy of current config.
        """
        return DataLoadingSpecsConfig(
            batch_size=self.batch_size,
            tensor_type=self.tensor_type,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            padding=self.padding,
            truncation=self.truncation,
            max_len=self.max_length
        )


@Config.register('data_config')
class DataConfig(Config):
    def __init__(self, **kwargs):
        """Initialize the DataConfig object.

        Args:
            train_dir (str): dir of train set.
            valid_dir (str): dir of valid set.
            test_dir (str): dir of test set.
            source_dir (str): dir of source data.
            data_format (str, optional): format of the data. Defaults to None.
            data_specs (dict, optional): specs of dataset. Defaults to None.
            tokenizer_config (dict, optional): tokenizer configuration. Defaults to None.
        """
        self.train_dir = kwargs.get("train_dir")
        self.valid_dir = kwargs.get('valid_dir')
        self.test_dir = kwargs.get("test_dir")
        self.source_dir = kwargs.get("source_dir")
        self.data_format = kwargs.get('data_format')
        
        if self.data_format is None:
            if self.train_dir is not None:
                self.data_format = self.train_dir.split('.')[-1]
            elif self.source_dir is not None:
                self.data_format = self.source_dir.split('.')[-1]

        try:
            data_loading_specs = kwargs.get("data_loading_specs", {})
            self.data_loading_specs = DataLoadingSpecsConfig.parse_from_yaml_config(data_loading_specs)
        except:
            self.data_loading_specs = kwargs.get("data_loading_specs")
        
        try:
            data_specs = kwargs.get("data_specs", {})
            self.data_specs = TokenizerConfig.parse_from_yaml_config(data_specs)
        except:
            self.data_specs = kwargs.get("data_loading_specs")
            
    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the data in dict format.
        """
        return {
            'train_dir': self.train_dir,
            'valid_dir': self.valid_dir,
            'test_dir': self.test_dir,
            'source_dir': self.source_dir,
            'data_format': self.data_format,
            'data_loading_specs': self.data_loading_specs.get_yaml_config(),
            'data_specs': self.data_specs.get_yaml_config(),
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config: dict, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            PLDataConfig: Config class for data.
        """
        experiment_id = kwargs.get('experiment_id')
        if experiment_id is not None : 
            exp_yaml_config = yaml_config[experiment_id]
        else:
            exp_yaml_config = yaml_config
        
        return DataConfig(**exp_yaml_config)

    def copy(self):
        """Copy the config.

        Returns:
            PLDataConfig: a copy of current config.
        """
        return DataConfig(
            train_dir=self.train_dir,
            valid_dir=self.valid_dir,
            test_dir=self.test_dir,
            source_dir=self.source_dir,
            data_format=self.data_format,
            data_loading_specs=self.data_loading_specs,
            data_specs=self.data_specs
        )

    def get_data_dir(self, split):
        """Get the dir of the source raw data.

        Args:
            split (str): dataset split notation, 'train', 'dev' or 'valid', 'test'.

        Returns:
            str: dir of the source raw data file.
        """
        try:
            split = split.lower()
            if split == 'train':
                return self.train_dir
            elif split in ['dev', 'valid']:
                return self.valid_dir
            else:
                return self.test_dir
        except:
            return self.source_dir

