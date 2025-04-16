from easy_tpp.config_factory.config import Config
from easy_tpp.preprocess.data_loader import TPPDataModule
from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.config_factory.model_config import ModelConfig
from easy_tpp.models import BaseModel


@Config.register("simulator_config")
class SimulatorConfig(Config):
    """
    Configuration class for the simulator.
    
    Attributes:
        save_dir (str): Directory to save simulation results.
        start_time (int): Start time for the simulation.
        end_time (int): End time for the simulation.
        history_data (TPPDataModule): Data module for history data.
        pretrained_model (BaseModel): Pretrained model for simulation.
        num_simulations (int): Number of simulations to run.
        splits (dict): Dictionary of dataset splits and their ratios.
        seed (int): Random seed for reproducibility.
        experiment_id (str): Identifier for the experiment.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initialize the SimulatorConfig with the provided configuration dictionary.
        
        Args:
            kwargs (dict): Configuration dictionary containing simulator settings.
        """
        
        self.save_dir = kwargs.get('save_dir', 'simulated_data')
        self.start_time = kwargs.get('start_time')
        self.end_time = kwargs.get('end_time')
        self.history_dataset_id = kwargs.get('history_dataset_id', None)
        self.splits = kwargs.get('splits', {'train': 0.6, 'test': 0.2, 'dev': 0.2})
        self.seed = kwargs.get('seed', None)
        self.num_simulations = kwargs.get('num_simulations', 100)
        self.experiment_id = kwargs.get('experiment_id', None)
        
        # Build Config for history data
        history_config = kwargs.get('history_config', None)
        history_config = DataConfig.parse_from_yaml_config(history_config) if history_config else None
        
        model_config = kwargs.get('model_config', None)
        model_config = ModelConfig.parse_from_yaml_config(model_config) if model_config else None
        
        self.history_data = None
        if history_config:
            self.history_data = TPPDataModule(data_config = history_config)
            
        self.pretrained_model = None
        if model_config:
            self.pretrained_model = BaseModel.generate_model_from_config(model_config)
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str): The key to retrieve
            default: The default value to return if key is not found
            
        Returns:
            The value associated with the key, or the default value
        """
        return getattr(self, key, default)
    
    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """
        Parse the configuration from a YAML dictionary, supporting multiple experiments.
        
        Args:
            yaml_config (dict): Configuration dictionary from YAML file.
            **kwargs: Additional keyword arguments, including experiment_id.
            
        Returns:
            SimulatorConfig: A new SimulatorConfig instance
        """
        # Check if direct_parse is specified to bypass experiment_id lookup
        direct_parse = kwargs.get('direct_parse', False)
        
        # Get experiment-specific configuration if not direct parsing
        if not direct_parse:
            experiment_id = kwargs.get('experiment_id')
            if experiment_id is not None and experiment_id in yaml_config:
                # Extract experiment-specific config
                config_dict = yaml_config[experiment_id].copy()
                # Add the experiment_id to the config
                config_dict['experiment_id'] = experiment_id
            else:
                # If no experiment ID specified or not found, use the top-level config
                config_dict = yaml_config.copy()
        else:
            # Use the provided config directly
            config_dict = yaml_config.copy()
            
        # Remove pipeline_config_id if present (not used in __init__)
        if 'pipeline_config_id' in config_dict:
            del config_dict['pipeline_config_id']
            
        # Update with any additional kwargs
        if kwargs:
            for key, value in kwargs.items():
                if key not in ['direct_parse', 'experiment_id']:
                    config_dict[key] = value
                    
        # Validate required fields
        required_keys = ['save_dir', 'end_time', 'start_time']
        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
            
        return SimulatorConfig(**config_dict)
    
    def get_yaml_config(self):
        """
        Return the config as a dictionary suitable for YAML serialization.
        
        Returns:
            dict: Configuration dictionary.
        """
        yaml_config = {
            'save_dir': self.save_dir,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'num_simulations': self.num_simulations,
            'seed': self.seed,
            'splits': self.splits,
        }
        
        if self.history_dataset_id:
            yaml_config['history_dataset_id'] = self.history_dataset_id
            
        if self.experiment_id:
            yaml_config['experiment_id'] = self.experiment_id
            
        return yaml_config