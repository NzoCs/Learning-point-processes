from easy_tpp.config_factory.config import Config
from easy_tpp.preprocess.data_loader import TPPDataModule
from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.config_factory.model_config import ModelConfig
from easy_tpp.models.basemodel import BaseModel

import os
import copy


@Config.register("simulator_config")
class SimulatorConfig(Config):
    """
    Configuration class for the simulator.
    
    Attributes:
        save_dir (str): Directory to save simulation results.
        start_time (float): Start time for the simulation.
        end_time (float): End time for the simulation.
        history_data_module (TPPDataModule): Data module for history data.
        pretrained_model (BaseModel): Pretrained model for simulation.
        split (str): Dataset split to use for simulation.
        seed (int): Random seed for reproducibility.
        experiment_id (str): Identifier for the experiment.
        dataset_id (str): Identifier for the dataset.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initialize the SimulatorConfig with the provided configuration dictionary.
        
        Args:
            kwargs (dict): Configuration dictionary containing simulator settings.
                Required keys:
                - end_time: End time for the simulation
                - start_time: Start time for the simulation
                - model_config: Model configuration dictionary
                - history_config: History data configuration dictionary
        
        Raises:
            ValueError: If required configuration keys are missing
        """
        # Validate required fields
        required_keys = ['model_config', 'hist_data_config']
        missing_keys = [key for key in required_keys if key not in kwargs]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

        # Basic configuration attributes
        dataset_id = kwargs.get('dataset_id', None)
        self.seed = int(kwargs.get('seed', 2002))
        self.split = kwargs.get('split', 'test')
        
        # Build Config for history data
        history_config = kwargs.get('hist_data_config', {})
        self.history_config = DataConfig.parse_from_yaml_config(history_config) 
        self.max_size = kwargs.get("max_size", 10**4)
        
        # Build model configuration
        model_config = kwargs.get('model_config', {})
        ckpt_path = f"../train/checkpoints/{model_config['model_id']}/{dataset_id}/trained_models/best.ckpt"
        model_config['model_path'] = ckpt_path
        model_config = ModelConfig.parse_from_yaml_config(model_config)
        
        # Create data module and model from configs
        self.history_data_module = TPPDataModule(data_config=self.history_config)
        self.pretrained_model = BaseModel.generate_model_from_config(model_config)
        
        model_id = model_config.model_id
        self.save_dir = kwargs.get("save_dir", f"./simul/{model_id}/{dataset_id}")
        os.makedirs(self.save_dir, exist_ok=True)
    
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
    def parse_from_yaml_config(yaml_config: dict, **kwargs) -> 'SimulatorConfig':
        """Parse configuration from YAML dictionary.
        
        Args:
            yaml_config (dict): Configuration dictionary from YAML
            **kwargs: Additional keyword arguments
            
        Returns:
            SimulatorConfig: Configured simulator instance
            
        Raises:
            ValueError: If required configurations are missing
        """
        
        experiment_id = kwargs.get('experiment_id')
        dataset_id = kwargs.get('dataset_id')
        
        if experiment_id is not None:
            exp_yaml_config = yaml_config[experiment_id]
        else:
            exp_yaml_config = yaml_config
        
        # Initialize history data config
        data_loading_specs = exp_yaml_config.get('data_loading_specs', {})
        
        hist_data_config = yaml_config.get('data', {}).get(dataset_id, {})
        hist_data_config['data_loading_specs'] = data_loading_specs
        hist_data_config['dataset_id'] = dataset_id
        exp_yaml_config['dataset_id'] = dataset_id
        
        exp_yaml_config['hist_data_config'] = hist_data_config
        # Initialize model config
        model_config = exp_yaml_config.get('model_config', {})
        if 'num_event_types' not in model_config:
            # If we have access to hist_data_config, we can set num_event_types
            if hasattr(hist_data_config, 'data_specs') and hasattr(hist_data_config.data_specs, 'num_event_types'):
                model_config['num_event_types'] = hist_data_config.data_specs.num_event_types
        
        
        return SimulatorConfig(**exp_yaml_config)
    
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
            'seed': self.seed,
            'split': self.split,
            'model_config': self.model_config.get_yaml_config() if self.model_config else None,
            'history_config': self.history_config.get_yaml_config() if self.history_config else None
        }
        
        if hasattr(self, 'experiment_id') and self.experiment_id:
            yaml_config['experiment_id'] = self.experiment_id
            
        if hasattr(self, 'dataset_id') and self.dataset_id:
            yaml_config['dataset_id'] = self.dataset_id
            
        return {k: v for k, v in yaml_config.items() if v is not None}
    
    def copy(self):
        """
        Get a deep copy of the current configuration.
        
        Returns:
            SimulatorConfig: A new instance with the same configuration values
        """
        config_dict = copy.deepcopy(self.get_yaml_config())
        return SimulatorConfig(**config_dict)