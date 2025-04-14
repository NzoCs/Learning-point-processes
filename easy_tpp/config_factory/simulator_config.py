from easy_tpp.config_factory.config import Config
from easy_tpp.preprocess.data_loader import TPPDataModule
from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.config_factory.model_config import ModelConfig
from easy_tpp.models import BaseModel


class SimulatorConfig(Config):
    """
    Configuration class for the simulator.
    
    Attributes:
        save_dir (str): Directory to save simulation results.
        start_time (int): Start time for the simulation.
        end_time (int): End time for the simulation.
        history_batch_dir (str): Directory for history batch files.
        pretrained_model (str): Path to the pretrained model file.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initialize the SimulatorConfig with the provided configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary containing simulator settings.
        """
        
        self.save_dir = kwargs.get('save_dir', None)
        self.start_time = kwargs.get('start_time')
        self.end_time = kwargs.get('end_time')
        self.history_dataset_id = kwargs.get('history_dataset_id', None)
        
        # Build Config for history data
        history_config = kwargs.get('history_config', None)
        history_config = DataConfig.parse_from_yaml_config(history_config) if history_config else None
        
        
        model_config = kwargs.get('model_config', None)
        model_config = ModelConfig.parse_from_yaml_config(model_config) if model_config else None
        
        # Build history batch 
        try :
            self.history_data_module = TPPDataModule(data_config = history_config, batch_size = history_config.batch_size)
        except Exception as e:
            self.history_data_module = None
            
        try:
            self.model = BaseModel.generate_model_from_config(model_config)
        except Exception as e:
            self.model = None
        
    def parse_from_yaml_dict(self, config : dict) -> None:
        """
        Parse the configuration from a YAML dictionary.
        
        Args:
            config (dict): Configuration dictionary containing simulator settings.
        """
         
        
        required_keys = ['save_dir', 'end_time', 'start_time', 'history_config', 'model_config']
        
        # Check if all required keys are present in the config
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
            
        return SimulatorConfig(**config)