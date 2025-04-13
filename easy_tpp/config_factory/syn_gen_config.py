import copy
from easy_tpp.config_factory.config import Config

import torch

@Config.register('syn_gen_config')
class SynGenConfig(Config):
    
    def __init__(self, config_dict : dict):
        
        self.save_dir = config_dict.get("save_dir", None)
        
        # Generator configurations
        self.num_mark = config_dict.get("num_mark", 1)
        self.dtime_max = config_dict.get("dtime_max", 5.0)
        self.num_samples_boundary = config_dict.get("num_samples_boundary", 100)
        self.num_samples = config_dict.get("num_samples", 20)
        self.end_time = config_dict.get("end_time", None)
        self.test_end_time = config_dict.get("test_end_time", None)
        self.start_time = config_dict.get("start_time", 0.0)
        self.num_seq = config_dict.get("num_seq", 100)
        self.num_batch = config_dict.get("num_batch", 10)
        self.use_mc_samples = config_dict.get("use_mc_samples", True)
        self.batch_size = config_dict.get("batch_size", 32)
        self.device = config_dict.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model ID for generator selection
        self.model_id = config_dict.get("model_id", "Hawkes")
        self.model_config = config_dict.get("model_config", {})
        
        # Event sampler configuration
        self.sampler_config = config_dict.get("sampler_config", {})
        self.experiment_id = config_dict.get("experiment_id", "synthetic_data_gen")
    
    def get_yaml_config(self):
        """Get the yaml format config from self.

        Returns:
            dict: Configuration au format yaml
        """
        yaml_config = {
            'save_dir': self.save_dir,
            'num_mark': self.num_mark,
            'dtime_max': self.dtime_max,
            'num_samples_boundary': self.num_samples_boundary,
            'num_samples': self.num_samples,
            'end_time': self.end_time,
            'test_end_time': self.test_end_time,
            'start_time': self.start_time,
            'num_seq': self.num_seq,
            'num_batch': self.num_batch,
            'use_mc_samples': self.use_mc_samples,
            'batch_size': self.batch_size,
            'model_id': self.model_id,
            'sampler_config': self.sampler_config,
            'experiment_id': self.experiment_id
        }
        return yaml_config
    
    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            SynGenConfig: Config class for synthetic data generation.
        """
        # Create a copy of the yaml_config to avoid modifying the original
        config_dict = dict(yaml_config)
        
        # Remove pipeline_config_id as it's not used in __init__
        if 'pipeline_config_id' in config_dict:
            del config_dict['pipeline_config_id']
        
        
        # Update with any additional kwargs
        direct_parse = kwargs.get('direct_parse', False)
        if not direct_parse:
            exp_id = kwargs.get('experiment_id')
            config_dict = config_dict[exp_id]
        if kwargs:
            config_dict.update(kwargs)
            
        return SynGenConfig(config_dict)
    
    def copy(self):
        """Get a same and freely modifiable copy of self.

        Returns:
            SynGenConfig: A copy of the current config.
        """
        return SynGenConfig(copy.deepcopy(self.get_yaml_config()))


