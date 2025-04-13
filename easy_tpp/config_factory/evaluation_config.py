from copy import deepcopy
from omegaconf import OmegaConf
from typing import Dict, Any

from easy_tpp.config_factory.config import Config
from easy_tpp.utils import logger

@Config.register('evaluation_config')
class EvaluationConfig(Config):
    
    def __init__(self, **kwargs):
        """Initialize evaluation configuration.
        
        Args:
            **kwargs: Configuration options including:
                - mode: Evaluation mode ("prediction" or "simulation")
                - true_data_config: Configuration for ground truth data
                - pred_data_config: Configuration for prediction data
                - data_specs: Specifications for the dataset
                - label_split: Data split for labels (e.g., "test")
                - pred_split: Data split for predictions (e.g., "test")
        """
        self.mode = kwargs.get("mode", "prediction")
        self.label_data_config = kwargs.get("label_data_config", {})
        self.pred_data_config = kwargs.get("pred_data_config", {})
        self.data_specs = kwargs.get("data_specs", {})
        self.label_split = kwargs.get("label_split", "test")
        self.pred_split = kwargs.get("pred_split", "test")
        self.batch_size = kwargs.get("batch_size", 1)
        
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        
        logger.info(f"EvaluatorConfig initialized with mode: {self.mode}")
    
    def get_yaml_config(self):
        """Get the yaml format config from self.

        Returns:
            OmegaConf: Configuration in OmegaConf format.
        """
        config_dict = {
            "mode": self.mode,
            "true_data_config": self.true_data_config,
            "pred_data_config": self.pred_data_config,
            "data_specs": self.data_specs,
            "label_split": self.label_split,
            "pred_split": self.pred_split
        }
        
        # Add any additional attributes
        for key, value in vars(self).items():
            if key not in config_dict:
                config_dict[key] = value
                
        return OmegaConf.create(config_dict)
    
    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (OmegaConf): configs from yaml file.
            **kwargs: Additional arguments to override yaml configs.

        Returns:
            EvaluatorConfig: Config class for evaluation.
        """
        config_dict = OmegaConf.to_container(yaml_config, resolve=True)
        
        # Override with kwargs
        for key, value in kwargs.items():
            config_dict[key] = value
            
        return EvaluationConfig(**config_dict)
    
    def copy(self):
        """Get a same and freely modifiable copy of self.

        Returns:
            EvaluatorConfig: A copy of the current config.
        """
        return deepcopy(self)