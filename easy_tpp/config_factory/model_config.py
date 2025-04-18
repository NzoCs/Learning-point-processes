from easy_tpp.config_factory.config import Config
from easy_tpp.utils.const import Backend

from typing import Optional, Union

def get_available_gpu():
    """Detect available GPUs on the machine.
    
    Returns:
        int: GPU ID (0 if available, -1 if no GPU available)
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 0
    except ImportError:
        pass
    
    return -1

class ThinningConfig(Config):
    """Configuration class for the thinning algorithms.
    
    Attributes:
        num_sample (int): Number of sampled next event times (default: 1)
        num_exp (int): Number of i.i.d. Exp(intensity_bound) draws (default: 500)
        num_steps (int): Number of steps for multiple step predictions (default: 10)
        over_sample_rate (float): Multiplier for intensity upper bound (default: 5.0)
        num_samples_boundary (int): Samples for intensity boundary computation (default: 5)
        dtime_max (float): Maximum delta time in sampling (default: 5.0)
    """
    
    def __init__(self, 
                 num_sample: int = 1,
                 num_exp: int = 500,
                 num_steps: int = 10,
                 over_sample_rate: float = 5.0,
                 num_samples_boundary: int = 5,
                 dtime_max: float = 5.0) -> None:
        """Initialize ThinningConfig with type-safe parameters."""
        self.num_sample = int(num_sample)
        self.num_exp = int(num_exp)
        self.num_steps = int(num_steps)
        self.over_sample_rate = float(over_sample_rate)
        self.num_samples_boundary = int(num_samples_boundary)
        self.dtime_max = float(dtime_max)

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate configuration parameters."""
        if self.num_sample <= 0:
            raise ValueError("num_sample must be positive")
        if self.num_exp <= 0:
            raise ValueError("num_exp must be positive")
        if self.over_sample_rate <= 0:
            raise ValueError("over_sample_rate must be positive")
        if self.num_samples_boundary <= 0:
            raise ValueError("num_samples_boundary must be positive")
        if self.dtime_max <= 0:
            raise ValueError("dtime_max must be positive")

    def get_yaml_config(self) -> dict:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the thinning specs in dict format.
        """
        return {'num_sample': self.num_sample,
                'num_exp': self.num_exp,
                'over_sample_rate': self.over_sample_rate,
                'num_samples_boundary': self.num_samples_boundary,
                'dtime_max': self.dtime_max}

    @staticmethod
    def parse_from_yaml_config(yaml_config) -> 'ThinningConfig':
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.ThinningConfig: Config class for thinning algorithms.
        """
        return ThinningConfig(**yaml_config) if yaml_config is not None else None

    def copy(self):
        """Copy the config.

        Returns:
            EasyTPP.ThinningConfig: a copy of current config.
        """
        return ThinningConfig(num_sample=self.num_sample,
                              num_exp=self.num_exp,
                              over_sample_rate=self.over_sample_rate,
                              num_samples_boundary=self.num_samples_boundary,
                              dtime_max=self.dtime_max)

class SimulationConfig(Config):
    """Configuration class for simulation parameters.
    
    Attributes:
        start_time (float): Start time for the simulation.
        end_time (float): End time for the simulation.
        batch_size (int): Batch size for training/testing.
    """
    
    def __init__(self, **kwargs):
        """Initialize the Config class."""
        
        required_keys = ['start_time', 'end_time']
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required config key: {key}")
            
        self.start_time = kwargs.get('start_time')
        self.end_time = kwargs.get('end_time')
        self.batch_size = kwargs.get('batch_size', 32)

    def get_yaml_config(self) -> dict:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the thinning specs in dict format.
        """
        return {'start_time': self.start_time,
                'end_time': self.end_time,
                'batch_size': self.batch_size}
    
    def copy(self) -> 'SimulationConfig':
        """Copy the config.

        Returns:
            SimulationConfig: a copy of current config.
        """
        return SimulationConfig(start_time=self.start_time,
                                 end_time=self.end_time,
                                 batch_size=self.batch_size)
        
    @staticmethod
    def parse_from_yaml_config(yaml_config) -> 'SimulationConfig':
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            SimulationConfig: Config class for simulation specs.
        """
        return SimulationConfig(**yaml_config) if yaml_config is not None else None

class BaseConfig(Config):
    """Base configuration for the EasyTPP library.
    
    Attributes:
        stage (str): Training stage ('train' or 'test')
        backend (Backend): Computation backend (Torch or TensorFlow)
        dataset_id (str): Dataset identifier
        lr (float): Learning rate
        lr_scheduler (bool): Enable learning rate scheduling
        max_epochs (int | None): Maximum training epochs
        base_dir (str): Base directory for model/logs
    """
    
    VALID_STAGES = {'train', 'test', 'val'}
    
    def __init__(self,
                 stage: str = 'train',
                 backend: str = 'torch',
                 dataset_id: Optional[str] = None,
                dropout_rate: float = 0.0,
                 lr: float = 0.001,
                 lr_scheduler: bool = False,
                 max_epochs: Optional[int] = None,
                 base_dir: Optional[str] = None) -> None:
        
        """Initialize BaseConfig with type-safe parameters."""
        self.stage = self._validate_stage(stage)
        self.lr = float(lr)
        self.lr_scheduler = bool(lr_scheduler)
        
        self.dropout = float(dropout_rate)
        self.max_epochs = int(max_epochs) if max_epochs is not None else None
        self.dataset_id = dataset_id
        self.base_dir = base_dir
        self.backend = self.set_backend(backend)

        self._validate_parameters()

    def _validate_stage(self, stage: str) -> str:
        """Validate training stage."""
        stage = stage.lower()
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Stage must be one of {self.VALID_STAGES}")
        return stage

    def _validate_parameters(self) -> None:
        """Validate configuration parameters."""
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.lr_scheduler and self.max_epochs is None:
            raise ValueError("max_epochs required when lr_scheduler is enabled")
        if self.max_epochs is not None and self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")

    @staticmethod
    def set_backend(backend):
        if backend.lower() in ['torch', 'pytorch']:
            return Backend.Torch
        elif backend.lower() in ['tf', 'tensorflow']:
            return Backend.TF
        else:
            raise ValueError(
                f"Backend should be selected between 'torch or pytorch' and 'tf or tensorflow', "
                f"current value: {backend}"
            )

    def get_yaml_config(self) -> dict:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the base config specs in dict format.
        """
        return {'stage': self.stage,
                'backend': str(self.backend),
                'dataset_id': self.dataset_id,
                'base_dir': self.base_dir,
                'lr': self.lr,
                'lr_scheduler': self.lr_scheduler,
                'max_epochs': self.max_epochs,
                'dropout_rate': self.dropout}

    @staticmethod
    def parse_from_yaml_config(yaml_config) -> 'BaseConfig':
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            BaseConfig: Config class for trainer specs.
        """
        return BaseConfig(**yaml_config)

    def copy(self):
        """Copy the config.

        Returns:
            BaseConfig: a copy of current config.
        """
        return BaseConfig(stage=self.stage,
                          backend=self.backend,
                          dataset_id=self.dataset_id,
                          base_dir=self.base_dir,
                          dropout_rate=self.dropout,
                          lr=self.lr,
                          lr_scheduler=self.lr_scheduler,
                          max_epochs=self.max_epochs)


class ModelSpecsConfig:
    """Configuration class for the model specifications.
    This class is used to define the configuration for the model specifications.
    
    Attributes:
        rnn_type (str): Type of RNN to be used (e.g., 'LSTM', 'GRU').
        hidden_size (int): Size of the hidden layer.
        time_emb_size (int): Size of the time embedding layer.
        num_layers (int): Number of layers in the RNN.
        num_heads (int): Number of attention heads.
        sharing_param_layer (bool): Whether to share parameters across layers.
        use_mc_samples (bool): Whether to use Monte Carlo samples for log-likelihood computation.
        loss_integral_num_sample_per_step (int): Number of samples per step for loss integral computation.
    """
    
    def __init__(self, **kwargs):
        """Initialize the Config class."""
        self.rnn_type = kwargs.get('rnn_type', 'LSTM')
        self.hidden_size = kwargs.get('hidden_size', 32)
        self.time_emb_size = kwargs.get('time_emb_size', 16)
        self.num_layers = kwargs.get('num_layers', 2)
        self.num_heads = kwargs.get('num_heads', 2)
        self.sharing_param_layer = kwargs.get('sharing_param_layer', False)
        self.loss_integral_num_sample_per_step = kwargs.get('loss_integral_num_sample_per_step', 20)  # mc_num_sample_per_step
        self.use_ln = kwargs.get('use_norm', False)

        #for cumulative hazard function network of the FullyNN model
        self.num_mlp_layers = kwargs.get('num_mlp_layers', 2)
        self.proper_marked_intensities = kwargs.get('proper_marked_intensities', False)
        
        #for IntensityFree model
        self.num_mix_components = kwargs.get('num_mix_components', 1)
        self.mean_log_inter_time = kwargs.get('mean_log_inter_time', 0.0)
        self.std_log_inter_time = kwargs.get('std_log_inter_time', 1.0)

        #for ODETPP model
        self.ode_num_sample_per_step = kwargs.get('ode_num_sample_per_step', 20)

    @staticmethod
    def parse_from_yaml_config(yaml_config) -> 'ModelSpecsConfig':
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            ModelConfig: Config class for trainer specs.
        """
        return ModelSpecsConfig(**yaml_config)
    
    def copy(self) -> 'ModelSpecsConfig':
        """Copy the config.

        Returns:
            ModelSpecsConfig: a copy of current config.
        """
        return ModelSpecsConfig(rnn_type=self.rnn_type,
                                hidden_size=self.hidden_size,
                                time_emb_size=self.time_emb_size,
                                num_layers=self.num_layers,
                                num_heads=self.num_heads,
                                sharing_param_layer=self.sharing_param_layer,
                                loss_integral_num_sample_per_step=self.loss_integral_num_sample_per_step,
                                use_norm=self.use_ln)
    
    def get_yaml_config(self) -> dict:
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the model specs in dict format.
        """
        return {'rnn_type': self.rnn_type,
                'hidden_size': self.hidden_size,
                'time_emb_size': self.time_emb_size,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'sharing_param_layer': self.sharing_param_layer,
                'loss_integral_num_sample_per_step': self.loss_integral_num_sample_per_step,
                'use_norm': self.use_ln}
                
                
@Config.register('model_config')
class ModelConfig(Config):
    """
    Configuration class for the model.
    This class is used to define the configuration for the model.
    
    Attributes:
        dropout_rate (float): Dropout rate for the model.
        use_ln (bool): Whether to use layer normalization.
        thinning (ThinningConfig): Configuration for the thinning process.  
        is_training (bool): Whether the model is in training mode.
        num_event_types_pad (int): Number of event types for padding.
        num_event_types (int): Number of event types.
        pad_token_id (int): Padding token ID.
        model_id (str): Model ID.
        gpu (int): GPU ID to be used for training.
        model_specs (ModelSpecsConfig): Configuration for the model specifications."""
    
    def __init__(self, **kwargs):
        """Initialize the Config class."""
        
        required_keys = ['num_event_types', 'model_id']
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required config key: {key}")
        
        self.is_training = kwargs.get('training', False)
        self.num_event_types = kwargs.get('num_event_types')
        self.num_event_types_pad = kwargs.get('num_event_types_pad', self.num_event_types + 1)
        self.pad_token_id = kwargs.get('event_pad_index', self.num_event_types)
        # Use available GPU if not specified
        self.gpu = kwargs.get('gpu', get_available_gpu())
        self.model_id = kwargs.get('model_id')
        
        self.use_mc_samples = kwargs.get('use_mc_samples', True)  # if using MC samples in computing log-likelihood
        
        self.device = kwargs.get('device', 'cuda' if self.gpu >= 0 else 'cpu')
        
        self.thinning = ThinningConfig.parse_from_yaml_config(kwargs.get('thinning', {}))
        
        self.specs = ModelSpecsConfig.parse_from_yaml_config(kwargs.get('specs', {}))

        self.base_config = BaseConfig.parse_from_yaml_config(kwargs.get('base_config', {}))
        
        self.simulation_config = kwargs.get('simulation', None)
        
        if self.simulation_config is not None:
            self.simulation_config = SimulationConfig.parse_from_yaml_config(self.simulation_config)
        
        
    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the model config specs in dict format.
        """
        
        return {
            'use_mc_samples': self.use_mc_samples,
            'thinning': self.thinning.get_yaml_config(),
            'training': self.is_training,
            'num_event_types_pad': self.num_event_types_pad,
            'num_event_types': self.num_event_types,
            'event_pad_index': self.pad_token_id,
            'model_id': self.model_id,
            'gpu': self.gpu,
            'model_specs': self.specs.get_yaml_config(),
            'base_config': self.base_config.get_yaml_config()
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            ModelConfig: Config class for trainer specs.
        """
        return ModelConfig(**yaml_config)

    def copy(self):
        """Copy the config.

        Returns:
            ModelConfig: a copy of current config.
        """
        return ModelConfig(
            use_mc_samples=self.use_mc_samples,
            thinning=self.thinning.copy(),
            num_event_types_pad=self.num_event_types_pad,
            num_event_types=self.num_event_types,
            event_pad_index=self.pad_token_id,
            gpu=self.gpu,
            model_specs=self.specs.copy(),
            base_config=self.base_config.copy()
        )
