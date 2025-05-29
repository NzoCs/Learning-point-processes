
from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner.trainer import Trainer
from easy_tpp.utils import logger
from typing import Optional, Union, List
import copy

class Runner:
    """
    Runner class that manages the training, validation, testing, and prediction phases
    of a temporal point process model with intelligent logging management.
    """
    
    def __init__(self, config: RunnerConfig, checkpoint_path: Optional[str] = None, output_dir: Optional[str] = None, **kwargs):
        """
        Initialize the Runner.
        
        Args:
            config (RunnerConfig): Configuration object containing all necessary parameters.
            checkpoint_path (str, optional): Path to a checkpoint file to resume training from.
            output_dir (str, optional): Directory to save outputs.
            **kwargs: Additional keyword arguments passed to the Trainer.
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.kwargs = kwargs
        
        # Store original logger config to restore it later
        self.original_logger_config = config.trainer_config.logger_config
        
        logger.info(f"Runner initialized for model: {config.model_config.model_id} on dataset: {config.data_config.dataset_id}")
    
    def _create_trainer(self, enable_logging: bool = True) -> Trainer:
        """
        Create a trainer with logging enabled or disabled.
        
        Args:
            enable_logging (bool): Whether to enable logging for this trainer.
            
        Returns:
            Trainer: Configured trainer instance.
        """
        # Create a copy of the config to avoid modifying the original
        config_copy = copy.deepcopy(self.config)
        
        if not enable_logging:
            # Disable logging by setting logger_config to None
            config_copy.trainer_config.logger_config = None
            logger.debug("Logging disabled for this phase")
        else:
            logger.debug("Logging enabled for this phase")
        
        return Trainer(
            config=config_copy,
            checkpoint_path=self.checkpoint_path,
            output_dir=self.output_dir,
            **self.kwargs
        )
    
    def train(self) -> None:
        """Execute the training phase with logging enabled."""
        logger.info("=== TRAINING PHASE ===")
        trainer = self._create_trainer(enable_logging=True)
        trainer.train()

        return None

    def test(self) -> Optional[dict]:
        """
        Execute the testing phase with logging disabled.
        
        Returns:
            dict: Test results if available, None otherwise.
        """
        logger.info("=== TESTING PHASE ===")
        trainer = self._create_trainer(enable_logging=False)
        trainer.test()  # This method doesn't return results but saves them to file
        
        return None
    
    def predict(self) -> Optional[str]:
        """
        Execute the prediction phase with logging disabled.
        
        Returns:
            str: Path to the directory where predictions are saved, None if error.
        """
        logger.info("=== PREDICTION PHASE ===")
        trainer = self._create_trainer(enable_logging=False)
        trainer.predict()  # This method doesn't return predictions but saves them
        
        return None
    
    def run(self, phase: Union[str, List[str]] = "all") -> dict:
        """
        Execute one or multiple phases based on the phase parameter.
        
        Args:
            phase (Union[str, List[str]]): Phase(s) to execute. Options:
                - "train": Only training
                - "test": Only testing
                - "predict": Only prediction
                - "validation": Only validation (warning: not fully implemented)
                - "all": All phases in sequence (train -> test -> predict)
                - List of phases: e.g., ["train", "test"]
        
        Returns:
            dict: Results from executed phases.
        """
        results = {}
        
        # Convert single phase to list for uniform processing
        if isinstance(phase, str):
            if phase == "all":
                phases = ["train", "test", "predict"]
            else:
                phases = [phase]
        else:
            phases = phase
        
        logger.info(f"Runner executing phases: {phases}")
        
        for current_phase in phases:
            try:
                if current_phase == "train":
                    self.train()
                
                elif current_phase == "test":
                    self.test()
                
                elif current_phase == "predict":
                    self.predict()
                
                else:
                    logger.error(f"Unknown phase: {current_phase}")
                    results[current_phase] = "error: unknown phase"
            
            except Exception as e:
                logger.error(f"Error in phase '{current_phase}': {str(e)}")
                # Continue with other phases instead of stopping
                continue
        
        logger.info(f"Runner completed.")
