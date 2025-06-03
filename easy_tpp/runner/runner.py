from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner.trainer import Trainer
from easy_tpp.utils import logger
from typing import Optional, Union, List
import copy
import traceback

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
        
        logger.critical(f"Runner initialized for model: {config.model_config.model_id} on dataset: {config.data_config.dataset_id}")
    
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
        logger.critical("=== TESTING PHASE ===")
        trainer = self._create_trainer(enable_logging=False)
        trainer.test()  # This method doesn't return results but saves them to file
        
        return None
    
    def predict(self) -> Optional[str]:
        """
        Execute the prediction phase with logging disabled.
        
        Returns:
            str: Path to the directory where predictions are saved, None if error.
        """
        logger.critical("=== PREDICTION PHASE ===")
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
        
        logger.info(f"Runner executing phases: {phases}")
        
        for current_phase in phases:
            try:
                if current_phase == "train":
                    result = self.train()
                    results[current_phase] = "completed"
                
                elif current_phase == "test":
                    result = self.test()
                    results[current_phase] = "completed"
                
                elif current_phase == "predict":
                    result = self.predict()
                    results[current_phase] = "completed"
                
                else:
                    logger.error(f"Unknown phase: {current_phase}")
                    results[current_phase] = "error: unknown phase"
            
            except Exception as e:
                # Capture detailed error information
                exc_type, exc_value, exc_traceback = traceback.sys.exc_info()
                
                # Get the traceback as a formatted string
                tb_str = traceback.format_exc()
                
                # Get the specific line where the error occurred
                tb_lines = traceback.format_tb(exc_traceback)
                if tb_lines:
                    # Get the last frame (where the actual error occurred)
                    last_frame = tb_lines[-1].strip()
                    # Extract file and line info from the last frame
                    frame_info = traceback.extract_tb(exc_traceback)[-1]
                    filename = frame_info.filename
                    line_number = frame_info.lineno
                    function_name = frame_info.name
                    code_line = frame_info.line if frame_info.line else "N/A"
                    
                    error_location = f"File: {filename}, Line: {line_number}, Function: {function_name}"
                    error_code = f"Code: {code_line}"
                else:
                    error_location = "Unknown location"
                    error_code = "Unknown code"
                
                # Log detailed error information
                logger.error(f"Error in phase '{current_phase}': {str(e)}")
                logger.error(f"Error location: {error_location}")
                logger.error(f"Error code: {error_code}")
                logger.debug(f"Full traceback:\n{tb_str}")
                
                # Store error information in results
                results[current_phase] = {
                    "status": "error",
                    "error_message": str(e),
                    "error_type": exc_type.__name__ if exc_type else "Unknown",
                    "error_location": error_location,
                    "error_code": error_code,
                    "full_traceback": tb_str
                }
                
                # Continue with other phases instead of stopping
                continue       

        return results

# Convenience function for backward compatibility and simple usage
def run_experiment(config: RunnerConfig, 
                  phase: Union[str, List[str]] = "all",
                  checkpoint_path: Optional[str] = None,
                  output_dir: Optional[str] = None,
                  **kwargs) -> dict:
    """
    Convenience function to run an experiment with a single function call.
    
    Args:
        config (RunnerConfig): Configuration object.
        phase (Union[str, List[str]]): Phase(s) to execute.
        checkpoint_path (str, optional): Path to checkpoint file.
        output_dir (str, optional): Output directory.
        **kwargs: Additional arguments for the Trainer.
    
    Returns:
        dict: Results from executed phases.
    """
    runner = Runner(config, checkpoint_path, output_dir, **kwargs)
    return runner.run(phase)
