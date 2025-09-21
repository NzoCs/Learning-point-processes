from easy_tpp.configs import RunnerConfig
from easy_tpp.runners.model_runner import Trainer
from easy_tpp.utils import logger
from typing import Optional, Union, List
import copy
import traceback


class Runner:
    """
    Runner class that manages the training, validation, testing, and prediction phases
    of a temporal point process model with intelligent logging management.
    """


    def __init__(
        self,
        config: RunnerConfig,
        checkpoint_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.kwargs = kwargs
        self.original_logger_config = config.trainer_config.logger_config
        self.is_setup = False
        self.trainer = None
        self.enable_logging = True
        logger.critical(
            f"Runner initialized for model: {config.model_config.model_id} on dataset: {config.data_config.dataset_id}"
        )


    def setup_trainer(self, enable_logging: bool = True):
        if not self.is_setup:
            # Create trainer for the first time
            config_copy = copy.deepcopy(self.config)
            self.trainer = Trainer(
                config=config_copy,
                enable_logging=enable_logging,
                checkpoint_path=self.checkpoint_path,
                output_dir=self.output_dir,
                **self.kwargs,
            )
            self.is_setup = True
            self.enable_logging = enable_logging
        elif self.enable_logging != enable_logging:
            # Just change the logging configuration
            self.trainer.set_logging(enable_logging)
            self.enable_logging = enable_logging


    def train(self) -> None:
        logger.info("=== TRAINING PHASE ===")
        self.setup_trainer(enable_logging=True)
        self.trainer.train()
        return None


    def test(self) -> Optional[dict]:
        logger.critical("=== TESTING PHASE ===")
        self.setup_trainer(enable_logging=False)
        self.trainer.test()
        return None


    def predict(self) -> Optional[str]:
        logger.critical("=== PREDICTION PHASE ===")
        self.setup_trainer(enable_logging=False)
        self.trainer.predict()
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

        return results