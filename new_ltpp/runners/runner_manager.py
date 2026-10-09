import os
from typing import List, Optional, Union

from new_ltpp.configs import RunnerConfig
from new_ltpp.runners.model_runner import Runner
from new_ltpp.runners.run_manifest import RunManifest
from new_ltpp.utils import logger


class RunnerManager:
    """
    Runner class that manages the training, validation, testing, and prediction phases
    of a temporal point process model with intelligent logging management.
    """

    def __init__(
        self,
        config: RunnerConfig,
        checkpoint_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.original_logger_config = config.logger_config
        self.is_setup = False
        self.enable_logging = True
        logger.critical(
            f"Runner initialized for model: {self.config.model_id} on dataset: {self.config.data_config.dataset_id}"
        )

    def setup_runner(self, enable_logging: bool = True):
        if not self.is_setup:
            # Create runner for the first time
            logger.critical(
                f"Setting up runner with logging {'enabled' if enable_logging else 'disabled'}."
            )

            self.runner = Runner(
                config=self.config,
                enable_logging=enable_logging,
                checkpoint_path=self.checkpoint_path,
            )

            self.is_setup = True
            self.enable_logging = enable_logging

        elif self.enable_logging != enable_logging:
            # Just change the logging configuration
            self.runner.set_logging(enable_logging)
            self.enable_logging = enable_logging

    def train(self) -> None:
        logger.info("=== TRAINING PHASE ===")
        self.setup_runner(enable_logging=True)
        self.runner.train()
        return None

    def test(self) -> None:
        logger.critical("=== TESTING PHASE ===")
        self.setup_runner(enable_logging=False)
        self.runner.test()
        return None

    def predict(self) -> None:
        logger.critical("=== PREDICTION PHASE ===")
        self.setup_runner(enable_logging=False)
        self.runner.predict()
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
        else:
            phases = phase

        logger.info(f"Runner executing phases: {phases}")

        rank = int(
            os.environ.get(
                "RANK",
                os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", "0")),
            )
        )
        manifest = RunManifest(self.config) if rank == 0 else None
        for current_phase in phases:
            if current_phase not in ("train", "test", "predict"):
                raise ValueError(f"Unknown phase: {current_phase}")
            if manifest is not None:
                manifest.start(current_phase)
                if self.checkpoint_path:
                    from new_ltpp.runners.run_manifest import sha256_file

                    manifest.value["resume_from"] = {
                        "path": self.checkpoint_path,
                        "sha256": sha256_file(self.checkpoint_path),
                    }
                    manifest.save()
            try:
                getattr(self, current_phase)()
            except Exception as error:
                if manifest is not None:
                    manifest.finish(current_phase, error=error)
                raise
            if manifest is not None:
                manifest.finish(current_phase, checkpoint=self.runner.checkpoint_path)
            results[current_phase] = "completed"

        return results
