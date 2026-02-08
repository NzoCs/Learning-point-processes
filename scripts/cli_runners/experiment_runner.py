"""
Experiment Runner

Runner to execute TPP experiments using configuration builders.
Inspired by run_all_phase.py for loading full configuration from YAML.
"""

from pathlib import Path
from typing import Optional, Union

from new_ltpp.configs.config_builders import RunnerConfigBuilder
from new_ltpp.runners.runner_manager import RunnerManager

from .cli_base import CONFIG_MAP, CLIRunnerBase


from new_ltpp.configs.config_loaders.runner_config_loader import RunnerConfigYamlLoader


class ExperimentRunner(CLIRunnerBase):
    """
    Runner for executing TPP experiments.
    Uses RunnerConfigBuilder to load full configuration from YAML.
    Allows specifying each configuration type individually.
    """

    def __init__(self, debug: bool = False):
        super().__init__("ExperimentRunner", debug=debug)

    def _build_config_paths(self, **config_kwargs) -> dict:
        """
        Build configuration paths following the standard pattern.

        Pattern: {config_type}_configs.{config_name}

        Args:
            **config_kwargs: Dictionary of configurations {type: name}

        Returns:
            Dictionary of formatted configuration paths
        """

        config_paths = {}

        for config_type, config_name in config_kwargs.items():
            if config_name is not None:  # Skip None values
                if config_type in CONFIG_MAP:
                    prefix = CONFIG_MAP[config_type]
                    config_paths[f"{config_type}_config_path"] = (
                        f"{prefix}.{config_name}"
                    )
                else:
                    self.print_error(
                        f"Type de configuration non reconnu: {config_type}"
                    )

        return config_paths

    def run_experiment(
        self,
        config_path: Optional[Union[str, Path]] = None,
        phase: str = "train",
        max_epochs: Optional[int] = None,
        data_config: Optional[str] = None,
        general_specs_config: Optional[str] = None,
        training_config: Optional[str] = None,
        data_loading_config: Optional[str] = None,
        simulation_config: Optional[str] = None,
        thinning_config: Optional[str] = None,
        logger_config: Optional[str] = None,
        model_id: str = "NHP",
        save_dir: Optional[Union[str, Path]] = None,
        model_specs_config: Optional[str] = None,
        debug: bool = False,
    ) -> bool:
        """
        Run a TPP experiment with the given parameters.

        Args:
            config_path: Path to the YAML configuration file
            data_config: Data configuration (e.g., test, large, synthetic)
            general_specs_config: General model configuration (e.g., h16, h32)
            training_config: Training configuration (e.g., quick_test, full_training)
            data_loading_config: Data loading configuration
            simulation_config: Simulation configuration (optional)
            thinning_config: Thinning configuration (optional)
            logger_config: Logger configuration (e.g., mlflow, tensorboard)
            model_id: Model identifier (default: NHP)
            phase: Execution phase (train, test, predict, all)
            max_epochs: Maximum epochs to override YAML
            save_dir: Save directory to override YAML
            model_specs_config: Model-specific configuration (optional)
            debug: Debug mode

        Returns:
            True if the experiment completed successfully
        """
        # Enable debug mode if requested
        self.set_debug(debug)

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(config_path, str):
            config_path = Path(config_path)

        # Check required modules
        required_modules = ["new_ltpp.configs", "new_ltpp.runners"]
        if not self.check_dependencies(required_modules):
            return False

        self.print_info(f"Starting TPP experiment - Phase: {phase}")

        # Default configuration file if none provided
        if config_path is None:
            config_path = str(self.get_config_path())
            self.print_info(f"Using default configuration: {config_path}")

        # Validate configuration file
        if not Path(config_path).exists():
            self.print_error(f"Configuration file not found: {config_path}")
            return False

        # Build runner configuration from YAML (comme dans run_all_phase.py)
        config_builder = RunnerConfigBuilder()

        # Construire les chemins de configuration avec la fonction utilitaire
        # Build configuration paths with the helper function
        config_paths = self._build_config_paths(
            data=data_config,
            general_specs=general_specs_config,
            model_specs=model_specs_config,
            training=training_config,
            data_loading=data_loading_config,
            simulation=simulation_config,
            thinning=thinning_config,
            logger=logger_config,
        )

        self.print_info("Configurations used:")
        for path_key, path_value in config_paths.items():
            config_type = path_key.replace("_config_path", "").replace("_", " ").title()
            self.print_info(f"  • {config_type}: {path_value}")

        # Charger la configuration complète depuis le YAML via le Loader
        # Load full configuration from YAML via the Loader
        loader = RunnerConfigYamlLoader()
        config_dict = loader.load(
            yaml_path=config_path,
            training_config_path=config_paths.get("training_config_path"),  # type: ignore
            data_config_path=config_paths.get("data_config_path"),  # type: ignore
            model_config_path=config_paths.get("model_config_path"),  # type: ignore
            data_loading_config_path=config_paths.get("data_loading_config_path"),  # type: ignore
            simulation_config_path=config_paths.get("simulation_config_path"),
            thinning_config_path=config_paths.get("thinning_config_path"),
            logger_config_path=config_paths.get("logger_config_path"),
            general_specs_config_path=config_paths.get("general_specs_config_path"),
            model_specs_config_path=config_paths.get("model_specs_config_path"),
        )

        config_builder.from_dict(config_dict)

        self.print_info("YAML configuration loaded successfully")

        # Appliquer les overrides de paramètres CLI en utilisant les méthodes du builder
        # Apply CLI parameter overrides using the builder's methods
        if max_epochs is not None:
            config_builder.set_max_epochs(max_epochs)
            self.print_info(f"Override: max_epochs = {max_epochs}")

        # Ne passer save_dir que s'il est explicitement fourni par l'utilisateur
        # Only pass save_dir if explicitly provided by the user
        if save_dir:
            config_builder.set_save_dir(save_dir)
            self.print_info(f"Override: save_dir = {save_dir}")
        # Sinon, laisser les sous-couches générer leur propre save_dir par défaut
        # qui sera plus intelligent (model_id/dataset_id/etc.)
        # Otherwise, let subcomponents generate their own default save_dir
        # which will be more informative (model_id/dataset_id/etc.)

        # Créer la configuration finale avec la factory
        # Build the final configuration using the factory
        config = config_builder.build(model_id=model_id)

        # Validate phase
        valid_phases = ["train", "test", "predict", "all"]
        if phase not in valid_phases:
            self.print_error(f"Invalid phase: {phase}. Valid phases: {valid_phases}")
            return False

        # Créer et lancer le runner
        # Create and start the runner
        runner_manager = RunnerManager(config=config)

        if phase == "all":
            self.print_info("Full run: train → test → predict")

            # Exécuter chaque phase séparément comme dans run_all_phase.py
            # Execute each phase separately as in run_all_phase.py
            self.print_info("Phase 1/3: Training")
            train_results = runner_manager.run(phase="train")

            self.print_info("Phase 2/3: Testing")
            test_results = runner_manager.run(phase="test")

            self.print_info("Phase 3/3: Prediction")
            predict_results = runner_manager.run(phase="predict")

            # Combiner les résultats
            # Combine the results
            results = {
                "train": train_results,
                "test": test_results,
                "predict": predict_results,
            }

        else:
            self.print_info(f"Exécution phase: {phase}")
            self.print_info(f"Running phase: {phase}")
            results = runner_manager.run(phase=phase)

        self.print_success(f"Experiment completed successfully - Phase: {phase}")

        # Afficher les résultats
        # Display the results
        if results and self.console:
            from rich.table import Table

            table = Table(title="Experiment Results")
            table.add_column("Phase", style="cyan")
            table.add_column("Status", style="green")

            if phase == "all" and isinstance(results, dict):
                for phase_name, phase_results in results.items():
                    status = "✓ Finished" if phase_results else "✗ Failed"
                    table.add_row(phase_name, status)
            else:
                status = "✓ Finished" if results else "✗ Failed"
                table.add_row(phase, status)

            self.console.print(table)

        return True
