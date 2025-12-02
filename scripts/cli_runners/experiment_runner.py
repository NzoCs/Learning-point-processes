"""
Experiment Runner

Runner pour l'exécution d'expériences TPP avec configuration builder.
Inspiré de run_all_phase.py pour charger toute la configuration depuis YAML.
"""

from pathlib import Path
from typing import Optional, Union

from new_ltpp.configs.config_builders import RunnerConfigBuilder
from new_ltpp.runners.runner_manager import RunnerManager

from .cli_base import CONFIG_MAP, CLIRunnerBase


class ExperimentRunner(CLIRunnerBase):
    """
    Runner pour l'exécution d'expériences TPP.
    Utilise RunnerConfigBuilder pour charger toute la configuration depuis YAML.
    Permet de spécifier individuellement chaque type de configuration.
    """

    def __init__(self, debug: bool = False):
        super().__init__("ExperimentRunner", debug=debug)

    def _build_config_paths(self, **config_kwargs) -> dict:
        """
        Construit les chemins de configuration en suivant le pattern standard.

        Pattern: {config_type}_configs.{config_name}

        Args:
            **config_kwargs: Dictionnaire des configurations {type: name}

        Returns:
            Dictionnaire des chemins de configuration formatés
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
        max_epochs: int | None,
        config_path: Union[str, Path],
        data_config: str,
        general_specs_config: str,
        training_config: str,
        data_loading_config: str,
        simulation_config: str,
        thinning_config: str,
        logger_config: str,
        model_id: str,
        phase: str,
        save_dir: Union[str, Path],
        model_specs_config: Optional[str] = None,
        debug: bool = False,
    ) -> bool:
        """
        Lance une expérience TPP avec les paramètres spécifiés.

        Args:
            config_path: Chemin vers le fichier de configuration YAML
            data_config: Configuration des données (ex: test, large, synthetic)
            general_specs_config: Configuration générale du modèle (ex: h16, h32, h64)
            training_config: Configuration d'entraînement (ex: quick_test, full_training)
            data_loading_config: Configuration du chargement de données
            simulation_config: Configuration de simulation (optionnel)
            thinning_config: Configuration de thinning (optionnel)
            logger_config: Configuration du logger (ex: mlflow, tensorboard)
            model_id: Identifiant du modèle (default: NHP)
            phase: Phase d'exécution (train, test, predict, all)
            max_epochs: Nombre maximum d'époques (override config)
            save_dir: Répertoire de sauvegarde (override config)
            model_specs_config: Configuration spécifique au modèle (optionnel)
            debug: Mode debug

        Returns:
            True si l'expérience s'est déroulée avec succès
        """
        # Activer le mode debug si demandé
        self.set_debug(debug)

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(config_path, str):
            config_path = Path(config_path)

        # Vérifier les dépendances
        required_modules = ["new_ltpp.configs", "new_ltpp.runners"]
        if not self.check_dependencies(required_modules):
            return False

        self.print_info(f"Démarrage de l'expérience TPP - Phase: {phase}")

        # Configuration par défaut si aucun fichier spécifié
        if config_path is None:
            config_path = str(self.get_config_path())
            self.print_info(
                f"Utilisation de la configuration par défaut: {config_path}"
            )

        # Validation du fichier de configuration
        if not Path(config_path).exists():
            self.print_error(f"Fichier de configuration non trouvé: {config_path}")
            return False

        # Build runner configuration from YAML (comme dans run_all_phase.py)
        config_builder = RunnerConfigBuilder()

        # Construire les chemins de configuration avec la fonction utilitaire
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

        self.print_info(f"Configurations utilisées:")
        for path_key, path_value in config_paths.items():
            config_type = (
                path_key.replace("_config_path", "").replace("_", " ").title()
            )
            self.print_info(f"  • {config_type}: {path_value}")

        # Charger la configuration complète depuis le YAML (comme run_all_phase.py)
        config_builder.load_from_yaml(
            yaml_file_path=config_path,
            **config_paths,  # Unpacking des chemins construits dynamiquement
        )

        self.print_info("Configuration YAML chargée avec succès")

        # Récupérer le dictionnaire de configuration
        config_dict = config_builder.config_dict

        # Appliquer les overrides de paramètres CLI en utilisant les méthodes du builder
        if max_epochs is not None:
            config_builder.set_max_epochs(max_epochs)
            self.print_info(f"Override: max_epochs = {max_epochs}")

        # Ne passer save_dir que s'il est explicitement fourni par l'utilisateur
        if save_dir:
            config_builder.set_save_dir(save_dir)
            self.print_info(f"Override: save_dir = {save_dir}")
        # Sinon, laisser les sous-couches générer leur propre save_dir par défaut
        # qui sera plus intelligent (model_id/dataset_id/etc.)

        # Créer la configuration finale avec la factory
        config = config_builder.build(model_id=model_id)

        # Validation de la phase
        valid_phases = ["train", "test", "predict", "all"]
        if phase not in valid_phases:
            self.print_error(
                f"Phase invalide: {phase}. Phases valides: {valid_phases}"
            )
            return False

        # Créer et lancer le runner
        runner_manager = RunnerManager(config=config)

        if phase == "all":
            self.print_info("Exécution complète: train → test → predict")

            # Exécuter chaque phase séparément comme dans run_all_phase.py
            self.print_info("Phase 1/3: Training")
            train_results = runner_manager.run(phase="train")

            self.print_info("Phase 2/3: Testing")
            test_results = runner_manager.run(phase="test")

            self.print_info("Phase 3/3: Prediction")
            predict_results = runner_manager.run(phase="predict")

            # Combiner les résultats
            results = {
                "train": train_results,
                "test": test_results,
                "predict": predict_results,
            }

        else:
            self.print_info(f"Exécution phase: {phase}")
            results = runner_manager.run(phase=phase)

        self.print_success(f"Expérience terminée avec succès - Phase: {phase}")

        # Afficher les résultats
        if results and self.console:
            from rich.table import Table

            table = Table(title="Résultats de l'expérience")
            table.add_column("Phase", style="cyan")
            table.add_column("Statut", style="green")

            if phase == "all" and isinstance(results, dict):
                for phase_name, phase_results in results.items():
                    status = "✓ Terminé" if phase_results else "✗ Échec"
                    table.add_row(phase_name, status)
            else:
                status = "✓ Terminé" if results else "✗ Échec"
                table.add_row(phase, status)

            self.console.print(table)

        return True
