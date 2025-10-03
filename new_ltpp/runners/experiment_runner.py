"""
Experiment Runner

Runner pour l'exécution d'expériences TPP avec configuration builder.
Inspiré de run_all_phase.py pour charger toute la configuration depuis YAML.
"""

from typing import Optional, List
from pathlib import Path

from .cli_base import CLIRunnerBase, CONFIG_MAP

try:
    from new_ltpp.configs import ConfigFactory, ConfigType
    from new_ltpp.configs.config_builder import RunnerConfigBuilder
    from new_ltpp.runners import RunnerManager
except ImportError as e:
    ConfigFactory = None
    ConfigType = None
    RunnerConfigBuilder = None
    RunnerManager = None
    IMPORT_ERROR = str(e)

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
                    config_paths[f"{config_type}_config_path"] = f"{prefix}.{config_name}"
                else:
                    self.print_error(f"Type de configuration non reconnu: {config_type}")
        
        return config_paths
        
    def run_experiment(
        self,
        config_path: Optional[str] = None,
        data_config: str = "test",
        model_config: str = "neural_small", 
        training_config: str = "quick_test",
        data_loading_config: str = "quick_test",
        simulation_config: Optional[str] = "simulation_fast",
        thinning_config: Optional[str] = "thinning_fast",
        logger_config: Optional[str] = "tensorboard",
        model_id: str = "NHP",
        phase: str = "all",
        max_epochs: Optional[int] = None,
        save_dir: Optional[str] = None,
        gpu_id: Optional[int] = None,
        debug: bool = False
    ) -> bool:
        """
        Lance une expérience TPP avec les paramètres spécifiés.
        
        Args:
            config_path: Chemin vers le fichier de configuration YAML
            data_config: Configuration des données (ex: test, large, synthetic)
            model_config: Configuration du modèle (ex: neural_small, neural_large)
            training_config: Configuration d'entraînement (ex: quick_test, full_training)
            data_loading_config: Configuration du chargement de données
            simulation_config: Configuration de simulation (optionnel)
            thinning_config: Configuration de thinning (optionnel)
            logger_config: Configuration du logger (ex: mlflow, tensorboard)
            model_id: Identifiant du modèle (default: NHP)
            phase: Phase d'exécution (train, test, predict, all)
            max_epochs: Nombre maximum d'époques (override config)
            save_dir: Répertoire de sauvegarde (override config) 
            gpu_id: ID du GPU à utiliser (override config)
            debug: Mode debug
            
        Returns:
            True si l'expérience s'est déroulée avec succès
        """
        # Activer le mode debug si demandé
        self.set_debug(debug)
        
        # Vérifier les dépendances
        required_modules = ["new_ltpp.configs", "new_ltpp.runners"]
        if not self.check_dependencies(required_modules):
            return False
            
        try:
            self.print_info(f"Démarrage de l'expérience TPP - Phase: {phase}")
            
            # Configuration par défaut si aucun fichier spécifié
            if config_path is None:
                config_path = self.get_config_path()
                self.print_info(f"Utilisation de la configuration par défaut: {config_path}")
            
            # Validation du fichier de configuration
            if not Path(config_path).exists():
                self.print_error(f"Fichier de configuration non trouvé: {config_path}")
                return False
            
            # Build runner configuration from YAML (comme dans run_all_phase.py)
            config_builder = RunnerConfigBuilder()
            
            # Construire les chemins de configuration avec la fonction utilitaire
            config_paths = self._build_config_paths(
                data=data_config,
                model=model_config,
                training=training_config,
                data_loading=data_loading_config,
                simulation=simulation_config,
                thinning=thinning_config,
                logger=logger_config
            )
            
            self.print_info(f"Configurations utilisées:")
            for path_key, path_value in config_paths.items():
                config_type = path_key.replace('_config_path', '').replace('_', ' ').title()
                self.print_info(f"  • {config_type}: {path_value}")
            
            try:
                # Charger la configuration complète depuis le YAML (comme run_all_phase.py)
                config_builder.load_from_yaml(
                    yaml_file_path=config_path,
                    **config_paths  # Unpacking des chemins construits dynamiquement
                )
                
                self.print_info("Configuration YAML chargée avec succès")
                
            except Exception as e:
                self.print_error_with_traceback(f"Erreur lors du chargement de la configuration: {e}", e)
                return False
            
            # Récupérer le dictionnaire de configuration
            config_dict = config_builder.config_dict
            
            # Appliquer les overrides de paramètres CLI en utilisant les méthodes du builder
            if max_epochs:
                config_builder.override_max_epochs(max_epochs)
                self.print_info(f"Override: max_epochs = {max_epochs}")
                
            # Ne passer save_dir que s'il est explicitement fourni par l'utilisateur
            if save_dir:
                config_builder.set_field("save_dir", save_dir)
                self.print_info(f"Override: save_dir = {save_dir}")
            # Sinon, laisser les sous-couches générer leur propre save_dir par défaut
            # qui sera plus intelligent (model_id/dataset_id/etc.)
                
            if gpu_id is not None:
                config_builder.override_devices(gpu_id)
                self.print_info(f"Override: devices = {gpu_id}")
            
            # Créer la configuration finale avec la factory
            config = config_builder.build(model_id=model_id)
            
            # Validation de la phase
            valid_phases = ["train", "test", "predict", "all"]
            if phase not in valid_phases:
                self.print_error(f"Phase invalide: {phase}. Phases valides: {valid_phases}")
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
                    "predict": predict_results
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
            
        except Exception as e:
            self.print_error_with_traceback(f"Erreur lors de l'exécution: {e}", e)
            if self.debug:
                self.logger.exception("Détails de l'erreur:")
            return False