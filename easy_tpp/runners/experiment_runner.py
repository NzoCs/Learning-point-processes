"""
Experiment Runner

Runner pour l'exécution d'expériences TPP avec configuration builder.
"""

from typing import Optional, List
from pathlib import Path

from .cli_base import CLIRunnerBase

try:
    from easy_tpp.configs import ConfigFactory, ConfigType
    from easy_tpp.configs import RunnerConfigBuilder
    from easy_tpp.runners import RunnerManager
except ImportError as e:
    ConfigFactory = None
    ConfigType = None
    RunnerConfigBuilder = None
    RunnerManager = None
    IMPORT_ERROR = str(e)

class ExperimentRunner(CLIRunnerBase):
    """
    Runner pour l'exécution d'expériences TPP.
    Utilise RunnerConfigBuilder pour la configuration.
    """
    
    def __init__(self):
        super().__init__("ExperimentRunner")
        
    def run_experiment(
        self,
        config_path: Optional[str] = None,
        dataset_id: Optional[str] = None,
        model_id: Optional[str] = None,
        phase: str = "all",
        max_epochs: Optional[int] = None,
        save_dir: Optional[str] = None,
        gpu_id: Optional[int] = None,
        debug: bool = False
    ) -> bool:
        """
        Lance une expérience TPP avec les paramètres spécifiés.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            dataset_id: Identifiant du dataset
            model_id: Identifiant du modèle
            phase: Phase d'exécution (train, test, predict, all)
            max_epochs: Nombre maximum d'époques
            save_dir: Répertoire de sauvegarde
            gpu_id: ID du GPU à utiliser
            debug: Mode debug
            
        Returns:
            True si l'expérience s'est déroulée avec succès
        """
        # Vérifier les dépendances
        required_modules = ["easy_tpp.config_factory", "easy_tpp.runners"]
        if not self.check_dependencies(required_modules):
            return False
            
        try:
            self.print_info(f"Démarrage de l'expérience TPP - Phase: {phase}")
            
            # Configuration via builder
            if config_path:
                config = ConfigFactory.from_yaml(config_path, ConfigType.RUNNER)
            else:
                # Utiliser le builder pour créer la configuration
                builder = RunnerConfigBuilder()
                
                if dataset_id:
                    builder.dataset_id(dataset_id)
                if model_id:
                    builder.model_id(model_id)
                if max_epochs:
                    builder.max_epochs(max_epochs)
                if save_dir:
                    builder.save_dir(save_dir)
                if gpu_id is not None:
                    builder.gpu_id(gpu_id)
                    
                config = builder.build()
            
            # Validation de la phase
            valid_phases = ["train", "test", "predict", "all"]
            if phase not in valid_phases:
                self.print_error(f"Phase invalide: {phase}. Phases valides: {valid_phases}")
                return False
            
            # Lancer l'expérience
            runner_manager = RunnerManager(config)
            
            if phase == "all":
                self.print_info("Exécution complète: train → validate → test")
                results = runner_manager.run()
            else:
                self.print_info(f"Exécution phase: {phase}")
                results = runner_manager.run(phase=phase)
            
            self.print_success(f"Expérience terminée avec succès - Phase: {phase}")
            
            if results and self.console:
                from rich.table import Table
                table = Table(title="Résultats de l'expérience")
                table.add_column("Métrique", style="cyan")
                table.add_column("Valeur", style="magenta")
                
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        table.add_row(key, f"{value:.4f}")
                    else:
                        table.add_row(key, str(value))
                        
                self.console.print(table)
            
            return True
            
        except Exception as e:
            self.print_error(f"Erreur lors de l'exécution: {e}")
            if debug:
                self.logger.exception("Détails de l'erreur:")
            return False