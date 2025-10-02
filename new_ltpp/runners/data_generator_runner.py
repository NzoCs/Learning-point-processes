"""
Data Generator Runner

Runner pour la génération de données synthétiques TPP.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

from .cli_base import CLIRunnerBase

try:
    from new_ltpp.data.generation import HawkesSimulator, SelfCorrecting
except ImportError as e:
    HawkesSimulator = None
    SelfCorrecting = None
    IMPORT_ERROR = str(e)

class DataGenerator(CLIRunnerBase):
    """
    Runner pour la génération de données synthétiques.
    Utilise SynGenConfigBuilder pour la configuration.
    """
    
    def __init__(self):
        super().__init__("DataGenerator")
        
    def generate_data(
        self,
        output_dir: Optional[str] = None,
        num_simulations: int = 50,
        generation_method: str = "hawkes",
        splits: Optional[Dict[str, float]] = None,
        start_time: float = 0,
        end_time: float = 100,
        dim_process: int = 2,
        mu: Optional[List[float]] = None,
        alpha: Optional[List[List[float]]] = None,
        beta: Optional[List[List[float]]] = None,
        **kwargs
    ) -> bool:
        """
        Génère des données synthétiques TPP.
        
        Args:
            output_dir: Répertoire de sortie
            num_simulations: Nombre de simulations à générer
            generation_method: Méthode de génération (hawkes, self_correcting)
            splits: Répartition des données (train/test/dev)
            start_time: Temps de début
            end_time: Temps de fin
            dim_process: Dimension du processus
            mu: Paramètres mu pour Hawkes
            alpha: Paramètres alpha pour Hawkes
            beta: Paramètres beta pour Hawkes
            **kwargs: Paramètres additionnels
            
        Returns:
            True si la génération s'est déroulée avec succès
        """
        # Vérifier les dépendances
        required_modules = ["easy_tpp.data.generation"]
        if not self.check_dependencies(required_modules):
            return False
            
        try:
            self.print_info(f"Génération de {num_simulations} simulations - Méthode: {generation_method}")
            
            # Créer le répertoire de sortie par défaut si nécessaire
            if output_dir is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = str(self.get_output_path("data_generation", f"generated_{timestamp}"))
                self.print_info(f"Répertoire de sortie: {output_dir}")
            
            # Valeurs par défaut pour les splits
            if splits is None:
                splits = {"train": 0.6, "test": 0.2, "dev": 0.2}
            
            # Génération selon la méthode
            if generation_method.lower() == "hawkes":
                # Paramètres par défaut pour Hawkes
                if mu is None:
                    mu = [0.2] * dim_process
                if alpha is None:
                    if dim_process == 2:
                        alpha = [[0.4, 0], [0, 0.8]]
                    else:
                        alpha = [[0.3 if i == j else 0.1 for j in range(dim_process)] for i in range(dim_process)]
                if beta is None:
                    if dim_process == 2:
                        beta = [[1, 0], [0, 20]]
                    else:
                        beta = [[2.0 if i == j else 1.0 for j in range(dim_process)] for i in range(dim_process)]
                
                generator = HawkesSimulator(
                    mu=mu,
                    alpha=alpha,
                    beta=beta,
                    dim_process=dim_process,
                    start_time=start_time,
                    end_time=end_time,
                )
                
            elif generation_method.lower() == "self_correcting":
                generator = SelfCorrecting(
                    dim_process=dim_process,
                    start_time=start_time,
                    end_time=end_time,
                )
                
            else:
                self.print_error(f"Méthode de génération non supportée: {generation_method}")
                self.print_info("Méthodes disponibles: hawkes, self_correcting")
                return False
            
            # Générer et sauvegarder les données
            self.print_info("Génération en cours...")
            
            if self.console:
                with self.console.status("[bold green]Génération en cours...") as status:
                    generator.generate_and_save(
                        output_dir=output_dir,
                        num_simulations=num_simulations,
                        splits=splits,
                    )
                    status.update("[bold green]Sauvegarde terminée")
            else:
                generator.generate_and_save(
                    output_dir=output_dir,
                    num_simulations=num_simulations,
                    splits=splits,
                )
            
            # Statistiques des données générées
            stats = {
                "generation_method": generation_method,
                "num_simulations": num_simulations,
                "dim_process": dim_process,
                "time_range": f"{start_time} - {end_time}",
                "splits": splits,
                "output_directory": output_dir
            }
            
            # Afficher les statistiques
            if self.console:
                from rich.table import Table
                table = Table(title="Données synthétiques générées")
                table.add_column("Statistique", style="cyan")
                table.add_column("Valeur", style="magenta")
                
                for key, value in stats.items():
                    if isinstance(value, (dict, list)):
                        table.add_row(key, str(value))
                    elif isinstance(value, float):
                        table.add_row(key, f"{value:.2f}")
                    else:
                        table.add_row(key, str(value))
                        
                self.console.print(table)
            
            # Sauvegarder les métadonnées
            metadata = {
                "generation_config": {
                    "generation_method": generation_method,
                    "num_simulations": num_simulations,
                    "dim_process": dim_process,
                    "start_time": start_time,
                    "end_time": end_time,
                    "mu": mu,
                    "alpha": alpha,
                    "beta": beta
                },
                "statistics": stats
            }
            
            import json
            metadata_path = Path(output_dir) / "generation_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.print_success(f"Données générées avec succès dans: {output_dir}")
            self.print_success(f"Métadonnées: {metadata_path}")
            
            return True
            
        except Exception as e:
            self.print_error(f"Erreur lors de la génération: {e}")
            self.logger.exception("Détails de l'erreur:")
            return False