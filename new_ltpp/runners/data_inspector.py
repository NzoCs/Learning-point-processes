"""
Data Inspector Runner

Runner pour l'inspection et visualisation de données TPP.
"""

from typing import Optional, List
from pathlib import Path

from .cli_base import CLIRunnerBase

try:
    from new_ltpp.configs import DataConfigBuilder
    from new_ltpp.data.preprocess import Visualizer
except ImportError as e:
    DataConfigBuilder = None
    Visualizer = None
    IMPORT_ERROR = str(e)

class DataInspector(CLIRunnerBase):
    """
    Runner pour l'inspection et visualisation de données.
    Utilise DataConfigBuilder et le nouveau Visualizer API.
    """
    
    def __init__(self):
        super().__init__("DataInspector")
        
    def inspect_data(
        self,
        data_dir: str,
        data_format: str = "json",
        output_dir: Optional[str] = None,
        save_graphs: bool = True,
        show_graphs: bool = False,
        max_sequences: Optional[int] = None,
        event_types: Optional[List[str]] = None
    ) -> bool:
        """
        Inspecte et visualise des données TPP.
        
        Args:
            data_dir: Répertoire contenant les données
            data_format: Format des données (json, csv, etc.)
            output_dir: Répertoire de sortie pour les graphiques
            save_graphs: Sauvegarder les graphiques
            show_graphs: Afficher les graphiques
            max_sequences: Nombre maximum de séquences à analyser
            event_types: Types d'événements à analyser
            
        Returns:
            True si l'inspection s'est déroulée avec succès
        """
        # Vérifier les dépendances
        required_modules = ["easy_tpp.configs", "easy_tpp.data.preprocess"]
        if not self.check_dependencies(required_modules):
            return False
            
        try:
            self.print_info(f"Inspection des données: {data_dir}")
            
            # Configuration des données via builder
            builder = DataConfigBuilder()
            builder.set_src_dir(data_dir)  # Utilise set_src_dir qui définit train/valid/test
            builder.set_dataset_id("test")
            builder.set_data_format(data_format)
            
            # Spécifications de chargement par défaut
            builder.set_data_loading_specs({
                "batch_size": 32,
                "num_workers": 1,
                "shuffle": False
            })
            
            # Spécifications de données par défaut (dépendront des données réelles)
            builder.set_tokenizer_specs({
                "num_event_types": 10,  # Sera mis à jour après lecture des données
                "padding_side": "left",
                "truncation_side": "left"
            })
            
            data_config = builder.build()
            
            # Créer le data module (comme dans l'exemple)
            from new_ltpp.data.preprocess import TPPDataModule
            datamodule = TPPDataModule(data_config)
            
            try:
                datamodule.setup(stage="test")
            except Exception as e:
                self.print_error(f"Erreur setup data module: {e}")
                # Fallback vers analyse directe
                return self._fallback_direct_analysis(
                    data_dir, output_dir, save_graphs, show_graphs, max_sequences
                )
            
            # Créer le visualizer (comme dans l'exemple)
            if output_dir:
                save_dir = output_dir
                Path(save_dir).mkdir(parents=True, exist_ok=True)
            else:
                # Utiliser le répertoire par défaut dans artifacts/
                data_name = Path(data_dir).name
                save_dir = str(self.get_output_path("data_inspection", data_name))
            
            # Utiliser train si disponible, sinon test
            split_to_use = "train"
            try:
                visualizer = Visualizer(
                    data_module=datamodule, 
                    split=split_to_use, 
                    save_dir=save_dir,
                    dataset_size=max_sequences if max_sequences else 10000
                )
                
                self.print_info("Génération des visualisations avec Visualizer...")
                
                # Générer toutes les distributions (comme dans l'exemple)
                visualizer.show_all_distributions(save_graph=save_graphs, show_graph=show_graphs)
                
                # Générer les visualisations individuelles aussi
                visualizer.delta_times_distribution(save_graph=save_graphs, show_graph=show_graphs)
                visualizer.event_type_distribution(save_graph=save_graphs, show_graph=show_graphs)
                visualizer.sequence_length_distribution(save_graph=save_graphs, show_graph=show_graphs)
                
                results = {
                    "show_all_distributions": True,
                    "delta_times_distribution": True,
                    "event_type_distribution": True,
                    "sequence_length_distribution": True
                }
                
                self.print_success("✓ Toutes les visualisations générées")
                
            except Exception as e:
                self.print_error(f"Erreur avec Visualizer: {e}")
                # Fallback vers analyse directe
                return self._fallback_direct_analysis(
                    data_dir, output_dir, save_graphs, show_graphs, max_sequences
                )
                    
            # Sauvegarde des métadonnées
            if save_graphs and output_dir:
                metadata = {
                    "data_dir": data_dir,
                    "data_format": data_format,
                    "max_sequences": max_sequences,
                    "event_types": event_types,
                    "visualizations_generated": list(results.keys()),
                    "timestamp": str(Path().absolute())
                }
                
                import json
                metadata_path = Path(output_dir) / "inspection_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                self.print_success(f"Métadonnées sauvegardées: {metadata_path}")
            
            # Résumé des résultats
            if self.console and results:
                from rich.table import Table
                table = Table(title="Résultats de l'inspection")
                table.add_column("Visualisation", style="cyan")
                table.add_column("Statut", style="green")
                
                for viz_name, result in results.items():
                    status = "✓ Généré" if result else "✗ Échec"
                    table.add_row(viz_name, status)
                    
                self.console.print(table)
            
            self.print_success("Inspection des données terminée")
            return True
            
        except Exception as e:
            self.print_error(f"Erreur lors de l'inspection: {e}")
            self.logger.exception("Détails de l'erreur:")
            return False
    
    def _analyze_data_directly(self, data, save_dir, save_graphs, show_graphs, max_sequences):
        """
        Analyse directe des données JSON sans passer par le data loader complet.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from collections import Counter
        
        results = {}
        
        try:
            # Les données sont une liste d'objets (séquences)
            if not isinstance(data, list):
                self.print_error("Format de données non reconnu - attendu une liste")
                return {}
            
            # Extraire les séquences
            time_seqs = []
            type_seqs = []
            
            for sequence in data:
                if isinstance(sequence, dict):
                    if "time_since_start" in sequence and "type_event" in sequence:
                        time_seqs.append(sequence["time_since_start"])
                        type_seqs.append(sequence["type_event"])
            
            if not time_seqs or not type_seqs:
                self.print_error("Aucune séquence valide trouvée")
                return {}
            
            # Limiter le nombre de séquences si demandé
            if max_sequences:
                time_seqs = time_seqs[:max_sequences]
                type_seqs = type_seqs[:max_sequences]
            
            # Calculer les statistiques
            seq_lengths = [len(seq) for seq in type_seqs]
            all_event_types = []
            all_time_deltas = []
            
            for type_seq, time_seq in zip(type_seqs, time_seqs):
                all_event_types.extend(type_seq)
                
                # Calculer les intervalles de temps
                if len(time_seq) > 1:
                    deltas = np.diff(time_seq)
                    all_time_deltas.extend(deltas)
            
            self.print_info(f"Séquences analysées: {len(seq_lengths)}")
            self.print_info(f"Événements totaux: {len(all_event_types)}")
            self.print_info(f"Types d'événements: {len(set(all_event_types))}")
            
            # 1. Distribution des longueurs de séquences
            self.print_info("Génération: distribution des longueurs")
            plt.figure(figsize=(10, 6))
            plt.hist(seq_lengths, bins=30, alpha=0.7, edgecolor='black')
            plt.title("Distribution des longueurs de séquences")
            plt.xlabel("Longueur de séquence")
            plt.ylabel("Fréquence")
            
            if save_graphs:
                filepath = Path(save_dir) / "sequence_lengths.png"
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                self.print_success(f"Sauvegardé: {filepath}")
                
            if show_graphs:
                try:
                    plt.show()
                except:
                    pass
            plt.close()
            results["sequence_lengths"] = True
            
            # 2. Distribution des types d'événements
            self.print_info("Génération: distribution des types d'événements")
            event_counts = Counter(all_event_types)
            types = list(event_counts.keys())
            counts = list(event_counts.values())
            
            plt.figure(figsize=(10, 6))
            plt.bar(types, counts, alpha=0.7)
            plt.title("Distribution des types d'événements")
            plt.xlabel("Type d'événement")
            plt.ylabel("Fréquence")
            
            if save_graphs:
                filepath = Path(save_dir) / "event_types.png"
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                self.print_success(f"Sauvegardé: {filepath}")
                
            if show_graphs:
                try:
                    plt.show()
                except:
                    pass
            plt.close()
            results["event_types"] = True
            
            # 3. Distribution des intervalles de temps
            if all_time_deltas:
                self.print_info("Génération: distribution des intervalles de temps")
                plt.figure(figsize=(10, 6))
                
                # Utiliser log scale pour les intervalles de temps
                positive_deltas = [d for d in all_time_deltas if d > 0]
                
                if positive_deltas:
                    plt.hist(positive_deltas, bins=50, alpha=0.7, edgecolor='black')
                    plt.title("Distribution des intervalles de temps")
                    plt.xlabel("Intervalle de temps")
                    plt.ylabel("Fréquence")
                    plt.yscale('log')
                    
                    if save_graphs:
                        filepath = Path(save_dir) / "time_intervals.png"
                        plt.savefig(filepath, dpi=300, bbox_inches="tight")
                        self.print_success(f"Sauvegardé: {filepath}")
                        
                    if show_graphs:
                        try:
                            plt.show()
                        except:
                            pass
                
                plt.close()
                results["time_intervals"] = True
            
            # Générer un rapport de résumé
            self._generate_summary_report(
                save_dir, seq_lengths, all_event_types, all_time_deltas, event_counts
            )
            
            return results
            
        except Exception as e:
            self.print_error(f"Erreur dans l'analyse directe: {e}")
            return {}
    
    def _generate_summary_report(self, save_dir, seq_lengths, all_event_types, all_time_deltas, event_counts):
        """Génère un rapport de résumé."""
        import numpy as np
        
        try:
            summary = {
                "total_sequences": int(len(seq_lengths)),
                "total_events": int(len(all_event_types)),
                "unique_event_types": int(len(set(all_event_types))),
                "avg_sequence_length": float(np.mean(seq_lengths)),
                "median_sequence_length": float(np.median(seq_lengths)),
                "min_sequence_length": int(np.min(seq_lengths)),
                "max_sequence_length": int(np.max(seq_lengths)),
                "event_type_distribution": {str(k): int(v) for k, v in event_counts.items()}
            }
            
            if all_time_deltas:
                positive_deltas = [d for d in all_time_deltas if d > 0]
                if positive_deltas:
                    summary.update({
                        "avg_time_interval": float(np.mean(positive_deltas)),
                        "median_time_interval": float(np.median(positive_deltas)),
                        "min_time_interval": float(np.min(positive_deltas)),
                        "max_time_interval": float(np.max(positive_deltas))
                    })
            
            # Sauvegarder le rapport
            import json
            report_path = Path(save_dir) / "summary_report.json"
            with open(report_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            self.print_success(f"Rapport de résumé: {report_path}")
            
            # Afficher le résumé dans la console
            if self.console:
                from rich.table import Table
                table = Table(title="Résumé de l'analyse")
                table.add_column("Métrique", style="cyan")
                table.add_column("Valeur", style="green")
                
                table.add_row("Total séquences", str(summary["total_sequences"]))
                table.add_row("Total événements", str(summary["total_events"]))
                table.add_row("Types d'événements", str(summary["unique_event_types"]))
                table.add_row("Longueur moyenne", f"{summary['avg_sequence_length']:.2f}")
                table.add_row("Longueur médiane", f"{summary['median_sequence_length']:.2f}")
                
                self.console.print(table)
                
        except Exception as e:
            self.print_error(f"Erreur génération rapport: {e}")
    
    def _fallback_direct_analysis(self, data_dir, output_dir, save_graphs, show_graphs, max_sequences):
        """
        Fallback vers analyse directe si le Visualizer ne fonctionne pas.
        """
        self.print_info("Utilisation de l'analyse directe (fallback)...")
        
        import json
        from pathlib import Path
        
        data_path = Path(data_dir)
        
        # Chercher les fichiers de données
        json_files = {}
        for split in ["train", "test", "dev"]:
            json_file = data_path / f"{split}.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    json_files[split] = json.load(f)
                    
        if not json_files:
            self.print_error("Aucun fichier de données trouvé (train.json, test.json, dev.json)")
            return False
            
        # Utiliser le premier fichier disponible
        split_name = list(json_files.keys())[0]
        data = json_files[split_name]
        
        self.print_info(f"Données chargées depuis {split_name}.json")
        
        # Créer un répertoire de sortie
        if output_dir:
            save_dir = output_dir
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        else:
            # Utiliser le répertoire par défaut dans artifacts/
            data_name = Path(data_dir).name
            save_dir = str(self.get_output_path("data_inspection", data_name))
        
        # Analyser les données directement
        results = self._analyze_data_directly(
            data, save_dir, save_graphs, show_graphs, max_sequences
        )
        
        return len(results) > 0