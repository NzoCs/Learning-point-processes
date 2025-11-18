"""
Interactive Setup Runner

Runner pour la configuration guidée d'expériences TPP.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import typer
import yaml
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from .cli_base import CLIRunnerBase


class InteractiveSetup(CLIRunnerBase):
    """
    Runner pour la configuration interactive d'expériences.
    Guide l'utilisateur à travers la création de configurations.
    """

    def __init__(self, debug: bool = False):
        super().__init__("InteractiveSetup", debug=debug)

    def run_interactive_setup(
        self,
        setup_type: str = "experiment",
        output_path: Optional[Union[str, Path]] = None,
        quick_mode: bool = False,
    ) -> bool:
        """
        Lance la configuration interactive.

        Args:
            setup_type: Type de setup (experiment, data, model)
            output_path: Chemin de sauvegarde de la configuration
            quick_mode: Mode rapide avec valeurs par défaut

        Returns:
            True si la configuration a été créée avec succès
        """

        self.print_info(f"Configuration interactive - Type: {setup_type}")

        if setup_type == "experiment":
            config = self._setup_experiment_config(quick_mode)
        elif setup_type == "data":
            config = self._setup_data_config(quick_mode)
        elif setup_type == "model":
            config = self._setup_model_config(quick_mode)
        else:
            self.print_error(f"Type de setup non supporté: {setup_type}")
            return False

        # Afficher la configuration finale
        self._display_final_config(config)

        # Confirmer et sauvegarder
        if Confirm.ask("Sauvegarder cette configuration?"):
            if output_path is None:
                output_path_str = Prompt.ask(
                    "Chemin de sauvegarde", default=f"{setup_type}_config.yaml"
                )
                output_path_obj = Path(output_path_str)
            else:
                output_path_obj = Path(output_path)

            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path_obj, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            self.print_success(f"Configuration sauvegardée: {output_path_obj}")

            # Proposer de lancer directement l'expérience
            if setup_type == "experiment" and Confirm.ask(
                "Lancer l'expérience maintenant?"
            ):
                return self._launch_experiment(output_path_obj)

        return True

    def _setup_experiment_config(self, quick_mode: bool) -> Dict[str, Any]:
        """Configuration interactive d'une expérience complète."""
        config = {}

        if self.console:
            self.console.print("[bold blue]Configuration d'Expérience TPP[/bold blue]")
            self.console.print()

        # Configuration des données
        if not quick_mode or Confirm.ask("Configurer les données?", default=True):
            config["data_config"] = self._setup_data_section(quick_mode)

        # Configuration du modèle
        if not quick_mode or Confirm.ask("Configurer le modèle?", default=True):
            config["model_config"] = self._setup_model_section(quick_mode)

        # Configuration du runner
        if not quick_mode or Confirm.ask("Configurer l'entraînement?", default=True):
            config["runner_config"] = self._setup_runner_section(quick_mode)

        return config

    def _setup_data_config(self, quick_mode: bool) -> Dict[str, Any]:
        """Configuration interactive des données."""
        if self.console:
            self.console.print("[bold green]Configuration des Données[/bold green]")

        data_config = {}

        # Répertoire des données
        data_dir = Prompt.ask("Répertoire des données", default="./data/")
        data_config["data_dir"] = data_dir

        # Format des données
        if quick_mode:
            data_format = "json"
        else:
            data_format = Prompt.ask(
                "Format des données", choices=["json", "csv", "pickle"], default="json"
            )
        data_config["data_format"] = data_format

        # Spécifications des données
        tokenizer_specs = {}

        num_event_types = IntPrompt.ask("Nombre de types d'événements", default=5)
        tokenizer_specs["num_event_types"] = num_event_types

        max_seq_len = IntPrompt.ask("Longueur maximale des séquences", default=100)
        tokenizer_specs["max_seq_len"] = max_seq_len

        data_config["tokenizer_specs"] = tokenizer_specs

        # Configuration du chargement
        if not quick_mode:
            loading_specs = {}

            batch_size = IntPrompt.ask("Taille des batches", default=64)
            loading_specs["batch_size"] = batch_size

            shuffle = Confirm.ask("Mélanger les données?", default=True)
            loading_specs["shuffle"] = shuffle

            data_config["data_loading_specs"] = loading_specs

        return data_config

    def _setup_model_config(self, quick_mode: bool) -> Dict[str, Any]:
        """Configuration interactive du modèle."""
        if self.console:
            self.console.print("[bold green]Configuration du Modèle[/bold green]")

        model_config = {}

        # Type de modèle
        if quick_mode:
            model_type = "NHP"
        else:
            model_type = Prompt.ask(
                "Type de modèle",
                choices=["NHP", "THP", "RMTPP", "FullyNN", "LogNormMix"],
                default="NHP",
            )
        model_config["model_type"] = model_type

        # Spécifications du modèle
        model_specs = {}

        hidden_size = IntPrompt.ask("Taille des couches cachées", default=128)
        model_specs["hidden_size"] = hidden_size

        if not quick_mode:
            num_layers = IntPrompt.ask("Nombre de couches", default=2)
            model_specs["num_layers"] = num_layers

            dropout = FloatPrompt.ask("Taux de dropout", default=0.1)
            model_specs["dropout"] = dropout

        model_config["model_specs"] = model_specs

        return model_config

    def _setup_data_section(self, quick_mode: bool) -> Dict[str, Any]:
        """Section données pour la configuration d'expérience."""
        return self._setup_data_config(quick_mode)

    def _setup_model_section(self, quick_mode: bool) -> Dict[str, Any]:
        """Section modèle pour la configuration d'expérience."""
        return self._setup_model_config(quick_mode)

    def _setup_runner_section(self, quick_mode: bool) -> Dict[str, Any]:
        """Section runner pour la configuration d'expérience."""
        if self.console:
            self.console.print(
                "[bold green]Configuration de l'Entraînement[/bold green]"
            )

        runner_config = {}

        # Identifiants
        dataset_id = Prompt.ask("ID du dataset", default="experiment")
        runner_config["dataset_id"] = dataset_id

        model_id = Prompt.ask("ID du modèle", default="baseline")
        runner_config["model_id"] = model_id

        # Paramètres d'entraînement
        max_epochs = IntPrompt.ask("Nombre maximum d'époques", default=100)
        runner_config["max_epochs"] = max_epochs

        if not quick_mode:
            patience = IntPrompt.ask("Patience pour early stopping", default=20)
            runner_config["patience"] = patience

            val_freq = IntPrompt.ask("Fréquence de validation", default=5)
            runner_config["val_freq"] = val_freq

            save_dir = Prompt.ask("Répertoire de sauvegarde", default="./experiments/")
            runner_config["save_dir"] = save_dir

        return runner_config

    def _display_final_config(self, config: Dict[str, Any]):
        """Affiche la configuration finale."""
        if not self.console:
            print("\\n=== Configuration Finale ===")
            for key, value in config.items():
                print(f"{key}: {value}")
            return

        from rich.panel import Panel
        from rich.tree import Tree

        tree = Tree("Configuration Générée")

        for section_name, section_config in config.items():
            section_node = tree.add(f"[bold cyan]{section_name}[/bold cyan]")

            if isinstance(section_config, dict):
                for key, value in section_config.items():
                    if isinstance(value, dict):
                        sub_node = section_node.add(f"[yellow]{key}[/yellow]")
                        for sub_key, sub_value in value.items():
                            sub_node.add(f"{sub_key}: [green]{sub_value}[/green]")
                    else:
                        section_node.add(f"{key}: [green]{value}[/green]")
            else:
                tree.add(f"{section_name}: [green]{section_config}[/green]")

        self.console.print(
            Panel(tree, title="Configuration Finale", border_style="blue")
        )

    def _launch_experiment(self, config_path: Path) -> bool:
        """Lance directement une expérience avec la configuration créée."""
        try:
            # Importer et utiliser l'ExperimentRunner
            from .experiment_runner import ExperimentRunner

            runner = ExperimentRunner()
            success = runner.run_experiment(config_path=str(config_path), phase="all")

            if success:
                self.print_success("Expérience lancée avec succès!")
            else:
                self.print_error("Échec du lancement de l'expérience")

            return success

        except Exception as e:
            self.print_error(f"Erreur lors du lancement: {e}")
            return False
