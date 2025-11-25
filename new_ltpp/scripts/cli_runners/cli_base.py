"""
Base CLI Runner

Classe de base pour tous les runners CLI avec fonctionnalités communes.
"""

import logging
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich.table import Table

from new_ltpp.globals import CONFIGS_FILE, OUTPUT_DIR, ROOT_DIR

# Configuration globale pour les répertoires de sortie (dans artifacts/)
OUTPUT_DIRS = {
    "experiments": "experiments",
    "models": "models",
    "checkpoints": "checkpoints",
    "results": "results",
    "benchmarks": "benchmarks",
    "data_generation": "generated_data",
    "data_inspection": "data_inspection",
    "logs": "logs",
}

CONFIG_MAP = {
    "data": "data_configs",
    "general_specs": "general_specs",
    "model_specs": "model_specs",
    "runner": "runner_configs",
    "simulation": "simulation_configs",
    "hpo": "hpo_configs",
    "training": "training_configs",
    "data_loading": "data_loading_configs",
    "thinning": "thinning_configs",
    "logger": "logger_configs",
}


class CLIRunnerBase:
    """
    Classe de base pour tous les runners CLI.
    Fournit logging, console Rich, et vérification des dépendances.
    """

    def __init__(self, name: str = "CLIRunner", debug: bool = False):
        self.name = name
        self.debug = debug
        self.console = Console()
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configuration du logging avec Rich si disponible."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # Éviter les doublons de handlers
        if logger.handlers:
            return logger

        handler = RichHandler(console=self.console, rich_tracebacks=True)
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

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

    def check_dependencies(self, required_modules: List[str]) -> bool:
        """Vérifie que les modules requis sont disponibles."""
        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)

        if missing:
            error_msg = f"Modules manquants: {', '.join(missing)}"
            if self.console:
                self.console.print(f"[bold red]Erreur:[/bold red] {error_msg}")
            else:
                print(f"Erreur: {error_msg}")
            return False
        return True

    def print_success(self, message: str):
        """Affiche un message de succès."""
        if self.console:
            self.console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"✓ {message}")

    def print_error(self, message: str):
        """Affiche un message d'erreur."""
        if self.console:
            self.console.print(f"[bold red]✗[/bold red] {message}")
        else:
            print(f"✗ {message}")

    def print_info(self, message: str):
        """Affiche un message d'information."""
        if self.console:
            self.console.print(f"[bold blue]ℹ[/bold blue] {message}")
        else:
            print(f"ℹ {message}")

    def print_error_with_traceback(
        self, message: str, exception: Optional[Exception] = None
    ):
        """Affiche un message d'erreur avec traceback complet si debug activé."""
        self.print_error(message)

        if self.debug and exception:
            import traceback

            if self.console:
                self.console.print("[yellow]📋 Traceback complet:[/yellow]")
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
            else:
                print("📋 Traceback complet:")
                print(traceback.format_exc())

    def set_debug(self, debug: bool):
        """Active ou désactive le mode debug."""
        self.debug = debug

    def get_config_path(self, config_name: Optional[str] = None) -> Path:
        """Retourne le chemin vers un fichier de configuration."""
        base_dir = ROOT_DIR / "yaml_configs"

        if config_name:
            return base_dir / f"{config_name}.yaml"

        return CONFIGS_FILE

    def get_output_path(self) -> Path:
        """Retourne le chemin vers un répertoire de sortie dans artifacts/."""

        return OUTPUT_DIR

    def get_root_dir(self) -> Path:
        """Retourne le répertoire racine du projet."""
        return ROOT_DIR
