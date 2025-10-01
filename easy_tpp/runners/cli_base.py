"""
Base CLI Runner

Classe de base pour tous les runners CLI avec fonctionnalités communes.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    
try:
    import typer
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False

# Configuration globale pour les chemins de config
CONFIG_DIR = "yaml_configs"
CONFIG_PATHS = {
    "data": "data_configs",
    "model": "model_configs", 
    "runner": "runner_configs",
    "simulation": "simulator_configs",
    "hpo": "hpo_configs"
}

DEFAULT_CONFIGS = {
    "data": "default_data_config.yaml",
    "model": "default_model_config.yaml", 
    "runner": "default_runner_config.yaml",
    "simulation": "default_simulator_config.yaml",
    "hpo": "default_hpo_config.yaml"
}

class CLIRunnerBase:
    """
    Classe de base pour tous les runners CLI.
    Fournit logging, console Rich, et vérification des dépendances.
    """
    
    def __init__(self, name: str = "CLIRunner"):
        self.name = name
        self.console = Console() if RICH_AVAILABLE else None
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Configuration du logging avec Rich si disponible."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        # Éviter les doublons de handlers
        if logger.handlers:
            return logger
            
        if RICH_AVAILABLE:
            handler = RichHandler(console=self.console, rich_tracebacks=True)
            formatter = logging.Formatter(
                "%(message)s",
                datefmt="[%X]"
            )
        else:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
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
            
    def get_config_path(self, config_type: str, config_name: str = None) -> Path:
        """Retourne le chemin vers un fichier de configuration."""
        base_dir = Path(CONFIG_DIR)
        
        if config_type in CONFIG_PATHS:
            config_dir = base_dir / CONFIG_PATHS[config_type]
        else:
            config_dir = base_dir
            
        if config_name is None:
            config_name = DEFAULT_CONFIGS.get(config_type, "config.yaml")
            
        return config_dir / config_name