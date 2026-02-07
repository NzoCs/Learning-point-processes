"""
Base CLI Runner

Base class for all CLI runners providing common functionality.
"""

import logging
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.logging import RichHandler

from new_ltpp.globals import CONFIGS_FILE, OUTPUT_DIR, ROOT_DIR

# Global configuration for output directories (in artifacts/)
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
    "training": "training_configs",
    "data_loading": "data_loading_configs",
    "thinning": "thinning_configs",
    "logger": "logger_configs",
}


class CLIRunnerBase:
    """
    Base class for all CLI runners.
    Provides logging setup, a Rich console, and dependency checks.
    """

    def __init__(self, name: str = "CLIRunner", debug: bool = False):
        self.name = name
        self.debug = debug
        self.console = Console()
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure logging with Rich if available."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        handler = RichHandler(console=self.console, rich_tracebacks=True)
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _build_config_paths(self, **config_kwargs) -> dict[str, str]:
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
                        f"Unrecognized configuration type: {config_type}"
                    )

        return config_paths

    def check_dependencies(self, required_modules: List[str]) -> bool:
        """Check that required modules are available."""
        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)

        if missing:
            error_msg = f"Missing modules: {', '.join(missing)}"
            if self.console:
                self.console.print(f"[bold red]Error:[/bold red] {error_msg}")
            else:
                print(f"Error: {error_msg}")
            return False
        return True

    def print_success(self, message: str):
        """Print a success message."""
        if self.console:
            self.console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"✓ {message}")

    def print_error(self, message: str):
        """Print an error message."""
        if self.console:
            self.console.print(f"[bold red]✗[/bold red] {message}")
        else:
            print(f"✗ {message}")

    def print_info(self, message: str):
        """Print an informational message."""
        if self.console:
            self.console.print(f"[bold blue]ℹ[/bold blue] {message}")
        else:
            print(f"ℹ {message}")

    def print_error_with_traceback(
        self, message: str, exception: Optional[Exception] = None
    ):
        """Print an error message with full traceback if debug is enabled."""
        self.print_error(message)

        if self.debug and exception:
            import traceback

            if self.console:
                self.console.print("[yellow]📋 Full traceback:[/yellow]")
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
            else:
                print("📋 Full traceback:")
                print(traceback.format_exc())

    def set_debug(self, debug: bool):
        """Enable or disable debug mode."""
        self.debug = debug

    def get_config_path(self, config_name: Optional[str] = None) -> Path:
        """Return the path to a configuration file."""
        base_dir = ROOT_DIR / "yaml_configs"

        if config_name:
            return base_dir / f"{config_name}.yaml"

        return CONFIGS_FILE

    def get_output_path(self) -> Path:
        """Return the path to an output directory under artifacts/."""

        return OUTPUT_DIR

    def get_root_dir(self) -> Path:
        """Return the project's root directory."""
        return ROOT_DIR
