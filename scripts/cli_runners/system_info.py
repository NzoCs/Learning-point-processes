"""
System Info Runner

Runner to display system and environment information.
"""

import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .cli_base import CLIRunnerBase


class SystemInfo(CLIRunnerBase):
    """
    Runner to display system and environment information.
    Diagnoses the new_ltpp installation and its dependencies.
    """

    def __init__(self, debug: bool = False):
        super().__init__("SystemInfo", debug=debug)

    def display_system_info(
        self,
        include_deps: bool = True,
        include_hardware: bool = True,
        output_file: Optional[str] = None,
    ) -> bool:
        """
        Display system and environment information.

        Args:
            include_deps: Include dependency information
            include_hardware: Include hardware information
            output_file: Output file path to save the information

        Returns:
            True if the display completed successfully
        """
        try:
            self.print_info("Collecting system information...")

            # Basic information
            system_info = self._collect_system_info()

            # Python information
            python_info = self._collect_python_info()

            # Dependency information
            deps_info = {}
            if include_deps:
                deps_info = self._collect_dependencies_info()

            # Hardware information
            hardware_info = {}
            if include_hardware:
                hardware_info = self._collect_hardware_info()

            # New_LTPP specific information
            new_ltpp_info = self._collect_new_ltpp_info()

            # Affichage
            all_info = {
                "system": system_info,
                "python": python_info,
                "dependencies": deps_info,
                "hardware": hardware_info,
                "new_ltp": new_ltpp_info,
            }

            self._display_info_tables(all_info)

            # Save to file if requested
            if output_file:
                self._save_system_info(all_info, output_file)

            return True

        except Exception as e:
            self.print_error_with_traceback(f"Error displaying system info: {e}", e)
            if self.debug:
                self.logger.exception("Error details:")
            return False

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect basic system information."""
        return {
            "OS": platform.system(),
            "OS Version": platform.version(),
            "Architecture": platform.architecture()[0],
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "Platform": platform.platform(),
            "Node": platform.node(),
        }

    def _collect_python_info(self) -> Dict[str, Any]:
        """Collect Python runtime information."""
        return {
            "Version": sys.version,
            "Executable": sys.executable,
            "Path": sys.path[:3],  # First 3 paths
            "Prefix": sys.prefix,
            "API Version": sys.api_version if hasattr(sys, "api_version") else "N/A",
        }

    def _collect_dependencies_info(self) -> Dict[str, str]:
        """Collect information about installed dependencies."""
        dependencies = [
            "torch",
            "numpy",
            "pandas",
            "matplotlib",
            "scikit-learn",
            "pytorch_lightning",
            "wandb",
            "tensorboard",
            "optuna",
            "rich",
            "typer",
            "pydantic",
            "omegaconf",
            "hydra-core",
        ]

        deps_info = {}
        for dep in dependencies:
            try:
                module = __import__(dep)
                version = getattr(module, "__version__", "Unknown")
                deps_info[dep] = version
            except ImportError:
                deps_info[dep] = "Not installed"
            except Exception:
                deps_info[dep] = "Error"

        return deps_info

    def _collect_hardware_info(self) -> Dict[str, Any]:
        """Collect hardware information."""
        hardware_info = {}

        # GPU information (if PyTorch available)
        try:
            import torch

            hardware_info["CUDA Available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                hardware_info["GPU Count"] = torch.cuda.device_count()
                hardware_info["GPU Names"] = [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
        except ImportError:
            hardware_info["PyTorch"] = "Not available"

        # Memory information
        try:
            import psutil

            memory = psutil.virtual_memory()
            hardware_info["RAM Total"] = f"{memory.total / (1024**3):.1f} GB"
            hardware_info["RAM Available"] = f"{memory.available / (1024**3):.1f} GB"
            hardware_info["CPU Count"] = psutil.cpu_count()
        except ImportError:
            hardware_info["Memory Info"] = "psutil not available"

        return hardware_info

    def _collect_new_ltpp_info(self) -> Dict[str, Any]:
        """Collect information specific to new_ltpp."""
        new_ltpp_info = {}

        try:
            # Check New_LTPP installation
            import new_ltpp

            new_ltpp_info["new_ltpp Version"] = getattr(
                new_ltpp, "__version__", "Unknown"
            )
            new_ltpp_info["Installation Path"] = str(Path(new_ltpp.__file__).parent) # type: ignore

            # Check main modules
            modules_to_check = [
                "new_ltpp.configs",
                "new_ltpp.runners",
                "new_ltpp.models",
                "new_ltpp.data",
                "new_ltpp.data.preprocess",
            ]

            for module_name in modules_to_check:
                    try:
                        __import__(module_name)
                        new_ltpp_info[f"Module {module_name}"] = "✓ Available"
                    except ImportError:
                        new_ltpp_info[f"Module {module_name}"] = "✗ Missing"

        except ImportError:
            new_ltpp_info["new_ltpp"] = "Not installed"

        return new_ltpp_info

    def _display_info_tables(self, all_info: Dict[str, Dict[str, Any]]):
        """Display information as tables."""
        if not self.console:
            # Simple text version
            for category, info in all_info.items():
                if info:  # Only if there is info
                    print(f"\n=== {category.upper()} ===")
                    for key, value in info.items():
                        print(f"{key}: {value}")
            return

        # Rich version with tables
        from rich.columns import Columns
        from rich.table import Table

        tables = []

        for category, info in all_info.items():
            if not info:  # Skip empty categories
                continue

            table = Table(title=category.title())
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")

            for key, value in info.items():
                # Special handling for lists
                if isinstance(value, list):
                    value_str = "\\n".join(
                        str(item) for item in value[:3]
                    )  # Max 3 items
                    if len(value) > 3:
                        value_str += f"\\n... ({len(value)} total)"
                else:
                    value_str = str(value)

                # Colorization for status strings
                if "✓" in value_str:
                    value_str = f"[green]{value_str}[/green]"
                elif "✗" in value_str or "Not" in value_str:
                    value_str = f"[red]{value_str}[/red]"

                table.add_row(key, value_str)

            tables.append(table)

        # Affichage en colonnes pour un meilleur layout
        if len(tables) <= 2:
            self.console.print(Columns(tables))
        else:
            for table in tables:
                self.console.print(table)
                self.console.print()  # Espace entre les tableaux

    def _save_system_info(self, all_info: Dict[str, Dict[str, Any]], output_file: str):
        """Save system information to a file."""
        import json
        from datetime import datetime

        # Add a timestamp
        report = {"timestamp": datetime.now().isoformat(), "system_info": all_info}

        # Serialize avoiding type errors
        def serialize_value(obj):
            if isinstance(obj, (list, tuple)):
                return [str(item) for item in obj]
            return str(obj)

        # Convert recursively
        serializable_info = {}
        for category, info in all_info.items():
            serializable_info[category] = {
                key: serialize_value(value) for key, value in info.items()
            }

        report["system_info"] = serializable_info

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.print_success(f"System information saved: {output_file}")
