"""
Interactive Setup Runner

Runner for guided configuration of TPP experiments.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from .cli_base import CLIRunnerBase


class InteractiveSetup(CLIRunnerBase):
    """
    Runner for interactive experiment configuration.
    Guides the user through creating configuration files.
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
        Start the interactive configuration.

        Args:
            setup_type: Type of setup (experiment, data, model)
            output_path: Path to save the configuration
            quick_mode: Quick mode using default values

        Returns:
            True if the configuration was created successfully
        """

        self.print_info(f"Configuration interactive - Type: {setup_type}")

        if setup_type == "experiment":
            config = self._setup_experiment_config(quick_mode)
        elif setup_type == "data":
            config = self._setup_data_config(quick_mode)
        elif setup_type == "model":
            config = self._setup_model_config(quick_mode)
        else:
            self.print_error(f"Unsupported setup type: {setup_type}")
            return False

        # Display final configuration
        self._display_final_config(config)

        # Confirm and save
        if Confirm.ask("Save this configuration?"):
            if output_path is None:
                output_path_str = Prompt.ask(
                    "Save path", default=f"{setup_type}_config.yaml"
                )
                output_path_obj = Path(output_path_str)
            else:
                output_path_obj = Path(output_path)

            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path_obj, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            self.print_success(f"Configuration saved: {output_path_obj}")

            # Offer to launch the experiment directly
            if setup_type == "experiment" and Confirm.ask(
                "Launch the experiment now?"
            ):
                return self._launch_experiment(output_path_obj)

        return True

    def _setup_experiment_config(self, quick_mode: bool) -> Dict[str, Any]:
        """Interactive configuration for a full experiment."""
        config = {}

        if self.console:
            self.console.print("[bold blue]TPP Experiment Configuration[/bold blue]")
            self.console.print()

        # Data configuration
        if not quick_mode or Confirm.ask("Configure data?", default=True):
            config["data_config"] = self._setup_data_section(quick_mode)

        # Model configuration
        if not quick_mode or Confirm.ask("Configure model?", default=True):
            config["model_config"] = self._setup_model_section(quick_mode)

        # Runner configuration
        if not quick_mode or Confirm.ask("Configure training?", default=True):
            config["runner_config"] = self._setup_runner_section(quick_mode)

        return config

    def _setup_data_config(self, quick_mode: bool) -> Dict[str, Any]:
        """Interactive data configuration."""
        if self.console:
            self.console.print("[bold green]Data Configuration[/bold green]")

        data_config = {}

        # Data directory
        data_dir = Prompt.ask("Data directory", default="./data/")
        data_config["data_dir"] = data_dir

        # Data format
        if quick_mode:
            data_format = "json"
        else:
            data_format = Prompt.ask(
                "Data format", choices=["json", "csv", "pickle"], default="json"
            )
        data_config["data_format"] = data_format

        # Data specifications
        tokenizer_specs = {}

        num_event_types = IntPrompt.ask("Number of event types", default=5)
        tokenizer_specs["num_event_types"] = num_event_types

        max_seq_len = IntPrompt.ask("Maximum sequence length", default=100)
        tokenizer_specs["max_seq_len"] = max_seq_len

        data_config["tokenizer_specs"] = tokenizer_specs

        # Loading configuration
        if not quick_mode:
            loading_specs = {}

            batch_size = IntPrompt.ask("Batch size", default=64)
            loading_specs["batch_size"] = batch_size

            shuffle = Confirm.ask("Shuffle data?", default=True)
            loading_specs["shuffle"] = shuffle

            data_config["data_loading_specs"] = loading_specs

        return data_config

    def _setup_model_config(self, quick_mode: bool) -> Dict[str, Any]:
        """Interactive model configuration."""
        if self.console:
            self.console.print("[bold green]Model Configuration[/bold green]")

        model_config = {}

        # Model type
        if quick_mode:
            model_type = "NHP"
        else:
            model_type = Prompt.ask(
                "Model type",
                choices=["NHP", "THP", "RMTPP", "FullyNN", "LogNormMix"],
                default="NHP",
            )
        model_config["model_type"] = model_type

        # Model specifications
        model_specs = {}

        hidden_size = IntPrompt.ask("Hidden layer size", default=128)
        model_specs["hidden_size"] = hidden_size

        if not quick_mode:
            num_layers = IntPrompt.ask("Number of layers", default=2)
            model_specs["num_layers"] = num_layers

            dropout = FloatPrompt.ask("Dropout rate", default=0.1)
            model_specs["dropout"] = dropout

        model_config["model_specs"] = model_specs

        return model_config

    def _setup_data_section(self, quick_mode: bool) -> Dict[str, Any]:
        """Data section for experiment configuration."""
        return self._setup_data_config(quick_mode)

    def _setup_model_section(self, quick_mode: bool) -> Dict[str, Any]:
        """Model section for experiment configuration."""
        return self._setup_model_config(quick_mode)

    def _setup_runner_section(self, quick_mode: bool) -> Dict[str, Any]:
        """Runner section for experiment configuration."""
        if self.console:
            self.console.print(
                "[bold green]Training Configuration[/bold green]"
            )

        runner_config = {}

        # Identifiers
        dataset_id = Prompt.ask("Dataset ID", default="experiment")
        runner_config["dataset_id"] = dataset_id
        model_id = Prompt.ask("Model ID", default="baseline")
        runner_config["model_id"] = model_id

        # Training parameters
        max_epochs = IntPrompt.ask("Maximum number of epochs", default=100)
        runner_config["max_epochs"] = max_epochs

        if not quick_mode:
            patience = IntPrompt.ask("Early stopping patience", default=20)
            runner_config["patience"] = patience

            val_freq = IntPrompt.ask("Validation frequency", default=5)
            runner_config["val_freq"] = val_freq

            save_dir = Prompt.ask("Save directory", default="./experiments/")
            runner_config["save_dir"] = save_dir

        return runner_config

    def _display_final_config(self, config: Dict[str, Any]):
        """Display the final configuration."""
        if not self.console:
            print("\n=== Final Configuration ===")
            for key, value in config.items():
                print(f"{key}: {value}")
            return

        from rich.panel import Panel
        from rich.tree import Tree

        tree = Tree("Generated Configuration")

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
            Panel(tree, title="Final Configuration", border_style="blue")
        )

    def _launch_experiment(self, config_path: Path) -> bool:
        """Launch an experiment directly using the created configuration."""
        try:
            # Import and use the ExperimentRunner
            from .experiment_runner import ExperimentRunner

            runner = ExperimentRunner()
            success = runner.run_experiment(config_path=str(config_path), phase="all")

            if success:
                self.print_success("Experiment launched successfully!")
            else:
                self.print_error("Experiment launch failed")

            return success

        except Exception as e:
            self.print_error(f"Error during launch: {e}")
            return False
