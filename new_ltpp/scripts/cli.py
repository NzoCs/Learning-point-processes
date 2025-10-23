#!/usr/bin/env python3
"""
EasyTPP CLI v4.0 - Runners Process Architecture

Command line interface for EasyTPP using modular runners
for each process (experiment, inspection, generation, etc.).

Usage:
    python easytpp_cli_runners.py --help
    python easytpp_cli_runners.py run --config config.yaml
    python easytpp_cli_runners.py inspect --data-dir ./data/
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

# Import runners
from .cli_runners import (
    BenchmarkRunner,
    DataGenerator,
    DataInspector,
    ExperimentRunner,
    InteractiveSetup,
    SystemInfo,
)

from new_ltpp.globals import CONFIGS_FILE, OUTPUT_DIR

app = typer.Typer(
    name="new-ltpp",
    help="New-LTPP CLI v4.0 - Temporal Point Processes with runners architecture",
    no_args_is_help=True,
)
console = Console()


@app.command("run")
def run_experiment(
    config: str = typer.Option(
        str(CONFIGS_FILE), "--config", "-c", help="YAML configuration file [default: yaml_configs/configs.yaml]"
    ),
    data_config: str = typer.Option(
        "test", "--data-config", help="Data configuration (test, large, synthetic) [default: test]"
    ),
    model_config: str = typer.Option(
        "neural_small",
        "--model-config",
        help="Model configuration (neural_small, neural_large) [default: neural_small]"
    ),
    training_config: str = typer.Option(
        "quick_test",
        "--training-config",
        help="Training configuration (quick_test, full_training) [default: quick_test]",
    ),
    data_loading_config: str = typer.Option(
        "quick_test", "--data-loading-config", help="Data loading configuration [default: quick_test]"
    ),
    simulation_config: str = typer.Option(
        "simulation_fast", "--simulation-config", help="Configuration de simulation [default: simulation_fast]"
    ),
    thinning_config: str = typer.Option(
        "thinning_fast", "--thinning-config", help="Configuration de thinning [default: thinning_fast]"
    ),
    logger_config: str = typer.Option(
        "tensorboard",
        "--logger-config",
        help="Configuration du logger (mlflow, tensorboard) [default: tensorboard]",
    ),
    model_id: str = typer.Option(
        "NHP", "--model", "-m", help="ID du modèle (NHP, RMTPP, etc.) [default: NHP]"
    ),
    phase: str = typer.Option(
        "all", "--phase", "-p", help="Phase d'exécution (train/test/predict/all)"
    ),
    max_epochs: int = typer.Option(
        100, "--epochs", "-e", help="Maximum number of epochs [default: 100]",
    ),
    save_dir: str = typer.Option(
        str(OUTPUT_DIR), "--save-dir", "-s", help="Save directory [default: artifacts]"
    ),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
):
    """Run a TPP experiment with ExperimentRunner."""
    runner = ExperimentRunner(debug=debug)
    success = runner.run_experiment(
        config_path=config,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        data_loading_config=data_loading_config,
        simulation_config=simulation_config,
        thinning_config=thinning_config,
        logger_config=logger_config,
        model_id=model_id,
        phase=phase,
        max_epochs=max_epochs,
        save_dir=save_dir,
        debug=debug,
    )

    if not success:
        raise typer.Exit(1)


@app.command("inspect")
def inspect_data(
    data_dir: str = typer.Argument(..., help="Directory containing the data"),
    data_format: str = typer.Option("json", "--format", "-f", help="Data format"),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    save_graphs: bool = typer.Option(True, "--save/--no-save", help="Save graphs"),
    show_graphs: bool = typer.Option(False, "--show/--no-show", help="Show graphs"),
    max_sequences: Optional[int] = typer.Option(
        None, "--max-seq", help="Maximum number of sequences"
    ),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
):
    """Inspect and visualize TPP data with DataInspector."""
    runner = DataInspector(debug=debug)
    success = runner.inspect_data(
        data_dir=data_dir,
        data_format=data_format,
        output_dir=output_dir,
        save_graphs=save_graphs,
        show_graphs=show_graphs,
        max_sequences=max_sequences,
    )

    if not success:
        raise typer.Exit(1)


@app.command("generate")
def generate_data(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory [default: artifacts/generated_data/TIMESTAMP]",
    ),
    num_sequences: int = typer.Option(
        1000, "--num-seq", "-n", help="Number of sequences"
    ),
    max_seq_len: int = typer.Option(
        100, "--max-len", "-l", help="Maximum sequence length"
    ),
    num_event_types: int = typer.Option(
        5, "--event-types", "-t", help="Number of event types"
    ),
    method: str = typer.Option("nhp", "--method", "-m", help="Generation method"),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Configuration file"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed for reproducibility"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
):
    """Generate synthetic TPP data with DataGenerator."""
    runner = DataGenerator(debug=debug)
    success = runner.generate_data(
        output_dir=output_dir,
        num_sequences=num_sequences,
        max_seq_len=max_seq_len,
        num_event_types=num_event_types,
        generation_method=method,
        config_path=config,
        seed=seed,
    )

    if not success:
        raise typer.Exit(1)


# ConfigValidator removed - use built-in validation in configs


@app.command("info")
def system_info(
    include_deps: bool = typer.Option(
        True, "--deps/--no-deps", help="Include dependencies"
    ),
    include_hardware: bool = typer.Option(
        True, "--hw/--no-hw", help="Include hardware info"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
):
    """Display system information with SystemInfo."""
    runner = SystemInfo(debug=debug)
    success = runner.display_system_info(
        include_deps=include_deps, include_hardware=include_hardware, output_file=output
    )

    if not success:
        raise typer.Exit(1)


# ConfigManager removed - use built-in configuration tools


@app.command("setup")
def interactive_setup(
    setup_type: str = typer.Option("experiment", "--type", "-t", help="Setup type"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
):
    """Interactive configuration with InteractiveSetup."""
    runner = InteractiveSetup(debug=debug)
    success = runner.run_interactive_setup(
        setup_type=setup_type, output_path=output, quick_mode=quick
    )

    if not success:
        raise typer.Exit(1)


@app.command("benchmark")
def benchmark_performance(
    config_path: str = typer.Option(
        ..., "--config", "-c", help="Fichier de configuration YAML"
    ),
    data_config: Optional[List[str]] = typer.Option(
        None,
        "--data-config",
        help="Configuration(s) des données (ex: test, large). Peut être répété pour plusieurs configs",
    ),
    data_loading_config: str = typer.Option(
        "quick_test",
        "--data-loading-config",
        help="Configuration du chargement des données",
    ),
    benchmarks: Optional[List[str]] = typer.Option(
        None, "--benchmarks", "-b", help="Liste des benchmarks à exécuter"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Répertoire de sortie"
    ),
    run_all: bool = typer.Option(False, "--all", help="Exécuter tous les benchmarks"),
    run_all_configs: bool = typer.Option(
        False, "--all-configs", help="Exécuter sur toutes les configurations"
    ),
    list_benchmarks: bool = typer.Option(
        False, "--list", help="Lister les benchmarks disponibles"
    ),
    debug: bool = typer.Option(False, "--debug", help="Mode debug"),
):
    """
    Run TPP benchmarks with BenchmarkRunner.

    Examples:
        # Simple benchmark
        benchmark --data-config test --all

        # Multiple configurations
        benchmark --data-config test --data-config large --all

        # All benchmarks on all configs
        benchmark --data-config test --data-config large --all --all-configs
    """
    runner = BenchmarkRunner(debug=debug)

    if list_benchmarks:
        runner.list_available_benchmarks()
        return

    # If no config specified, use 'test' by default
    if data_config is None:
        data_config = ["test"]

    success = runner.run_benchmark(
        config_path=config_path,
        data_config=data_config,
        data_loading_config=data_loading_config,
        benchmarks=benchmarks,
        output_dir=output_dir,
        run_all=run_all,
        run_all_configs=run_all_configs,
    )

    if not success:
        raise typer.Exit(1)


@app.command("version")
def show_version():
    """Show CLI version."""
    console.print("[bold blue]EasyTPP CLI v4.0[/bold blue]")
    console.print("Architecture: Runners Process")
    console.print("Available runners:")

    runners = [
        "ExperimentRunner - Run TPP experiments",
        "DataInspector - Inspect and visualize data",
        "DataGenerator - Generate synthetic data",
        "SystemInfo - System information",
        "InteractiveSetup - Guided setup",
        "BenchmarkRunner - Performance benchmarks",
    ]

    for runner in runners:
        console.print(f"  • [green]{runner}[/green]")


def main():
    """Point d'entrée principal."""
    if len(sys.argv) == 1:
        console.print(
            "[bold blue]🚀 EasyTPP CLI v4.0 - Runners Architecture[/bold blue]"
        )
        console.print()
        console.print("Available commands:")

        commands = [
            ("run", "Run a TPP experiment"),
            ("inspect", "Inspect and visualize data"),
            ("generate", "Generate synthetic data"),
            ("info", "System information"),
            ("setup", "Interactive configuration"),
            ("benchmark", "Performance benchmarks"),
            ("version", "Show version"),
        ]

        table = Table()
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        console.print(table)
        console.print()
        console.print("Use [bold]--help[/bold] with each command for more details")
        return

    app()


if __name__ == "__main__":
    main()
