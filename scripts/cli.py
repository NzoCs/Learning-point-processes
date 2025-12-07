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
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from new_ltpp.globals import CONFIGS_FILE, OUTPUT_DIR

# Import runners
from .cli_runners import (
    BenchmarkRunner,
    DataGenerator,
    DataInspector,
    ExperimentRunner,
    InteractiveSetup,
    SystemInfo,
)

app = typer.Typer(
    name="new-ltpp",
    help="New-LTPP CLI v4.0 - Temporal Point Processes with runners architecture",
    no_args_is_help=True,
)
console = Console()


@app.command("run")
def run_experiment(
    config: str = typer.Option(
        str(CONFIGS_FILE),
        "--config",
        "-c",
        help="YAML configuration file [default: yaml_configs/configs.yaml]",
    ),
    data_config: str = typer.Option(
        "test",
        "--data-config",
        help="Data configuration (taxi, taobao, amazon, retweet, volcano, earthquake, stackoverflow, test, hawkes1, hawkes2, H2expi, H2expc) [default: test]",
    ),
    general_specs_config: str = typer.Option(
        "quick_test",
        "--general-specs-config",
        help="General model specs configuration (quick_test, debug, h16, h32, h64, h128) [default: quick_test]",
    ),
    model_specs_config: Optional[str] = typer.Option(
        None,
        "--model-specs-config",
        help="Model-specific specs configuration (optional, depends on model) [default: None]",
    ),
    training_config: str = typer.Option(
        "quick_test",
        "--training-config",
        help="Training configuration (quick_test, debug, e500_b1, e500_b4, e1000_b2, e1000_b4) [default: quick_test]",
    ),
    data_loading_config: str = typer.Option(
        "quick_test",
        "--data-loading-config",
        help="Data loading configuration (debug, quick_test, b32_w1, b64_w2, b128_w4, b64_w4, b32_w2, b16_w1, b128_w6, b512_w8) [default: quick_test]",
    ),
    simulation_config: str = typer.Option(
        "quick_test",
        "--simulation-config",
        help="Simulation configuration (quick_test, debug, tw30_b5000_b16, tw70_b15000_b32, tw100_b50000_b64, tw200_b100000_b128, tw300_b150000_b256, tw80_b30000_b64, tw70_b15000_b32) [default: quick_test]",
    ),
    thinning_config: str = typer.Option(
        "quick_test",
        "--thinning-config",
        help="Thinning configuration (quick_test, debug, e50_s15, e100_s30, e150_s50, e200_s60, e20_s10) [default: quick_test]",
    ),
    logger_config: str = typer.Option(
        "tensorboard",
        "--logger-config",
        help="Logger configuration (tensorboard, csv, wandb) [default: tensorboard]",
    ),
    model_id: str = typer.Option(
        "NHP", "--model", "-m", help="Model ID (NHP, RMTPP, etc.) [default: NHP]"
    ),
    phase: str = typer.Option(
        "all", "--phase", "-p", help="Execution phase (train/test/predict/all)"
    ),
    max_epochs: int | None = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Maximum number of epochs [default: 100]",
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
        general_specs_config=general_specs_config,
        model_specs_config=model_specs_config,
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
    num_simulations: int = typer.Option(
        1000, "--num-sim", "-n", help="Number of simulations to generate"
    ),
    method: str = typer.Option(
        "hawkes", "--method", "-m", help="Generation method (hawkes, self_correcting)"
    ),
    dim_process: int = typer.Option(
        2, "--dim", "-d", help="Number of event types/dimensions"
    ),
    start_time: float = typer.Option(
        0.0, "--start", help="Start time for simulation"
    ),
    end_time: float = typer.Option(
        100.0, "--end", help="End time for simulation"
    ),
    train_ratio: float = typer.Option(
        0.6, "--train-ratio", help="Train split ratio"
    ),
    test_ratio: float = typer.Option(
        0.2, "--test-ratio", help="Test split ratio"
    ),
    dev_ratio: float = typer.Option(
        0.2, "--dev-ratio", help="Dev split ratio"
    ),
    push_to_hub: bool = typer.Option(
        False, "--push", help="Push dataset to Hugging Face Hub"
    ),
    repo_id: Optional[str] = typer.Option(
        None, "--repo-id", help="Hugging Face repo ID (username/dataset-name)"
    ),
    private: bool = typer.Option(
        False, "--private", help="Make Hugging Face dataset private"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed for reproducibility"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
):
    """
    Generate synthetic TPP data with DataGenerator.
    
    Examples:
        # Generate Hawkes process data
        new-ltpp generate --method hawkes --num-sim 1000 --dim 2
        
        # Generate self-correcting process data
        new-ltpp generate --method self_correcting --num-sim 500 --dim 3
        
        # Generate and push to Hugging Face
        new-ltpp generate --method hawkes --push --repo-id username/my-dataset
    """
    # Prepare splits
    splits = {
        "train": train_ratio,
        "test": test_ratio,
        "dev": dev_ratio
    }
    
    # Validate splits
    if abs(sum(splits.values()) - 1.0) > 1e-10:
        console.print("[red]Error: Split ratios must sum to 1.0[/red]")
        raise typer.Exit(1)
    
    # If push to hub is requested, check repo_id
    if push_to_hub and not repo_id:
        console.print("[red]Error: --repo-id is required when using --push[/red]")
        raise typer.Exit(1)
    
    try:
        # Create simulator based on method
        import numpy as np
        from new_ltpp.data.generation import (
            HawkesSimulator,
            IOSimulator,
            SelfCorrecting,
            SimulationManager,
        )
        
        if method.lower() == "hawkes":
            simulator = HawkesSimulator(
                mu=np.array([0.2] * dim_process),
                alpha=np.array([[0.3 if i == j else 0.1 for j in range(dim_process)] for i in range(dim_process)]),
                beta=np.array([[2.0 if i == j else 1.0 for j in range(dim_process)] for i in range(dim_process)]),
                dim_process=dim_process,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
            )
        elif method.lower() == "self_correcting":
            simulator = SelfCorrecting(
                dim_process=dim_process,
                mu=1.0,
                alpha=1.0,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
            )
        else:
            console.print(f"[red]Unknown method: {method}[/red]")
            console.print("[yellow]Available methods: hawkes, self_correcting[/yellow]")
            raise typer.Exit(1)
        
        # Create simulation manager
        console.print(f"[bold blue]Generating {num_simulations} simulations with {method} method[/bold blue]")
        sim_manager = SimulationManager(
            simulation_func=simulator.simulate,
            dim_process=dim_process,
            start_time=start_time,
            end_time=end_time,
        )
        
        # Generate and format simulations
        formatted_data = sim_manager.bulk_simulate(num_simulations)
        metadata = simulator.get_metadata(num_simulations)
        
        # Create IO handler
        io_handler = IOSimulator()
        
        # Push to Hugging Face Hub or save locally
        if push_to_hub and repo_id:
            console.print(f"[bold blue]Pushing dataset to Hugging Face Hub: {repo_id}[/bold blue]")
            io_handler.push_to_hub(
                formatted_data=formatted_data,
                repo_id=repo_id,
                splits=splits,
                metadata=metadata,
                private=private,
            )
            console.print(f"[green]✓ Dataset successfully pushed to https://huggingface.co/datasets/{repo_id}[/green]")
        else:
            # Use output_dir or create default
            if output_dir is None:
                from datetime import datetime
                from pathlib import Path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = str(Path("artifacts") / "generated_data" / f"generated_{timestamp}")
                console.print(f"Output directory: {output_dir}")
            
            io_handler.save_to_json(
                formatted_data=formatted_data,
                output_dir=output_dir,
                splits=splits,
                metadata=metadata,
            )
            console.print(f"[green]✓ Dataset successfully generated in {output_dir}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during generation: {e}[/red]")
        if debug:
            import traceback
            traceback.print_exc()
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
        CONFIGS_FILE, "--config", "-c", help="YAML configuration file"
    ),
    data_config: Optional[List[str]] = typer.Option(
        None,
        "--data-config",
        help="Data configuration(s) (e.g., test, large). Can be repeated for multiple configs",
    ),
    data_loading_config: str = typer.Option(
        "quick_test",
        "--data-loading-config",
        help="Data loading configuration",
    ),
    benchmarks: Optional[List[str]] = typer.Option(
        None, "--benchmarks", "-b", help="List of benchmarks to run"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    run_all: bool = typer.Option(False, "--all", help="Run all benchmarks"),
    run_all_configs: bool = typer.Option(
        False, "--all-configs", help="Run on all configurations"
    ),
    list_benchmarks: bool = typer.Option(
        False, "--list", help="List available benchmarks"
    ),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
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
