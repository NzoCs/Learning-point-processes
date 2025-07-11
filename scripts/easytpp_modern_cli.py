#!/usr/bin/env python3
"""
EasyTPP CLI v2.1 - Modern CLI with Typer

An alternative implementation using Typer for a more modern CLI experience.
This version provides the same functionality as the main CLI but with a 
different framework approach.

Usage:
    python easytpp_modern_cli.py --help
    python easytpp_modern_cli.py run --help
    python easytpp_modern_cli.py interactive
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try to import required packages
try:
    import typer
    from typing_extensions import Annotated
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from easy_tpp.config_factory import RunnerConfig
    from easy_tpp.runner import Runner
    from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config
    EASYTPP_AVAILABLE = True
except ImportError:
    EASYTPP_AVAILABLE = False


# Initialize Typer app
if TYPER_AVAILABLE:
    app = typer.Typer(
        name="easytpp",
        help="🚀 EasyTPP - Professional Temporal Point Process CLI",
        add_completion=False,
        rich_markup_mode="rich" if RICH_AVAILABLE else None
    )
else:
    app = None

# Initialize console
console = Console() if RICH_AVAILABLE else None


def check_dependencies():
    """Check if required dependencies are available"""
    missing = []
    
    if not TYPER_AVAILABLE:
        missing.append("typer")
    if not EASYTPP_AVAILABLE:
        missing.append("easy_tpp")
    
    if missing:
        error_msg = f"❌ Missing required dependencies: {', '.join(missing)}"
        if RICH_AVAILABLE:
            console.print(error_msg, style="bold red")
        else:
            print(error_msg)
        
        print("\n📦 Install missing packages:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        return False
    
    return True


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    if RICH_AVAILABLE:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[RichHandler(console=console, rich_tracebacks=True)]
        )
    else:
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )


def print_header():
    """Print application header"""
    if RICH_AVAILABLE:
        header = Panel.fit(
            "[bold blue]EasyTPP CLI v2.1[/bold blue]\n"
            "[cyan]Modern Temporal Point Process Tool[/cyan]",
            border_style="blue"
        )
        console.print(header)
    else:
        print("=" * 50)
        print("       EasyTPP CLI v2.1")
        print("  Modern Temporal Point Process Tool")
        print("=" * 50)


@app.command()
def run(
    config: Annotated[str, typer.Option("--config", "-c", help="Configuration YAML file path")] = "./runner_config.yaml",
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="Experiment ID")] = "THP",
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset ID")] = "H2expc",
    phase: Annotated[str, typer.Option("--phase", "-p", help="Execution phase")] = typer.Option("test", case_sensitive=False),
    checkpoint: Annotated[Optional[str], typer.Option("--checkpoint", help="Checkpoint file path")] = None,
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Output directory")] = None,
    device: Annotated[str, typer.Option("--device", help="Computation device")] = "auto",
    seed: Annotated[Optional[int], typer.Option("--seed", help="Random seed")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False
):
    """🚀 Run TPP experiments with comprehensive configuration options."""
    
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Validate phase
    valid_phases = ["train", "test", "predict", "validation", "all"]
    if phase.lower() not in valid_phases:
        if RICH_AVAILABLE:
            console.print(f"❌ Invalid phase: {phase}. Valid options: {valid_phases}", style="bold red")
        else:
            print(f"❌ Invalid phase: {phase}. Valid options: {valid_phases}")
        raise typer.Exit(1)
    
    # Check if config file exists
    config_path = Path(config)
    if not config_path.exists():
        if RICH_AVAILABLE:
            console.print(f"❌ Configuration file not found: {config}", style="bold red")
        else:
            print(f"❌ Configuration file not found: {config}")
        raise typer.Exit(1)
    
    try:
        if RICH_AVAILABLE:
            with console.status("[bold green]Running experiment...") as status:
                status.update("[blue]Loading configuration...")
                config_dict = parse_runner_yaml_config(config, experiment, dataset)
                
                status.update("[blue]Creating runner...")
                runner_config = RunnerConfig.from_dict(config_dict)
                
                runner = Runner(
                    config=runner_config,
                    checkpoint_path=checkpoint,
                    output_dir=output
                )
                
                status.update(f"[green]Executing {phase} phase...")
                runner.run(phase=phase.lower())
            
            console.print("✅ Experiment completed successfully!", style="bold green")
        else:
            print(f"Loading configuration from {config}...")
            config_dict = parse_runner_yaml_config(config, experiment, dataset)
            
            print("Creating runner...")
            runner_config = RunnerConfig.from_dict(config_dict)
            
            runner = Runner(
                config=runner_config,
                checkpoint_path=checkpoint,
                output_dir=output
            )
            
            print(f"Running {phase} phase...")
            runner.run(phase=phase.lower())
            
            print("✅ Experiment completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Experiment failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def interactive():
    """🎯 Launch interactive configuration mode for guided experiment setup."""
    
    if not RICH_AVAILABLE:
        print("❌ Interactive mode requires the 'rich' package.")
        print("Install it with: pip install rich")
        raise typer.Exit(1)
    
    from rich.prompt import Prompt, Confirm
    
    console.print("🎯 Starting Interactive Mode", style="bold blue")
    console.print("Follow the prompts to configure your experiment.\n")
    
    # Get configuration details
    config_file = Prompt.ask(
        "📁 Configuration file path",
        default="./runner_config.yaml"
    )
    
    if not Path(config_file).exists():
        console.print(f"❌ Configuration file not found: {config_file}", style="bold red")
        raise typer.Exit(1)
    
    experiment_id = Prompt.ask(
        "🧪 Experiment ID",
        default="THP"
    )
    
    dataset_id = Prompt.ask(
        "📊 Dataset ID",
        default="H2expc"
    )
    
    phase = Prompt.ask(
        "⚙️ Execution phase",
        choices=["train", "test", "predict", "validation", "all"],
        default="test"
    )
    
    # Optional parameters
    use_checkpoint = Confirm.ask("📂 Use checkpoint file?", default=False)
    checkpoint_path = None
    if use_checkpoint:
        checkpoint_path = Prompt.ask("   Checkpoint path")
    
    custom_output = Confirm.ask("📁 Specify custom output directory?", default=False)
    output_dir = None
    if custom_output:
        output_dir = Prompt.ask("   Output directory")
    
    device = Prompt.ask(
        "💻 Computation device",
        choices=["auto", "cpu", "gpu"],
        default="auto"
    )
    
    use_seed = Confirm.ask("🌱 Set random seed?", default=False)
    seed = None
    if use_seed:
        seed = int(Prompt.ask("   Random seed", default="42"))
    
    # Show summary
    table = Table(title="Experiment Configuration Summary")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Config File", config_file)
    table.add_row("Experiment", experiment_id)
    table.add_row("Dataset", dataset_id)
    table.add_row("Phase", phase)
    table.add_row("Device", device)
    
    if checkpoint_path:
        table.add_row("Checkpoint", checkpoint_path)
    if output_dir:
        table.add_row("Output Dir", output_dir)
    if seed:
        table.add_row("Seed", str(seed))
    
    console.print(table)
    
    # Confirm execution
    if Confirm.ask("\n🚀 Proceed with this configuration?", default=True):
        # Execute using the run command
        run(
            config=config_file,
            experiment=experiment_id,
            dataset=dataset_id,
            phase=phase,
            checkpoint=checkpoint_path,
            output=output_dir,
            device=device,
            seed=seed,
            verbose=False
        )
    else:
        console.print("Operation cancelled.", style="yellow")


@app.command("list-configs")
def list_configs(
    directory: Annotated[str, typer.Option("--dir", "-d", help="Directory to search")] = "./configs"
):
    """📋 List available configuration files in a directory."""
    
    config_dir = Path(directory)
    
    if not config_dir.exists():
        if RICH_AVAILABLE:
            console.print(f"❌ Directory not found: {directory}", style="bold red")
        else:
            print(f"❌ Directory not found: {directory}")
        raise typer.Exit(1)
    
    # Find YAML files
    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    
    if not yaml_files:
        if RICH_AVAILABLE:
            console.print(f"📁 No YAML configuration files found in {directory}", style="yellow")
        else:
            print(f"📁 No YAML configuration files found in {directory}")
        return
    
    if RICH_AVAILABLE:
        table = Table(title=f"Configuration Files in {directory}")
        table.add_column("File", style="cyan")
        table.add_column("Size", style="magenta")
        table.add_column("Modified", style="yellow")
        
        for file in yaml_files:
            stat = file.stat()
            size = f"{stat.st_size:,} bytes"
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            table.add_row(file.name, size, modified)
        
        console.print(table)
    else:
        print(f"Configuration files in {directory}:")
        for file in yaml_files:
            print(f"  - {file.name}")


@app.command()
def validate(
    config: Annotated[str, typer.Option("--config", "-c", help="Configuration file path")],
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="Experiment ID")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset ID")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False
):
    """✅ Validate a configuration file and display summary."""
    
    setup_logging(verbose)
    
    config_path = Path(config)
    if not config_path.exists():
        if RICH_AVAILABLE:
            console.print(f"❌ Configuration file not found: {config}", style="bold red")
        else:
            print(f"❌ Configuration file not found: {config}")
        raise typer.Exit(1)
    
    try:
        config_dict = parse_runner_yaml_config(config, experiment, dataset)
        runner_config = RunnerConfig.from_dict(config_dict)
        
        if RICH_AVAILABLE:
            console.print("✅ Configuration is valid!", style="bold green")
            
            # Show configuration summary
            table = Table(title="Configuration Validation Summary")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="yellow")
            
            table.add_row("Config File", config)
            table.add_row("Experiment ID", experiment)
            table.add_row("Dataset ID", dataset)
            
            # Try to extract model information
            try:
                model_name = getattr(runner_config.model_config, 'model_name', 'Unknown')
                table.add_row("Model Type", model_name)
            except:
                pass
            
            console.print(table)
        else:
            print("✅ Configuration is valid!")
            print(f"Config File: {config}")
            print(f"Experiment ID: {experiment}")
            print(f"Dataset ID: {dataset}")
            
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"❌ Configuration validation failed: {str(e)}", style="bold red")
        else:
            print(f"❌ Configuration validation failed: {str(e)}")
        
        if verbose:
            import traceback
            traceback.print_exc()
        
        raise typer.Exit(1)


@app.command()
def info():
    """ℹ️ Display system and environment information."""
    
    try:
        import torch
        import pytorch_lightning as pl
        import platform
        
        if RICH_AVAILABLE:
            table = Table(title="System Information")
            table.add_column("Component", style="cyan")
            table.add_column("Version/Info", style="yellow")
            
            table.add_row("Platform", platform.platform())
            table.add_row("Python", sys.version.split()[0])
            table.add_row("PyTorch", torch.__version__)
            table.add_row("PyTorch Lightning", pl.__version__)
            table.add_row("CUDA Available", "✅ Yes" if torch.cuda.is_available() else "❌ No")
            
            if torch.cuda.is_available():
                table.add_row("CUDA Version", torch.version.cuda or "Unknown")
                table.add_row("GPU Count", str(torch.cuda.device_count()))
                if torch.cuda.device_count() > 0:
                    table.add_row("Primary GPU", torch.cuda.get_device_name(0))
            
            # Memory information
            if torch.cuda.is_available():
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                table.add_row("GPU Memory", f"{memory_gb:.1f} GB")
            
            console.print(table)
        else:
            print("System Information:")
            print(f"Platform: {platform.platform()}")
            print(f"Python: {sys.version.split()[0]}")
            print(f"PyTorch: {torch.__version__}")
            print(f"PyTorch Lightning: {pl.__version__}")
            print(f"CUDA Available: {'Yes' if torch.cuda.is_available() else 'No'}")
            
            if torch.cuda.is_available():
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"GPU Count: {torch.cuda.device_count()}")
        
    except ImportError as e:
        if RICH_AVAILABLE:
            console.print(f"❌ Error importing packages: {str(e)}", style="bold red")
        else:
            print(f"❌ Error importing packages: {str(e)}")
        raise typer.Exit(1)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
    version: Annotated[bool, typer.Option("--version", help="Show version")] = False
):
    """
    🚀 EasyTPP CLI v2.1 - Modern Temporal Point Process Tool
    
    A professional command-line interface for running Temporal Point Process
    experiments with EasyTPP. Built with Typer for a modern CLI experience.
    """
    
    if version:
        if RICH_AVAILABLE:
            console.print("EasyTPP CLI v2.1", style="bold blue")
        else:
            print("EasyTPP CLI v2.1")
        raise typer.Exit()
    
    if verbose:
        setup_logging(True)
    
    # Print header only if we're running a command
    print_header()


def cli_main():
    """Main entry point for the CLI"""
    if not check_dependencies():
        sys.exit(1)
    
    if not TYPER_AVAILABLE:
        print("❌ This CLI requires Typer. Please install it with: pip install typer")
        sys.exit(1)
    
    app()


if __name__ == "__main__":
    cli_main()
