#!/usr/bin/env python3
"""
EasyTPP CLI - Professional Command Line Interface

A comprehensive command-line tool for running Temporal Point Process experiments
with EasyTPP. This tool provides multiple subcommands for training, evaluation,
benchmarking, and data generation.

Author: EasyTPP Team
Version: 2.0.0
"""

import argparse
import logging
import sys
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import time

# Rich imports for beautiful CLI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Confirm, Prompt
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

# Project imports
try:
    from easy_tpp.config_factory import RunnerConfig
    from easy_tpp.runner import Runner
    from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config
except ImportError:
    print("❌ Error: EasyTPP modules not found. Please install the package or check your PYTHONPATH.")
    sys.exit(1)


class CLIConfig:
    """Configuration class for CLI settings"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.log_level = logging.INFO
        self.output_dir = Path("./outputs")
        self.config_dir = Path("./configs")
        self.verbose = False
        
    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration"""
        level = logging.DEBUG if verbose else self.log_level
        
        if RICH_AVAILABLE:
            logging.basicConfig(
                level=level,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(console=self.console, rich_tracebacks=True)]
            )
        else:
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution"""
    config_dir: str
    experiment_id: str
    dataset_id: str
    phase: str
    checkpoint_path: Optional[str] = None
    output_dir: Optional[str] = None
    device: Optional[str] = None
    seed: Optional[int] = None


class EasyTPPCLI:
    """Main CLI class for EasyTPP operations"""
    
    def __init__(self):
        self.config = CLIConfig()
        self.logger = logging.getLogger(__name__)
        
    def print_header(self):
        """Print CLI header"""
        if RICH_AVAILABLE:
            header_text = """
╔═══════════════════════════════════════════════════════════════╗
║                        EasyTPP CLI v2.0                      ║
║              Professional Temporal Point Process Tool         ║
╚═══════════════════════════════════════════════════════════════╝
            """
            panel = Panel(
                header_text,
                style="bold blue",
                expand=False
            )
            self.config.console.print(panel)
        else:
            print("=" * 60)
            print("            EasyTPP CLI v2.0")
            print("     Professional Temporal Point Process Tool")
            print("=" * 60)
    
    def run_experiment(self, exp_config: ExperimentConfig):
        """Run a single experiment"""
        try:
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.config.console,
                ) as progress:
                    task = progress.add_task("Loading configuration...", total=None)
                    
                    # Build configuration from YAML
                    config_dict = parse_runner_yaml_config(
                        exp_config.config_dir, 
                        exp_config.experiment_id, 
                        exp_config.dataset_id
                    )
                    
                    progress.update(task, description="Creating runner...")
                    config = RunnerConfig.from_dict(config_dict)
                    
                    runner = Runner(
                        config=config,
                        checkpoint_path=exp_config.checkpoint_path,
                        output_dir=exp_config.output_dir
                    )
                    
                    progress.update(task, description=f"Running {exp_config.phase} phase...")
                    runner.run(phase=exp_config.phase)
                    
                    progress.update(task, description="✅ Experiment completed!", completed=True)
            else:
                print(f"Loading configuration from {exp_config.config_dir}...")
                config_dict = parse_runner_yaml_config(
                    exp_config.config_dir, 
                    exp_config.experiment_id, 
                    exp_config.dataset_id
                )
                
                print("Creating runner...")
                config = RunnerConfig.from_dict(config_dict)
                
                runner = Runner(
                    config=config,
                    checkpoint_path=exp_config.checkpoint_path,
                    output_dir=exp_config.output_dir
                )
                
                print(f"Running {exp_config.phase} phase...")
                runner.run(phase=exp_config.phase)
                print("✅ Experiment completed!")
                
        except Exception as e:
            self.logger.error(f"❌ Experiment failed: {str(e)}")
            raise
    
    def list_configs(self, config_dir: str):
        """List available configuration files"""
        config_path = Path(config_dir)
        
        if not config_path.exists():
            self.logger.error(f"❌ Configuration directory not found: {config_dir}")
            return
        
        yaml_files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))
        
        if RICH_AVAILABLE:
            table = Table(title="Available Configuration Files")
            table.add_column("File", style="cyan")
            table.add_column("Size", style="magenta")
            table.add_column("Modified", style="yellow")
            
            for file in yaml_files:
                stat = file.stat()
                size = f"{stat.st_size} bytes"
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                table.add_row(file.name, size, modified)
            
            self.config.console.print(table)
        else:
            print("Available configuration files:")
            for file in yaml_files:
                print(f"  - {file.name}")
    
    def validate_config(self, config_path: str, experiment_id: str, dataset_id: str):
        """Validate configuration file"""
        try:
            config_dict = parse_runner_yaml_config(config_path, experiment_id, dataset_id)
            config = RunnerConfig.from_dict(config_dict)
            
            if RICH_AVAILABLE:
                self.config.console.print("✅ Configuration is valid!", style="bold green")
                
                # Show config summary
                table = Table(title="Configuration Summary")
                table.add_column("Parameter", style="cyan")
                table.add_column("Value", style="yellow")
                
                table.add_row("Experiment ID", experiment_id)
                table.add_row("Dataset ID", dataset_id)
                table.add_row("Model Type", getattr(config.model_config, 'model_name', 'Unknown'))
                
                self.config.console.print(table)
            else:
                print("✅ Configuration is valid!")
                print(f"Experiment ID: {experiment_id}")
                print(f"Dataset ID: {dataset_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {str(e)}")
            return False
        
        return True
    
    def interactive_mode(self):
        """Interactive configuration mode"""
        if not RICH_AVAILABLE:
            print("Interactive mode requires the 'rich' package. Please install it with: pip install rich")
            return
        
        self.config.console.print("🚀 Starting interactive mode...", style="bold blue")
        
        # Get configuration file
        config_dir = Prompt.ask("Configuration file path", default="./runner_config.yaml")
        
        if not Path(config_dir).exists():
            self.config.console.print(f"❌ Configuration file not found: {config_dir}", style="bold red")
            return
        
        # Get experiment details
        experiment_id = Prompt.ask("Experiment ID", default="THP")
        dataset_id = Prompt.ask("Dataset ID", default="H2expc")
        
        # Get phase
        phase_choices = ["train", "test", "predict", "validation", "all"]
        phase = Prompt.ask("Phase", choices=phase_choices, default="test")
        
        # Optional parameters
        checkpoint_path = Prompt.ask("Checkpoint path (optional)", default="")
        output_dir = Prompt.ask("Output directory (optional)", default="")
        
        # Confirm execution
        summary_table = Table(title="Execution Summary")
        summary_table.add_column("Parameter", style="cyan")
        summary_table.add_column("Value", style="yellow")
        
        summary_table.add_row("Config File", config_dir)
        summary_table.add_row("Experiment ID", experiment_id)
        summary_table.add_row("Dataset ID", dataset_id)
        summary_table.add_row("Phase", phase)
        if checkpoint_path:
            summary_table.add_row("Checkpoint", checkpoint_path)
        if output_dir:
            summary_table.add_row("Output Dir", output_dir)
        
        self.config.console.print(summary_table)
        
        if Confirm.ask("Do you want to proceed with this configuration?"):
            exp_config = ExperimentConfig(
                config_dir=config_dir,
                experiment_id=experiment_id,
                dataset_id=dataset_id,
                phase=phase,
                checkpoint_path=checkpoint_path if checkpoint_path else None,
                output_dir=output_dir if output_dir else None
            )
            
            self.run_experiment(exp_config)
        else:
            self.config.console.print("Operation cancelled.", style="yellow")


def create_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="EasyTPP CLI - Professional Temporal Point Process Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run training phase
  easytpp run --config ./config.yaml --experiment THP --dataset H2expc --phase train
  
  # Run evaluation with checkpoint
  easytpp run --config ./config.yaml --experiment THP --dataset H2expc --phase test --checkpoint ./model.ckpt
  
  # Interactive mode
  easytpp interactive
  
  # List available configurations
  easytpp list-configs --dir ./configs
  
  # Validate configuration
  easytpp validate --config ./config.yaml --experiment THP --dataset H2expc
  
For more information, visit: https://github.com/ant-research/EasyTemporalPointProcess
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="EasyTPP CLI v2.0.0"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run TPP experiments")
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    run_parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment ID in the config file"
    )
    run_parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Dataset ID in the config file"
    )
    run_parser.add_argument(
        "--phase", "-p",
        type=str,
        default="test",
        choices=["train", "test", "predict", "validation", "all"],
        help="Phase to execute"
    )
    run_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file"
    )
    run_parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results"
    )
    run_parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="Device to use for computation"
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive configuration mode")
    
    # List configs command
    list_parser = subparsers.add_parser("list-configs", help="List available configuration files")
    list_parser.add_argument(
        "--dir",
        type=str,
        default="./configs",
        help="Directory to search for config files"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    validate_parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment ID in the config file"
    )
    validate_parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Dataset ID in the config file"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system and environment information")
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize CLI
    cli = EasyTPPCLI()
    cli.config.setup_logging(verbose=args.verbose)
    
    # Show header
    cli.print_header()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "run":
            exp_config = ExperimentConfig(
                config_dir=args.config,
                experiment_id=args.experiment,
                dataset_id=args.dataset,
                phase=args.phase,
                checkpoint_path=args.checkpoint,
                output_dir=args.output,
                device=args.device,
                seed=args.seed
            )
            cli.run_experiment(exp_config)
            
        elif args.command == "interactive":
            cli.interactive_mode()
            
        elif args.command == "list-configs":
            cli.list_configs(args.dir)
            
        elif args.command == "validate":
            cli.validate_config(args.config, args.experiment, args.dataset)
            
        elif args.command == "info":
            if RICH_AVAILABLE:
                table = Table(title="System Information")
                table.add_column("Component", style="cyan")
                table.add_column("Version/Info", style="yellow")
                
                import torch
                import pytorch_lightning as pl
                
                table.add_row("Python", sys.version.split()[0])
                table.add_row("PyTorch", torch.__version__)
                table.add_row("PyTorch Lightning", pl.__version__)
                table.add_row("CUDA Available", str(torch.cuda.is_available()))
                if torch.cuda.is_available():
                    table.add_row("CUDA Version", torch.version.cuda)
                    table.add_row("GPU Count", str(torch.cuda.device_count()))
                
                cli.config.console.print(table)
            else:
                print("System Information:")
                print(f"Python: {sys.version.split()[0]}")
                import torch
                print(f"PyTorch: {torch.__version__}")
                print(f"CUDA Available: {torch.cuda.is_available()}")
        
    except KeyboardInterrupt:
        cli.logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        cli.logger.error(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
