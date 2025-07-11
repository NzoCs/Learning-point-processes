#!/usr/bin/env python3
"""
Setup script for EasyTPP CLI

This script sets up the EasyTPP CLI tool by:
1. Installing required dependencies
2. Creating a symbolic link or entry point
3. Setting up configuration directories
4. Providing initial configuration templates
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform


def run_command(command, check=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check,
            capture_output=True,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Enhanced requirements for the CLI
    cli_requirements = [
        "rich>=13.0.0",  # For beautiful CLI output
        "click>=8.0.0",  # Alternative CLI framework (optional)
        "pyyaml>=6.0",   # YAML parsing
        "tabulate>=0.9.0",  # Table formatting fallback
        "colorama>=0.4.0",  # Cross-platform colored output
        "typer>=0.9.0",  # Modern CLI framework (optional)
        "inquirer>=3.0.0",  # Interactive prompts (optional)
    ]
    
    for package in cli_requirements:
        print(f"  Installing {package}...")
        result = run_command(f"pip install {package}", check=False)
        if result.returncode != 0:
            print(f"    ⚠️  Warning: Could not install {package}")
        else:
            print(f"    ✅ {package} installed successfully")


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "configs",
        "outputs",
        "logs",
        "checkpoints",
        "data",
        "experiments"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"  ✅ Created {dir_name}/")


def create_config_templates():
    """Create configuration templates"""
    print("⚙️  Creating configuration templates...")
    
    # Basic runner config template
    runner_config_template = """# EasyTPP Runner Configuration Template
# This is a template configuration file for EasyTPP experiments

base_config:
  seed: 42
  device: auto  # auto, cpu, gpu
  log_level: INFO

experiments:
  THP:  # Transformer Hawkes Process
    model_config:
      model_name: THP
      hidden_size: 64
      num_layers: 4
      num_heads: 8
      dropout: 0.1
      
    data_config:
      dataset_name: H2expc
      batch_size: 32
      train_split: 0.8
      valid_split: 0.1
      test_split: 0.1
      
    training_config:
      optimizer: Adam
      learning_rate: 0.001
      max_epochs: 100
      patience: 10
      monitor: val_loss
      
    logging_config:
      use_wandb: false
      project_name: easytpp_experiments
      experiment_name: thp_baseline
      
datasets:
  H2expc:
    path: ./data/h2expc
    type: point_process
    preprocessing:
      normalize: true
      standardize: false
      
  taxi:
    path: ./data/taxi
    type: point_process
    preprocessing:
      normalize: true
      standardize: true
"""
    
    with open("configs/runner_config_template.yaml", "w") as f:
        f.write(runner_config_template)
    
    print("  ✅ Created configs/runner_config_template.yaml")
    
    # CLI configuration
    cli_config = """# EasyTPP CLI Configuration
# Global settings for the CLI tool

cli:
  default_config_dir: ./configs
  default_output_dir: ./outputs
  default_log_level: INFO
  auto_create_dirs: true
  
  # Rich terminal settings
  use_rich: true
  color_theme: default
  progress_bars: true
  
  # Logging settings
  log_file: ./logs/easytpp_cli.log
  log_rotation: daily
  log_retention: 30  # days
  
# Default values for commands
defaults:
  experiment_id: THP
  dataset_id: H2expc
  phase: test
  device: auto
  
# Shortcuts for common configurations
shortcuts:
  quick_train:
    phase: train
    device: auto
    
  quick_test:
    phase: test
    device: auto
    
  gpu_train:
    phase: train
    device: gpu
    
  cpu_test:
    phase: test
    device: cpu
"""
    
    with open("configs/cli_config.yaml", "w") as f:
        f.write(cli_config)
    
    print("  ✅ Created configs/cli_config.yaml")


def create_entry_point():
    """Create entry point script"""
    print("🔗 Creating entry point...")
    
    # Get the absolute path of the CLI script
    cli_script_path = Path(__file__).parent / "easytpp_cli.py"
    
    if platform.system() == "Windows":
        # Create batch file for Windows
        batch_content = f"""@echo off
python "{cli_script_path}" %*
"""
        with open("easytpp.bat", "w") as f:
            f.write(batch_content)
        print("  ✅ Created easytpp.bat (Windows entry point)")
        
        # Add to PATH instructions
        print("  📝 To use 'easytpp' command globally on Windows:")
        print(f"     1. Add {Path().absolute()} to your PATH environment variable")
        print("     2. Or copy easytpp.bat to a directory already in PATH")
        
    else:
        # Create shell script for Unix-like systems
        shell_content = f"""#!/bin/bash
python3 "{cli_script_path}" "$@"
"""
        with open("easytpp", "w") as f:
            f.write(shell_content)
        
        # Make executable
        os.chmod("easytpp", 0o755)
        print("  ✅ Created easytpp (Unix entry point)")
        
        # Add to PATH instructions
        print("  📝 To use 'easytpp' command globally on Unix/Linux/macOS:")
        print(f"     1. Add {Path().absolute()} to your PATH")
        print("     2. Or create a symlink: ln -s {Path().absolute()}/easytpp /usr/local/bin/easytpp")


def create_readme():
    """Create README for CLI usage"""
    print("📚 Creating documentation...")
    
    readme_content = """# EasyTPP CLI - Professional Command Line Interface

A comprehensive and professional command-line tool for running Temporal Point Process experiments with EasyTPP.

## 🚀 Features

- **Professional CLI Interface**: Clean, intuitive commands with comprehensive help
- **Rich Terminal Output**: Beautiful tables, progress bars, and colored output
- **Interactive Mode**: Guided configuration setup
- **Configuration Management**: Template system and validation
- **Multiple Execution Modes**: Train, test, predict, validate, and more
- **Error Handling**: Robust error reporting and recovery
- **Logging**: Professional logging with rotation and retention
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 📦 Installation

1. **Install Dependencies**:
   ```bash
   python setup_cli.py
   ```

2. **Verify Installation**:
   ```bash
   python easytpp_cli.py --version
   ```

## 🎯 Quick Start

### Basic Usage

```bash
# Run a basic experiment
python easytpp_cli.py run --config configs/runner_config.yaml --experiment THP --dataset H2expc --phase test

# Interactive mode (recommended for beginners)
python easytpp_cli.py interactive

# List available configurations
python easytpp_cli.py list-configs --dir configs

# Validate a configuration
python easytpp_cli.py validate --config configs/runner_config.yaml --experiment THP --dataset H2expc
```

### Advanced Usage

```bash
# Training with specific checkpoint and output directory
python easytpp_cli.py run \\
  --config configs/advanced_config.yaml \\
  --experiment TransformerHP \\
  --dataset taxi \\
  --phase train \\
  --checkpoint checkpoints/model_epoch_50.ckpt \\
  --output experiments/run_001 \\
  --device gpu \\
  --seed 42

# Run all phases sequentially
python easytpp_cli.py run \\
  --config configs/full_pipeline.yaml \\
  --experiment THP \\
  --dataset H2expc \\
  --phase all

# Get system information
python easytpp_cli.py info
```

## 📋 Commands

### `run` - Execute TPP Experiments

Run temporal point process experiments with comprehensive configuration options.

```bash
python easytpp_cli.py run [OPTIONS]
```

**Options:**
- `--config, -c`: Path to configuration YAML file (required)
- `--experiment, -e`: Experiment ID in the config file (required)
- `--dataset, -d`: Dataset ID in the config file (required)
- `--phase, -p`: Phase to execute [train|test|predict|validation|all] (default: test)
- `--checkpoint`: Path to checkpoint file
- `--output`: Output directory for results
- `--device`: Device to use [cpu|gpu|auto] (default: auto)
- `--seed`: Random seed for reproducibility

### `interactive` - Interactive Configuration

Launch an interactive mode for guided experiment setup.

```bash
python easytpp_cli.py interactive
```

This mode will guide you through:
- Configuration file selection
- Experiment and dataset selection
- Phase and parameter configuration
- Execution confirmation

### `list-configs` - List Available Configurations

Display all available configuration files in a directory.

```bash
python easytpp_cli.py list-configs [--dir DIRECTORY]
```

**Options:**
- `--dir`: Directory to search for config files (default: ./configs)

### `validate` - Validate Configuration

Validate a configuration file and show a summary.

```bash
python easytpp_cli.py validate --config CONFIG --experiment EXP --dataset DATA
```

### `info` - System Information

Display system and environment information including PyTorch, CUDA, and hardware details.

```bash
python easytpp_cli.py info
```

## ⚙️ Configuration

### Configuration File Structure

The CLI uses YAML configuration files with the following structure:

```yaml
base_config:
  seed: 42
  device: auto
  log_level: INFO

experiments:
  THP:
    model_config:
      model_name: THP
      hidden_size: 64
      num_layers: 4
      # ... more model parameters
      
    data_config:
      dataset_name: H2expc
      batch_size: 32
      # ... more data parameters
      
    training_config:
      optimizer: Adam
      learning_rate: 0.001
      # ... more training parameters

datasets:
  H2expc:
    path: ./data/h2expc
    type: point_process
    # ... dataset configuration
```

### CLI Configuration

Global CLI settings can be configured in `configs/cli_config.yaml`:

```yaml
cli:
  default_config_dir: ./configs
  default_output_dir: ./outputs
  use_rich: true
  color_theme: default

defaults:
  experiment_id: THP
  dataset_id: H2expc
  phase: test
```

## 🎨 Rich Terminal Features

When the `rich` package is installed, the CLI provides:

- **Colored Output**: Syntax highlighting and colored text
- **Progress Bars**: Real-time progress tracking
- **Tables**: Beautiful formatted tables for data display
- **Panels**: Organized information display
- **Interactive Prompts**: Enhanced user input

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   pip install rich typer
   ```

2. **Configuration Not Found**: Verify the config file path
   ```bash
   python easytpp_cli.py list-configs
   ```

3. **CUDA Issues**: Check device availability
   ```bash
   python easytpp_cli.py info
   ```

### Verbose Mode

Enable verbose logging for debugging:

```bash
python easytpp_cli.py --verbose run --config config.yaml --experiment THP --dataset H2expc
```

## 📄 Examples

### Example 1: Quick Training

```bash
# Start interactive mode for guided setup
python easytpp_cli.py interactive
```

### Example 2: Automated Pipeline

```bash
# Train a model
python easytpp_cli.py run -c configs/thp_config.yaml -e THP -d taxi -p train --output experiments/taxi_run_001

# Evaluate the trained model
python easytpp_cli.py run -c configs/thp_config.yaml -e THP -d taxi -p test --checkpoint experiments/taxi_run_001/checkpoints/best.ckpt
```

### Example 3: Configuration Management

```bash
# List all available configurations
python easytpp_cli.py list-configs

# Validate a specific configuration
python easytpp_cli.py validate -c configs/new_config.yaml -e MyExperiment -d MyDataset

# Run with validation
python easytpp_cli.py run -c configs/new_config.yaml -e MyExperiment -d MyDataset -p train
```

## 🤝 Contributing

To contribute to the CLI tool:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📧 Support

For support and questions:
- Check the documentation
- Open an issue on GitHub
- Contact the EasyTPP team

---

**EasyTPP CLI v2.0** - Making Temporal Point Process research accessible and professional.
"""
    
    with open("CLI_README.md", "w") as f:
        f.write(readme_content)
    
    print("  ✅ Created CLI_README.md")


def create_example_scripts():
    """Create example usage scripts"""
    print("📝 Creating example scripts...")
    
    # Create examples directory
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Basic training example
    basic_example = """#!/usr/bin/env python3
\"\"\"
Basic EasyTPP CLI Example
This script demonstrates basic usage of the EasyTPP CLI tool.
\"\"\"

import subprocess
import sys

def run_command(cmd):
    \"\"\"Run a command and print output\"\"\"
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)
    print("✅ Command completed successfully\\n")

def main():
    print("🚀 EasyTPP CLI Basic Example")
    print("=" * 40)
    
    # Step 1: Validate configuration
    print("Step 1: Validating configuration...")
    run_command("python ../easytpp_cli.py validate --config ../configs/runner_config_template.yaml --experiment THP --dataset H2expc")
    
    # Step 2: List available configurations
    print("Step 2: Listing available configurations...")
    run_command("python ../easytpp_cli.py list-configs --dir ../configs")
    
    # Step 3: Get system info
    print("Step 3: Getting system information...")
    run_command("python ../easytpp_cli.py info")
    
    # Step 4: Run a test experiment (if config exists)
    print("Step 4: Running test experiment...")
    run_command("python ../easytpp_cli.py run --config ../configs/runner_config_template.yaml --experiment THP --dataset H2expc --phase test")
    
    print("🎉 Example completed successfully!")

if __name__ == "__main__":
    main()
"""
    
    with open("examples/basic_example.py", "w") as f:
        f.write(basic_example)
    
    # Advanced pipeline example
    advanced_example = """#!/usr/bin/env python3
\"\"\"
Advanced EasyTPP CLI Pipeline Example
This script demonstrates an advanced ML pipeline using the EasyTPP CLI.
\"\"\"

import subprocess
import sys
import yaml
from pathlib import Path
import argparse

class EasyTPPPipeline:
    def __init__(self, config_path, experiment_id, dataset_id, output_dir):
        self.config_path = config_path
        self.experiment_id = experiment_id
        self.dataset_id = dataset_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_command(self, cmd):
        \"\"\"Run a command and handle errors\"\"\"
        print(f"🔧 Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            sys.exit(1)
        
        print("✅ Command completed successfully")
        return result.stdout
    
    def validate_config(self):
        \"\"\"Validate the configuration\"\"\"
        print("\\n📋 Step 1: Validating configuration...")
        cmd = f"python ../easytpp_cli.py validate --config {self.config_path} --experiment {self.experiment_id} --dataset {self.dataset_id}"
        self.run_command(cmd)
    
    def train_model(self):
        \"\"\"Train the model\"\"\"
        print("\\n🎯 Step 2: Training model...")
        cmd = f"python ../easytpp_cli.py run --config {self.config_path} --experiment {self.experiment_id} --dataset {self.dataset_id} --phase train --output {self.output_dir}/training"
        self.run_command(cmd)
    
    def evaluate_model(self):
        \"\"\"Evaluate the trained model\"\"\"
        print("\\n📊 Step 3: Evaluating model...")
        checkpoint_path = self.output_dir / "training" / "checkpoints" / "best.ckpt"
        cmd = f"python ../easytpp_cli.py run --config {self.config_path} --experiment {self.experiment_id} --dataset {self.dataset_id} --phase test --checkpoint {checkpoint_path} --output {self.output_dir}/evaluation"
        self.run_command(cmd)
    
    def generate_predictions(self):
        \"\"\"Generate predictions\"\"\"
        print("\\n🔮 Step 4: Generating predictions...")
        checkpoint_path = self.output_dir / "training" / "checkpoints" / "best.ckpt"
        cmd = f"python ../easytpp_cli.py run --config {self.config_path} --experiment {self.experiment_id} --dataset {self.dataset_id} --phase predict --checkpoint {checkpoint_path} --output {self.output_dir}/predictions"
        self.run_command(cmd)
    
    def run_full_pipeline(self):
        \"\"\"Run the complete pipeline\"\"\"
        print("🚀 Starting Advanced EasyTPP Pipeline")
        print("=" * 50)
        
        try:
            self.validate_config()
            self.train_model()
            self.evaluate_model()
            self.generate_predictions()
            
            print("\\n🎉 Pipeline completed successfully!")
            print(f"📁 Results saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Advanced EasyTPP CLI Pipeline")
    parser.add_argument("--config", default="../configs/runner_config_template.yaml", help="Config file path")
    parser.add_argument("--experiment", default="THP", help="Experiment ID")
    parser.add_argument("--dataset", default="H2expc", help="Dataset ID")
    parser.add_argument("--output", default="./pipeline_output", help="Output directory")
    
    args = parser.parse_args()
    
    pipeline = EasyTPPPipeline(
        config_path=args.config,
        experiment_id=args.experiment,
        dataset_id=args.dataset,
        output_dir=args.output
    )
    
    pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()
"""
    
    with open("examples/advanced_pipeline.py", "w") as f:
        f.write(advanced_example)
    
    print("  ✅ Created examples/basic_example.py")
    print("  ✅ Created examples/advanced_pipeline.py")


def main():
    """Main setup function"""
    print("🔧 EasyTPP CLI Setup")
    print("===================")
    
    try:
        install_dependencies()
        create_directories()
        create_config_templates()
        create_entry_point()
        create_readme()
        create_example_scripts()
        
        print("\n" + "=" * 50)
        print("✅ EasyTPP CLI Setup Complete!")
        print("=" * 50)
        
        print("\n📋 Next Steps:")
        print("1. Test the CLI: python easytpp_cli.py --help")
        print("2. Try interactive mode: python easytpp_cli.py interactive")
        print("3. Read the documentation: CLI_README.md")
        print("4. Run examples: cd examples && python basic_example.py")
        
        print("\n🎯 Quick Start:")
        print("python easytpp_cli.py run --config configs/runner_config_template.yaml --experiment THP --dataset H2expc --phase test")
        
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
