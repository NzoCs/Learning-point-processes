# Scripts Directory

This directory contains all the CLI tools and utility scripts for the EasyTPP project.

## 📋 Available Scripts

### 🚀 Main CLI Tools

| Script | Description | Platform | Type |
|--------|-------------|----------|------|
| [`easytpp_cli.py`](easytpp_cli.py) | **Professional CLI** - Main command-line interface with rich features | Cross-platform | Python |
| [`easytpp_modern_cli.py`](easytpp_modern_cli.py) | **Modern CLI** - Alternative implementation using Typer framework | Cross-platform | Python |
| [`quick_start.py`](quick_start.py) | **Quick Start** - Simple script to get started quickly | Cross-platform | Python |

### 🔧 Platform-Specific Wrappers

| Script | Description | Platform | Type |
|--------|-------------|----------|------|
| [`easytpp.bat`](easytpp.bat) | **Windows Batch** - Simple batch wrapper for Windows users | Windows | Batch |
| [`easytpp.ps1`](easytpp.ps1) | **PowerShell** - Professional PowerShell interface with colored output | Windows | PowerShell |

### ⚙️ Setup and Utilities

| Script | Description | Platform | Type |
|--------|-------------|----------|------|
| [`setup_cli.py`](setup_cli.py) | **CLI Setup** - Installation and configuration script | Cross-platform | Python |
| [`test_cli.py`](test_cli.py) | **CLI Tests** - Test suite for CLI functionality | Cross-platform | Python |

## 🎯 Quick Usage Examples

### Professional CLI (Recommended)

```bash
# Interactive mode (best for beginners)
python scripts/easytpp_cli.py interactive

# Direct execution
python scripts/easytpp_cli.py run --config configs/runner_config.yaml --experiment THP --dataset H2expc --phase test

# List available configurations
python scripts/easytpp_cli.py list-configs --dir configs

# Validate configuration
python scripts/easytpp_cli.py validate --config configs/runner_config.yaml --experiment THP --dataset H2expc

# System information
python scripts/easytpp_cli.py info
```

### Modern CLI (Typer-based)

```bash
# Similar commands with modern syntax
python scripts/easytpp_modern_cli.py run --config configs/runner_config.yaml --experiment THP --dataset H2expc

# Interactive mode
python scripts/easytpp_modern_cli.py interactive
```

### Windows Users

**PowerShell (Recommended for Windows):**
```powershell
# Navigate to project root first
./scripts/easytpp.ps1 interactive
./scripts/easytpp.ps1 run --config configs/runner_config.yaml --experiment THP --dataset H2expc
```

**Batch:**
```cmd
scripts\easytpp.bat interactive
scripts\easytpp.bat run --config configs/runner_config.yaml --experiment THP --dataset H2expc
```

### Quick Start

```bash
# Simple execution with minimal configuration
python scripts/quick_start.py --config_dir configs/experiment_config.yaml --experiment_id NHP_train
```

## 📚 Documentation

- **[CLI_PROFESSIONAL_README.md](CLI_PROFESSIONAL_README.md)** - Comprehensive CLI documentation
- **[Main README.md](../README.md)** - Project overview and general documentation

## 🔧 Setup

1. **Install CLI dependencies:**
   ```bash
   python scripts/setup_cli.py
   ```

2. **Test the installation:**
   ```bash
   python scripts/test_cli.py
   ```

3. **Start using the CLI:**
   ```bash
   python scripts/easytpp_cli.py --help
   ```

## ⚡ Features

### Professional CLI Features
- 🎨 **Rich Terminal Output** - Colored tables, progress bars, and styled text
- 🎯 **Interactive Mode** - Guided step-by-step configuration
- 📁 **Configuration Management** - List, validate, and manage configs
- 🛡️ **Error Handling** - Robust error reporting and recovery
- 📊 **Logging** - Professional logging with rotation and retention
- 🔄 **Multi-Platform** - Works on Windows, macOS, and Linux

### Modern CLI Features
- 🚀 **Typer Framework** - Modern CLI with automatic help generation
- 📝 **Type Safety** - Full type hints and validation
- 🎨 **Rich Integration** - Beautiful terminal output
- 🔧 **Extensible** - Easy to add new commands

## 🛠️ Development

### Adding New Scripts

1. Create your script in this directory
2. Update this README with script information
3. Add appropriate documentation
4. Test the script with `test_cli.py`

### Script Conventions

- Use `#!/usr/bin/env python3` shebang for Python scripts
- Include proper error handling and logging
- Provide `--help` documentation
- Follow the existing code style

## 🐛 Troubleshooting

### Common Issues

1. **Python not found**: Make sure Python 3.8+ is installed and in PATH
2. **Import errors**: Install dependencies with `pip install -r requirements.txt`
3. **Permission errors**: On Unix systems, make scripts executable with `chmod +x script.py`

### Getting Help

- Use `--help` flag with any script for usage information
- Check the [CLI_PROFESSIONAL_README.md](CLI_PROFESSIONAL_README.md) for detailed documentation
- Run `python scripts/easytpp_cli.py info` to check system configuration

## 📞 Support

For issues with these scripts:
1. Check the documentation in this directory
2. Run the test script: `python scripts/test_cli.py`
3. Open an issue in the project repository
4. Check the main project README for additional help
