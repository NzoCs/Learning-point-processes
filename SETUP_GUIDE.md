# EasyTPP - Quick Setup Guide

This guide helps you quickly set up the EasyTPP project with the new `pyproject.toml` system.

## Prerequisites

- Python 3.8 or higher
- pip 21.3+ (for full pyproject.toml support)
- Git

## Quick Installation

### 1. Clone the project

```bash
git clone https://github.com/ant-research/EasyTemporalPointProcess.git
cd EasyTemporalPointProcess
```

### 2. Create a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install the project

```bash
# Full installation (recommended for development)
pip install -e ".[all]"

# Or minimal installation
pip install -e .
```

### 4. Verify installation

```bash
python check_installation.py
```

## Installation Options

Choose the installation that matches your needs:

```bash
# Basic installation (main dependencies only)
pip install -e .

# Development tools (tests, linting, formatting)
pip install -e ".[dev]"

# CLI tools (command line interfaces)
pip install -e ".[cli]"

# Documentation tools
pip install -e ".[docs]"

# Install everything
pip install -e ".[all]"
```

## Development Tools Configuration

The project includes preconfigured tools for development:

### Makefile (Unix/Linux/Windows with Git Bash)

The project uses a Makefile to automate common tasks. On Windows, you must have installed:

- `make`
- Git Bash Unix tools (to use make on windows too)

**Automatic setup on Windows:**

```bash
# Add make and Unix tools to PATH
$makePath = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe\bin"
$gitUnixPath = "C:\Program Files\Git\usr\bin"
$env:PATH += ";$makePath;$gitUnixPath"
```

**Available Makefile commands:**

```bash
make help          # Show help
make install-all   # Full installation
make check         # Installation verification
make test          # Run tests
make format        # Format code
make lint          # Check code
make clean         # Clean temporary files
make demo          # Quick demonstration
```

### Pre-commit hooks

```bash
# Install pre-commit hooks (after installing dev dependencies)
pre-commit install
```

### Available tools

- **black**: Automatic code formatting
- **isort**: Import organization
- **flake8**: Code quality checking
- **mypy**: Static type checking
- **pytest**: Testing with coverage

### Using the tools

```bash
# Format code
black .

# Organize imports
isort .

# Check code
flake8

# Check types
mypy easy_tpp

# Run tests
pytest
```

## Project Structure

```
EasyTemporalPointProcess/
├── pyproject.toml          # Main project configuration
├── check_installation.py   # Verification script
├── README.md              # Main documentation
├── easy_tpp/              # Main source code
├── examples/              # Usage examples
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## pyproject.toml Configuration

All project configuration is centralized in `pyproject.toml`:

- Build system configuration
- Dependencies and optional dependency groups
- Tool configuration (black, isort, pytest, mypy, etc.)
- Project metadata and URLs

## Dependency Groups

- **`cli`**: Rich terminal interfaces, command line tools
- **`docs`**: Sphinx documentation system, themes and extensions
- **`dev`**: Development workflow tools (tests, linting, formatting)
- **`all`**: Installs all optional dependencies

## Troubleshooting

### Python version error

```bash
# Check your Python version
python --version

# Recommended upgrade to Python 3.8+
```

### pip error

```bash
# Update pip
python -m pip install --upgrade pip

# Use python -m pip instead of pip directly
python -m pip install -e ".[all]"
```

## Support

If you encounter problems:

1. Check that Python 3.8+ is installed
2. Run `python check_installation.py` to diagnose
3. Consult the complete documentation in README.md
4. Create an issue on GitHub if the problem persists

---
