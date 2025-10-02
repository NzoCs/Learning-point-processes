
# EasyTPP - Quick Setup Guide

This guide helps you quickly set up the EasyTPP project using `uv` - the fast Python package manager.

## Prerequisites

- **uv** (replaces pip and venv) - [Install from here](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.8 or higher (uv can install this automatically)
- Git

## Quick Installation with uv

### 1. Clone the project

```bash
git clone https://github.com/ant-research/EasyTemporalPointProcess.git
cd EasyTemporalPointProcess
```

### 2. Setup project with uv (automatically installs in editable mode)

```bash
# Initialize and install dependencies in one command
# This automatically installs the project in editable mode
uv sync

# Or if you want to specify Python version
uv python install 3.11  # Install specific Python version
uv sync --python 3.11   # Use specific Python version
```

### 3. Verify installation

```bash
# Run with uv (automatically uses project environment)
uv run python check_installation.py
```

## Installation Options with uv

Choose the dependency groups that match your needs:

```bash
# Basic installation (main dependencies only)
uv sync

# Include development tools (tests, linting, formatting)
uv sync --group dev

# Include CLI tools (command line interfaces)
uv sync --group cli

# Include documentation tools
uv sync --group docs

# Install all optional dependencies
uv sync --all-extras
```

Note: With `uv sync`, the project is automatically installed in editable mode, so any changes to the source code are immediately reflected.

## Working with uv

### Running commands

```bash
# Run Python scripts (automatically uses project environment)
uv run python your_script.py

# Run pytest
uv run pytest

# Run any command in the project environment
uv run black .
uv run isort .
uv run flake8
```

### Adding new dependencies

```bash
# Add a new dependency
uv add numpy pandas

# Add development dependencies
uv add --dev pytest black isort

# Add optional dependencies to a group
uv add --optional cli rich typer
```

### Environment management

```bash
# Activate shell with project environment
uv shell

# Show project info
uv show

# Update dependencies
uv sync --upgrade
```

## Development Tools Configuration

The project includes preconfigured tools for development, now optimized for `uv`:

### Using uv for development tasks

```bash
# Install development dependencies
uv sync --group dev

# Run development tools with uv
uv run black .          # Format code
uv run isort .          # Organize imports
uv run flake8           # Check code quality
uv run mypy easy_tpp    # Static type checking
uv run pytest          # Run tests with coverage
```

### Makefile (Unix/Linux/Windows with Git Bash)

The project still supports Makefile commands, but they now use `uv`:

**Available Makefile commands:**

```bash
make help          # Show help
make install-all   # Full installation with uv
make check         # Installation verification
make test          # Run tests with uv
make format        # Format code with uv
make lint          # Check code with uv
make clean         # Clean temporary files
make demo          # Quick demonstration
```

### Pre-commit hooks

```bash
# Install pre-commit hooks (uv will handle the environment)
uv run pre-commit install
```

### Available tools

- **black**: Automatic code formatting
- **isort**: Import organization
- **flake8**: Code quality checking
- **mypy**: Static type checking
- **pytest**: Testing with coverage
- **pre-commit**: Git hooks for code quality

### Development workflow with uv

```bash
# One-time setup
uv sync --group dev

# Daily development workflow
uv run black .          # Format your code
uv run isort .          # Sort imports
uv run flake8           # Check code style
uv run mypy easy_tpp    # Check types
uv run pytest          # Run tests

# Add new dependencies as you work
uv add requests         # Add runtime dependency
uv add --dev pytest-mock # Add development dependency
```

## Project Structure

```bash
EasyTemporalPointProcess/
├── pyproject.toml          # Main project configuration
├── uv.lock                 # Lockfile for reproducible builds
├── check_installation.py   # Verification script
├── README.md              # Main documentation
├── easy_tpp/              # Main source code
├── examples/              # Usage examples
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## uv and pyproject.toml Configuration

All project configuration is centralized in `pyproject.toml`, and `uv` uses this for:

- Build system configuration
- Dependencies and optional dependency groups
- Tool configuration (black, isort, pytest, mypy, etc.)
- Project metadata and URLs

`uv` creates a `uv.lock` file for reproducible installations across different environments.

## Dependency Groups

- **`cli`**: Rich terminal interfaces, command line tools
- **`docs`**: Sphinx documentation system, themes and extensions
- **`dev`**: Development workflow tools (tests, linting, formatting)
- **`all`**: Installs all optional dependencies

## Why uv?

`uv` is significantly faster than pip and provides:

- **Fast installations**: 10-100x faster than pip
- **Better dependency resolution**: More reliable than pip
- **Built-in virtual environment management**: No need for separate venv
- **Lock files**: Reproducible builds with `uv.lock`
- **Project management**: Commands like `uv add`, `uv remove`, `uv sync`
- **Python version management**: Can install and switch Python versions
- **Automatic editable installs**: Projects are installed in editable mode by default

## Troubleshooting

### uv not found

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix/macOS
# or
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

### Python version error

```bash
# Let uv install Python for you
uv python install 3.11

# Use specific Python version
uv sync --python 3.11
```

### Dependency conflicts

```bash
# Reset and reinstall everything
rm -rf .venv uv.lock  # Unix/macOS
rmdir /s .venv; del uv.lock  # Windows
uv sync
```

### Migration from pip/venv

```bash
# Remove old virtual environment
rm -rf .venv  # Unix/macOS
rmdir /s .venv  # Windows

# Initialize with uv
uv sync

# If you have requirements.txt
uv add -r requirements.txt
```

## Support

If you encounter problems:

1. Make sure `uv` is installed and up to date: `uv --version`
2. Run `uv run python check_installation.py` to diagnose
3. Try `uv sync --reinstall` for a fresh installation
4. Consult the complete documentation in README.md
5. Check [uv documentation](https://docs.astral.sh/uv/) for uv-specific issues
6. Create an issue on GitHub if the problem persists

---
