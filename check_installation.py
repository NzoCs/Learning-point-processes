from typing import Optional

#!/usr/bin/env python3
"""
Quick installation verification script for New_LTPP.

This script checks if the installation was successful and all core dependencies are available.
Run this after installation to verify everything is working correctly.
"""

import importlib.util
import sys
from typing import List, Tuple


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print(f"✅ Python {major}.{minor} is compatible")
        return True
    else:
        print(f"❌ Python {major}.{minor} is not compatible (requires Python 3.8+)")
        return False


def check_package(package_name: str, import_name: Optional[str] = None) -> bool:
    """Check if a package is installed and importable."""
    import_name = import_name or package_name
    try:
        module = importlib.import_module(import_name)
        # Try to get version if available
        version = getattr(module, '__version__', 'installed')
        print(f"✅ {package_name} ({version})")
        return True
    except ImportError:
        print(f"❌ {package_name} (not found)")
        return False


def check_installation() -> Tuple[int, int, int, int]:
    """Check the installation status of new_ltpp and its dependencies."""
    print("🔍 Checking new_ltpp Installation\n")

    # Check Python version
    python_ok = check_python_version()
    print()

    # Core dependencies
    print("📦 Checking Core Dependencies:")
    core_deps = [
        ("new_ltpp", "new_ltpp"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("pytorch-lightning", "pytorch_lightning"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
    ]

    core_success = 0
    for package, import_name in core_deps:
        if check_package(package, import_name):
            core_success += 1

    print(f"\nCore dependencies: {core_success}/{len(core_deps)} installed")
    
    # Test specific new_ltpp modules
    print("\n🧪 Testing new_ltpp Modules:")
    new_ltpp_modules = [
        ("config_factory", "new_ltpp.config_factory"),
        ("models", "new_ltpp.models"),
        ("data", "new_ltpp.data"),
        ("runner", "new_ltpp.runner"),
        ("evaluation", "new_ltpp.evaluation"),
        ("utils", "new_ltpp.utils"),
        ("hpo", "new_ltpp.hpo"),
    ]
    
    new_ltpp_success = 0
    for module_name, import_name in new_ltpp_modules:
        if check_package(f"  {module_name}", import_name):
            new_ltpp_success += 1
    
    print(f"\nnew_ltpp modules: {new_ltpp_success}/{len(new_ltpp_modules)} importable")

    # Optional dependencies
    print("\n🔧 Checking Optional Dependencies:")
    optional_deps = [
        (
            "Development tools",
            [
                ("pytest", "pytest"),
                ("black", "black"),
                ("flake8", "flake8"),
                ("isort", "isort"),
                ("mypy", "mypy"),
                ("pre-commit", "pre_commit"),
                ("pytest-cov", "pytest_cov"),
            ],
        ),
        (
            "CLI tools",
            [
                ("rich", "rich"),
                ("typer", "typer"),
                ("click", "click"),
                ("colorama", "colorama"),
                ("questionary", "questionary"),
                ("tabulate", "tabulate"),
            ],
        ),
        (
            "Documentation",
            [
                ("sphinx", "sphinx"),
                ("myst-parser", "myst_parser"),
                ("sphinx-rtd-theme", "sphinx_rtd_theme"),
                ("nbsphinx", "nbsphinx"),
            ],
        ),
        (
            "HPO tools",
            [
                ("optuna", "optuna"),
            ],
        ),
    ]

    optional_success = 0
    optional_total = 0

    for category, deps in optional_deps:
        print(f"\n  {category}:")
        category_success = 0
        for package, import_name in deps:
            if check_package(f"    {package}", import_name):
                category_success += 1
                optional_success += 1
            optional_total += 1
        print(f"    → {category_success}/{len(deps)} installed")

    print(f"\nOptional dependencies: {optional_success}/{optional_total} installed")

    # Summary
    print("\n" + "=" * 50)
    if python_ok and core_success == len(core_deps) and new_ltpp_success >= 6:  # Allow hpo to fail
        print("🎉 Installation successful! new_ltpp is ready to use.")
        print("\nNext steps:")
        print("1. Check out the examples/ directory")
        print("2. Read the documentation")
        print("3. Try running: make demo")
        print("4. Or start with: make quick-start")
        if new_ltpp_success < len(new_ltpp_modules):
            print("\n⚠️  Note: Some optional modules (like HPO) may need additional dependencies.")
    else:
        print("⚠️  Installation incomplete.")
        if not python_ok:
            print("- Upgrade to Python 3.8 or higher")
        if core_success < len(core_deps):
            print("- Install missing core dependencies with: make uv-sync")

    if optional_success < optional_total:
        print(f"\n💡 To install optional dependencies with uv:")
        print('   make install-dev     # Development tools')
        print('   make install-cli     # CLI tools')
        print('   make install-docs    # Documentation')
        print('   make install-all     # All optional dependencies')
        print("\n   Or manually with uv:")
        print('   uv sync --group dev      # Development tools')
        print('   uv sync --group cli      # CLI tools')
        print('   uv sync --group docs     # Documentation')
        print('   uv sync --all-groups     # All groups')

    return core_success, len(core_deps), new_ltpp_success, len(new_ltpp_modules)


if __name__ == "__main__":
    try:
        core_success, core_total, new_ltpp_success, new_ltpp_total = check_installation()
        # Consider installation successful if core deps are OK and most new_ltpp modules work
        success = (core_success == core_total) and (new_ltpp_success >= new_ltpp_total - 1)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Installation check interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during installation check: {e}")
        sys.exit(1)
