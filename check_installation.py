from typing import Optional

#!/usr/bin/env python3
"""
Quick installation verification script for EasyTPP.

This script checks if the installation was successful and all core dependencies are available.
Run this after installation to verify everything is working correctly.
"""

import sys
import importlib.util
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
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} (not found)")
        return False


def check_installation() -> Tuple[int, int]:
    """Check the installation status of EasyTPP and its dependencies."""
    print("🔍 Checking EasyTPP Installation\n")
    
    # Check Python version
    python_ok = check_python_version()
    print()
    
    # Core dependencies
    print("📦 Checking Core Dependencies:")
    core_deps = [
        ("easy_tpp", "easy_tpp"),
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
    
    # Optional dependencies
    print("\n🔧 Checking Optional Dependencies:")
    optional_deps = [
        ("Development tools", [
            ("pytest", "pytest"),
            ("black", "black"),
            ("flake8", "flake8"),
            ("isort", "isort"),
            ("mypy", "mypy"),
        ]),
        ("CLI tools", [
            ("rich", "rich"),
            ("typer", "typer"),
            ("click", "click"),
        ]),
        ("Documentation", [
            ("sphinx", "sphinx"),
            ("myst_parser", "myst_parser"),
        ]),
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
    print("\n" + "="*50)
    if python_ok and core_success == len(core_deps):
        print("🎉 Installation successful! EasyTPP is ready to use.")
        print("\nNext steps:")
        print("1. Check out the examples/ directory")
        print("2. Read the documentation")
        print("3. Try running a simple model")
    else:
        print("⚠️  Installation incomplete.")
        if not python_ok:
            print("- Upgrade to Python 3.8 or higher")
        if core_success < len(core_deps):
            print("- Install missing core dependencies with: pip install -e .")
    
    if optional_success < optional_total:
        print(f"\n💡 To install optional dependencies:")
        print("   pip install -e \".[dev]\"    # Development tools")
        print("   pip install -e \".[cli]\"    # CLI tools")
        print("   pip install -e \".[docs]\"   # Documentation")
        print("   pip install -e \".[all]\"    # All optional dependencies")
    
    return core_success, len(core_deps)


if __name__ == "__main__":
    try:
        success, total = check_installation()
        sys.exit(0 if success == total else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Installation check interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during installation check: {e}")
        sys.exit(1)
