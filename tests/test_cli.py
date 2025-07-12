#!/usr/bin/env python3
"""
Test Script for EasyTPP CLI

This script tests all main CLI functionalities
to ensure they work correctly.
"""

import subprocess
import sys
import os
from pathlib import Path
import tempfile
import yaml


class CLITester:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.cli_script = self.project_root / "easytpp_cli.py"
        self.config_dir = self.project_root / "configs"
        self.test_passed = 0
        self.test_failed = 0

    def run_command(self, command, expect_success=True):
        """Execute a command and verify the result"""
        try:
            print(f"+ Testing: {command}")
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )

            if expect_success:
                if result.returncode == 0:
                    print("+ PASSED")
                    self.test_passed += 1
                    return True
                else:
                    print(f"- FAILED: {result.stderr}")
                    self.test_failed += 1
                    return False
            else:
                if result.returncode != 0:
                    print("+ PASSED (Expected failure)")
                    self.test_passed += 1
                    return True
                else:
                    print("- FAILED: Command should have failed")
                    self.test_failed += 1
                    return False

        except subprocess.TimeoutExpired:
            print("- FAILED: Command timed out")
            self.test_failed += 1
            return False
        except Exception as e:
            print(f"- FAILED: Exception {str(e)}")
            self.test_failed += 1
            return False

    def test_basic_commands(self):
        """Test basic commands"""
        print("\n== Testing Basic Commands ==")
        print("=" * 40)

        # Test help
        self.run_command(f"python {self.cli_script} --help")

        # Test version
        self.run_command(f"python {self.cli_script} --version")

        # Test info command
        self.run_command(f"python {self.cli_script} info")

    def test_config_commands(self):
        """Test configuration commands"""
        print("\n== Testing Configuration Commands ==")
        print("=" * 40)

        # Test list-configs
        self.run_command(
            f"python {self.cli_script} list-configs --dir {self.config_dir}"
        )

        # Test list-configs with non-existent directory (should fail)
        self.run_command(
            f"python {self.cli_script} list-configs --dir /non/existent/dir",
            expect_success=False,
        )

    def test_validation(self):
        """Test configuration validation"""
        print("\n== Testing Configuration Validation ==")
        print("=" * 40)

        # Create a minimal test config
        test_config = {
            "base_config": {"seed": 42, "device": "auto"},
            "experiments": {
                "THP": {
                    "model_config": {"model_name": "THP", "hidden_size": 64},
                    "data_config": {"dataset_name": "H2expc", "batch_size": 32},
                    "training_config": {"optimizer": "Adam", "learning_rate": 0.001},
                }
            },
            "datasets": {"H2expc": {"path": "./data/h2expc", "type": "point_process"}},
        }

        # Write test config
        test_config_path = self.config_dir / "test_config.yaml"
        with open(test_config_path, "w") as f:
            yaml.dump(test_config, f)

        try:
            # Test validation with valid config
            self.run_command(
                f"python {self.cli_script} validate --config {test_config_path} --experiment THP --dataset H2expc"
            )

            # Test validation with invalid experiment (should fail)
            self.run_command(
                f"python {self.cli_script} validate --config {test_config_path} --experiment INVALID --dataset H2expc",
                expect_success=False,
            )

            # Test validation with non-existent config (should fail)
            self.run_command(
                f"python {self.cli_script} validate --config /non/existent/config.yaml --experiment THP --dataset H2expc",
                expect_success=False,
            )

        finally:
            # Clean up
            if test_config_path.exists():
                test_config_path.unlink()

    def test_error_handling(self):
        """Test error handling"""
        print("\n== Testing Error Handling ==")
        print("=" * 40)

        # Test missing required arguments
        self.run_command(f"python {self.cli_script} run", expect_success=False)

        # Test invalid phase
        self.run_command(
            f"python {self.cli_script} run --config test.yaml --experiment THP --dataset H2expc --phase invalid",
            expect_success=False,
        )

        # Test non-existent config file
        self.run_command(
            f"python {self.cli_script} run --config /non/existent.yaml --experiment THP --dataset H2expc",
            expect_success=False,
        )

    def test_verbose_mode(self):
        """Test verbose mode"""
        print("\n== Testing Verbose Mode ==")
        print("=" * 40)

        # Test verbose flag
        self.run_command(f"python {self.cli_script} --verbose info")

    def test_modern_cli(self):
        """Test modern CLI (Typer)"""
        print("\n== Testing Modern CLI (Typer) ==")
        print("=" * 40)

        modern_cli_script = self.project_root / "easytpp_modern_cli.py"

        if modern_cli_script.exists():
            # Test basic commands
            self.run_command(f"python {modern_cli_script} --help")
            self.run_command(f"python {modern_cli_script} --version")

            # Test info command
            self.run_command(f"python {modern_cli_script} info")

            # Test list-configs
            self.run_command(
                f"python {modern_cli_script} list-configs --dir {self.config_dir}"
            )
        else:
            print("! Modern CLI script not found, skipping tests")

    def test_installation_scripts(self):
        """Test installation scripts"""
        print("\n== Testing Installation Scripts ==")
        print("=" * 40)

        setup_script = self.project_root / "setup_cli.py"

        if setup_script.exists():
            print("+ Setup script found")
            self.test_passed += 1
        else:
            print("- Setup script not found")
            self.test_failed += 1

        # Test wrapper scripts
        wrappers = ["easytpp.bat", "easytpp.ps1"]
        for wrapper in wrappers:
            wrapper_path = self.project_root / wrapper
            if wrapper_path.exists():
                print(f"+ {wrapper} found")
                self.test_passed += 1
            else:
                print(f"- {wrapper} not found")
                self.test_failed += 1

    def test_documentation(self):
        """Test documentation"""
        print("\n== Testing Documentation ==")
        print("=" * 40)

        docs = ["CLI_PROFESSIONAL_README.md", "CLI_README.md"]

        for doc in docs:
            doc_path = self.project_root / doc
            if doc_path.exists():
                print(f"+ {doc} found")
                self.test_passed += 1
            else:
                print(f"- {doc} not found")
                self.test_failed += 1

    def run_all_tests(self):
        """Execute all tests"""
        print("EasyTPP CLI Test Suite")
        print("=" * 50)

        # Check if CLI script exists
        if not self.cli_script.exists():
            print(f"- CLI script not found: {self.cli_script}")
            return False

        # Execute all tests
        self.test_basic_commands()
        self.test_config_commands()
        self.test_validation()
        self.test_error_handling()
        self.test_verbose_mode()
        self.test_modern_cli()
        self.test_installation_scripts()
        self.test_documentation()

        # Display summary
        print("\n" + "=" * 50)
        print("Test Results Summary")
        print("=" * 50)
        print(f"+ Tests Passed: {self.test_passed}")
        print(f"- Tests Failed: {self.test_failed}")
        print(
            f"Success Rate: {self.test_passed / (self.test_passed + self.test_failed) * 100:.1f}%"
        )

        if self.test_failed == 0:
            print("\nAll tests passed! CLI is working correctly.")
            return True
        else:
            print(f"\n{self.test_failed} tests failed. Please check the issues above.")
            return False


def main():
    """Main function"""
    tester = CLITester()
    success = tester.run_all_tests()

    if success:
        print("\nCLI is ready for use!")
        print("\nQuick start commands:")
        print("  python easytpp_cli.py --help")
        print("  python easytpp_cli.py interactive")
        print("  python easytpp_cli.py info")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please fix the issues before using the CLI.")
        sys.exit(1)


if __name__ == "__main__":
    main()
