#!/usr/bin/env python3
"""
Test runner script for KoboldAlignX v2.0

Provides convenient commands for running different test suites.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: list[str]) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main test runner."""
    if len(sys.argv) < 2:
        print("""
KoboldAlignX Test Runner
========================

Usage: python run_tests.py [command]

Commands:
  all          - Run all tests
  unit         - Run unit tests only
  integration  - Run integration tests only
  fast         - Run fast tests (exclude slow tests)
  coverage     - Run tests with detailed coverage report
  specific     - Run specific test file (requires file path)
  watch        - Run tests in watch mode (re-run on changes)
  
Examples:
  python run_tests.py all
  python run_tests.py unit
  python run_tests.py specific tests/unit/test_email_service.py
        """)
        return 1
    
    command = sys.argv[1]
    
    # Base pytest command
    base_cmd = ["pytest"]
    
    if command == "all":
        # Run all tests
        return run_command(base_cmd + ["tests/"])
    
    elif command == "unit":
        # Run unit tests only
        return run_command(base_cmd + ["tests/unit/", "-m", "unit or not integration"])
    
    elif command == "integration":
        # Run integration tests only
        return run_command(base_cmd + ["tests/integration/", "-m", "integration"])
    
    elif command == "fast":
        # Run fast tests (exclude slow)
        return run_command(base_cmd + ["tests/", "-m", "not slow"])
    
    elif command == "coverage":
        # Run with detailed coverage
        return run_command(base_cmd + [
            "tests/",
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    elif command == "specific":
        # Run specific test file
        if len(sys.argv) < 3:
            print("Error: Please provide test file path")
            print("Example: python run_tests.py specific tests/unit/test_email_service.py")
            return 1
        test_file = sys.argv[2]
        return run_command(base_cmd + [test_file])
    
    elif command == "watch":
        # Watch mode (requires pytest-watch)
        try:
            return run_command(["ptw", "--", "tests/"])
        except FileNotFoundError:
            print("Error: pytest-watch not installed")
            print("Install with: pip install pytest-watch")
            return 1
    
    elif command == "parallel":
        # Run tests in parallel (requires pytest-xdist)
        try:
            return run_command(base_cmd + ["tests/", "-n", "auto"])
        except:
            print("Error: pytest-xdist not installed")
            print("Install with: pip install pytest-xdist")
            return 1
    
    else:
        print(f"Unknown command: {command}")
        print("Run 'python run_tests.py' for help")
        return 1


if __name__ == "__main__":
    sys.exit(main())