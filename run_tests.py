#!/usr/bin/env python3
"""
One-click test runner for LLM Factory.
"""

import os
import sys
import pytest
import argparse
from typing import List, Optional
from termcolor import colored


def run_tests(test_paths: Optional[List[str]] = None, verbose: bool = False, coverage: bool = True) -> bool:
    """
    Run pytest with specified test paths.
    
    Args:
        test_paths: List of test paths to run, or None to run all tests
        verbose: Whether to show verbose output
        coverage: Whether to generate coverage reports
    
    Returns:
        bool: True if all tests passed
    """
    pytest_args = []
    
    # Add verbosity flag
    if verbose:
        pytest_args.append("-v")
    
    # Add test paths or use default
    if test_paths:
        pytest_args.extend(test_paths)
    else:
        pytest_args.append("tests/")
    
    # Add coverage reporting if requested
    if coverage:
        try:
            import pytest_cov
            pytest_args.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:coverage_report"
            ])
        except ImportError:
            print(colored("Warning: pytest-cov not installed. Coverage report will be skipped.", "yellow"))
    
    try:
        return pytest.main(pytest_args) == 0
    except Exception as e:
        print(colored(f"Error running tests: {e}", "red"))
        return False


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(colored(text.center(80), "cyan", attrs=["bold"]))
    print("=" * 80 + "\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run LLM Factory tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--test", "-t", nargs="+", help="Specific test files or directories to run")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    args = parser.parse_args()
    
    print_header("LLM Factory Test Suite")
    
    # Ensure we're running from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Add project root to Python path
    sys.path.insert(0, project_root)
    
    # Run tests
    success = run_tests(
        test_paths=args.test,
        verbose=args.verbose,
        coverage=not args.no_coverage
    )
    
    if success:
        print("\n" + colored("✓ All tests passed!", "green", attrs=["bold"]))
        sys.exit(0)
    else:
        print("\n" + colored("✗ Some tests failed!", "red", attrs=["bold"]))
        sys.exit(1)


if __name__ == "__main__":
    main() 