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


def run_tests(test_paths: Optional[List[str]] = None, verbose: bool = False) -> bool:
    """
    Run pytest with specified test paths.
    
    Args:
        test_paths: List of test paths to run, or None to run all tests
        verbose: Whether to show verbose output
    
    Returns:
        bool: True if all tests passed
    """
    args = ["-v"] if verbose else []
    
    if test_paths:
        args.extend(test_paths)
    else:
        args.append("tests/")
    
    # Add coverage reporting
    args.extend([
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_report"
    ])
    
    return pytest.main(args) == 0


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
    args = parser.parse_args()
    
    print_header("LLM Factory Test Suite")
    
    # Ensure we're running from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Add project root to Python path
    sys.path.insert(0, project_root)
    
    # Run tests
    success = run_tests(args.test, args.verbose)
    
    if success:
        print("\n" + colored("✓ All tests passed!", "green", attrs=["bold"]))
        sys.exit(0)
    else:
        print("\n" + colored("✗ Some tests failed!", "red", attrs=["bold"]))
        sys.exit(1)


if __name__ == "__main__":
    main() 