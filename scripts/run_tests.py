#!/usr/bin/env python
"""
Run all tests and generate coverage report.
"""

import sys
import os
import subprocess


def run_tests():
    """Run all tests with coverage."""
    print("=" * 60)
    print("Running Retail Price Optimization Test Suite")
    print("=" * 60)

    # Run pytest with coverage
    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        'tests/',
        '-v',
        '--cov=.',
        '--cov-report=term-missing',
        '--cov-report=html'
    ])

    return result.returncode


def main():
    """Main entry point."""
    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Run tests
    test_result = run_tests()

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests: {'PASSED' if test_result == 0 else 'FAILED'}")

    sys.exit(test_result)


if __name__ == '__main__':
    main()
