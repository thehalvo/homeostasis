#!/usr/bin/env python3
"""
Identify slow or hanging tests in the test suite.
Run this to find which tests might be causing timeouts on GitHub Actions.
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Timeout for individual tests (seconds)
TEST_TIMEOUT = 30

def run_single_test(test_path, timeout=TEST_TIMEOUT):
    """Run a single test with timeout."""
    print(f"\nRunning: {test_path}")
    start_time = time.time()

    env = os.environ.copy()
    env['USE_MOCK_TESTS'] = 'true'
    env['DISABLE_PERFORMANCE_TRACKING'] = 'true'

    try:
        # Run the test with a timeout
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-xvs', '--tb=short', str(test_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ PASSED in {duration:.2f}s")
            return True, duration
        else:
            print(f"✗ FAILED in {duration:.2f}s")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}...")
            return False, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⚠ TIMEOUT after {duration:.2f}s - This test might be hanging!")
        # Kill any remaining processes
        try:
            subprocess.run(['pkill', '-f', str(test_path)], capture_output=True)
        except:
            pass
        return False, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"✗ ERROR: {str(e)} after {duration:.2f}s")
        return False, duration


def main():
    """Find all test files and run them individually to identify slow/hanging tests."""
    print("Identifying slow or hanging tests...")
    print(f"Using timeout of {TEST_TIMEOUT} seconds per test")
    print("-" * 60)

    # Find all test files
    test_dir = Path('tests')
    test_files = list(test_dir.rglob('test_*.py'))

    # Skip the known problematic test
    test_files = [f for f in test_files if 'test_concurrent_error_processing_performance' not in str(f)]

    print(f"Found {len(test_files)} test files to check")

    slow_tests = []
    failed_tests = []
    hanging_tests = []

    for i, test_file in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] ", end='')
        success, duration = run_single_test(test_file)

        if not success and duration >= TEST_TIMEOUT - 1:
            hanging_tests.append((test_file, duration))
        elif not success:
            failed_tests.append((test_file, duration))
        elif duration > 10:  # Tests taking more than 10 seconds
            slow_tests.append((test_file, duration))

    # Report results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if hanging_tests:
        print(f"\n⚠ HANGING/TIMEOUT TESTS ({len(hanging_tests)}):")
        for test, duration in sorted(hanging_tests, key=lambda x: x[1], reverse=True):
            print(f"  - {test}: {duration:.2f}s")

    if slow_tests:
        print(f"\n🐌 SLOW TESTS ({len(slow_tests)}):")
        for test, duration in sorted(slow_tests, key=lambda x: x[1], reverse=True):
            print(f"  - {test}: {duration:.2f}s")

    if failed_tests:
        print(f"\n✗ FAILED TESTS ({len(failed_tests)}):")
        for test, duration in sorted(failed_tests, key=lambda x: x[1], reverse=True):
            print(f"  - {test}: {duration:.2f}s")

    if not (hanging_tests or slow_tests or failed_tests):
        print("\n✓ All tests completed successfully within reasonable time!")

    print("\n" + "-" * 60)
    print("Next steps:")
    if hanging_tests:
        print("1. Fix or skip the hanging tests listed above")
        print("2. These tests are likely causing the GitHub Actions timeout")
    if slow_tests:
        print("3. Consider optimizing slow tests or marking them with @pytest.mark.slow")

    return len(hanging_tests) + len(failed_tests)


if __name__ == '__main__':
    sys.exit(main())