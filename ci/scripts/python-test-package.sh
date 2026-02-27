#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Test packages from packages/ directory
#
# Usage:
#   python-test-package.sh                           # Test all packages
#   python-test-package.sh <package>                 # Test one package
#   python-test-package.sh <pkg1> <pkg2> ...         # Test multiple packages
#   python-test-package.sh <package> -- <pytest-args>  # Test with extra pytest args
#   python-test-package.sh --list                    # List available packages

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common-python.sh"

# Handle --list flag
if [ "$1" = "--list" ]; then
    echo "Available packages:"
    list_packages | while read pkg; do
        echo "  - $pkg"
    done
    exit 0
fi

# Parse arguments: packages before --, pytest args after --
PACKAGES_TO_TEST=()
PYTEST_ARGS=()
PARSING_PACKAGES=true

for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        PARSING_PACKAGES=false
        continue
    fi
    
    if [ "$PARSING_PACKAGES" = true ]; then
        PACKAGES_TO_TEST+=("$arg")
    else
        PYTEST_ARGS+=("$arg")
    fi
done

# If no packages specified, test all
if [ ${#PACKAGES_TO_TEST[@]} -eq 0 ]; then
    while read pkg; do
        PACKAGES_TO_TEST+=("$pkg")
    done < <(list_packages)
fi

# Validate all packages exist
for pkg in "${PACKAGES_TO_TEST[@]}"; do
    if ! package_exists "$pkg"; then
        echo "Error: Package '$pkg' not found in packages/ directory" >&2
        echo ""
        echo "Available packages:"
        list_packages | while read p; do
            echo "  - $p"
        done
        exit 1
    fi
done

if [ ${#PACKAGES_TO_TEST[@]} -eq 0 ]; then
    echo "No packages found to test"
    exit 1
fi

# Test each package
TESTED_COUNT=0
for PACKAGE in "${PACKAGES_TO_TEST[@]}"; do
    # Install dependencies for each package
    echo "Installing dependencies for $PACKAGE..."
    install_python_deps "$PACKAGE" "dev"
    
    PACKAGE_PATH=$(get_package_path "$PACKAGE")
    TEST_DIR=$(get_test_dir "$PACKAGE")
    
    echo ""
    echo "=========================================="
    echo "Testing: $PACKAGE"
    echo "Package path: $PACKAGE_PATH"
    echo "Test directory: tests/$TEST_DIR"
    echo "Minimum coverage: $DEFAULT_MIN_COVERAGE%"
    echo "=========================================="
    
    run_tests_with_coverage "$PACKAGE" "$PACKAGE_PATH" "$TEST_DIR" "$DEFAULT_MIN_COVERAGE" "${PYTEST_ARGS[@]}" || {
        EXIT_CODE=$?
        echo ""
        echo "=========================================="
        echo "ERROR: Tests failed for $PACKAGE (exit code: ${EXIT_CODE})"
        echo "Stopping test run (tested $TESTED_COUNT of ${#PACKAGES_TO_TEST[@]} packages)"
        echo "=========================================="
        exit 1
    }
    TESTED_COUNT=$((TESTED_COUNT + 1))
done

echo ""
echo "=========================================="
if [ ${#PACKAGES_TO_TEST[@]} -gt 1 ]; then
    echo "Tested ${#PACKAGES_TO_TEST[@]} packages"
fi

echo "All tests passed!"
