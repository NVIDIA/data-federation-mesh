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

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Default minimum coverage for all packages
DEFAULT_MIN_COVERAGE="${DFM_CI_MINIMUM_PYTHON_COVERAGE:-60}"

# Auto-discover packages from packages/ directory
# Returns package names (directory names like nv-dfm-core)
function list_packages {
    for dir in "$REPO_ROOT"/packages/*/; do
        if [ -f "$dir/pyproject.toml" ]; then
            basename "$dir"
        fi
    done
}

# Get package source path for coverage
# Usage: get_package_path <package_name>
# Example: get_package_path nv-dfm-core -> packages/nv-dfm-core/nv_dfm_core
function get_package_path {
    local PACKAGE_NAME="$1"
    # Convert package name to module name (nv-dfm-core -> nv_dfm_core)
    local MODULE_NAME="${PACKAGE_NAME//-/_}"
    echo "packages/${PACKAGE_NAME}/${MODULE_NAME}"
}

# Get test directory for a package
# Usage: get_test_dir <package_name>
# Example: get_test_dir nv-dfm-core -> unit/nv_dfm_core
function get_test_dir {
    local PACKAGE_NAME="$1"
    # Convert package name to module name (nv-dfm-core -> nv_dfm_core)
    local MODULE_NAME="${PACKAGE_NAME//-/_}"
    echo "unit/${MODULE_NAME}"
}

# Check if a package exists
function package_exists {
    local PACKAGE_NAME="$1"
    [ -f "$REPO_ROOT/packages/${PACKAGE_NAME}/pyproject.toml" ]
}

# Install requirements
# Usage: install_python_deps <package_name> <dev_mode>
#   package_name: Package name to install (e.g., "nv-dfm-core") or "all" for all packages
#   dev_mode: "dev" to install dev extras, "nodev" otherwise
# Example: install_python_deps "nv-dfm-core" "dev"
# Example: install_python_deps "all" "nodev"
function install_python_deps {
    local PACKAGE_NAME="${1:?Package name is required (use 'all' for all packages)}"
    local DEV_MODE="${2:?Dev mode is required (dev/nodev)}"
    
    # Build UV_ARGS based on package name
    local UV_ARGS=""
    if [ "$PACKAGE_NAME" = "all" ]; then
        UV_ARGS="--all-packages --extra all"
    else
        UV_ARGS="--package $PACKAGE_NAME"
    fi
    
    # Add --dev flag if requested
    if [ "$DEV_MODE" = "dev" ]; then
        UV_ARGS="${UV_ARGS} --dev"
    fi
    
    # Add test and docs extras for packages that have test dependencies
    if [ "$DEV_MODE" = "dev" ] && [ "$PACKAGE_NAME" = "nv-dfm-core" ]; then
        UV_ARGS="${UV_ARGS} --extra test"
    fi
    
    echo "Installing dependencies with: uv sync $UV_ARGS"
    
    # Set longer network timeout for uv (default is 30s)
    export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
    
    # Use uv to install the dependencies with retry logic
    # Note: We disable set -e temporarily to allow retries
    local RETRIES=3
    local SUCCESS=0
    for i in $(seq 1 $RETRIES); do
        echo "Attempt $i of $RETRIES..."
        if uv sync $UV_ARGS; then
            SUCCESS=1
            break
        fi
        echo "Attempt $i failed. Retrying in 5 seconds..."
        sleep 5
    done

    if [[ $SUCCESS -eq 0 ]]; then
        echo "Failed to install dependencies after $RETRIES attempts."
        exit 1
    fi
}

# Run tests with coverage for a package
# Usage: run_tests_with_coverage <package_name> <package_path> <test_dir> [min_coverage] [extra_args...]
# Example: run_tests_with_coverage "nv-dfm-core" "packages/nv-dfm-core/nv_dfm_core" "unit/nv_dfm_core" 60
function run_tests_with_coverage {
    local PACKAGE_NAME="$1"
    local PACKAGE_PATH="$2"
    local TEST_DIR="$3"
    local MIN_COVERAGE="${4:-60}"
    shift 4 2>/dev/null || true  # Remove first 4 args, remaining are extra pytest args
    
    echo "Running tests for package: $PACKAGE_NAME"
    echo "Package path: $PACKAGE_PATH"
    echo "Test directory: tests/$TEST_DIR"
    echo "Minimum coverage: $MIN_COVERAGE%"
    
    mkdir -p artifacts/coverage
    
    uv run pytest -v -s "tests/$TEST_DIR" \
        --cov="$PACKAGE_PATH" \
        --cov-config=pyproject.toml \
        --cov-report=term \
        --cov-report=xml:"artifacts/coverage/coverage-${PACKAGE_NAME}.xml" \
        --cov-report=html:"artifacts/coverage/html/${PACKAGE_NAME}" \
        --cov-fail-under="$MIN_COVERAGE" \
        "$@"
    return $?
}
