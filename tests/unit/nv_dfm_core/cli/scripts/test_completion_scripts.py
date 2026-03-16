#!/usr/bin/env python3
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

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def temp_home():
    """Create a temporary home directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()


@pytest.fixture
def setup_script_path():
    """Get the path to the setup-completion.sh script."""
    # Go up from tests/unit/nv_dfm_core/cli/scripts/ to project root, then to packages/nv-dfm-core/nv_dfm_core/cli/scripts/
    return (
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "packages"
        / "nv-dfm-core"
        / "nv_dfm_core"
        / "cli"
        / "scripts"
        / "setup-completion.sh"
    )


@pytest.fixture
def test_script_path():
    """Get the path to the test-completion.sh script."""
    # Go up from tests/unit/nv_dfm_core/cli/scripts/ to project root, then to packages/nv-dfm-core/nv_dfm_core/cli/scripts/
    return (
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "packages"
        / "nv-dfm-core"
        / "nv_dfm_core"
        / "cli"
        / "scripts"
        / "test-completion.sh"
    )


class TestCompletionScripts:
    """Test cases for completion setup and test scripts."""

    def test_setup_script_exists(self, setup_script_path):
        """Test that the setup script exists and is executable."""
        assert setup_script_path.exists()
        assert os.access(setup_script_path, os.X_OK)

    def test_test_script_exists(self, test_script_path):
        """Test that the test script exists and is executable."""
        assert test_script_path.exists()
        assert os.access(test_script_path, os.X_OK)

    def test_setup_script_help(self, setup_script_path):
        """Test that the setup script shows help when run with --help."""
        try:
            result = subprocess.run(
                [str(setup_script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Script might not support --help, so we just check it doesn't crash
            assert result.returncode in [0, 1, 2]  # Common exit codes for help/usage
        except subprocess.TimeoutExpired:
            pytest.fail("Setup script timed out")

    @pytest.mark.skip(reason="Strange problems")
    def test_test_script_help(self, test_script_path):
        """Test that the test script shows help when run with --help."""
        # Clear any environment variables that might interfere
        env = os.environ.copy()
        env.pop("DFM_CLI_CONFIG", None)
        env.pop("PYTHONPATH", None)

        try:
            result = subprocess.run(
                [str(test_script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            # Script might not support --help, so we just check it doesn't crash
            assert result.returncode in [0, 1, 2]  # Common exit codes for help/usage
        except subprocess.TimeoutExpired as e:
            pytest.fail(
                f"Test script timed out after {e.timeout} seconds. This might be due to test interference or environment issues."
            )

    def test_setup_script_with_mock_dfm(self, setup_script_path, temp_home):
        """Test the setup script with a mocked dfm command."""
        # Create a mock dfm command
        mock_dfm_path = temp_home / "dfm"
        mock_dfm_path.write_text("""#!/bin/bash
echo "Mock DFM CLI"
exit 0
""")
        mock_dfm_path.chmod(0o755)

        # Update PATH to include our mock dfm
        env = os.environ.copy()
        env["PATH"] = str(temp_home) + ":" + env["PATH"]

        try:
            result = subprocess.run(
                [str(setup_script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=temp_home,
            )

            # The script should run without crashing
            # It might fail due to missing dependencies, but shouldn't crash
            assert result.returncode in [0, 1, 2]

            # Check that the script produced some output
            assert len(result.stdout) > 0 or len(result.stderr) > 0

        except subprocess.TimeoutExpired:
            pytest.fail("Setup script timed out")

    def test_test_script_with_mock_dfm(self, test_script_path, temp_home):
        """Test the test script with a mocked dfm command."""
        # Create a mock dfm command
        mock_dfm_path = temp_home / "dfm"
        mock_dfm_path.write_text("""#!/bin/bash
if [ "$_DFM_COMPLETE" = "zsh_source" ]; then
    echo "#compdef dfm"
    echo "_dfm_completion() { echo 'test'; }"
    echo "compdef _dfm_completion dfm"
elif [ "$_DFM_COMPLETE" = "bash_source" ]; then
    echo "Shell completion is not supported for Bash versions older than 4.4."
    exit 1
elif [ "$_DFM_COMPLETE" = "fish_source" ]; then
    echo "complete -c dfm -f"
else
    echo "Mock DFM CLI"
fi
exit 0
""")
        mock_dfm_path.chmod(0o755)

        # Update PATH to include our mock dfm
        env = os.environ.copy()
        env["PATH"] = str(temp_home) + ":" + env["PATH"]

        try:
            result = subprocess.run(
                [str(test_script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=temp_home,
            )

            # The script should run without crashing
            assert result.returncode in [0, 1, 2]

            # Check that the script produced some output
            assert len(result.stdout) > 0 or len(result.stderr) > 0

        except subprocess.TimeoutExpired:
            pytest.fail("Test script timed out")

    def test_setup_script_content(self, setup_script_path):
        """Test that the setup script contains expected content."""
        content = setup_script_path.read_text()

        # Check for key functions and features
        assert "check_dfm_available" in content
        assert "detect_shell" in content
        assert "setup_bash_completion" in content
        assert "setup_zsh_completion" in content
        assert "test_completion" in content
        assert "main" in content

        # Check for color output functions
        assert "print_status" in content
        assert "print_success" in content
        assert "print_warning" in content
        assert "print_error" in content

    def test_test_script_content(self, test_script_path):
        """Test that the test script contains expected content."""
        content = test_script_path.read_text()

        # Check for key features and content
        assert "DFM CLI Shell Completion Test" in content
        assert "command -v dfm" in content
        assert "completion script generation" in content
        assert "built-in completion command" in content

        # Check for color output functions
        assert "print_info" in content
        assert "print_success" in content
        assert "print_warning" in content

    def test_script_shebang(self, setup_script_path, test_script_path):
        """Test that scripts have proper shebang."""
        for script_path in [setup_script_path, test_script_path]:
            content = script_path.read_text()
            assert content.startswith("#!/bin/bash")

    def test_script_copyright_headers(self, setup_script_path, test_script_path):
        """Test that scripts have proper copyright headers."""
        for script_path in [setup_script_path, test_script_path]:
            content = script_path.read_text()
            assert "SPDX-FileCopyrightText" in content
            assert "NVIDIA CORPORATION" in content
            assert "SPDX-License-Identifier" in content

    def test_script_error_handling(self, setup_script_path, temp_home):
        """Test that scripts handle errors gracefully."""
        # Test with non-existent dfm command
        env = os.environ.copy()
        # Remove dfm from PATH
        env["PATH"] = "/usr/bin:/bin"

        try:
            result = subprocess.run(
                [str(setup_script_path)],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
                cwd=temp_home,
            )

            # Should exit with error code when dfm is not found
            assert result.returncode != 0
            assert (
                "DFM CLI tool not found" in result.stderr
                or "DFM CLI tool not found" in result.stdout
            )

        except subprocess.TimeoutExpired:
            pytest.fail("Setup script timed out")

    def test_script_shell_detection(self, setup_script_path, temp_home):
        """Test that scripts can detect shell types."""
        # Create a mock dfm command
        mock_dfm_path = temp_home / "dfm"
        mock_dfm_path.write_text("""#!/bin/bash
echo "Mock DFM CLI"
exit 0
""")
        mock_dfm_path.chmod(0o755)

        # Test with different shell environments
        for shell_env in ["ZSH_VERSION=5.8", "BASH_VERSION=5.1.16"]:
            env = os.environ.copy()
            env["PATH"] = str(temp_home) + ":" + env["PATH"]
            env[shell_env.split("=")[0]] = shell_env.split("=")[1]

            try:
                result = subprocess.run(
                    [str(setup_script_path)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    env=env,
                    cwd=temp_home,
                )

                # Should run without crashing
                assert result.returncode in [0, 1, 2]

            except subprocess.TimeoutExpired:
                pytest.fail("Setup script timed out")
