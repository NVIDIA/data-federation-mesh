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
from click.testing import CliRunner

from nv_dfm_core.cli.main import cli


@pytest.fixture
def temp_home():
    """Create a temporary home directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()


@pytest.fixture
def runner():
    """Create a Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_completion_script():
    """Mock completion script content."""
    return """#compdef dfm

_dfm_completion() {
    local -a completions
    local -a completions_with_descriptions
    local -a response
    (( ! $+commands[dfm] )) && return 1

    response=("${(@f)$(env COMP_WORDS="${words[*]}" COMP_CWORD=$((CURRENT-1)) _DFM_COMPLETE=zsh_complete dfm)}")

    for type key descr in ${response}; do
        if [[ "$type" == "plain" ]]; then
            if [[ "$descr" == "_" ]]; then
                completions+=("$key")
            else
                completions_with_descriptions+=("$key":"$descr")
            fi
        elif [[ "$type" == "dir" ]]; then
            _path_files -/
        elif [[ "$type" == "file" ]]; then
            _path_files -f
        fi
    done

    if [ -n "$completions_with_descriptions" ]; then
        _describe -V unsorted completions_with_descriptions -U
    fi

    if [ -n "$completions" ]; then
        compadd -U -V unsorted -a completions
    fi
}

if [[ $zsh_eval_context[-1] == loadautofunc ]]; then
    _dfm_completion "$@"
else
    compdef _dfm_completion dfm
fi
"""


class TestCompletionIntegration:
    """Integration tests for completion functionality."""

    def test_completion_command_in_main_cli(self, runner):
        """Test that completion commands are available in main CLI."""
        # Clear any cached config paths from previous tests
        from nv_dfm_core.cli.config._cli import Utils

        Utils._cli_config_path = None
        Utils._repo_dir_cache = None

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "completion" in result.output

        result = runner.invoke(cli, ["completion", "--help"])
        assert result.exit_code == 0
        assert "Shell completion commands for DFM CLI" in result.output

    def test_completion_workflow_generate_and_setup(
        self, runner, temp_home, mock_completion_script
    ):
        """Test the complete workflow of generating and setting up completion."""
        # Clear any cached config paths from previous tests
        from nv_dfm_core.cli.config._cli import Utils

        Utils._cli_config_path = None
        Utils._repo_dir_cache = None

        # Mock the subprocess call to dfm command
        with patch("nv_dfm_core.cli.core._completion.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = mock_completion_script
            mock_run.return_value = mock_result

            # Step 1: Generate completion script
            output_path = temp_home / "test-completion.zsh"
            result = runner.invoke(
                cli,
                [
                    "completion",
                    "generate",
                    "--shell",
                    "zsh",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert f"Completion script written to {output_path}" in result.output
            assert output_path.exists()
            assert output_path.read_text() == mock_completion_script

            # Step 2: Setup completion
            with patch("pathlib.Path.home", return_value=temp_home):
                result = runner.invoke(cli, ["completion", "setup", "--shell", "zsh"])

                assert result.exit_code == 0
                assert "✓" in result.output
                assert "Completion setup complete for zsh!" in result.output

                # Check that completion script was created in home directory
                home_completion_path = temp_home / ".dfm-complete.zsh"
                assert home_completion_path.exists()

                # Check that config file was updated
                config_path = temp_home / ".zshrc"
                assert config_path.exists()
                assert str(home_completion_path) in config_path.read_text()

    def test_completion_status_workflow(
        self, runner, temp_home, mock_completion_script
    ):
        """Test the status checking workflow."""
        with patch("pathlib.Path.home", return_value=temp_home):
            # Initially, completion should not be configured
            result = runner.invoke(cli, ["completion", "status", "--shell", "zsh"])
            assert result.exit_code == 0
            assert "ZSH: ✗ Not configured" in result.output

            # Setup completion
            with patch("nv_dfm_core.cli.core._completion.subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = mock_completion_script
                mock_run.return_value = mock_result

                result = runner.invoke(cli, ["completion", "setup", "--shell", "zsh"])
                assert result.exit_code == 0

            # Now completion should be configured
            result = runner.invoke(cli, ["completion", "status", "--shell", "zsh"])
            assert result.exit_code == 0
            assert "ZSH: ✓ Configured" in result.output

    def test_completion_info_command(self, runner):
        """Test the info command provides comprehensive information."""
        result = runner.invoke(cli, ["completion", "info"])
        assert result.exit_code == 0

        # Check that all expected sections are present
        expected_sections = [
            "DFM CLI Shell Completion Information",
            "Supported Shells:",
            "Quick Setup:",
            "Manual Setup:",
            "Check Status:",
            "Usage Examples:",
        ]

        for section in expected_sections:
            assert section in result.output

        # Check that shell information is present
        assert "BASH:" in result.output
        assert "ZSH:" in result.output
        assert "FISH:" in result.output

    def test_completion_auto_detect_workflow(
        self, runner, temp_home, mock_completion_script
    ):
        """Test the auto-detect workflow."""
        # Clear any cached config paths from previous tests
        from nv_dfm_core.cli.config._cli import Utils

        Utils._cli_config_path = None
        Utils._repo_dir_cache = None

        with patch("pathlib.Path.home", return_value=temp_home):
            with patch("nv_dfm_core.cli.core._completion.subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = mock_completion_script
                mock_run.return_value = mock_result

                # Mock shell detection
                with patch(
                    "nv_dfm_core.cli.core._completion.CompletionGenerator.detect_shell",
                    return_value="zsh",
                ):
                    result = runner.invoke(cli, ["completion", "setup"])

                    assert result.exit_code == 0
                    assert "Detected shell: zsh" in result.output
                    assert "✓" in result.output
                    assert "Completion setup complete for zsh!" in result.output

    def test_completion_force_setup(self, runner, temp_home, mock_completion_script):
        """Test force setup overwrites existing configuration."""
        with patch("pathlib.Path.home", return_value=temp_home):
            # Create initial configuration
            config_path = temp_home / ".zshrc"
            completion_path = temp_home / ".dfm-complete.zsh"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                f"# Initial config\nsource {completion_path}\n# More config"
            )

            with patch("nv_dfm_core.cli.core._completion.subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = mock_completion_script
                mock_run.return_value = mock_result

                # Setup with force flag
                result = runner.invoke(
                    cli, ["completion", "setup", "--shell", "zsh", "--force"]
                )

                assert result.exit_code == 0
                assert "✓" in result.output

                # Verify that completion script was created
                assert completion_path.exists()

    def test_completion_error_handling(self, runner):
        """Test error handling in completion commands."""
        # Test generate command with subprocess error
        with patch("nv_dfm_core.cli.core._completion.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Command failed"
            mock_run.return_value = mock_result

            result = runner.invoke(cli, ["completion", "generate", "--shell", "zsh"])
            assert result.exit_code == 1
            assert "Error generating completion script" in result.output

        # Test setup command with unknown shell detection
        with patch(
            "nv_dfm_core.cli.core._completion.CompletionGenerator.detect_shell",
            return_value="unknown",
        ):
            result = runner.invoke(cli, ["completion", "setup"])
            assert result.exit_code == 1
            assert "Could not detect shell type" in result.output

    def test_completion_script_content_validation(self, runner, temp_home):
        """Test that generated completion scripts contain expected content."""
        # Clear any cached config paths from previous tests
        from nv_dfm_core.cli.config._cli import Utils

        Utils._cli_config_path = None
        Utils._repo_dir_cache = None

        with patch("nv_dfm_core.cli.core._completion.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = """#compdef dfm

_dfm_completion() {
    # Test completion function
}

compdef _dfm_completion dfm
"""
            mock_run.return_value = mock_result

            output_path = temp_home / "test-completion.zsh"
            result = runner.invoke(
                cli,
                [
                    "completion",
                    "generate",
                    "--shell",
                    "zsh",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()

            content = output_path.read_text()
            assert "#compdef dfm" in content
            assert "_dfm_completion()" in content
            assert "compdef _dfm_completion dfm" in content

    def test_completion_multiple_shells(self, runner, temp_home):
        """Test completion setup for multiple shells."""
        with patch("pathlib.Path.home", return_value=temp_home):
            with patch("nv_dfm_core.cli.core._completion.subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "test completion script"
                mock_run.return_value = mock_result

                # Setup for bash
                result = runner.invoke(cli, ["completion", "setup", "--shell", "bash"])
                assert result.exit_code == 0

                # Setup for zsh
                result = runner.invoke(cli, ["completion", "setup", "--shell", "zsh"])
                assert result.exit_code == 0

                # Check that both completion scripts exist
                bash_completion = temp_home / ".dfm-complete.bash"
                zsh_completion = temp_home / ".dfm-complete.zsh"

                assert bash_completion.exists()
                assert zsh_completion.exists()

                # Check that both config files were updated
                bashrc = temp_home / ".bashrc"
                zshrc = temp_home / ".zshrc"

                assert bashrc.exists()
                assert zshrc.exists()

                assert str(bash_completion) in bashrc.read_text()
                assert str(zsh_completion) in zshrc.read_text()

    def test_completion_status_all_shells(self, runner, temp_home):
        """Test status command for all shells."""
        with patch("pathlib.Path.home", return_value=temp_home):
            result = runner.invoke(cli, ["completion", "status"])
            assert result.exit_code == 0

            # Check that all supported shells are shown
            assert "BASH:" in result.output
            assert "ZSH:" in result.output
            assert "FISH:" in result.output

            # Check that status information is shown for each
            assert "Not configured" in result.output or "Configured" in result.output
            assert "Completion script:" in result.output
            assert "Config file:" in result.output
