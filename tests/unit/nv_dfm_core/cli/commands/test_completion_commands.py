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

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from nv_dfm_core.cli.commands._completion import completion


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


class TestCompletionCommands:
    """Test cases for completion CLI commands."""

    def test_completion_group_help(self, runner):
        """Test that the completion command group shows help."""
        result = runner.invoke(completion, ["--help"])
        assert result.exit_code == 0
        assert "Shell completion commands for DFM CLI" in result.output
        assert "generate" in result.output
        assert "setup" in result.output
        assert "status" in result.output
        assert "info" in result.output

    def test_completion_generate_help(self, runner):
        """Test that the generate command shows help."""
        result = runner.invoke(completion, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Generate shell completion script for DFM CLI" in result.output
        assert "--shell" in result.output
        assert "--output" in result.output

    def test_completion_setup_help(self, runner):
        """Test that the setup command shows help."""
        result = runner.invoke(completion, ["setup", "--help"])
        assert result.exit_code == 0
        assert "Set up shell completion for DFM CLI" in result.output
        assert "--shell" in result.output
        assert "--force" in result.output

    def test_completion_status_help(self, runner):
        """Test that the status command shows help."""
        result = runner.invoke(completion, ["status", "--help"])
        assert result.exit_code == 0
        assert "Check completion status for DFM CLI" in result.output
        assert "--shell" in result.output

    def test_completion_info_help(self, runner):
        """Test that the info command shows help."""
        result = runner.invoke(completion, ["info", "--help"])
        assert result.exit_code == 0
        assert "Show information about DFM CLI shell completion" in result.output


class TestCompletionGenerate:
    """Test cases for the completion generate command."""

    def test_generate_stdout(self, runner, mock_completion_script):
        """Test generating completion script to stdout."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.generate_completion_script.return_value = (
                mock_completion_script
            )
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["generate", "--shell", "zsh"])

            assert result.exit_code == 0
            assert mock_completion_script in result.output
            mock_generator.generate_completion_script.assert_called_once_with(
                "zsh", None
            )

    def test_generate_to_file(self, runner, mock_completion_script, temp_home):
        """Test generating completion script to file."""
        output_path = temp_home / "test-completion.zsh"

        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.generate_completion_script.return_value = (
                mock_completion_script
            )
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(
                completion, ["generate", "--shell", "zsh", "--output", str(output_path)]
            )

            assert result.exit_code == 0
            assert f"Completion script written to {output_path}" in result.output
            assert (
                "To enable completion, add this line to your ~/.zshrc:" in result.output
            )
            mock_generator.generate_completion_script.assert_called_once_with(
                "zsh", output_path
            )

    def test_generate_error(self, runner):
        """Test generate command with error."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.generate_completion_script.side_effect = RuntimeError(
                "Generation failed"
            )
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["generate", "--shell", "zsh"])

            assert result.exit_code == 1
            assert (
                "Error generating completion script: Generation failed" in result.output
            )


class TestCompletionSetup:
    """Test cases for the completion setup command."""

    def test_setup_auto_detect_success(self, runner, temp_home):
        """Test setup command with auto-detection."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.detect_shell.return_value = "zsh"
            mock_generator.setup_completion.return_value = (
                True,
                "Completion configured successfully",
            )
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["setup"])

            assert result.exit_code == 0
            assert "Detected shell: zsh" in result.output
            assert "✓ Completion configured successfully" in result.output
            assert "Completion setup complete for zsh!" in result.output
            mock_generator.setup_completion.assert_called_once_with("zsh", None, False)

    def test_setup_auto_detect_unknown_shell(self, runner):
        """Test setup command with unknown shell detection."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.detect_shell.return_value = "unknown"
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["setup"])

            assert result.exit_code == 1
            assert "Could not detect shell type" in result.output

    def test_setup_specific_shell(self, runner):
        """Test setup command with specific shell."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.setup_completion.return_value = (
                True,
                "Completion configured successfully",
            )
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["setup", "--shell", "bash"])

            assert result.exit_code == 0
            assert "✓ Completion configured successfully" in result.output
            mock_generator.setup_completion.assert_called_once_with("bash", None, False)

    def test_setup_with_force(self, runner):
        """Test setup command with force flag."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.setup_completion.return_value = (
                True,
                "Completion configured successfully",
            )
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["setup", "--shell", "zsh", "--force"])

            assert result.exit_code == 0
            assert "✓ Completion configured successfully" in result.output
            mock_generator.setup_completion.assert_called_once_with("zsh", None, True)

    def test_setup_with_home_dir(self, runner, temp_home):
        """Test setup command with custom home directory."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.setup_completion.return_value = (
                True,
                "Completion configured successfully",
            )
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(
                completion, ["setup", "--shell", "zsh", "--home-dir", str(temp_home)]
            )

            assert result.exit_code == 0
            assert "✓ Completion configured successfully" in result.output
            mock_generator.setup_completion.assert_called_once_with(
                "zsh", temp_home, False
            )

    def test_setup_failure(self, runner):
        """Test setup command with failure."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.setup_completion.return_value = (False, "Setup failed")
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["setup", "--shell", "zsh"])

            assert result.exit_code == 1
            assert "✗ Setup failed" in result.output

    def test_setup_exception(self, runner):
        """Test setup command with exception."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.setup_completion.side_effect = RuntimeError(
                "Unexpected error"
            )
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["setup", "--shell", "zsh"])

            assert result.exit_code == 1
            assert "Error setting up completion: Unexpected error" in result.output


class TestCompletionStatus:
    """Test cases for the completion status command."""

    def test_status_all_shells(self, runner):
        """Test status command for all shells."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.SUPPORTED_SHELLS = ["bash", "zsh", "fish"]
            mock_generator.is_completion_configured.side_effect = [False, True, False]
            mock_generator.get_completion_script_path.side_effect = [
                Path("/home/test/.dfm-complete.bash"),
                Path("/home/test/.dfm-complete.zsh"),
                Path("/home/test/.dfm-complete.fish"),
            ]
            mock_generator.get_shell_config_path.side_effect = [
                Path("/home/test/.bashrc"),
                Path("/home/test/.zshrc"),
                Path("/home/test/.config/fish/config.fish"),
            ]
            mock_generator.get_shell_requirements.side_effect = [
                {"min_version": "4.4", "note": "Bash 4.4 or later required"},
                {"min_version": "any", "note": "Requires compinit to be loaded"},
                {
                    "min_version": "any",
                    "note": "Completion scripts should be placed in ~/.config/fish/completions/",
                },
            ]
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["status"])

            assert result.exit_code == 0
            assert "DFM CLI Completion Status" in result.output
            assert "BASH: ✗ Not configured" in result.output
            assert "ZSH: ✓ Configured" in result.output
            assert "FISH: ✗ Not configured" in result.output

    def test_status_specific_shell(self, runner):
        """Test status command for specific shell."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.is_completion_configured.return_value = True
            mock_completion_path = Path("/home/test/.dfm-complete.zsh")
            mock_config_path = Path("/home/test/.zshrc")
            mock_generator.get_completion_script_path.return_value = (
                mock_completion_path
            )
            mock_generator.get_shell_config_path.return_value = mock_config_path
            mock_generator.get_shell_requirements.return_value = {
                "min_version": "any",
                "note": "Requires compinit to be loaded",
            }
            mock_generator_class.return_value = mock_generator

            # Mock the Path.exists() method
            with patch.object(Path, "exists", return_value=True):
                result = runner.invoke(completion, ["status", "--shell", "zsh"])

                assert result.exit_code == 0
                assert "DFM CLI Completion Status" in result.output
                assert "ZSH: ✓ Configured" in result.output
                assert (
                    "Completion script: /home/test/.dfm-complete.zsh" in result.output
                )
                assert "Config file: /home/test/.zshrc" in result.output
                assert "Note: Requires compinit to be loaded" in result.output

    def test_status_exception_handling(self, runner):
        """Test status command with exception handling."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.SUPPORTED_SHELLS = ["zsh"]
            mock_generator.is_completion_configured.side_effect = Exception(
                "Test error"
            )
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["status"])

            assert result.exit_code == 0
            assert "ZSH: Error checking status - Test error" in result.output


class TestCompletionInfo:
    """Test cases for the completion info command."""

    def test_info_command(self, runner):
        """Test info command output."""
        with patch(
            "nv_dfm_core.cli.commands._completion.CompletionGenerator"
        ) as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.SUPPORTED_SHELLS = ["bash", "zsh", "fish"]
            mock_generator.get_shell_requirements.side_effect = [
                {
                    "min_version": "4.4",
                    "note": "Bash 4.4 or later required for completion support",
                },
                {"min_version": "any", "note": "Requires compinit to be loaded"},
                {
                    "min_version": "any",
                    "note": "Completion scripts should be placed in ~/.config/fish/completions/",
                },
            ]
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(completion, ["info"])

            assert result.exit_code == 0
            assert "DFM CLI Shell Completion Information" in result.output
            assert "Supported Shells:" in result.output
            assert (
                "BASH: Bash 4.4 or later required for completion support"
                in result.output
            )
            assert "ZSH: Requires compinit to be loaded" in result.output
            assert (
                "FISH: Completion scripts should be placed in ~/.config/fish/completions/"
                in result.output
            )
            assert "Quick Setup:" in result.output
            assert "Manual Setup:" in result.output
            assert "Check Status:" in result.output
            assert "Usage Examples:" in result.output
            assert "dfm <TAB>" in result.output
