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
from unittest.mock import patch, mock_open, MagicMock

import pytest

from nv_dfm_core.cli.core._completion import CompletionGenerator


@pytest.fixture
def temp_home():
    """Create a temporary home directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()


@pytest.fixture
def completion_generator():
    """Create a CompletionGenerator instance for testing."""
    return CompletionGenerator()


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


class TestCompletionGenerator:
    """Test cases for CompletionGenerator class."""

    def test_init(self, completion_generator):
        """Test CompletionGenerator initialization."""
        assert completion_generator.env_var == "_DFM_COMPLETE"
        assert completion_generator.SUPPORTED_SHELLS == ["bash", "zsh", "fish"]

    def test_generate_completion_script_success(
        self, completion_generator, mock_completion_script
    ):
        """Test successful completion script generation."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = mock_completion_script
            mock_run.return_value = mock_result

            result = completion_generator.generate_completion_script("zsh")

            assert result == mock_completion_script
            mock_run.assert_called_once_with(
                ["dfm"],
                env={**os.environ, "_DFM_COMPLETE": "zsh_source"},
                capture_output=True,
                text=True,
            )

    def test_generate_completion_script_with_output_path(
        self, completion_generator, mock_completion_script, temp_home
    ):
        """Test completion script generation with output path."""
        output_path = temp_home / "test-completion.zsh"

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = mock_completion_script
            mock_run.return_value = mock_result

            result = completion_generator.generate_completion_script("zsh", output_path)

            assert result == mock_completion_script
            assert output_path.exists()
            assert output_path.read_text() == mock_completion_script

    def test_generate_completion_script_unsupported_shell(self, completion_generator):
        """Test completion script generation with unsupported shell."""
        with pytest.raises(ValueError, match="Unsupported shell: invalid"):
            completion_generator.generate_completion_script("invalid")

    def test_generate_completion_script_subprocess_error(self, completion_generator):
        """Test completion script generation with subprocess error."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Command failed"
            mock_run.return_value = mock_result

            with pytest.raises(
                RuntimeError, match="Failed to generate completion script"
            ):
                completion_generator.generate_completion_script("zsh")

    def test_generate_completion_script_file_not_found(self, completion_generator):
        """Test completion script generation when dfm command not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="DFM CLI tool not found in PATH"):
                completion_generator.generate_completion_script("zsh")

    def test_get_completion_script_path(self, completion_generator, temp_home):
        """Test getting completion script path."""
        path = completion_generator.get_completion_script_path("zsh", temp_home)
        expected = temp_home / ".dfm-complete.zsh"
        assert path == expected

    def test_get_completion_script_path_default_home(self, completion_generator):
        """Test getting completion script path with default home."""
        with patch("pathlib.Path.home", return_value=Path("/home/test")):
            path = completion_generator.get_completion_script_path("bash")
            expected = Path("/home/test/.dfm-complete.bash")
            assert path == expected

    def test_get_shell_config_path(self, completion_generator, temp_home):
        """Test getting shell configuration file paths."""
        # Test zsh
        zsh_path = completion_generator.get_shell_config_path("zsh", temp_home)
        assert zsh_path == temp_home / ".zshrc"

        # Test bash
        bash_path = completion_generator.get_shell_config_path("bash", temp_home)
        assert bash_path == temp_home / ".bashrc"

        # Test fish
        fish_path = completion_generator.get_shell_config_path("fish", temp_home)
        assert fish_path == temp_home / ".config" / "fish" / "config.fish"

    def test_get_shell_config_path_unknown_shell(self, completion_generator, temp_home):
        """Test getting shell config path for unknown shell."""
        path = completion_generator.get_shell_config_path("unknown", temp_home)
        assert path == temp_home / ".unknownrc"

    def test_is_completion_configured_true(self, completion_generator, temp_home):
        """Test completion configuration check when configured."""
        config_path = temp_home / ".zshrc"
        completion_path = temp_home / ".dfm-complete.zsh"

        # Create config file with completion source line
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            f"# Some config\nsource {completion_path}\n# More config"
        )

        assert completion_generator.is_completion_configured("zsh", temp_home) is True

    def test_is_completion_configured_false_no_config(
        self, completion_generator, temp_home
    ):
        """Test completion configuration check when config file doesn't exist."""
        assert completion_generator.is_completion_configured("zsh", temp_home) is False

    def test_is_completion_configured_false_no_source(
        self, completion_generator, temp_home
    ):
        """Test completion configuration check when source line not found."""
        config_path = temp_home / ".zshrc"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# Some config without completion source")

        assert completion_generator.is_completion_configured("zsh", temp_home) is False

    def test_is_completion_configured_exception_handling(
        self, completion_generator, temp_home
    ):
        """Test completion configuration check with exception handling."""
        config_path = temp_home / ".zshrc"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# Some config")

        # Mock Path.read_text to raise an exception
        with patch.object(Path, "read_text", side_effect=PermissionError):
            assert (
                completion_generator.is_completion_configured("zsh", temp_home) is False
            )

    def test_detect_shell_from_environment(self, completion_generator):
        """Test shell detection from environment variables."""
        with patch.dict(os.environ, {"ZSH_VERSION": "5.8"}):
            assert completion_generator.detect_shell() == "zsh"

        with patch.dict(os.environ, {"BASH_VERSION": "5.1.16"}):
            assert completion_generator.detect_shell() == "bash"

        with patch.dict(os.environ, {"FISH_VERSION": "3.5.1"}):
            assert completion_generator.detect_shell() == "fish"

    def test_detect_shell_from_parent_process(self, completion_generator):
        """Test shell detection from parent process."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("psutil.Process") as mock_process:
                mock_parent = MagicMock()
                mock_parent.name.return_value = "zsh"
                mock_process.return_value.parent.return_value = mock_parent

                assert completion_generator.detect_shell() == "zsh"

    def test_detect_shell_from_shell_env_var(self, completion_generator):
        """Test shell detection from SHELL environment variable."""
        # Test zsh
        with patch.dict(os.environ, {"SHELL": "/bin/zsh"}, clear=True):
            with patch("psutil.Process", side_effect=ImportError):
                assert completion_generator.detect_shell() == "zsh"

        # Test bash
        with patch.dict(os.environ, {"SHELL": "/bin/bash"}, clear=True):
            with patch("psutil.Process", side_effect=ImportError):
                assert completion_generator.detect_shell() == "bash"

        # Test fish
        with patch.dict(os.environ, {"SHELL": "/usr/bin/fish"}, clear=True):
            with patch("psutil.Process", side_effect=ImportError):
                assert completion_generator.detect_shell() == "fish"

    def test_detect_shell_unknown(self, completion_generator):
        """Test shell detection when shell cannot be determined."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("psutil.Process", side_effect=ImportError):
                assert completion_generator.detect_shell() == "unknown"

    def test_get_shell_requirements(self, completion_generator):
        """Test getting shell requirements."""
        bash_req = completion_generator.get_shell_requirements("bash")
        assert bash_req["min_version"] == "4.4"
        assert "Bash 4.4" in bash_req["note"]

        zsh_req = completion_generator.get_shell_requirements("zsh")
        assert zsh_req["min_version"] == "any"
        assert "compinit" in zsh_req["note"]

        fish_req = completion_generator.get_shell_requirements("fish")
        assert fish_req["min_version"] == "any"
        assert "completions" in fish_req["note"]

    def test_get_shell_requirements_unknown(self, completion_generator):
        """Test getting shell requirements for unknown shell."""
        req = completion_generator.get_shell_requirements("unknown")
        assert req["min_version"] == "unknown"
        assert req["note"] == "Unknown shell"

    def test_setup_completion_success(
        self, completion_generator, temp_home, mock_completion_script
    ):
        """Test successful completion setup."""

        def mock_generate_side_effect(shell, output_path=None):
            if output_path:
                output_path.write_text(mock_completion_script)
            return mock_completion_script

        with patch.object(
            completion_generator,
            "generate_completion_script",
            side_effect=mock_generate_side_effect,
        ):
            success, message = completion_generator.setup_completion("zsh", temp_home)

            assert success is True
            assert "configured" in message.lower()

            # Check that completion script was generated
            completion_path = temp_home / ".dfm-complete.zsh"
            assert completion_path.exists()
            assert completion_path.read_text() == mock_completion_script

            # Check that config file was updated
            config_path = temp_home / ".zshrc"
            assert config_path.exists()
            assert str(completion_path) in config_path.read_text()

    def test_setup_completion_already_configured(
        self, completion_generator, temp_home, mock_completion_script
    ):
        """Test completion setup when already configured."""
        # Pre-configure completion
        config_path = temp_home / ".zshrc"
        completion_path = temp_home / ".dfm-complete.zsh"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(f"source {completion_path}")

        with patch.object(
            completion_generator, "generate_completion_script"
        ) as mock_generate:
            mock_generate.return_value = mock_completion_script

            success, message = completion_generator.setup_completion(
                "zsh", temp_home, force=False
            )

            assert success is True
            assert "already configured" in message.lower()

    def test_setup_completion_force_overwrite(
        self, completion_generator, temp_home, mock_completion_script
    ):
        """Test completion setup with force flag."""
        # Pre-configure completion
        config_path = temp_home / ".zshrc"
        completion_path = temp_home / ".dfm-complete.zsh"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(f"source {completion_path}")

        with patch.object(
            completion_generator, "generate_completion_script"
        ) as mock_generate:
            mock_generate.return_value = mock_completion_script

            success, message = completion_generator.setup_completion(
                "zsh", temp_home, force=True
            )

            assert success is True
            assert "configured" in message.lower()

    def test_setup_completion_create_config_file(
        self, completion_generator, temp_home, mock_completion_script
    ):
        """Test completion setup when config file doesn't exist."""
        with patch.object(
            completion_generator, "generate_completion_script"
        ) as mock_generate:
            mock_generate.return_value = mock_completion_script

            success, message = completion_generator.setup_completion("zsh", temp_home)

            assert success is True

            # Check that config file was created
            config_path = temp_home / ".zshrc"
            assert config_path.exists()
            assert "DFM CLI completion" in config_path.read_text()

    def test_setup_completion_generation_error(self, completion_generator, temp_home):
        """Test completion setup when script generation fails."""
        with patch.object(
            completion_generator,
            "generate_completion_script",
            side_effect=RuntimeError("Generation failed"),
        ):
            success, message = completion_generator.setup_completion("zsh", temp_home)

            assert success is False
            assert "Failed to setup completion" in message
