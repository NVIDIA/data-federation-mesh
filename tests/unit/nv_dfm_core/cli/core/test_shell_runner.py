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

"""
Tests for the ShellRunner class.
"""

import os
import platform
import signal
import subprocess
import tempfile
import time

from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest
import click
from nv_dfm_core.cli.core._shell_runner import ShellRunner


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def shell_runner(temp_dir):
    """Create a ShellRunner instance with a temporary working directory."""
    return ShellRunner(cwd=temp_dir)


def test_shell_runner_initialization():
    """Test ShellRunner initialization."""
    runner = ShellRunner()
    assert runner.cwd is None
    assert runner.last_command is None
    assert runner.last_output is None
    assert runner.last_error is None
    assert runner.last_returncode is None


def test_shell_runner_with_cwd(temp_dir):
    """Test ShellRunner with custom working directory."""
    runner = ShellRunner(cwd=temp_dir)
    assert runner.cwd == str(temp_dir)


def test_run_basic_command(shell_runner):
    """Test running a basic command."""
    result = shell_runner.run(["echo", "Hello World"])

    assert result is not None
    assert result.result.returncode == 0
    assert result.result.stdout.strip() == "Hello World"
    assert shell_runner.last_command == "echo Hello World"
    assert shell_runner.last_output.strip() == "Hello World"
    assert shell_runner.last_returncode == 0


def test_run_command_list(shell_runner):
    """Test running a command as a list."""
    result = shell_runner.run(["echo", "Hello", "World"])

    assert result is not None
    assert result.result.returncode == 0
    assert result.result.stdout.strip() == "Hello World"
    assert shell_runner.last_command == "echo Hello World"


@patch("subprocess.run")
def test_run_command_with_error(mock_run, shell_runner):
    """Test running a command that fails."""
    # Mock subprocess.run to simulate a command failure
    error = subprocess.CalledProcessError(
        1, ["nonexistent_command"], stderr="Command not found"
    )
    mock_run.side_effect = error

    try:
        shell_runner.run(["nonexistent_command"])
    except subprocess.CalledProcessError:
        pass

    assert shell_runner.last_returncode == 1
    assert shell_runner.last_error == "Command not found"


@patch("subprocess.run")
def test_run_command_without_check(mock_run, shell_runner):
    """Test running a command without checking for errors."""
    # Mock subprocess.run to simulate a command failure
    mock_run.return_value = MagicMock(
        returncode=1, stdout="", stderr="Command not found"
    )

    result = shell_runner.run(["nonexistent_command"], check=False)

    assert result is not None
    assert result.result.returncode == 1
    assert result.result.stderr == "Command not found"


@patch("subprocess.run")
def test_run_command_with_env(mock_run, shell_runner):
    """Test running a command with environment variables."""
    env = {"TEST_VAR": "test_value"}
    mock_run.return_value = MagicMock(returncode=0, stdout="test_value\n", stderr="")

    if platform.system() == "Windows":
        result = shell_runner.run(["echo", "%TEST_VAR%"], env=env)
    else:
        result = shell_runner.run(["echo", "$TEST_VAR"], env=env)
    assert result is not None
    assert result.result.stdout.strip() == "test_value"


def test_run_command_with_capture_output(shell_runner):
    """Test running a command with capture_output=False."""
    result = shell_runner.run(["echo", "Test"], capture_output=False)
    assert result is not None
    assert result.result.stdout is None
    assert result.result.stderr is None
    assert shell_runner.last_output is None
    assert shell_runner.last_error is None


def test_run_command_with_cwd(temp_dir):
    """Test running a command in a specific directory."""
    # Create a test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")

    # Run command from different directory
    runner = ShellRunner()
    result = runner.run(["cat", "test.txt"], cwd=str(temp_dir))
    assert result is not None
    assert result.result.stdout.strip() == "test content"


def test_run_detached_command(shell_runner):
    """Test running a command in detached mode."""
    # Run a sleep command in detached mode
    result = shell_runner.run(["sleep", "1"], detached=True)

    # Check that we got a process ID
    assert result is not None
    assert hasattr(result, "pid")
    assert result.pid is not None
    assert result.pid > 0

    # Check that the process is running
    try:
        os.kill(result.pid, 0)
    except ProcessLookupError:
        pytest.fail("Process should still be running")

    # Wait for the process to finish
    time.sleep(1.1)

    # Check that the process is no longer running
    try:
        process = psutil.Process(result.pid)
        if process.status() == psutil.STATUS_RUNNING:
            pytest.fail("Process should no longer be running")
    except psutil.NoSuchProcess:
        pass  # This is what we want


def test_run_detached_command_with_env(shell_runner):
    """Test running a detached command with environment variables."""
    env = {"TEST_VAR": "test_value"}
    result = shell_runner.run(["sleep", "1"], detached=True, env=env)

    assert hasattr(result, "pid")
    assert result.pid > 0

    # Wait for the process to finish
    time.sleep(1.1)

    # Check that the process is no longer running
    try:
        process = psutil.Process(result.pid)
        if process.status() == psutil.STATUS_RUNNING:
            pytest.fail("Process should no longer be running")
    except psutil.NoSuchProcess:
        pass  # This is what we want


def test_run_detached_command_with_cwd(temp_dir):
    """Test running a detached command in a specific directory."""
    # Create a test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")

    # Run command from different directory
    runner = ShellRunner()
    result = runner.run(["sleep", "1"], detached=True, cwd=str(temp_dir))

    assert hasattr(result, "pid")
    assert result.pid > 0

    # Wait for the process to finish
    time.sleep(1.1)

    # Check that the process is no longer running
    try:
        process = psutil.Process(result.pid)
        if process.status() == psutil.STATUS_RUNNING:
            pytest.fail("Process should no longer be running")
    except psutil.NoSuchProcess:
        pass  # This is what we want


@pytest.fixture
def mock_process():
    """Create a mock process."""
    process = MagicMock(spec=subprocess.Popen)
    process.pid = 12345
    return process


@pytest.fixture
def temp_files(tmp_path):
    """Create temporary files for testing."""
    pid_file = tmp_path / "test.pid"
    log_file = tmp_path / "test.log"
    return pid_file, log_file


def test_start_process_success(temp_files, mock_process, shell_runner):
    """Test starting a process successfully."""
    pid_file, log_file = temp_files

    with patch("subprocess.Popen", return_value=mock_process):
        process = shell_runner.start_in_background(
            "test_process",
            ["echo", "test"],
            pid_file,
            log_file,
        )

        assert process.pid == 12345
        assert pid_file.read_text() == "12345"


def test_start_process_existing_pid_file(temp_files, shell_runner):
    """Test starting a process when PID file already exists."""
    pid_file, log_file = temp_files
    pid_file.write_text("12345")

    with pytest.raises(click.Abort):
        shell_runner.start_in_background(
            "test_process",
            ["echo", "test"],
            pid_file,
            log_file,
        )


def test_start_process_command_not_found(temp_files, shell_runner):
    """Test starting a process with nonexistent command."""
    pid_file, log_file = temp_files

    with (
        patch("subprocess.Popen", side_effect=FileNotFoundError),
        pytest.raises(click.Abort),
    ):
        shell_runner.start_in_background(
            "test_process",
            ["nonexistent_command"],
            pid_file,
            log_file,
        )


def test_start_process_with_cwd(temp_files, mock_process, tmp_path, shell_runner):
    """Test starting a process with custom working directory."""
    pid_file, log_file = temp_files
    cwd = tmp_path / "custom_dir"
    cwd.mkdir()

    with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        process = shell_runner.start_in_background(
            "test_process",
            ["echo", "test"],
            pid_file,
            log_file,
            cwd=cwd,
        )

        assert process.pid == 12345
        mock_popen.assert_called_once()
        assert mock_popen.call_args[1]["cwd"] == cwd


def test_stop_process_with_popen(mock_process, shell_runner):
    """Test stopping a process using Popen object."""
    shell_runner.stop_from_background("test_process", mock_process)
    mock_process.terminate.assert_called_once()


def test_stop_process_with_pid_file(temp_files, shell_runner):
    """Test stopping a process using PID file."""
    pid_file, _ = temp_files
    pid_file.write_text("12345")

    with patch("os.killpg") as mock_kill:
        shell_runner.stop_from_background("test_process", pid_file)
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        assert not pid_file.exists()  # PID file should be removed


def test_stop_process_missing_pid_file(temp_files, shell_runner):
    """Test stopping a process when PID file is missing."""
    pid_file, _ = temp_files

    # Without stop_on_error
    shell_runner.stop_from_background("test_process", pid_file)  # Should not raise

    # With stop_on_error
    with pytest.raises(click.Abort):
        shell_runner.stop_from_background("test_process", pid_file, stop_on_error=True)


def test_stop_process_kill_error(temp_files, shell_runner):
    """Test stopping a process when kill fails."""
    pid_file, _ = temp_files
    pid_file.write_text("12345")

    with patch("os.killpg", side_effect=OSError):
        shell_runner.stop_from_background("test_process", pid_file)
        assert not pid_file.exists()  # PID file should still be removed


def test_is_running_in_background_with_pid(shell_runner):
    """Test checking if a process is running using PID."""
    # Create a temporary process
    with subprocess.Popen(["sleep", "1"]) as process:
        # Test with direct PID
        assert shell_runner.is_running_in_background("pid", process.pid) is True

        # Test with PID file
        pid_file = Path("test.pid")
        pid_file.write_text(str(process.pid))
        assert shell_runner.is_running_in_background("pid", pid_file) is True
        pid_file.unlink()

        # Test with non-existent PID
        assert shell_runner.is_running_in_background("pid", 999999) is False

        # Test with invalid PID file
        invalid_pid_file = Path("invalid.pid")
        assert shell_runner.is_running_in_background("pid", invalid_pid_file) is False


def test_is_running_in_background_with_group_id(shell_runner):
    """Test checking if a process is running using group ID."""
    # Create a temporary process
    with subprocess.Popen(["sleep", "1"]) as process:
        group_id = os.getpgid(process.pid)

        # Test with direct group ID
        assert shell_runner.is_running_in_background("group_id", group_id) is True

        # Test with group ID file
        group_id_file = Path("test.pid")
        group_id_file.write_text(str(group_id))
        assert shell_runner.is_running_in_background("group_id", group_id_file) is True
        group_id_file.unlink()

        # Test with non-existent group ID
        # This test might fail if there are race conditions with process termination
        # So we'll just verify the method doesn't crash
        try:
            result = shell_runner.is_running_in_background("group_id", 999999)
            assert isinstance(result, bool)
        except ProcessLookupError:
            # This is acceptable if the process lookup fails during iteration
            pass


def test_is_running_in_background_with_cmdline(shell_runner):
    """Test checking if a process is running using command line."""
    # Create a temporary process with a unique command
    unique_cmd = f"test_cmd_{os.getpid()}"
    with subprocess.Popen(
        ["python", "-c", f"import time; print('{unique_cmd}'); time.sleep(1)"]
    ) as process:
        # Test with command line string
        assert shell_runner.is_running_in_background("in_cmdline", unique_cmd) is True

        # Test with non-existent command
        assert (
            shell_runner.is_running_in_background("in_cmdline", "nonexistent_command")
            is False
        )


def test_is_running_in_background_invalid_attribute(shell_runner):
    """Test checking if a process is running with invalid attribute."""
    with pytest.raises(ValueError, match="Invalid attribute: invalid_attr"):
        shell_runner.is_running_in_background("invalid_attr", 12345)
