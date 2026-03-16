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

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from nv_dfm_core.cli.core._flare_poc import FlarePoc
from nv_dfm_core.cli.config._federation import FederationConfig
from nv_dfm_core.cli.core._shell_runner import ShellRunner
import click
import os


@pytest.fixture
def workspace(tmp_path) -> Path:
    """Create a temporary workspace directory.

    Args:
        tmp_path: pytest fixture that provides a temporary directory.

    Returns:
        Path: Path to the created workspace directory.
    """
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def mock_federation_config() -> Mock:
    """Create a mock FederationConfig.

    Returns:
        Mock: Mock object with FederationConfig spec.
    """
    mock_cfg = Mock(spec=FederationConfig)
    mock_cfg.app_name = "test_app"
    mock_cfg.admin_package = Path("/tmp/admin/package")
    return mock_cfg


@pytest.fixture
def flare_poc(workspace) -> FlarePoc:
    """Create a FlarePoc instance.

    Args:
        workspace: pytest fixture that provides the workspace directory.

    Returns:
        FlarePoc: A FlarePoc instance initialized with the workspace.
    """
    return FlarePoc(workspace)


@pytest.fixture
def mock_shell_runner() -> Mock:
    """Create a mock ShellRunner.

    Returns:
        Mock: Mock object with ShellRunner spec.
    """
    return Mock(spec=ShellRunner)


def test_flare_poc_init(flare_poc, workspace):
    """Test FlarePoc initialization."""
    assert flare_poc._workspace == workspace
    assert flare_poc._fed_cfg is None


def test_set_federation_config(flare_poc, mock_federation_config):
    """Test setting federation config."""
    # Act
    flare_poc.set_federation_config(mock_federation_config)

    # Assert
    assert flare_poc._fed_cfg == mock_federation_config


def test_prepare_without_federation_config(flare_poc):
    """Test prepare method when federation config is not set."""
    with pytest.raises(RuntimeError, match="Federation config not set"):
        flare_poc.prepare()


def test_prepare_success(flare_poc, mock_federation_config):
    """Test successful preparation of Flare environment."""
    # Arrange
    config_path = Path("/tmp/config")
    project_path = Path("/tmp/project")
    workspace_path = Path("/tmp/workspace")

    mock_federation_config.config_path = config_path
    mock_federation_config.project_path = project_path
    mock_federation_config.federation_workspace_dir = workspace_path

    flare_poc.set_federation_config(mock_federation_config)

    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch("shutil.rmtree") as mock_rmtree,
        patch("sh.nvflare") as mock_nvflare,
        patch.dict(os.environ, {}, clear=True),
    ):
        flare_poc.prepare()

        # Verify echo messages
        mock_echo.assert_any_call(f"Preparing [configs: {config_path}, {project_path}]")
        mock_echo.assert_any_call(f"Workspace: {workspace_path}")
        mock_echo.assert_any_call("Removed old workspace directory.")
        mock_echo.assert_any_call("NFLARE POC directory ready.")

        # Verify workspace cleanup
        mock_rmtree.assert_called_once_with(workspace_path, ignore_errors=True)

        # Verify nvflare command
        mock_nvflare.assert_called_once_with(
            ["poc", "prepare", "-i", project_path.as_posix()]
        )

        # Verify environment variable
        assert os.environ["NVFLARE_POC_WORKSPACE"] == workspace_path.as_posix()


def test_start_success(flare_poc, mock_federation_config, mock_shell_runner):
    """Test successful start of Flare."""
    # Arrange
    mock_federation_config.app_name = "test_app"
    flare_poc.set_federation_config(mock_federation_config)

    mock_process = Mock()
    mock_shell_runner.start_in_background.return_value = mock_process

    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch(
            "nv_dfm_core.cli.core._flare_poc.ShellRunner",
            return_value=mock_shell_runner,
        ),
    ):
        process = flare_poc.start()

        # Verify process was started
        assert process == mock_process
        mock_shell_runner.start_in_background.assert_called_once_with(
            "NVFlare",
            ["nvflare", "poc", "start", "-ex", "test_app"],
            flare_poc._workspace / "flare.pid",
            flare_poc._workspace / "flare.log",
        )


def test_start_with_debug(flare_poc, mock_federation_config, mock_shell_runner):
    """Test start of Flare with debug mode."""
    # Arrange
    mock_federation_config.app_name = "test_app"
    flare_poc.set_federation_config(mock_federation_config)

    mock_process = Mock()
    mock_shell_runner.start_in_background.return_value = mock_process

    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch(
            "nv_dfm_core.cli.core._flare_poc.ShellRunner",
            return_value=mock_shell_runner,
        ),
    ):
        process = flare_poc.start(debug=True)

        # Verify process was started with debug flag
        assert process == mock_process
        mock_shell_runner.start_in_background.assert_called_once_with(
            "NVFlare",
            ["nvflare", "poc", "start", "-ex", "test_app", "--debug"],
            flare_poc._workspace / "flare.pid",
            flare_poc._workspace / "flare.log",
        )


def test_stop_with_process(flare_poc, mock_shell_runner):
    """Test stopping Flare with a provided process."""
    # Arrange
    mock_process = Mock()

    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch(
            "nv_dfm_core.cli.core._flare_poc.ShellRunner",
            return_value=mock_shell_runner,
        ),
    ):
        flare_poc.stop(mock_process)

        # Verify process was stopped
        mock_shell_runner.stop_from_background.assert_called_once_with(
            "NVFlare", mock_process
        )
        # Verify both messages were displayed
        mock_echo.assert_any_call("Stopping Flare...")
        mock_echo.assert_any_call(click.style("Flare stopped.", fg="green"))
        assert mock_echo.call_count == 2


def test_stop_with_pid_file(flare_poc, mock_shell_runner):
    """Test stopping Flare using PID file."""
    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch(
            "nv_dfm_core.cli.core._flare_poc.ShellRunner",
            return_value=mock_shell_runner,
        ),
    ):
        flare_poc.stop()

        # Verify process was stopped using PID file
        mock_shell_runner.stop_from_background.assert_called_once_with(
            "NVFlare", flare_poc._workspace / "flare.pid"
        )
        # Verify both messages were displayed
        mock_echo.assert_any_call("Stopping Flare...")
        mock_echo.assert_any_call(click.style("Flare stopped.", fg="green"))
        assert mock_echo.call_count == 2


def test_connect_success(flare_poc, mock_federation_config):
    """Test successful connection to Flare."""
    # Arrange
    flare_poc.set_federation_config(mock_federation_config)
    mock_session = Mock()

    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch(
            "nvflare.fuel.flare_api.flare_api.new_secure_session",
            return_value=mock_session,
        ) as mock_new_session,
    ):
        session = flare_poc.connect()

        # Verify session was created with correct parameters
        assert session == mock_session
        mock_new_session.assert_called_once_with(
            mock_federation_config.app_name,
            mock_federation_config.admin_package.as_posix(),
            debug=False,
        )
        mock_echo.assert_called_once_with(
            f"Connecting to Flare with admin package: {mock_federation_config.admin_package}"
        )


def test_connect_with_debug(flare_poc, mock_federation_config):
    """Test connection to Flare with debug mode."""
    # Arrange
    flare_poc.set_federation_config(mock_federation_config)
    mock_session = Mock()

    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch(
            "nvflare.fuel.flare_api.flare_api.new_secure_session",
            return_value=mock_session,
        ) as mock_new_session,
    ):
        session = flare_poc.connect(debug=True)

        # Verify session was created with debug flag
        assert session == mock_session
        mock_new_session.assert_called_once_with(
            mock_federation_config.app_name,
            mock_federation_config.admin_package.as_posix(),
            debug=True,
        )
        mock_echo.assert_called_once_with(
            f"Connecting to Flare with admin package: {mock_federation_config.admin_package}"
        )


def test_show_logs_when_file_exists(flare_poc, workspace):
    """Test showing logs when the log file exists."""
    # Arrange
    log_content = "line1\nline2\nline3\n"
    log_file = workspace / "flare.log"
    log_file.write_text(log_content)

    # Act & Assert
    with patch("click.echo") as mock_echo:
        flare_poc.show_logs()

        # Verify each line was echoed without trailing whitespace
        mock_echo.assert_any_call("line1")
        mock_echo.assert_any_call("line2")
        mock_echo.assert_any_call("line3")
        assert mock_echo.call_count == 3


def test_show_logs_when_file_does_not_exist(flare_poc, workspace):
    """Test showing logs when the log file doesn't exist."""
    # Act & Assert
    with patch("click.echo") as mock_echo:
        with pytest.raises(click.Abort):
            flare_poc.show_logs()

        # Verify error message was echoed
        mock_echo.assert_called_once_with(
            f"Sorry, no logs available ({workspace}/flare.log doesn't exist)."
        )


def test_wait_success(flare_poc, mock_federation_config):
    """Test successful wait for Flare to be ready."""
    # Arrange
    mock_federation_config.client_sites = ["client1", "client2"]
    flare_poc.set_federation_config(mock_federation_config)

    mock_session = Mock()
    mock_session.get_connected_client_list.return_value = ["client1", "client2"]

    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch.object(flare_poc, "connect", return_value=mock_session),
    ):
        flare_poc.wait()

        # Verify connection was attempted
        flare_poc.connect.assert_called_once()

        # Verify client list was checked
        mock_session.get_connected_client_list.assert_called_once()

        # Verify success message
        mock_echo.assert_any_call("Connected to Flare.")
        mock_echo.assert_any_call("Connected clients: 2/2")
        mock_echo.assert_any_call("Flare ready.")


def test_wait_with_retries(flare_poc, mock_federation_config):
    """Test wait with initial failures before success."""
    # Arrange
    mock_federation_config.client_sites = ["client1", "client2"]
    flare_poc.set_federation_config(mock_federation_config)

    mock_session = Mock()
    # First return empty list, then one client, then two clients
    mock_session.get_connected_client_list.side_effect = [
        [],
        ["client1"],
        ["client1", "client2"],
    ]

    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch.object(flare_poc, "connect", return_value=mock_session),
        patch("time.sleep") as mock_sleep,
    ):  # Mock sleep to speed up test
        flare_poc.wait()

        # Verify connection was attempted
        flare_poc.connect.assert_called_once()

        # Verify client list was checked multiple times
        assert mock_session.get_connected_client_list.call_count == 3

        # Verify progress messages
        mock_echo.assert_any_call("Connected to Flare.")
        mock_echo.assert_any_call("Connected clients: 0/2")
        mock_echo.assert_any_call("Connected clients: 1/2")
        mock_echo.assert_any_call("Connected clients: 2/2")
        mock_echo.assert_any_call("Flare ready.")

        # Verify sleep was called between retries
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1)


def test_wait_timeout(flare_poc, mock_federation_config):
    """Test wait timeout when not enough clients connect."""
    # Arrange
    mock_federation_config.client_sites = ["client1", "client2"]
    flare_poc.set_federation_config(mock_federation_config)

    mock_session = Mock()
    # Always return just one client
    mock_session.get_connected_client_list.return_value = ["client1"]

    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch.object(flare_poc, "connect", return_value=mock_session),
        patch("time.sleep") as mock_sleep,
        patch("time.monotonic", side_effect=[0, 15, 31]),
    ):  # Simulate timeout after 31 seconds
        with pytest.raises(click.Abort):
            flare_poc.wait(timeout=30)

        # Verify timeout message
        mock_echo.assert_any_call(click.style("Timeout waiting for Flare.", fg="red"))

        # Verify progress messages were shown
        mock_echo.assert_any_call("Connected to Flare.")
        mock_echo.assert_any_call("Connected clients: 1/2")


def test_wait_connection_error(flare_poc, mock_federation_config):
    """Test wait handling when connection fails."""
    # Arrange
    mock_federation_config.client_sites = ["client1", "client2"]
    flare_poc.set_federation_config(mock_federation_config)

    mock_session = Mock()
    mock_session.get_connected_client_list.side_effect = Exception("Connection failed")

    # Act & Assert
    with (
        patch("click.echo") as mock_echo,
        patch.object(flare_poc, "connect", return_value=mock_session),
        patch("time.sleep") as mock_sleep,
    ):
        with pytest.raises(click.Abort):
            flare_poc.wait(timeout=1)  # Short timeout for test

        # Verify error was logged
        mock_echo.assert_any_call("Error waiting for Flare: Connection failed")

        # Verify sleep was called after error
        mock_sleep.assert_called_with(5)


def test_cleanup_success(flare_poc, mock_federation_config):
    """Test successful cleanup of Flare environment."""
    # Arrange
    workspace_path = Path("/tmp/workspace")
    mock_federation_config.federation_workspace_dir = workspace_path
    flare_poc.set_federation_config(mock_federation_config)

    # Act & Assert
    with patch("click.echo") as mock_echo, patch("shutil.rmtree") as mock_rmtree:
        flare_poc.cleanup()

        # Verify workspace cleanup
        mock_rmtree.assert_called_once_with(workspace_path, ignore_errors=True)

        # Verify messages
        mock_echo.assert_any_call(f"Cleaning up POC workspace ({workspace_path})...")
        mock_echo.assert_any_call(click.style("POC workspace cleaned.", fg="green"))


def test_status(flare_poc, mock_federation_config, mock_shell_runner):
    """Test the status method for all status scenarios."""
    # Arrange
    mock_federation_config.project_path = "/tmp/project"
    mock_federation_config.config_path = "/tmp/config"
    mock_federation_config.admin_package = "/tmp/admin/package"
    mock_federation_config.app_name = "test_app"
    mock_federation_config.client_sites = ["client1", "client2"]
    flare_poc.set_federation_config(mock_federation_config)

    pid_file = flare_poc._workspace / "flare.pid"

    # Case 1: POC is running
    mock_shell_runner.is_running_in_background.side_effect = [True, True]
    with (
        patch("click.echo") as mock_echo,
        patch(
            "nv_dfm_core.cli.core._flare_poc.ShellRunner",
            return_value=mock_shell_runner,
        ),
    ):
        flare_poc.status()
        # Should print green running message
        mock_echo.assert_any_call(click.style("  POC is running.", fg="green"))

    # Case 2: POC is not running, no flare processes
    mock_shell_runner.is_running_in_background.side_effect = [False, False]
    with (
        patch("click.echo") as mock_echo,
        patch(
            "nv_dfm_core.cli.core._flare_poc.ShellRunner",
            return_value=mock_shell_runner,
        ),
    ):
        flare_poc.status()
        # Should print red not running message
        mock_echo.assert_any_call(click.style("  POC is not running.", fg="red"))
        # Should NOT print yellow warning
        assert not any(
            call.args and "Some flare processes are still alive" in call.args[0]
            for call in mock_echo.call_args_list
        )

    # Case 3: POC is not running, but some flare processes are alive
    mock_shell_runner.is_running_in_background.side_effect = [False, True]
    with (
        patch("click.echo") as mock_echo,
        patch(
            "nv_dfm_core.cli.core._flare_poc.ShellRunner",
            return_value=mock_shell_runner,
        ),
    ):
        flare_poc.status()
        # Should print red not running message
        mock_echo.assert_any_call(click.style("  POC is not running.", fg="red"))
        # Should print yellow warning
        mock_echo.assert_any_call(
            click.style(
                "  Some flare processes are still alive. Another POC instance might be running.",
                fg="yellow",
            )
        )
