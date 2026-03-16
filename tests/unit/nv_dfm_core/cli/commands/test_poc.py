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
from click import Abort
from click.testing import CliRunner
from unittest.mock import MagicMock, patch
import click

from nv_dfm_core.cli.commands._poc import poc
from nv_dfm_core.cli.config._federation import FederationConfig, FederationConfigManager
from nv_dfm_core.cli.core._context import CliContext
from nv_dfm_core.cli.core._flare_poc import FlarePoc

from nv_dfm_core.cli.main import cli


@pytest.fixture
def mock_context(tmp_path):
    with patch("nv_dfm_core.cli.main.CliContext") as mock:
        context = MagicMock(spec=CliContext)
        context.workspace_path = tmp_path
        context.debug = False
        context.add_config = MagicMock()  # Explicitly mock add_config
        mock.return_value = context
        yield context  # Return the context object, not the mock


@pytest.fixture
def mock_flare_poc():
    mock_flare_poc = MagicMock(spec=FlarePoc)

    # Patch FlarePoc to return our mock
    with patch("nv_dfm_core.cli.commands._poc.FlarePoc", return_value=mock_flare_poc):
        yield mock_flare_poc


@pytest.fixture
def mock_fed_config_mgr():
    with patch("nv_dfm_core.cli.commands._poc.FederationConfigManager") as mock:
        manager = MagicMock(spec=FederationConfigManager)
        mock.return_value = manager
        yield manager


@pytest.fixture
def mock_fed_config():
    with patch(
        "nv_dfm_core.cli.commands._poc.FederationConfig.add_to_context_and_get"
    ) as mock:
        yield mock


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_poc_logs_command(cli_runner, mock_context, mock_flare_poc):
    result = cli_runner.invoke(cli, ["poc", "logs"])
    assert result.exit_code == 0
    assert mock_flare_poc.show_logs.call_count == 1


def test_poc_cleanup_command(mock_fed_config, cli_runner, mock_context, mock_flare_poc):
    # Arrange
    federation_name = "test_federation"
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config

    # Act
    result = cli_runner.invoke(cli, ["poc", "cleanup", "-f", federation_name])

    # Assert
    assert result.exit_code == 0
    mock_flare_poc.cleanup.assert_called_once()


def test_poc_cleanup_command_federation_not_found(
    mock_fed_config, cli_runner, mock_context, mock_flare_poc
):
    # Arrange
    federation_name = "non_existent_federation"
    mock_fed_config.side_effect = click.Abort("Federation not found")

    # Act
    result = cli_runner.invoke(cli, ["poc", "cleanup", "-f", federation_name])

    # Assert
    assert result.exit_code != 0
    mock_flare_poc.cleanup.assert_not_called()


def test_poc_restart_command(mock_fed_config, cli_runner, mock_context, mock_flare_poc):
    # Arrange
    federation_name = "test_federation"
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config

    # Act
    result = cli_runner.invoke(cli, ["poc", "restart", "-f", federation_name])

    # Assert
    assert result.exit_code == 0
    # Should call stop and start, but not cleanup
    mock_flare_poc.stop.assert_called_once()
    mock_flare_poc.cleanup.assert_not_called()
    mock_flare_poc.prepare.assert_called_once()
    mock_flare_poc.start.assert_called_once()
    mock_flare_poc.wait.assert_called_once()


def test_poc_restart_command_with_cleanup(
    mock_fed_config, cli_runner, mock_context, mock_flare_poc
):
    # Arrange
    federation_name = "test_federation"
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config

    # Act
    result = cli_runner.invoke(
        cli, ["poc", "restart", "-f", federation_name, "--do-cleanup"]
    )

    # Assert
    assert result.exit_code == 0
    # Should call stop, cleanup, and start in order
    mock_flare_poc.stop.assert_called_once()
    mock_flare_poc.cleanup.assert_called_once()
    mock_flare_poc.prepare.assert_called_once()
    mock_flare_poc.start.assert_called_once()
    mock_flare_poc.wait.assert_called_once()


def test_poc_stop_command(cli_runner, mock_context, mock_flare_poc):
    # Act
    result = cli_runner.invoke(cli, ["poc", "stop"])

    # Assert
    assert result.exit_code == 0
    mock_flare_poc.stop.assert_called_once()


def test_poc_start_command(mock_fed_config, cli_runner, mock_context, mock_flare_poc):
    # Arrange
    federation_name = "test_federation"
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config
    mock_process = MagicMock()
    mock_flare_poc.start.return_value = mock_process

    # Act
    result = cli_runner.invoke(cli, ["poc", "start", "-f", federation_name])

    # Assert
    assert result.exit_code == 0
    mock_flare_poc.set_federation_config.assert_called_once_with(mock_config)
    mock_flare_poc.prepare.assert_called_once()
    mock_flare_poc.start.assert_called_once_with(False)  # debug=False by default
    mock_flare_poc.wait.assert_called_once()
    assert "Flare is ready!" in result.output


def test_poc_start_command_prepare_only(
    mock_fed_config, cli_runner, mock_context, mock_flare_poc
):
    # Arrange
    federation_name = "test_federation"
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config

    # Act
    result = cli_runner.invoke(
        cli, ["poc", "start", "-f", federation_name, "--prepare-only"]
    )

    # Assert
    assert result.exit_code == 0
    mock_flare_poc.set_federation_config.assert_called_once_with(mock_config)
    mock_flare_poc.prepare.assert_called_once()
    mock_flare_poc.start.assert_not_called()
    mock_flare_poc.wait.assert_not_called()
    assert "Flare workspace prepared." in result.output


def test_poc_start_command_debug(
    mock_fed_config, cli_runner, mock_context, mock_flare_poc
):
    # Arrange
    federation_name = "test_federation"
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config
    mock_process = MagicMock()
    mock_flare_poc.start.return_value = mock_process

    # Act
    result = cli_runner.invoke(cli, ["poc", "start", "-f", federation_name, "--debug"])

    # Assert
    assert result.exit_code == 0
    mock_flare_poc.set_federation_config.assert_called_once_with(mock_config)
    mock_flare_poc.prepare.assert_called_once()
    mock_flare_poc.start.assert_called_once_with(True)  # debug=True
    mock_flare_poc.wait.assert_called_once()
    assert "Flare is ready!" in result.output


def test_poc_start_command_wait_fails(
    mock_fed_config, cli_runner, mock_context, mock_flare_poc
):
    # Arrange
    federation_name = "test_federation"
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config
    mock_process = MagicMock()
    mock_flare_poc.start.return_value = mock_process
    mock_flare_poc.wait.side_effect = click.Abort("Wait failed")

    # Act
    result = cli_runner.invoke(cli, ["poc", "start", "-f", federation_name])

    # Assert
    assert result.exit_code != 0
    mock_flare_poc.set_federation_config.assert_called_once_with(mock_config)
    mock_flare_poc.prepare.assert_called_once()
    mock_flare_poc.start.assert_called_once_with(False)
    mock_flare_poc.wait.assert_called_once()
    mock_flare_poc.stop.assert_called_once_with(mock_process)


def test_poc_wait_command(mock_fed_config, cli_runner, mock_context, mock_flare_poc):
    # Arrange
    federation_name = "test_federation"
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config

    # Act
    result = cli_runner.invoke(cli, ["poc", "wait", "-f", federation_name])

    # Assert
    assert result.exit_code == 0
    mock_flare_poc.set_federation_config.assert_called_once_with(mock_config)
    mock_flare_poc.wait.assert_called_once()
    assert "Flare is ready!" in result.output


def test_poc_wait_command_fails(
    mock_fed_config, cli_runner, mock_context, mock_flare_poc
):
    # Arrange
    federation_name = "test_federation"
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config
    mock_flare_poc.wait.side_effect = click.Abort("Wait failed")

    # Act
    result = cli_runner.invoke(cli, ["poc", "wait", "-f", federation_name])

    # Assert
    assert result.exit_code != 0
    mock_flare_poc.set_federation_config.assert_called_once_with(mock_config)
    mock_flare_poc.wait.assert_called_once()


def test_poc_status_command(mock_fed_config, cli_runner, mock_context, mock_flare_poc):
    # Arrange
    federation_name = "test_federation"
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config

    # Act
    result = cli_runner.invoke(cli, ["poc", "status", "-f", federation_name])

    # Assert
    assert result.exit_code == 0
    mock_flare_poc.set_federation_config.assert_called_once_with(mock_config)
    mock_flare_poc.status.assert_called_once()


def test_poc_status_command_default_federation(
    mock_fed_config, cli_runner, mock_context, mock_flare_poc
):
    # Arrange
    mock_config = MagicMock(spec=FederationConfig)
    mock_fed_config.return_value = mock_config

    # Act
    result = cli_runner.invoke(cli, ["poc", "status"])

    # Assert
    assert result.exit_code == 0
    mock_flare_poc.set_federation_config.assert_called_once_with(mock_config)
    mock_flare_poc.status.assert_called_once()


def test_poc_status_command_federation_not_found(
    mock_fed_config, cli_runner, mock_context, mock_flare_poc
):
    # Arrange
    federation_name = "non_existent_federation"
    mock_fed_config.side_effect = click.Abort("Federation not found")

    # Act
    result = cli_runner.invoke(cli, ["poc", "status", "-f", federation_name])

    # Assert
    assert result.exit_code != 0
    mock_flare_poc.set_federation_config.assert_not_called()
    mock_flare_poc.status.assert_not_called()
