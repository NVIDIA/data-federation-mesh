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
from unittest.mock import patch
import tempfile
from pathlib import Path

import pytest
import click
from click.testing import CliRunner

from nv_dfm_core.cli.commands._config import config
from nv_dfm_core.cli.core._context import CliContext
from nv_dfm_core.cli.config._cli import CliConfig


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()


@pytest.fixture
def cli_context(temp_workspace):
    context = CliContext()
    context.workspace_path = temp_workspace
    return context


@pytest.fixture
def cli_config(temp_workspace):
    config = CliConfig(debug=True)
    config.initialize()  # Initialize the manager
    config.workspace.path = temp_workspace  # Set workspace path
    return config


def cli_config_with_custom_config_path(temp_workspace, custom_config_path):
    config = CliConfig(debug=True)

    config.initialize()  # Initialize the manager
    config.workspace.path = temp_workspace  # Set workspace path

    config.current_config_path = custom_config_path

    return config


@pytest.fixture
def runner():
    return CliRunner()


def test_config_show_no_config(runner, cli_context, temp_workspace):
    """Test config show command when no config file exists"""
    # Ensure no config file exists
    cli_config = cli_config_with_custom_config_path(temp_workspace, None)
    with patch(
        "nv_dfm_core.cli.config._cli.Utils.get_cli_config_path", return_value=None
    ):
        cli_context.add_config("cli", cli_config)
        result = runner.invoke(config, ["show"], obj=cli_context)
        assert result.exit_code == 0
        assert "No config file found, using defaults" in result.output


def test_config_show_with_config(runner, cli_context, temp_workspace):
    """Test config show command when config file exists"""
    # Create a sample config file
    config_path = temp_workspace / "config.yaml"
    cli_config = cli_config_with_custom_config_path(temp_workspace, config_path)
    with open(config_path, "w") as f:
        f.write("test: config")

    cli_context.add_config("cli", cli_config)

    result = runner.invoke(config, ["show"], obj=cli_context)
    assert result.exit_code == 0
    assert str(config_path.absolute()) in result.output
    assert "test: config" in result.output
