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

from pathlib import Path

import pytest
from unittest.mock import patch, mock_open
from nv_dfm_core.cli.core._context import CliContext
from nv_dfm_core.cli.config._cli import CliConfig


@pytest.fixture
def cli_context():
    """Fixture that provides a CliContext instance."""
    with (
        patch(
            "nv_dfm_core.cli.config._cli.Utils.get_cli_config_path"
        ) as mock_get_config_path,
        patch(
            "builtins.open",
            mock_open(
                read_data="federations_config_path: ./federations.yaml\nworkspace:\n  path: ./workspace\n"
            ),
        ),
    ):
        mock_get_config_path.return_value = Path("/mock/path/config.yaml")
        return CliContext()


def test_cli_context_initialization(monkeypatch):
    """Test that CliContext initializes correctly with different debug states."""
    with (
        patch(
            "nv_dfm_core.cli.config._cli.Utils.get_cli_config_path"
        ) as mock_get_config_path,
        patch(
            "builtins.open",
            mock_open(
                read_data="federations_config_path: ./federations.yaml\nworkspace:\n  path: ./workspace\n"
            ),
        ),
    ):
        mock_get_config_path.return_value = Path("/mock/path/config.yaml")

        # Test with debug disabled
        monkeypatch.setenv("DFM_CLI_DEBUG", "0")
        context_debug_off = CliContext()
        assert context_debug_off.debug is False
        assert isinstance(context_debug_off._configs, dict)
        assert "cli" in context_debug_off._configs
        assert isinstance(context_debug_off._configs["cli"], CliConfig)

        # Test with debug enabled
        monkeypatch.setenv("DFM_CLI_DEBUG", "1")
        context_debug_on = CliContext()
        assert context_debug_on.debug is True
        assert isinstance(context_debug_on._configs, dict)
        assert "cli" in context_debug_on._configs
        assert isinstance(context_debug_on._configs["cli"], CliConfig)


def test_add_config(cli_context):
    """Test adding a configuration to the context."""
    test_config = {"test": "config"}
    cli_context.add_config("test", test_config)
    assert "test" in cli_context._configs
    assert cli_context._configs["test"] == test_config


def test_get_config(cli_context):
    """Test retrieving configurations from the context."""
    # Test getting existing config
    cli_config = cli_context.get_config("cli")
    assert cli_config is not None
    assert isinstance(cli_config, CliConfig)

    # Test getting non-existent config
    with pytest.raises(RuntimeError):
        cli_context.get_config("non_existent")


def test_workspace_initialization(cli_context):
    """Test that workspace is properly initialized."""
    workspace_path = cli_context.workspace_path
    assert isinstance(workspace_path, Path)
    assert workspace_path.exists()
    assert workspace_path.is_dir()


def test_workspace_path_persistence(cli_context):
    """Test that workspace path is properly stored and accessible."""
    original_path = cli_context.workspace_path
    assert original_path == cli_context.get_config("cli").workspace.path
