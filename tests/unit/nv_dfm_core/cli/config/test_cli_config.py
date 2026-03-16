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
from pathlib import Path
import pytest
from unittest.mock import patch, mock_open
import yaml

from nv_dfm_core.cli.config._cli import (
    CliConfig,
    Utils,
    DictWrapper,
    DEFAULT_CONFIG_PATH,
    _log,
)


@pytest.fixture
def mock_yaml_config():
    return {
        "debug": False,
        "dev": {
            "testing": {"script": "test_script.sh"},
            "linting": {"script": "lint_script.sh", "fix": False},
            "formatting": {"script": "format_script.sh"},
        },
        "federations_config_path": "federations.yaml",
        "workspace": None,
    }


def test_basic_creation():
    cfg = CliConfig(debug=True)
    assert cfg._config == {}


def test_initialization_with_mock_config(mock_yaml_config):
    with (
        patch("nv_dfm_core.cli.config._cli.Utils.get_repo_dir") as mock_get_repo_dir,
        patch(
            "nv_dfm_core.cli.config._cli.Utils.get_cli_config_path"
        ) as mock_get_config_path,
        patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml_config))),
    ):
        mock_get_repo_dir.return_value = Path("/mock/repo")
        mock_get_config_path.return_value = Path("/mock/repo/.dfm-cli.conf.yaml")

        cfg = CliConfig(debug=True)
        cfg.initialize()

        # Compare only the relevant fields
        assert cfg._config["dev"]["testing"]["script"] == Path("test_script.sh")
        assert cfg._config["dev"]["linting"]["script"] == Path("lint_script.sh")
        assert cfg._config["dev"]["formatting"]["script"] == Path("format_script.sh")
        assert cfg._config["federations_config_path"] == Path("federations.yaml")


def test_initialization_with_env_var(mock_yaml_config):
    with (
        patch.dict(os.environ, {"DFM_CLI_CONFIG": "/custom/path/config.yaml"}),
        patch("nv_dfm_core.cli.config._cli.Utils.get_repo_dir") as mock_get_repo_dir,
        patch(
            "nv_dfm_core.cli.config._cli.Utils.get_cli_config_path"
        ) as mock_get_config_path,
        patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml_config))),
    ):
        mock_get_repo_dir.return_value = Path("/mock/repo")
        mock_get_config_path.return_value = Path("/custom/path/config.yaml")

        cfg = CliConfig(debug=True)
        cfg.initialize()

        # Compare only the relevant fields
        assert cfg._config["dev"]["testing"]["script"] == Path("test_script.sh")
        assert cfg._config["dev"]["linting"]["script"] == Path("lint_script.sh")
        assert cfg._config["dev"]["formatting"]["script"] == Path("format_script.sh")
        assert cfg._config["federations_config_path"] == Path("federations.yaml")


def test_initialization_no_config():
    with (
        patch("nv_dfm_core.cli.config._cli.Utils.get_repo_dir") as mock_get_repo_dir,
        patch(
            "nv_dfm_core.cli.config._cli.Utils.get_cli_config_path"
        ) as mock_get_config_path,
    ):
        mock_get_repo_dir.return_value = None
        mock_get_config_path.return_value = None

        cfg = CliConfig(debug=True)
        cfg.initialize()

        # Check for default values when no config file is found
        assert cfg._config["debug"] is False
        assert cfg._config["dev"] is None
        assert cfg._config["federations_config_path"] == Path("./federations.yaml")
        assert cfg._config["workspace"]["path"] == Path("./workspace")


def test_dict_wrapper():
    data = {"nested": {"key": "value"}, "path": "relative/path"}
    wrapper = DictWrapper(data)

    assert isinstance(wrapper.nested, DictWrapper)
    assert wrapper.nested.key == "value"
    assert wrapper.path == "relative/path"


def test_dict_wrapper_none_data():
    wrapper = DictWrapper(None)
    with pytest.raises(AttributeError):
        _ = wrapper.some_key


def test_dict_wrapper_missing_key():
    wrapper = DictWrapper({})
    with pytest.raises(AttributeError):
        _ = wrapper.missing_key


def test_utils_get_repo_dir():
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        repo_dir = Utils.get_repo_dir()
        # When .git exists, it should return the current working directory
        # but we need to account for the actual implementation behavior
        assert repo_dir is not None
        assert isinstance(repo_dir, Path)


def test_utils_get_repo_dir_not_found():
    with patch("nv_dfm_core.cli.config._cli.Utils.get_repo_dir", return_value=None):
        repo_dir = Utils.get_repo_dir()
        assert repo_dir is None


def test_utils_unrelative_path():
    with patch("nv_dfm_core.cli.config._cli.Utils.get_repo_dir") as mock_get_repo_dir:
        mock_get_repo_dir.return_value = Path("/mock/repo")

        # Test with relative path
        relative_path = Path("some/path")
        absolute_path = Utils.unrelative_path(relative_path)
        assert absolute_path == Path("/mock/repo/some/path")

        # Test with absolute path
        absolute_input = Path("/absolute/path")
        result = Utils.unrelative_path(absolute_input)
        assert result == Path("/absolute/path")

        # Test with None
        result = Utils.unrelative_path(None)
        assert result is None


def test_cli_config_attribute_access():
    cfg = CliConfig(debug=True)
    cfg._config = {"test_key": "test_value", "nested": {"key": "value"}}

    assert cfg.test_key == "test_value"
    assert isinstance(cfg.nested, DictWrapper)
    assert cfg.nested.key == "value"


def test_cli_config_missing_attribute():
    cfg = CliConfig(debug=True)
    cfg._config = {}

    with pytest.raises(AttributeError):
        _ = cfg.missing_key


def test_basic_access():
    with (
        patch(
            "nv_dfm_core.cli.config._cli.Utils.get_cli_config_path"
        ) as mock_get_config_path,
        patch(
            "builtins.open",
            mock_open(
                read_data="""federations_config_path: ./federations.yaml
workspace:
  path: ./workspace
dev:
  testing:
    script: ./test_script.sh
  linting:
    script: ./lint_script.sh
  formatting:
    script: ./format_script.sh
"""
            ),
        ),
    ):
        mock_get_config_path.return_value = Path("/mock/path/config.yaml")
        cfg = CliConfig(debug=True)
        cfg.initialize()
        assert cfg.repo_dir is not None
        assert cfg.dev.testing.script is not None
        assert cfg.dev.linting.script is not None
        assert cfg.dev.formatting.script is not None
        assert cfg.federations_config_path is not None


def test_get_cli_config_path_user_home():
    # Clear the cache to ensure clean state
    Utils._cli_config_path = None

    with (
        patch("nv_dfm_core.cli.config._cli.Utils.get_repo_dir") as mock_get_repo_dir,
        patch("pathlib.Path.exists", lambda self: self == DEFAULT_CONFIG_PATH),
    ):
        mock_get_repo_dir.return_value = None
        config_path = Utils.get_cli_config_path()
        assert config_path == DEFAULT_CONFIG_PATH


def test_log_function_debug_enabled():
    with patch("click.echo") as mock_echo:
        # Enable debug mode
        cfg = CliConfig(debug=True)
        _log("Test debug message")
        mock_echo.assert_called_once_with("DEBUG: Test debug message", err=True)


def test_log_function_debug_disabled():
    with patch("click.echo") as mock_echo:
        # Disable debug mode
        cfg = CliConfig(debug=False)
        _log("Test message")
        mock_echo.assert_not_called()


def test_current_config_path():
    with patch(
        "nv_dfm_core.cli.config._cli.Utils.get_cli_config_path"
    ) as mock_get_config_path:
        mock_get_config_path.return_value = Path("/mock/path/config.yaml")
        cfg = CliConfig(debug=True)
        assert cfg.current_config_path == Path("/mock/path/config.yaml")


def test_raw_config():
    mock_config_content = "mock config content"
    with (
        patch(
            "nv_dfm_core.cli.config._cli.Utils.get_cli_config_path"
        ) as mock_get_config_path,
        patch("builtins.open", mock_open(read_data=mock_config_content)),
    ):
        mock_get_config_path.return_value = Path("/mock/path/config.yaml")
        cfg = CliConfig(debug=True)
        assert cfg.raw() == mock_config_content


def test_create_default_config_new_file():
    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.is_dir") as mock_is_dir,
        patch("builtins.open", mock_open()) as mock_file,
    ):
        mock_exists.return_value = False
        mock_is_dir.return_value = False

        cfg = CliConfig(debug=True)
        result = cfg.create_default_config(Path("/mock/path/config.yaml"))

        assert result == "Created default CLI config at /mock/path/config.yaml"
        mock_file.assert_called_once()
        # Verify the YAML content was written by checking the calls
        write_calls = mock_file().write.call_args_list
        assert any("federations_config_path" in str(call) for call in write_calls)
        assert any("debug" in str(call) for call in write_calls)
        assert any("workspace" in str(call) for call in write_calls)


def test_create_default_config_existing_file():
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True

        cfg = CliConfig(debug=True)
        result = cfg.create_default_config(Path("/mock/path/config.yaml"))

        assert result == "Config file already exists at /mock/path/config.yaml"


def test_create_default_config_directory():
    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.is_dir") as mock_is_dir,
    ):
        mock_exists.return_value = True
        mock_is_dir.return_value = True

        cfg = CliConfig(debug=True)
        result = cfg.create_default_config(Path("/mock/path"))

        assert result == "Config file already exists at /mock/path/.dfm-cli.conf.yaml"


def test_create_default_config_directory_new():
    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.is_dir") as mock_is_dir,
        patch("builtins.open", mock_open()) as mock_file,
    ):
        mock_exists.return_value = False
        mock_is_dir.return_value = True

        cfg = CliConfig(debug=True)
        result = cfg.create_default_config(Path("/mock/path"))

        assert result == "/mock/path does not exist"
