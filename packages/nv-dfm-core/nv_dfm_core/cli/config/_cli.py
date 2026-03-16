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

"""
CLI Configuration Management Module

This module provides functionality for managing the Data Federation Mesh (DFM) CLI configuration.
It handles loading, saving, and managing configuration settings from various sources including
YAML files and environment variables. The configuration system follows a hierarchical approach
for determining the configuration source.
"""

import json
import os
from pathlib import Path
from typing import TypeAlias, cast

import click
import yaml
from typing_extensions import override

from .models._cli import Cli as CliModel
from .models._cli import Workspace as WorkspaceModel

# Default paths for configuration and workspace
DEFAULT_CONFIG_FILE = ".dfm-cli.conf.yaml"
DEFAULT_CONFIG_PATH = Path("~/.dfm/").expanduser() / DEFAULT_CONFIG_FILE


_debug = False


def _log(message: str) -> None:
    """
    Log a debug message if debug mode is enabled.

    Args:
        message (str): Message to log
    """
    if _debug:
        click.echo(f"DEBUG: {message}", err=True)


class Utils:
    # Class-level cache for repo_dir
    _repo_dir_cache: Path | None = None
    _cli_config_path: Path | None = None

    @classmethod
    def get_repo_dir(cls) -> Path | None:
        """
        Get the repository directory.
        """
        # Return cached value if available
        if cls._repo_dir_cache is not None:
            return cls._repo_dir_cache

        # Try to find the repo directory.
        repo_dir = Path.cwd()
        # Traverse up the directory tree until we find the config file or reach the root.
        while not (repo_dir / DEFAULT_CONFIG_FILE).exists():
            if repo_dir == Path("/"):
                cls._repo_dir_cache = None
                return None
            repo_dir = repo_dir.parent
        # Cache and return the repo directory.
        cls._repo_dir_cache = repo_dir
        return repo_dir

    @staticmethod
    def unrelative_path(path: Path | None) -> Path | None:
        """
        Convert a relative path to an absolute path.
        """
        if not path:
            return None
        assert isinstance(path, Path)
        repo_dir = Utils.get_repo_dir()
        if not path.is_absolute() and repo_dir:
            return repo_dir / path
        return path

    @staticmethod
    def set_cli_config_path(path: Path):
        """Set the path to the current configuration file."""
        _log(f"Setting CLI config path to {path}")
        Utils._cli_config_path = path

    @staticmethod
    def get_cli_config_path():
        """
        Determine the CLI configuration file path following the hierarchy:
        1. DFM_CLI_CONFIG envvar value
        2. .dfm-cli.conf.yaml file in the repo directory
        3. ${HOME}/.dfm/dfm-cli.conf.yaml (DEFAULT_CONFIG_PATH)

        Returns:
            Path | None: Path to the configuration file if found, None otherwise
        """

        if Utils._cli_config_path:
            return Utils._cli_config_path

        # Try to get the repo directory.
        repo_dir = Utils.get_repo_dir()

        # Check for config path in environment variable - we can use it to override the default config path
        # e.g. in CI.
        cli_config_path = None
        env_config_str = os.environ.get("DFM_CLI_CONFIG")
        if env_config_str:
            env_config_path = Path(env_config_str)
            _log(f"Checking for CLI config at {env_config_str} (DFM_CLI_CONFIG)")
            if env_config_path.exists():
                _log(f"Found CLI config at {env_config_path}")
                cli_config_path = env_config_path
        elif repo_dir:
            cli_config_path = repo_dir / DEFAULT_CONFIG_FILE
        else:
            # Now check the default user config path in home directory
            user_config_path = DEFAULT_CONFIG_PATH
            _log(f"Checking for CLI config at {user_config_path}")
            if user_config_path.exists():
                _log(f"Found CLI config at {user_config_path}")
                cli_config_path = user_config_path

        if not cli_config_path:
            # No configuration file found in any of the expected locations
            _log("No CLI config found")
            return None

        if not cli_config_path.exists():
            _log(f"CLI config file {cli_config_path} does not exist")
            return None

        Utils.set_cli_config_path(cli_config_path)
        return cli_config_path


ConfigBaseT: TypeAlias = str | int | float | bool | Path | None
ConfigDictT: TypeAlias = dict[str, ConfigBaseT] | None


class DictWrapper:
    """
    Wrapper class for a dictionary that allows for dynamic attribute access.
    """

    def __init__(self, data: ConfigDictT):
        self._data: ConfigDictT = data

    def __getattr__(self, name: str) -> "ConfigBaseT | DictWrapper":
        """Dynamically access attributes from the loaded config dictionary.

        This allows accessing nested attributes like dev.testing.script
        which will return self._config["dev"]["testing"]["script"].

        Args:
            name (str): Name of the attribute to access

        Returns:
            The requested attribute value from self._config

        Raises:
            AttributeError: If the attribute doesn't exist in self._config
        """
        if self._data is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' does not contain config data"
            )
        if name not in self._data:
            raise AttributeError(
                f"'{self.__class__.__name__}' does not contain config data for {name}"
            )

        attr = self._data[name]

        if isinstance(attr, dict):
            return DictWrapper(attr)

        # if it's a path, convert it to an absolute one.
        if isinstance(attr, Path):
            attr = Utils.unrelative_path(attr)

        return attr


class CliConfig:
    """
    Runtime configuration model for CLI settings.
    """

    def __init__(self, debug: bool = False):
        global _debug
        _debug = debug
        self._config: ConfigDictT = {}
        self.repo_dir: Path | None = None

    def initialize(self) -> None:
        # Try to find the repo directory.
        self.repo_dir = Utils.get_repo_dir()

        # Get the path to the CLI configuration file using the established hierarchy
        config_path = Utils.get_cli_config_path()

        # If a CLI config file was found, load and parse it
        if config_path:
            _log(f"Loading CLI config from {config_path}")
            with open(config_path, "r") as f:
                # Do we have Python++ now?
                cli_config: ConfigDictT = cast(ConfigDictT, yaml.safe_load(f))
            # Convert the loaded YAML into our configuration model to enforce the model
            # and validate the data.
            validated = CliModel.model_validate(cli_config)
        else:
            # No CLI config found, create a default configuration
            validated = CliModel(
                federations_config_path=Path("./federations.yaml"),
                workspace=WorkspaceModel(path=Path("./workspace")),
            )
            _log("No CLI config found, created default configuration")

        # Now store the validated data in a dictionary.
        # This might seem strange, but we can build a nice,
        # elastic accessing mechanism on top of it.
        self._config = validated.model_dump()
        _log("Successfully loaded CLI config")

    def __getattr__(self, name: str) -> ConfigBaseT | DictWrapper:
        """Dynamically access attributes from the loaded config dictionary."""
        if self._config is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' does not contain config data"
            )
        if name not in self._config:
            raise AttributeError(
                f"'{self.__class__.__name__}' does not contain config data for {name}"
            )

        attr: ConfigBaseT = self._config[name]

        if isinstance(attr, dict):
            return DictWrapper(attr)

        # if it's a path, convert it to an absolute one.
        if isinstance(attr, Path):
            attr = Utils.unrelative_path(attr)
        return attr

    @property
    def current_config_path(self) -> Path | None:
        """Get the path to the current configuration file."""
        return Utils.get_cli_config_path()

    @current_config_path.setter
    def current_config_path(self, path: Path) -> None:
        """Set the path to the current configuration file."""
        Utils.set_cli_config_path(path)

    def raw(self) -> str:
        """
        Get the raw contents of the current configuration file.

        Returns:
            str: Raw contents of the configuration file

        Raises:
            RuntimeError: If configuration has not been initialized
        """

        if self.current_config_path is None:
            raise RuntimeError("No configuration file path set")

        with open(self.current_config_path, "r") as f:
            return f.read()

    @override
    def __str__(self) -> str:
        global _debug
        return f"""CliConfig(
                    repo_dir={self.repo_dir},
                    config={json.dumps(self._config, indent=2)}
                    debug={_debug}
               )"""

    @override
    def __repr__(self) -> str:
        return self.__str__()

    def create_default_config(self, path: Path = DEFAULT_CONFIG_PATH) -> str:
        """
        Create a default CLI configuration file if it doesn't exist.

        Returns:
            str: Message indicating the result of the operation
        """

        # If the path is a directory, use the default config file name.
        if path.is_dir():
            if not path.exists():
                return f"{path} does not exist"
            path = path / DEFAULT_CONFIG_FILE

        # If the path already exists, return an error.
        if path.exists():
            return f"Config file already exists at {path}"

        config = CliModel(
            federations_config_path=Path("./federations.yaml"),
            workspace=WorkspaceModel(path=Path("./workspace")),
        )

        with open(path, "w") as f:
            yaml.dump(
                config.model_dump(mode="json"),
                f,
                indent=2,
                default_flow_style=False,
                sort_keys=False,
            )

        return f"Created default CLI config at {path}"
