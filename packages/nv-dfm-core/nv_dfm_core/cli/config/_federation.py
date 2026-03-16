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
Federation Configuration Management Module

This module provides functionality for managing Data Federation Mesh (DFM) federation configurations.
It handles loading, saving, and managing federation-specific settings including workspace paths,
configuration files, and project settings. The module supports multiple federations and provides
methods for managing federation-specific configurations.
"""

import json
from pathlib import Path
from typing import Callable, ParamSpec, TypeAlias, TypeVar, cast

import click
import yaml
from pydantic import BaseModel
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import YAMLError as RuamelYAMLError
from typing_extensions import override

from ...gen.apigen._config_models import FederationObject, ProvisionInfoObject
from ..config._cli import CliConfig
from ..core._context import CliContext

GenericConfigDataT: TypeAlias = dict[str, object] | None

# Type variables for the decorator
P = ParamSpec("P")
R = TypeVar("R")


class FederationConfigStored(BaseModel):
    """
    Data model for federation configuration settings.

    Attributes:
        federation_dir (Path): Main federation directory containing configuration, project files, code, etc.
                               Can be relative to the repo root or absolute.
        config_path (Path): Path to federation configuration file, relative to federation_dir
        project_path (Path): Path to federation project file, relative to federation_dir
    """

    federation_dir: Path  # Main federation directory; contains configuration, project files, code, etc.
    config_path: Path  # relative to federation_dir
    project_path: Path  # relative to federation_dir


class FederationConfig:
    """
    Manages configuration for a single DFM Federation.

    This class handles loading, parsing, and managing configuration for a specific federation.
    It supports configuration from both federation-specific config files and project files.

    Attributes:
        name (str): Name of the federation
        federation_workspace_dir (Path): Path to federation-specific workspace
        federation_dir (Path): Main federation directory
        config_path (Path): Path to federation configuration file
        project_path (Path): Path to federation project file
        module (str): Code package module name
        server_sites (int): Number of server sites
        client_sites (int): Number of client sites
        app_name (str): Name of the admin application
        admin_package (Path): Path to the admin package
    """

    @staticmethod
    def add_to_context_and_get(dc: CliContext, federation: str) -> "FederationConfig":
        """
        Add federation configuration to CLI context and return it.

        Args:
            dc (CliContext): CLI context to add configuration to
            federation (str): Name of the federation

        Returns:
            FederationConfig: The federation configuration

        Raises:
            click.Abort: If the specified federation is not configured
        """
        fed_cfg_mgr = FederationConfigManager.add_to_context_and_get(dc)
        if federation not in fed_cfg_mgr.get_federation_names():
            click.echo(f"Federation {federation} not configured")
            click.echo("Use 'dfm fed config set' to configure a federation.")
            raise click.Abort()
        return fed_cfg_mgr.get_config(federation)

    def __init__(self, debug: bool = False):
        """
        Initialize the federation configuration.

        Args:
            debug (bool): Enable debug mode for verbose logging
        """
        # We prefer to initialize in a separate function to allow for
        # lazy initialization of config.
        self._debug: bool = debug
        # Initialize all fields to None to indicate what fields will be available.
        # All fields will be set in the initialize() method.
        self.name: str | None = None
        self.module: str | None = None
        self.server_sites: list[str] | None = None
        self.client_sites: list[str] | None = None
        self.app_name: str | None = None
        self.version: str | None = None
        self.flare_project_name: str | None = None
        # Raw data from federation config file.
        self._stored: FederationConfigStored | None = None
        self._workspace_dir: Path | None = None
        self.has_federation_dir: bool = False
        self.has_config_path: bool = False
        self.has_project_path: bool = False
        self.poc_mode: bool = False
        self.provision_info: ProvisionInfoObject | None = None

    def _log(self, message: str) -> None:
        """
        Log a debug message if debug mode is enabled.

        Args:
            message (str): Message to log
        """
        if self._debug:
            click.echo(f"DEBUG[{self.__class__.__name__}]: {message}", err=True)

    def initialize(
        self,
        name: str,
        workspace_dir: Path | str,
        federation_dir: Path | str,
        config_path: Path | str,
        project_path: Path | str,
    ) -> None:
        """
        Initialize federation configuration with the provided parameters.

        Args:
            name (str): Name of the federation
            workspace_dir (Path | str): Path to the DFM CLI workspace directory
            federation_dir (Path | str): Path to the federation directory
            config_path (Path | str | None): Path to federation config file
            project_path (Path | str | None): Path to federation project file

        Raises:
            FileNotFoundError: If required files or directories are not found
        """
        # Log the initialization parameters for debugging
        self._log(f"""Initializing federation config with:
            name: {name},
            workspace dir: {workspace_dir},
            federation dir: {federation_dir},
            config path: {config_path},
            project path: {project_path}
        """)

        # Store the federation name
        self.name = name

        self._stored = FederationConfigStored(
            federation_dir=Path(federation_dir),
            config_path=Path(config_path),
            project_path=Path(project_path),
        )
        self._workspace_dir = Path(workspace_dir)

        # Check which paths exist
        self._check_paths()

        # Load configuration from both federation and project files
        federation_config, project_config = self._load_config_files()

        # Parse the loaded configurations to extract relevant settings
        self._parse_federation_config(federation_config)
        self._parse_federation_project(project_config)

        # Log successful initialization with the complete configuration
        self._log(f"Initialized federation config: {self}")

    @property
    def federation_workspace_dir(self) -> Path:
        """
        Get the federation-specific workspace directory.

        Returns:
            Path: Path to the federation-specific workspace directory
        """
        # Create the federation-specific workspace path by appending the federation name
        # to the base workspace path. This ensures each federation has its own isolated workspace.
        assert self.name
        assert self._workspace_dir
        name = self.name
        if self.poc_mode:
            name = name + "_poc"
        return (self._workspace_dir / name).expanduser().resolve()

    @property
    def federation_dir(self) -> Path:
        """
        Get the path to the federation directory.
        """
        assert self._stored
        return self._stored.federation_dir.expanduser().resolve()

    @property
    def config_path(self) -> Path:
        """
        Get the path to the federation configuration file.
        The config path points to the federation's main configuration file.
        """
        # Use default if not provided.
        assert self._stored
        config_path = self._stored.config_path or self._default_config_path(
            self._stored.federation_dir
        )
        # Handle relative path by making it relative to the federation directory
        if not config_path.is_absolute():
            config_path = self._stored.federation_dir / config_path
        # Ensure path is absolute and expanded (resolve any symlinks)
        return config_path.expanduser().resolve()

    @property
    def project_path(self) -> Path:
        """
        Get the path to the federation project file.
        The project path points to the federation's project-specific configuration.
        """
        assert self._stored
        project_path = self._stored.project_path or self._default_project_path(
            self._stored.federation_dir
        )
        # Handle relative path by making it relative to the federation directory
        if not project_path.is_absolute():
            project_path = self._stored.federation_dir / project_path
        # Ensure path is absolute and expanded (resolve any symlinks)
        return project_path.expanduser().resolve()

    @property
    def admin_package(self) -> Path:
        """
        Get the path to the admin package.
        """
        assert self.name
        assert self.module
        assert self.app_name
        return self.federation_workspace_dir / self.module / "prod_00" / self.app_name

    def patch(self, values: GenericConfigDataT | str | Path | list[Path]) -> None:
        """
        Patch the federation configuration with values from file(s) or dictionary.

        This method allows overriding provisioning information for various deployment targets.
        It supports patching from a dictionary, a single values file, or multiple values files.

        Args:
            values (dict | str | Path | list[Path]): Configuration values to patch with.
                Can be:
                - A dictionary containing provision info under 'provision' key
                - A string path to a YAML values file
                - A Path object to a YAML values file
                - A list of Path objects to multiple YAML values files

        Raises:
            ValueError: If dictionary doesn't contain 'provision' key
            FileNotFoundError: If any specified values file doesn't exist
        """
        self._log(f"Patching federation config with {values}")
        if isinstance(values, str):
            values = Path(values)
        if isinstance(values, dict):
            if "provision" not in values:
                raise ValueError("Provision info not found in values")
            self._patch_provision_info(values["provision"])
        if isinstance(values, Path):
            values = [values]
        if isinstance(values, list):
            for value in values:
                if not value.exists():
                    raise FileNotFoundError(f"Values file not found: {value}")
                with open(value, "r") as f:
                    self.patch(cast(GenericConfigDataT, yaml.safe_load(f)))

    @classmethod
    def from_stored(
        cls,
        stored: FederationConfigStored,
        workspace_dir: Path,
        name: str,
        debug: bool = False,
    ) -> "FederationConfig":
        """
        Create a FederationConfig instance from FederationConfigData.

        Args:
            data (FederationConfigData): Configuration data
            workspace_dir (Path): Path to the workspace
            name (str): Name of the federation

        Returns:
            FederationConfig: New federation configuration instance
        """
        obj = cls(debug)
        obj.initialize(
            name=name,
            workspace_dir=workspace_dir,
            federation_dir=stored.federation_dir,
            config_path=stored.config_path,
            project_path=stored.project_path,
        )
        return obj

    def to_stored(self) -> FederationConfigStored:
        """
        Convert the federation configuration to FederationConfigStored.

        Returns:
            FederationConfigStored: Configuration data
        """
        assert self._stored
        return FederationConfigStored(
            federation_dir=self._stored.federation_dir,
            config_path=self._stored.config_path,
            project_path=self._stored.project_path,
        )

    @override
    def __str__(self) -> str:
        """
        Get a string representation of the federation configuration.

        Returns:
            str: String representation of the configuration
        """
        return f"""FederationConfig(
            name={self.name},
            workspace_dir={self.federation_workspace_dir},
            federation_dir={self.federation_dir},
            config_path={self.config_path},
            project_path={self.project_path},
            module={self.module},
            server_sites={self.server_sites},
            client_sites={self.client_sites},
            app_name={self.app_name}
        )"""

    @override
    def __repr__(self) -> str:
        return self.__str__()

    def _default_config_path(self, path: Path) -> Path:
        """
        Get the default path for federation configuration file.

        Args:
            path (Path): Base federation directory path

        Returns:
            Path: Default configuration file path
        """
        return path / "configs" / "federation.dfm.yaml"

    def _default_project_path(self, path: Path) -> Path:
        """
        Get the default path for federation project file.

        Args:
            path (Path): Base federation directory path

        Returns:
            Path: Default project file path
        """
        return path / "configs" / "project.yaml"

    def _check_paths(self) -> None:
        """
        Verify that required paths exist and set availability flags.

        This method checks the existence of federation directory, config file, and project file,
        then sets corresponding boolean flags (has_federation_dir, has_config_path, has_project_path)
        that can be used by other methods to determine what resources are available.

        Side Effects:
            Sets the following instance attributes:
            - has_federation_dir (bool): True if federation directory exists
            - has_config_path (bool): True if config file exists
            - has_project_path (bool): True if project file exists
        """
        self.has_federation_dir = self.federation_dir.exists()
        self.has_config_path = self.config_path.exists()
        self.has_project_path = self.project_path.exists()

    def _load_config_files(self) -> tuple[GenericConfigDataT, GenericConfigDataT]:
        """
        Load configuration from federation config and project files.

        Returns:
            tuple[dict, dict]: Tuple containing federation config and project config
        """
        if self.has_config_path:
            self._log(f"Loading federation config from {self.config_path}")
            with open(self.config_path, "r") as f:
                federation_config: GenericConfigDataT = cast(
                    GenericConfigDataT, yaml.safe_load(f)
                )
                self._log("Successfully loaded federation config")
        else:
            self._log(f"No federation config file found at {self.config_path}")
            federation_config = {}

        if self.has_project_path:
            with open(self.project_path, "r") as f:
                project_config: GenericConfigDataT = cast(
                    GenericConfigDataT, yaml.safe_load(f)
                )
                self._log("Successfully loaded federation project config")
        else:
            self._log(f"No federation project file found at {self.project_path}")
            project_config = {}

        return federation_config, project_config

    def _parse_federation_config(self, federation_config: GenericConfigDataT) -> None:
        """
        Parse federation configuration from the loaded config file.

        Args:
            federation_config (dict): Loaded federation configuration
        """
        if not federation_config:
            self._log("No federation config found, skipping")
            return
        self._log("Parsing federation config")
        fed_obj = FederationObject.model_validate(federation_config)
        self.module = fed_obj.info.code_package
        self.provision_info = fed_obj.provision
        self.version = fed_obj.info.api_version
        self._log(f"Parsed federation config: {self}")

    def _patch_provision_info(self, provision_info: object) -> None:
        """
        Update the federation's provision information with new values.

        This method validates and applies provisioning configuration updates
        to the federation. The provision info is used for deployment-specific
        settings across different environments.

        Args:
            provision_info (dict): Dictionary containing provision configuration
                that will be validated against ProvisionInfoObject model

        Raises:
            ValidationError: If provision_info doesn't match expected schema
        """
        self.provision_info = ProvisionInfoObject.model_validate(provision_info)

    def _parse_federation_project(self, project_config: GenericConfigDataT) -> None:
        """
        Parse federation project configuration from the loaded project file.

        Args:
            project_config (dict): Loaded project configuration

        Raises:
            RuntimeError: If multiple admin participants are found or unknown participant type
        """
        if not project_config:
            self._log("No project config found, skipping")
            return
        self._log("Parsing federation project config")
        pj = project_config.get("name")
        assert isinstance(pj, str) or pj is None
        self.flare_project_name = pj
        self.server_sites = []
        self.client_sites = []
        self.app_name = None
        participants = cast(list[dict[str, str]], project_config["participants"])
        for participant in participants:
            if participant["type"] == "server":
                self.server_sites.append(participant["name"])
            elif participant["type"] == "client":
                self.client_sites.append(participant["name"])
            elif participant["type"] == "admin":
                if self.app_name is not None:
                    raise RuntimeError(
                        f"Multiple admin participants found in project config: {project_config['participants']}"
                    )
                self.app_name = participant["name"]
            else:
                raise RuntimeError(f"Unknown participant type: {participant['type']}")
        self._log("Parsed federation project config")


class CombinedFederationConfigStored(BaseModel):
    """
    Data model for combined federation configurations.

    Attributes:
        federations (dict[str, FederationConfigData]): Dictionary mapping federation names to their configuration data
    """

    federations: dict[str, FederationConfigStored]


def check_initialized(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator to check if federation config manager is initialized.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that checks initialization status

    Raises:
        RuntimeError: If federation config manager is not initialized
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # First argument should be self
        if not args:
            raise RuntimeError(
                "check_initialized decorator requires at least one argument (self)"
            )
        self = args[0]
        if not hasattr(self, "_initialized"):
            raise RuntimeError("Object does not have _initialized attribute")
        if not getattr(self, "_initialized", False):
            raise RuntimeError("Federation config not initialized")
        return func(*args, **kwargs)

    return wrapper


class FederationConfigManager:
    """
    Manages multiple federation configurations.

    This class handles loading, saving, and managing configurations for multiple federations.
    It provides methods for adding, removing, and accessing federation configurations.

    Attributes:
        _workspace_dir (Path): Path to the workspace directory
        _config_path (Path): Path to the federation configuration file
        _debug (bool): Debug mode flag
        _config (CombinedFederationConfig | None): Combined federation configurations
    """

    @staticmethod
    def add_to_context_and_get(dc: CliContext) -> "FederationConfigManager":
        """
        Add federation configuration manager to CLI context and return it.

        Args:
            dc (CliContext): CLI context to add configuration to

        Returns:
            FederationConfigManager: The federation configuration manager
        """
        fed_cfg_mgr = FederationConfigManager(dc)
        fed_cfg_mgr.initialize()
        dc.add_config("fed", fed_cfg_mgr)
        return fed_cfg_mgr

    def __init__(self, dc: CliContext):
        """
        Initialize the federation configuration manager.

        Args:
            dc (CliContext): CLI context containing workspace path and debug settings

        Raises:
            FileNotFoundError: If workspace path does not exist
        """

        self._initialized: bool = False
        if not dc.workspace_path.exists():
            raise FileNotFoundError(f"Workspace path not found: {dc.workspace_path}")

        self._workspace_dir: Path = dc.workspace_path
        cli_config: CliConfig = cast(CliConfig, dc.get_config("cli"))
        self._config_path: Path = dc.workspace_path / "fed_config.yaml"
        if cli_config.federations_config_path:
            assert isinstance(cli_config.federations_config_path, Path)
            self._config_path = cli_config.federations_config_path

        self._debug: bool = dc.debug

        self._config: dict[str, FederationConfig] = {}
        self._config_document: object | None = (
            None  # ruamel round-trip doc for comment preservation
        )
        self._log(
            f"Created FederationConfigManager with workspace path: {self._workspace_dir} and config path: {self._config_path}"
        )

    def _log(self, message: str) -> None:
        """
        Log a debug message if debug mode is enabled.

        Args:
            message (str): Message to log
        """
        if self._debug:
            click.echo(f"DEBUG[{self.__class__.__name__}]: {message}", err=True)

    def initialize(self) -> None:
        """
        Initialize the federation configuration manager.

        This method loads existing configuration or creates a default configuration
        if none exists.

        Raises:
            RuntimeError: If configuration is already initialized
        """
        if self._initialized:
            raise RuntimeError("Federation config already initialized")
        self._log(
            f"Initializing federation config with config path: {self._config_path}"
        )
        if not self._config_path.exists():
            click.echo(
                f"Federation config path does not exist: {self._config_path}\n"
                + "Use 'dfm fed config create-default' command to create it."
            )
            raise click.Abort()
        else:
            self._log("Config path exists, loading config")
            self._load_config()
        self._initialized = True
        self._log("Successfully initialized federation config")

    @check_initialized
    def add_config(self, name: str, config: FederationConfig) -> None:
        """
        Add or update a federation configuration.

        Args:
            name (str): Name of the federation
            config (FederationConfig): Federation configuration to add
        """
        if name in self._config:
            self._log(f"Federation {name} already exists, updating")
        self._config[name] = config
        self.__class__.save_config(
            self._config_path, self._config, existing_document=self._config_document
        )
        self._log(f"Successfully added federation {name}")

    @check_initialized
    def get_config(self, name: str) -> FederationConfig:
        """
        Get a federation configuration by name.

        Args:
            name (str): Name of the federation

        Returns:
            FederationConfig: The requested federation configuration
        """
        if name not in self._config:
            raise RuntimeError(f"Federation {name} not found")
        return self._config[name]

    @check_initialized
    def del_config(self, name: str) -> None:
        """
        Delete a federation configuration by name.

        Args:
            name (str): Name of the federation to delete
        """
        if name not in self._config:
            self._log(f"Federation {name} not found, nothing to delete")
            return
        del self._config[name]
        self.__class__.save_config(
            self._config_path, self._config, existing_document=self._config_document
        )
        self._log(f"Successfully deleted federation {name}")

    @check_initialized
    def get_federation_names(self) -> list[str]:
        """
        Get a list of all federation names.

        Returns:
            list[str]: List of federation names
        """
        return list(self._config.keys())

    @classmethod
    def create_default_config(cls, path: Path | str, debug: bool = False) -> None:
        """
        Create default federation configurations when no config file exists.

        Creates example federation configuration:
        - 'examplefed': Basic example federation

        Configuration points to the same federation directory ('examplefed')
        with standard config and project file paths. The configurations are
        automatically saved to the config file after creation.
        """
        path = Path(path)
        # Prevent overwriting existing config file
        if path.exists():
            click.echo(f"Federation config already exists at {path}.")
            raise click.Abort()
        federations: dict[str, FederationConfig] = {}
        name = "examplefed"
        config = FederationConfig(debug)
        config.initialize(
            name=name,
            workspace_dir=name,
            federation_dir="tutorials/example-fed/examplefed",
            config_path="configs/federation.dfm.yaml",
            project_path="configs/project.yaml",
        )
        federations[name] = config
        cls.save_config(path, federations)

    def _load_config(self) -> None:
        """
        Load federation configurations from the configuration file.
        Uses ruamel.yaml round-trip when the file exists so comments (e.g. license header) are preserved on save.
        """
        self._log(f"Loading federation config from {self._config_path}")
        self._config = {}
        with open(self._config_path, "r") as f:
            yaml_rt = YAML()
            try:
                doc = yaml_rt.load(f)
            except RuamelYAMLError as e:
                raise yaml.YAMLError(str(e)) from e
            data = CombinedFederationConfigStored.model_validate(doc)
            for fed, config in data.federations.items():
                self._config[fed] = FederationConfig.from_stored(
                    stored=config,
                    workspace_dir=self._workspace_dir,
                    name=fed,
                    debug=self._debug,
                )
            self._config_document = doc
        self._log("Successfully loaded federation config")

    @classmethod
    def save_config(
        cls,
        path: Path,
        config: dict[str, FederationConfig],
        existing_document: object | None = None,
    ) -> None:
        """
        Save federation configurations to the configuration file.
        If existing_document is provided (ruamel round-trip doc), comments are preserved.

        Args:
            path (Path): Path to save the configuration file
            config (dict[str, FederationConfig]): Dictionary mapping federation names to their configurations
            existing_document (object | None): Optional ruamel round-trip document to preserve comments
        """
        # Allow overwriting existing config file (used by 'set' command to update)
        # The 'create-default' command has its own explicit check to prevent overwriting

        data = CombinedFederationConfigStored(
            federations={name: cfg.to_stored() for name, cfg in config.items()}
        )
        data_dict = json.loads(data.model_dump_json())
        path.parent.mkdir(parents=True, exist_ok=True)

        if existing_document is not None:
            # Update in place to preserve comments and key order within each federation
            existing_federations = existing_document["federations"]
            new_federations = data_dict["federations"]
            for name in list(existing_federations.keys()):
                if name not in new_federations:
                    del existing_federations[name]
            for name, new_fed in new_federations.items():
                if name in existing_federations:
                    existing_fed = existing_federations[name]
                    for k in list(existing_fed.keys()):
                        if k in new_fed:
                            existing_fed[k] = new_fed[k]
                        else:
                            del existing_fed[k]
                    for k, v in new_fed.items():
                        if k not in existing_fed:
                            existing_fed[k] = v
                else:
                    cm = CommentedMap()
                    for key in ("federation_dir", "config_path", "project_path"):
                        if key in new_fed:
                            cm[key] = new_fed[key]
                    existing_federations[name] = cm
            yaml_rt = YAML()
            with open(path, "w") as f:
                yaml_rt.dump(existing_document, f)
        else:
            with open(path, "w") as f:
                yaml.dump(data_dict, f, indent=2)
