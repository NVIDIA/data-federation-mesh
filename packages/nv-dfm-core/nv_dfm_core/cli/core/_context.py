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
CliContext module provides the development context for DFM CLI operations.

This module implements the core context management system for the Data Federation Mesh (DFM)
Command Line Interface. It handles workspace configuration, session management, and provides
access to various configuration sources and workspace directories needed for CLI operations.

The context system is designed to be extensible, allowing different components to register
their configurations while maintaining a centralized access point for all CLI-related settings
and resources.
"""

import os
from pathlib import Path

from ..config._cli import CliConfig


class CliContext:
    """Development context for DFM CLI operations.

    This class serves as the central management system for DFM CLI operations, providing
    access to all necessary configurations and resources. It manages:

    - Project configuration from pyproject.toml
    - Workspace directories for POC and job execution
    - Configuration management for different CLI components
    - Workspace initialization and management

    The context maintains a registry of configuration objects that can be accessed
    throughout the CLI application lifecycle.
    """

    def __init__(self):
        """Initialize CliContext.

        Sets up the CLI context by:
        1. Reading environment variables for debug mode
        2. Initializing the configuration registry
        3. Setting up the CLI configuration manager
        4. Creating necessary workspace directories

        The initialization process ensures all required configurations and
        workspace structures are in place for CLI operations.
        """
        # Enable debug mode if DFM_CLI_DEBUG environment variable is set to "1"
        self.debug: bool = os.getenv("DFM_CLI_DEBUG", "0").upper() in [
            "1",
            "TRUE",
            "YES",
            "Y",
        ]
        # Initialize empty configuration registry
        self._configs: dict[str, object] = {}

        # Set up CLI configuration manager and initialize it
        cli_config = CliConfig(self.debug)
        cli_config.initialize()

        # Register the CLI configuration as it's required for all operations
        self.add_config("cli", cli_config)

        # Initialize workspace structure
        self._init_workspace()

    def add_config(self, name: str, config: object) -> None:
        """Add a configuration to the context registry.

        Args:
            name: The unique identifier for the configuration.
            config: The configuration object to register.

        Note:
            This method allows different components to register their configurations
            with the context system for centralized access.
        """
        self._configs[name] = config

    def get_config(self, name: str) -> object:
        """Retrieve a configuration from the context registry.

        Args:
            name: The identifier of the configuration to retrieve.

        Returns:
            The requested configuration object if found, None otherwise.
        """
        if name not in self._configs:
            raise RuntimeError(f"Configuration {name} not found")
        return self._configs[name]

    def _init_workspace(self) -> None:
        """Initialize the workspace directory structure.

        Creates the main workspace directory if it doesn't exist and stores
        its path for easy access throughout the context.

        The workspace path is stored as an instance variable for convenient
        access by other components.
        """
        cli_config = self.get_config("cli")

        assert isinstance(cli_config, CliConfig)

        # Store workspace path for easy access
        self.workspace_path = cli_config.workspace.path
        assert isinstance(self.workspace_path, Path)
        # Debug mode enabled in env variable overrides debug mode in config.
        assert isinstance(cli_config.debug, bool)
        self.debug = self.debug or cli_config.debug

        # Ensure workspace directory exists
        os.makedirs(self.workspace_path.as_posix(), exist_ok=True)
