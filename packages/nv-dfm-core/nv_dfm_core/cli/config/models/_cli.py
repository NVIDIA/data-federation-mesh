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

from pathlib import Path

from pydantic import BaseModel
from typing_extensions import override


class Workspace(BaseModel):
    """
    Stored configuration model for workspace settings.

    Attributes:
        path (Path): Path to the workspace directory
    """

    path: Path | None = None


class DevTesting(BaseModel):
    """
    Stored configuration model for development testing settings.
    """

    script: Path | None = None


class DevLinting(BaseModel):
    """
    Stored configuration model for development linting settings.
    """

    fix: bool = False
    script: Path | None = None


class DevFormatting(BaseModel):
    """
    Stored configuration model for development formatting settings.
    """

    script: Path | None = None


class DevTypeChecking(BaseModel):
    """
    Stored configuration model for development type checking settings.
    """

    script: Path | None = None


class Dev(BaseModel):
    """
    Stored configuration model for development settings.
    """

    testing: DevTesting | None = None
    linting: DevLinting | None = None
    formatting: DevFormatting | None = None
    type_checking: DevTypeChecking | None = None


class Cli(BaseModel):
    """
    Stored configuration model for CLI settings.
    """

    federations_config_path: Path | None = None
    debug: bool = False
    workspace: Workspace | None = None
    dev: Dev | None = None

    @override
    def __str__(self) -> str:
        return f"""Cli(
            federations_config_path={self.federations_config_path},
            debug={self.debug},
            workspace={self.workspace},
            dev={self.dev}
        )"""

    @override
    def __repr__(self) -> str:
        return self.__str__()
