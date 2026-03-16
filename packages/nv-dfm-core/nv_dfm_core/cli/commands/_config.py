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
Configuration commands for DFM CLI.

This module provides command-line interface commands for managing DFM configuration.
It allows users to view and create configuration files for the DFM CLI tool.
"""

from pathlib import Path

import click

from nv_dfm_core.cli.config._cli import CliConfig
from nv_dfm_core.cli.core._context import CliContext


@click.group()
def config():
    """
    Configuration commands group.

    This group contains commands for managing DFM CLI configuration settings.
    """
    pass


@config.command()
@click.pass_context
def show(ctx):
    """
    Show current configuration file or full configuration.

    This command displays either:
    1. The path to the current configuration file and its contents, or
    2. A message indicating no config file was found if using defaults

    Args:
        ctx: Click context object containing CLI context information
    """
    dc: CliContext = ctx.obj
    cli_cfg: CliConfig = dc.get_config("cli")
    config_path: Path = cli_cfg.current_config_path

    if not config_path:
        click.echo("No config file found, using defaults")
    else:
        click.echo(f"Current config file: {config_path.absolute().as_posix()}\n\n--")
        click.echo(cli_cfg.raw())


@config.command()
@click.option(
    "--path", type=click.Path(exists=False), help="Path to the configuration file"
)
@click.pass_context
def create_default(ctx, path):
    """
    Create a default configuration file.

    This command generates a new configuration file with default settings.
    The file will be created in the appropriate location based on the CLI context.

    Args:
        ctx: Click context object containing CLI context information
        path: Path to the configuration file

    """
    dc: CliContext = ctx.obj
    cli_cfg: CliConfig = dc.get_config("cli")

    path = path or Path.cwd()

    click.echo(cli_cfg.create_default_config(path=path))
