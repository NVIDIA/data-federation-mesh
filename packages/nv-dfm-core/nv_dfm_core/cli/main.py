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
Data Federation Mesh (DFM) Command Line Interface.

This module serves as the main entry point for the DFM CLI tool. It sets up the command
structure and provides access to various subcommands for managing federated learning
environments, configurations, and development workflows.

The CLI is built using Click and provides the following command groups:
- poc: Commands for managing Proof of Concept environments
- config: Commands for managing DFM configurations
- fed: Commands for federation-specific operations
- dev: Development-specific commands (only available in development mode)
- ui: Desktop interface for visualization and management

Each command group provides a set of subcommands for specific operations within that domain.
"""

import click

from nv_dfm_core.cli.commands._completion import completion
from nv_dfm_core.cli.commands._config import config
from nv_dfm_core.cli.commands._fed import fed
from nv_dfm_core.cli.commands._poc import poc
from nv_dfm_core.cli.core._context import CliContext

# Dev commands should not be available in production
try:
    from nv_dfm_core.cli.commands._dev import dev
except ImportError:
    dev = None


def dfm_version():
    import nv_dfm_core

    return nv_dfm_core.__version__


# Function to create and pass the context object
@click.group()
@click.version_option(version=dfm_version(), prog_name="DFM")
@click.pass_context
def cli(ctx):
    """CLI tool for DFM development.

    This is the main entry point for the Data Federation Mesh (DFM) CLI tool.
    It provides a set of commands for managing federated learning environments,
    configurations, and development workflows.

    The CLI uses a context object to maintain state and configuration across
    different commands. This context is automatically created and passed to
    all subcommands.
    """
    # Create and store the CLI context in the Click context object
    # This context will be available to all subcommands
    ctx.obj = CliContext()


# Register the main command groups
cli.add_command(poc)  # Proof of Concept environment management
cli.add_command(config)  # Configuration management
cli.add_command(fed)  # Federation operations
cli.add_command(completion)  # Shell completion management

# Register development commands if available
if dev:
    cli.add_command(dev)


if __name__ == "__main__":
    # Entry point when running the script directly
    cli()
