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
POC mode commands for DFM CLI.

This module provides commands for managing the DFM Proof of Concept (POC) mode,
which allows for local testing and development of federation features. It includes
functionality for:
- Starting and stopping the POC environment
- Managing Flare workspace
- Monitoring POC status and logs
- Cleaning up POC resources
"""

import click

from ..config._federation import FederationConfig
from ..core._context import CliContext
from ..core._flare_poc import FlarePoc


@click.group()
@click.pass_context
def poc(ctx):
    """
    Control DFM POC mode.

    This group contains commands for managing the DFM Proof of Concept environment,
    including starting, stopping, and monitoring the POC instance. The FlarePoc
    instance is initialized and made available to all subcommands.
    """
    # Initialize FlarePoc instance for all commands in this group
    ctx.obj.flare_poc = FlarePoc(ctx.obj.workspace_path)


@poc.command()
@click.option(
    "-f",
    "--federation",
    type=str,
    default="examplefed",
    help="Name of federation to wait for.",
)
@click.pass_context
def wait(ctx, federation: str):
    """
    Wait for DFM POC mode to be ready.

    Waits for the POC environment to be fully initialized and ready for use.
    This command is useful for ensuring the environment is ready before
    proceeding with other operations.

    Args:
        federation (str): Name of the federation to wait for

    The command will block until the POC environment is ready or until
    an error occurs.
    """
    dc: CliContext = ctx.obj
    fed_cfg = FederationConfig.add_to_context_and_get(dc, federation)
    # Enable POC mode
    fed_cfg.poc_mode = True

    # Configure and wait for Flare to be ready
    flare_poc: FlarePoc = ctx.obj.flare_poc
    flare_poc.set_federation_config(fed_cfg)
    flare_poc.wait()

    click.echo(click.style("Flare is ready!", fg="green"))


@poc.command()
@click.option(
    "-f",
    "--federation",
    type=str,
    default="examplefed",
    help="Name of federation to start.",
)
@click.option("--prepare-only", is_flag=True, help="Only prepare Flare workspace.")
@click.option("--skip-prepare", is_flag=True, help="Skip Flare workspace preparation.")
@click.option("--debug", is_flag=True, help="Run in debug mode.")
@click.pass_context
def start(ctx, federation: str, prepare_only: bool, skip_prepare: bool, debug: bool):
    """
    Start DFM POC mode.

    Initializes and starts the POC environment. This command can either
    prepare the workspace only or start the full POC environment.

    Args:
        federation (str): Name of the federation to start
        prepare_only (bool): If True, only prepare the workspace without starting
        debug (bool): If True, run in debug mode with additional logging

    The command will:
    1. Prepare the Flare workspace
    2. Optionally start the POC environment
    3. Wait for the environment to be ready
    """
    dc: CliContext = ctx.obj
    fed_cfg = FederationConfig.add_to_context_and_get(dc, federation)
    # Enable POC mode
    fed_cfg.poc_mode = True

    # Initialize and configure Flare POC
    flare_poc: FlarePoc = ctx.obj.flare_poc
    flare_poc.set_federation_config(fed_cfg)

    # Prepare workspace and optionally start Flare
    if not skip_prepare:
        flare_poc.prepare()
        click.echo("Flare workspace prepared.")
    if prepare_only:
        return

    # Start Flare and wait for it to be ready
    click.echo("Starting Flare...")
    flare_proc = flare_poc.start(debug)
    try:
        flare_poc.wait()
        click.echo(click.style("Flare is ready!", fg="green"))
    except click.Abort:
        # Ensure cleanup on failure
        flare_poc.stop(flare_proc)
        raise


@poc.command()
@click.pass_context
def stop(ctx):
    """
    Stop DFM POC mode.

    Gracefully shuts down the POC environment and cleans up any running
    processes. This command should be used to properly terminate the
    POC environment.
    """
    flare_poc: FlarePoc = ctx.obj.flare_poc
    flare_poc.stop()


@poc.command()
@click.option(
    "-f",
    "--federation",
    type=str,
    default="examplefed",
    help="Name of federation to start.",
)
@click.option("--do-cleanup", is_flag=True, help="Clean workspace before restart.")
@click.pass_context
def restart(ctx, federation: str, do_cleanup: bool):
    """
    Restart DFM POC mode.

    Restarts the POC environment, optionally cleaning the workspace first.
    This command is useful for resetting the environment to a clean state.

    Args:
        federation (str): Name of the federation to restart
        do_cleanup (bool): If True, clean the workspace before restarting

    The command will:
    1. Stop the current POC instance
    2. Optionally clean the workspace
    3. Start a new POC instance
    """
    # Stop current instance
    ctx.invoke(stop)

    # Clean workspace if requested
    if do_cleanup:
        ctx.invoke(cleanup, federation=federation)

    # Start new instance
    ctx.invoke(start, federation=federation)


@poc.command()
@click.option(
    "-f",
    "--federation",
    type=str,
    default="examplefed",
    help="Name of federation to clean.",
)
@click.pass_context
def cleanup(ctx, federation: str):
    """
    Clean DFM POC (remove workspace directory).

    Removes the workspace directory for the specified federation, effectively
    resetting the POC environment to its initial state.

    Args:
        federation (str): Name of the federation to clean

    This command will remove all files and directories associated with the
    POC workspace for the specified federation.
    """
    dc: CliContext = ctx.obj
    fed_cfg = FederationConfig.add_to_context_and_get(dc, federation)
    # Enable POC mode
    fed_cfg.poc_mode = True

    # Clean up Flare workspace
    flare_poc: FlarePoc = ctx.obj.flare_poc
    flare_poc.set_federation_config(fed_cfg)
    flare_poc.cleanup()


@poc.command()
@click.option(
    "-f",
    "--federation",
    type=str,
    default="examplefed",
    help="Name of federation to clean.",
)
@click.pass_context
def status(ctx, federation: str):
    """
    Show DFM POC status.

    Shows the current status of the POC environment.
    """
    dc: CliContext = ctx.obj
    fed_cfg = FederationConfig.add_to_context_and_get(dc, federation)
    # Enable POC mode
    fed_cfg.poc_mode = True

    # Clean up Flare workspace
    flare_poc: FlarePoc = ctx.obj.flare_poc
    flare_poc.set_federation_config(fed_cfg)
    flare_poc.status()


@poc.command()
@click.pass_context
def logs(ctx):
    """
    Show DFM POC logs.

    Displays the current logs from the POC environment. This is useful for
    debugging and monitoring the POC instance's operation.

    The logs will show the current state and any errors or warnings from
    the POC environment.
    """
    flare_poc: FlarePoc = ctx.obj.flare_poc
    flare_poc.show_logs()
