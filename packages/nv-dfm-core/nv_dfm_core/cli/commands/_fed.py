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

This module provides commands for managing and controlling DFM federation operations.
It includes functionality for:
- Federation configuration management (add, modify, delete, show)
- Listing configured federations
- Submitting pipelines to federations
- Generating federation API and runtime code
"""

from pathlib import Path

import click

from ..config._federation import FederationConfig, FederationConfigManager
from ..core._context import CliContext
from ..core._generator import Generator
from ..core._submitter import SubmitterConfig, get_submitter, list_submitters


@click.group()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Path to the config file.",
)
@click.pass_context
def fed(ctx, config: Path | None):
    """
    Control DFM federation.

    This group contains commands for managing and controlling DFM federation operations,
    including configuration management, pipeline submission, and code generation.
    """
    if config:
        dc: CliContext = ctx.obj
        cli_config = dc.get_config("cli")
        cli_config.federations_config_path_override = True
        cli_config.federations_config_path = Path(config)


@fed.group()
@click.pass_context
def config(ctx):
    """
    Manage DFM federation configuration.

    This group contains commands for managing federation configurations,
    including adding, modifying, deleting, and viewing configurations.
    """
    pass


# Register config group as a subcommand of fed
fed.add_command(config)


@config.command()
@click.argument("name", type=str)
@click.option(
    "--federation-dir",
    type=click.Path(exists=True),
    help="Path to the federation directory.",
)
@click.option("--config-path", type=str, help="Path to the federation config.")
@click.option("--project-path", type=str, help="Path to the federation project.")
@click.pass_context
def set(
    ctx,
    name: str,
    federation_dir: str | None,
    config_path: str | None,
    project_path: str | None,
):
    """
    Add or modify DFM federation configuration.

    This command allows adding a new federation configuration or modifying an existing one.
    If the federation already exists, it will be reconfigured with the new parameters.
    Missing parameters will be prompted for interactively.

    Args:
        name (str): Name of the federation configuration
        federation_dir (str, optional): Path to the federation directory
        config_path (str, optional): Path to the federation config file
        project_path (str, optional): Path to the federation project

    The command will:
    1. Check if the federation exists
    2. Use existing values as defaults for missing parameters
    3. Prompt for any remaining missing values
    4. Save the configuration
    """
    dc: CliContext = ctx.obj
    fed_cfg_mgr = FederationConfigManager.add_to_context_and_get(dc)

    # Extract command parameters for configuration
    # Explicitly list the parameters we care about
    defaults = {
        "federation_dir": federation_dir,
        "config_path": config_path,
        "project_path": project_path,
    }

    # Handle existing federation configuration
    if name in fed_cfg_mgr.get_federation_names():
        click.echo(f"Federation {name} already exists, reconfiguring it.")
        fed_cfg = fed_cfg_mgr.get_config(name)
        # Preserve existing values for parameters not provided in command
        for param in defaults:
            if defaults[param] is None:
                defaults[param] = getattr(fed_cfg, param)
    else:
        click.echo(f"Federation {name} does not exist, creating it.")
        fed_cfg = FederationConfig(dc.debug)

    # Prompt for any missing required parameters
    for param_name, param_value in defaults.items():
        if param_value is None:
            param_value = click.prompt(
                f"Please provide a value for {param_name} (default: {param_value})"
            )
            defaults[param_name] = param_value

    # Initialize and save the configuration
    fed_cfg.initialize(name=name, workspace_dir=dc.workspace_path, **defaults)
    fed_cfg_mgr.add_config(name, fed_cfg)

    click.echo(f"Federation {name} configured.")


@config.command()
@click.argument("name", type=str)
@click.pass_context
def delete(ctx, name: str):
    """
    Delete DFM federation configuration.

    Removes a federation configuration from the system. The command will verify
    that the federation exists before attempting to delete it.

    Args:
        name (str): Name of the federation configuration to delete

    Raises:
        click.Abort: If the federation configuration doesn't exist
    """
    dc: CliContext = ctx.obj
    fed_cfg_mgr = FederationConfigManager.add_to_context_and_get(dc)

    # Verify federation exists before deletion
    FederationConfig.add_to_context_and_get(dc, name)
    fed_cfg_mgr.del_config(name)
    click.echo(f"Federation {name} deleted.")


@config.command()
@click.argument("name", type=str)
@click.pass_context
def show(ctx, name: str):
    """
    Show DFM federation configuration.

    Displays the current configuration for a specific federation.

    Args:
        name (str): Name of the federation configuration to display
    """
    dc: CliContext = ctx.obj
    fed_cfg = FederationConfig.add_to_context_and_get(dc, name)
    click.echo("Federation config:")
    click.echo(fed_cfg)


@config.command()
@click.pass_context
def list_all(ctx):
    """
    List DFM configured federations.

    Displays all currently configured federations in the system.
    """
    dc: CliContext = ctx.obj
    fed_cfg_mgr = FederationConfigManager.add_to_context_and_get(dc)
    feds = ", ".join(fed_cfg_mgr.get_federation_names())
    click.echo(f"Configured federations: {feds}")


@config.command()
@click.option(
    "--path", type=click.Path(exists=False), help="Path to the federation config."
)
@click.pass_context
def create_default(ctx, path: Path | None):
    """
    List DFM configured federations.

    Displays all currently configured federations in the system.
    """
    dc: CliContext = ctx.obj
    cli_config = dc.get_config("cli")
    if path:
        path = Path(path)
        if (
            path != cli_config.federations_config_path
            and cli_config.federations_config_path_override
        ):
            click.echo("Arguments --config and --path point to different files.")
            raise click.Abort()
    else:
        path = cli_config.federations_config_path
        assert path
    FederationConfigManager.create_default_config(path, dc.debug)
    cli_config.federations_config_path = path
    click.echo(f"Default federation config created at {path}")


@fed.command()
@click.option(
    "--target",
    type=click.Choice(["flare", "local"]),
    default="flare",
    help="Execution target.",
)
@click.option("--timeout", type=int, default=3600, help="Timeout for the job.")
@click.option("--poc", is_flag=True, help="Run in POC mode.")
@click.argument("federation", type=str)
@click.argument("job_type", type=click.Choice(list_submitters()))
@click.argument("path", type=click.Path(exists=True))
@click.argument("job_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def submit(
    ctx,
    poc: bool,
    federation: str,
    job_type: str,
    path: Path,
    target: str,
    timeout: int,
    job_args: list[str],
):
    """
    Submit a pipeline to DFM federation.

    Submits a pipeline job to the specified federation for execution. The command
    performs preflight checks before submission and supports different execution
    targets and job types.

    Args:
        federation (str): Name of the target federation
        job_type (str): Type of job to submit (must be one of available submitters)
        path (Path): Path to the pipeline definition
        target (str): Execution target ('flare' or 'local')
        timeout (int): Job timeout in seconds
        job_args (list[str]): Additional arguments for the job.
            Use -- to separate arguments for the job from arguments for the submitter.

    Raises:
        click.Abort: If preflight checks fail or submission fails
    """
    dc: CliContext = ctx.obj
    path = Path(path)

    # Get federation configuration
    fed_cfg = FederationConfig.add_to_context_and_get(dc, federation)

    # Configure job submission
    submitter_config = SubmitterConfig(
        fed_cfg=fed_cfg, poc_mode=poc, timeout=timeout, job_args=job_args
    )

    # Get appropriate submitter for job type
    submitter = get_submitter(job_type, submitter_config)

    # Perform preflight checks
    preflight_check_result = submitter.preflight_check(path)
    click.echo(preflight_check_result.message)
    if not preflight_check_result.success:
        raise click.Abort()

    # Submit the job
    click.echo(f"Submitting {path} to {federation} ({target})...")
    submitter.submit(path, target)


@fed.group()
@click.pass_context
def gen(ctx):
    """
    Generate DFM federation code, dockerfile, etc.

    This group contains commands for generating DFM federation code, dockerfile, etc.
    """
    pass


# Register gen group as a subcommand of fed
fed.add_command(gen)


@gen.command()
@click.argument("name", type=str)
@click.option(
    "--output-dir", type=click.Path(exists=False), help="Path to the output directory."
)
@click.option("--cleanup", is_flag=True, help="Cleanup the output directory.")
@click.option("--no-api", is_flag=True, help="Don't generate the API code.")
@click.option(
    "--runtime-site",
    type=str,
    multiple=True,
    default=["all"],
    help="Generate runtime code for a specific site; none to skip runtime code generation.",
)
@click.pass_context
def code(
    ctx,
    name: str,
    output_dir: Path,
    cleanup: bool,
    no_api: bool,
    runtime_site: list[str],
):
    """
    Generate DFM federation API and runtime code.

    Generates API and runtime code for the specified federation. The command can
    generate both API and runtime code, or either one independently. Runtime code
    generation can be targeted for specific sites or skipped entirely.

    Args:
        name (str): Name of the federation
        output_dir (Path): Directory for generated code
        cleanup (bool): Whether to clean output directory before generation
        no_api (bool): Skip API code generation
        runtime_site (list[str]): Sites to generate runtime code for ('all' or specific sites)

    Raises:
        click.Abort: If invalid runtime site configuration is provided
    """
    dc: CliContext = ctx.obj

    # Handle runtime site configuration
    skip_runtime = False
    if "none" in runtime_site:
        if len(runtime_site) == 1:
            skip_runtime = True
        else:
            click.echo("None can only be specified alone, not with other sites.")
            raise click.Abort()

    if "all" in runtime_site:
        rt_site_to_use = None
    else:
        rt_site_to_use = runtime_site

    # Get federation configuration and generate code
    fed_cfg = FederationConfig.add_to_context_and_get(dc, name)
    generator = Generator(name, fed_cfg, dc.debug)
    generator.code(output_dir, cleanup, no_api, rt_site_to_use, skip_runtime)


@gen.command()
@click.argument("name", type=str)
@click.option("--force", is_flag=True, help="Force the generation of the Dockerfile.")
@click.option(
    "--patch",
    type=click.Path(exists=True),
    help="Path to the federation config patch file.",
)
@click.option(
    "--skip",
    type=click.Choice(["dockerfile", "scripts"]),
    help="Skip the generation of the dockerfile or scripts.",
)
@click.pass_context
def docker(ctx, name: str, force: bool, patch: Path, skip: str):
    """
    Generate a Dockerfile for the specified federation.

    Generates a Dockerfile for the specified federation. The command can optionally
    apply a configuration patch before generation and can force overwrite existing files.

    Args:
        name (str): Name of the federation
        force (bool): Force overwrite of existing Dockerfile
        patch (Path): Path to a federation config patch file to apply before generation
    """
    dc: CliContext = ctx.obj

    # Get federation configuration
    fed_cfg = FederationConfig.add_to_context_and_get(dc, name)
    if patch:
        fed_cfg.patch(patch)
    generator = Generator(name, fed_cfg, dc.debug)
    generator.dockerfile(force, skip=skip)


@gen.command()
@click.argument("name", type=str)
@click.option("--force", is_flag=True, help="Force the generation of the Helm chart.")
@click.option(
    "--patch",
    type=click.Path(exists=True),
    help="Path to the federation config patch file.",
)
@click.pass_context
def helm(
    ctx,
    name: str,
    force: bool,
    patch: Path,
):
    """
    Generate a Helm chart for the specified federation.

    Generates a Helm chart for the specified federation. The command can optionally
    apply a configuration patch before generation and can force overwrite existing files.

    Args:
        name (str): Name of the federation
        force (bool): Force overwrite of existing Helm chart
        patch (Path): Path to a federation config patch file to apply before generation
    """
    dc: CliContext = ctx.obj

    # Get federation configuration
    fed_cfg = FederationConfig.add_to_context_and_get(dc, name)
    if patch:
        fed_cfg.patch(patch)
    generator = Generator(name, fed_cfg, dc.debug)
    generator.helm_chart(force)


@gen.command()
@click.argument("name", type=str)
@click.option("--clean", is_flag=True, help="Clean the workspace before provisioning.")
@click.pass_context
def provision(
    ctx,
    name: str,
    clean: bool,
):
    """
    Provision the specified federation.

    Provisions the specified federation by creating packages necessary
    to deploy the federation. Since DFM is built on top of Flare, the
    provisioned packages are Flare workspace packages.

    Args:
        name (str): Name of the federation
        clean (bool): Clean the workspace before provisioning
    """
    dc: CliContext = ctx.obj

    # Get federation configuration
    fed_cfg = FederationConfig.add_to_context_and_get(dc, name)
    generator = Generator(name, fed_cfg, dc.debug)
    generator.provision(clean)
