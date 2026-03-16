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
Development commands for DFM CLI.

This module provides a set of development-related commands for the DFM CLI tool.
It includes functionality for running tests, linting, formatting, version management,
and package building. These commands are primarily used during development and
maintenance of the DFM project.
"""

import os
from pathlib import Path

import click

from ..core._context import CliContext
from ..core._shell_runner import ShellRunner
from ..core._version import BumpType
from ..core._version import bump as bump_version
from ..core._version import get as get_version


@click.group()
def dev():
    """
    Development commands group.

    This group contains various development-related commands for managing the DFM project,
    including testing, linting, formatting, version management, and package building.
    """
    pass


@dev.command()
@click.argument("package", required=False, default="all")
@click.option("--list", "list_packages", is_flag=True, help="List available packages.")
@click.pass_context
def test(ctx, package: str | None, list_packages: bool):
    """
    Run the test suite.

    Executes the project's test suite. If no PACKAGE is specified, runs tests for all
    packages. If PACKAGE is specified, runs tests only for that package.

    Examples:

        dfm dev test all                # Test all packages

        dfm dev test nv-dfm-core        # Test only nv-dfm-core package

        dfm dev test --list             # List available packages

    Raises:
        click.Abort: If the test script is not found or if tests fail.
    """

    dc: CliContext = ctx.obj
    config = dc.get_config("cli")

    # Determine testing script arguments
    script_path = config.dev.testing.script
    if list_packages:
        script_args = ["--list"]
    elif package == "all":
        script_args = []
    else:
        script_args = [package]

    if not script_path.exists():
        click.echo(
            click.style(f"Error: Test script not found at {script_path}", fg="red")
        )
        raise click.Abort()

    # Run the script using ShellRunner with test-specific environment
    runner = ShellRunner(config.repo_dir)
    cmd = [str(script_path)] + script_args
    result = runner.run(
        cmd,
        env=os.environ,
        capture_output=False,
        check=False,
    )

    if result is not None and result.result.returncode == 0:
        if not list_packages:
            click.echo(click.style("Tests completed successfully", fg="green"))
    else:
        click.echo(click.style("Tests failed", fg="red"))
        raise click.Abort()


@dev.command()
@click.option("--fix", is_flag=True, help="Fix lint errors.")
@click.pass_context
def lint(ctx, fix: bool):
    """
    Run linting checks on the codebase.

    Executes the project's linting script (python-run-lint.sh) to check for code style
    and quality issues. Optionally fixes automatically fixable issues.

    Args:
        fix (bool): If True, automatically fix linting issues where possible.

    Raises:
        click.Abort: If the lint script is not found or if linting fails.
    """
    dc: CliContext = ctx.obj
    config = dc.get_config("cli")
    project_root = config.repo_dir
    script_path = config.dev.linting.script

    if not script_path.exists():
        click.echo(
            click.style(f"Error: Lint script not found at {script_path}", fg="red")
        )
        raise click.Abort()

    # Build command with optional --fix flag - script should support or ignore it.
    cmd = [str(script_path)]
    if fix or config.dev.linting.fix:
        cmd.append("--fix")

    # Execute linting with dfm-specific environment
    runner = ShellRunner(project_root)
    result = runner.run(
        cmd,
        env=os.environ,
        capture_output=False,
        check=False,
    )

    if result is not None and result.result.returncode == 0:
        click.echo(click.style("Linting completed successfully", fg="green"))
    else:
        click.echo(click.style("Linting failed", fg="red"))
        raise click.Abort()


@dev.command()
@click.pass_context
def format(ctx):  # pylint: disable=redefined-builtin
    """
    Format code using ruff formatter.

    Runs the ruff formatter on the entire codebase to ensure consistent code style.

    Raises:
        click.Abort: If formatting fails.
    """
    dc: CliContext = ctx.obj
    config = dc.get_config("cli")
    project_root = config.repo_dir
    script_path = config.dev.formatting.script

    if not script_path.exists():
        click.echo(
            click.style(f"Error: Format script not found at {script_path}", fg="red")
        )
        raise click.Abort()

    # Execute linting with dfm-specific environment
    runner = ShellRunner(project_root)
    result = runner.run(
        str(script_path),
        env=os.environ,
        capture_output=False,
        check=False,
    )

    if result is not None and result.result.returncode == 0:
        click.echo(click.style("Formatting completed successfully", fg="green"))
    else:
        click.echo(click.style("Formatting failed", fg="red"))
        raise click.Abort()


@dev.command()
@click.option("--path", type=click.Path(exists=False), help="Path to check.")
@click.pass_context
def type_check(ctx, path: Path):
    """
    Run type checking.

    Runs the pyright type checker on the entire codebase to ensure consistent code style.

    Raises:
        click.Abort: If type checking fails.
    """
    dc: CliContext = ctx.obj
    config = dc.get_config("cli")
    project_root = config.repo_dir
    script_path = config.dev.type_checking.script

    if not script_path.exists():
        click.echo(
            click.style(
                f"Error: Type check script not found at {script_path}", fg="red"
            )
        )
        raise click.Abort()

    # Execute type checking with dfm-specific environment
    runner = ShellRunner(project_root)
    cmd = [str(script_path)]
    if path:
        cmd.append(str(path))

    result = runner.run(
        cmd,
        env=os.environ,
        capture_output=False,
        check=False,
    )
    assert result is not None
    if result.result.returncode == 0:
        click.echo(click.style("Type checking completed successfully", fg="green"))
    else:
        click.echo(click.style("Type checking failed", fg="red"))
        raise click.Abort()


@dev.command()
@click.argument("type", type=str, default="patch")
@click.option(
    "-f", "--force", is_flag=True, help="Force using already existing version."
)
@click.option("--dry-run", is_flag=True, help="Don't make any changes on disk.")
@click.pass_context
def bump(ctx, type: str, force: bool, dry_run: bool):
    """
    Bump DFM version or set an arbitrary version.

    Args:
        type (str): Version bump type ('major', 'minor', 'patch') or specific version.
        force (bool): If True, allows using an already existing version.
        dry_run (bool): If True, shows what would be changed without making changes.

    The command can either increment the version according to semantic versioning
    (major, minor, patch) or set a specific version number.
    """
    new_version = ""
    bump_type = BumpType.from_name(type)

    # Handle literal version numbers differently from semantic versioning
    if bump_type == BumpType.LITERAL:
        new_version = str(type)

    bump_version(type=bump_type, new_version=new_version, force=force, dry_run=dry_run)


@dev.command()
@click.pass_context
def version(ctx):
    """
    Display the current DFM version.

    Shows the current version of the DFM package as defined in the project.
    """
    version_str = get_version()
    click.echo(f"DFM version: {version_str}")


@dev.command()
@click.pass_context
@click.option(
    "--output-dir",
    type=click.Path(exists=False),
    help="Output directory for the wheel(s).",
)
def package(ctx, output_dir: str):
    """
    Build the DFM package.

    Creates a Python wheel package for the DFM project using the python-build-wheels.sh
    script. The wheel will be placed in the specified output directory or a default
    location if none is provided.

    Args:
        output_dir (str): Optional path to the output directory for the wheel package.

    Raises:
        click.Abort: If the build script is not found or if building fails.
    """
    dc: CliContext = ctx.obj
    config = dc.get_config("cli")

    project_root = config.repo_dir
    script_path = project_root / "ci" / "scripts" / "python-build-wheels.sh"

    if not script_path.exists():
        click.echo(
            click.style(f"Error: Build script not found at {script_path}", fg="red")
        )
        raise click.Abort()

    runner = ShellRunner()
    cmd = [str(script_path)]

    # Set up output directory and build environment
    output_dir_path = Path(output_dir) if output_dir else dc.workspace_path / "wheel"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Configure build environment with necessary variables
    env = os.environ | {
        "BUILD_WHEEL_OUTPUT_DIR": str(output_dir_path),
    }

    # Execute build process
    result = runner.run(
        cmd,
        env=env,
        check=False,
        capture_output=False,
    )

    if result is not None and result.result.returncode == 0:
        click.echo(click.style("Build completed successfully", fg="green"))
    else:
        click.echo(click.style("Build failed", fg="red"))
        raise click.Abort()
