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
Completion commands for DFM CLI.

This module provides commands for managing shell completion for the DFM CLI tool.
It includes functionality for generating completion scripts, setting up completion,
and managing completion configuration for different shells.
"""

from pathlib import Path

import click

from ..core._completion import CompletionGenerator


@click.group()
def completion():
    """
    Shell completion commands for DFM CLI.

    This group contains commands for managing shell completion functionality,
    including generating completion scripts and setting up completion for
    different shells (bash, zsh, fish).
    """
    pass


@completion.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    default="bash",
    help="Shell type for completion script generation.",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file path for the completion script. If not provided, prints to stdout.",
)
def generate(shell: str, output: str | None):
    """
    Generate shell completion script for DFM CLI.

    This command generates a shell completion script that enables tab completion
    for the DFM CLI tool. The generated script can be sourced in your shell
    configuration file to enable completion.

    Examples:
        # Generate bash completion script
        dfm completion generate --shell bash > ~/.dfm-complete.bash

        # Generate zsh completion script
        dfm completion generate --shell zsh > ~/.dfm-complete.zsh

        # Generate and save to file
        dfm completion generate --shell zsh --output ~/.dfm-complete.zsh

        # Add to your shell configuration:
        # For bash: echo "source ~/.dfm-complete.bash" >> ~/.bashrc
        # For zsh: echo "source ~/.dfm-complete.zsh" >> ~/.zshrc
    """
    generator = CompletionGenerator()

    try:
        output_path = Path(output) if output else None
        completion_script = generator.generate_completion_script(shell, output_path)

        if output:
            click.echo(f"Completion script written to {output}")
            click.echo(f"To enable completion, add this line to your ~/.{shell}rc:")
            click.echo(f"  source {output}")
        else:
            # Print to stdout
            click.echo(completion_script)

    except Exception as e:
        click.echo(f"Error generating completion script: {e}", err=True)
        raise click.Abort()


@completion.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    help="Shell type to setup completion for. If not provided, auto-detects current shell.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force setup even if completion is already configured.",
)
@click.option(
    "--home-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Home directory to use for configuration files.",
)
def setup(shell: str | None, force: bool, home_dir: str | None):
    """
    Set up shell completion for DFM CLI.

    This command automatically generates and configures shell completion for
    the DFM CLI tool. It will detect your current shell (if not specified),
    generate the appropriate completion script, and add it to your shell
    configuration file.

    Examples:
        # Auto-detect shell and setup completion
        dfm completion setup

        # Setup completion for specific shell
        dfm completion setup --shell zsh

        # Force setup (overwrite existing configuration)
        dfm completion setup --force
    """
    generator = CompletionGenerator()

    # Auto-detect shell if not provided
    if not shell:
        shell = generator.detect_shell()
        if shell == "unknown":
            click.echo(
                "Could not detect shell type. Please specify --shell option.", err=True
            )
            raise click.Abort()
        click.echo(f"Detected shell: {shell}")

    home_dir_path = Path(home_dir) if home_dir else None

    try:
        success, message = generator.setup_completion(shell, home_dir_path, force)

        if success:
            click.echo(click.style("✓ " + message, fg="green"))
            click.echo(f"Completion setup complete for {shell}!")
            click.echo("To activate completion, restart your shell or run:")
            click.echo(f"  source ~/.{shell}rc")
        else:
            click.echo(click.style("✗ " + message, fg="red"), err=True)
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error setting up completion: {e}", err=True)
        raise click.Abort()


@completion.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    help="Shell type to check. If not provided, checks all supported shells.",
)
def status(shell: str | None):
    """
    Check completion status for DFM CLI.

    This command checks whether shell completion is configured and working
    for the DFM CLI tool. It can check a specific shell or all supported shells.

    Examples:
        # Check completion status for all shells
        dfm completion status

        # Check completion status for specific shell
        dfm completion status --shell zsh
    """
    generator = CompletionGenerator()

    shells_to_check = [shell] if shell else generator.SUPPORTED_SHELLS

    click.echo("DFM CLI Completion Status")
    click.echo("=" * 30)

    for shell_type in shells_to_check:
        try:
            is_configured = generator.is_completion_configured(shell_type)
            completion_path = generator.get_completion_script_path(shell_type)
            config_path = generator.get_shell_config_path(shell_type)

            status_icon = "✓" if is_configured else "✗"
            status_color = "green" if is_configured else "red"

            click.echo(f"{shell_type.upper()}: ", nl=False)
            click.echo(click.style(status_icon, fg=status_color), nl=False)
            click.echo(f" {'Configured' if is_configured else 'Not configured'}")

            if completion_path.exists():
                click.echo(f"  Completion script: {completion_path}")
            else:
                click.echo("  Completion script: Not found")

            if config_path.exists():
                click.echo(f"  Config file: {config_path}")
            else:
                click.echo("  Config file: Not found")

            # Show requirements
            requirements = generator.get_shell_requirements(shell_type)
            if requirements["note"]:
                click.echo(f"  Note: {requirements['note']}")

            click.echo()

        except Exception as e:
            click.echo(f"{shell_type.upper()}: Error checking status - {e}")
            click.echo()


@completion.command()
def info():
    """
    Show information about DFM CLI shell completion.

    This command displays information about shell completion support,
    including supported shells, requirements, and usage examples.
    """
    generator = CompletionGenerator()

    click.echo("DFM CLI Shell Completion Information")
    click.echo("=" * 40)
    click.echo()

    click.echo("Supported Shells:")
    for shell in generator.SUPPORTED_SHELLS:
        requirements = generator.get_shell_requirements(shell)
        click.echo(f"  • {shell.upper()}: {requirements['note']}")

    click.echo()
    click.echo("Quick Setup:")
    click.echo("  dfm completion setup                    # Auto-detect and setup")
    click.echo("  dfm completion setup --shell zsh       # Setup for specific shell")
    click.echo()
    click.echo("Manual Setup:")
    click.echo("  dfm completion generate --shell zsh > ~/.dfm-complete.zsh")
    click.echo("  echo 'source ~/.dfm-complete.zsh' >> ~/.zshrc")
    click.echo()
    click.echo("Check Status:")
    click.echo("  dfm completion status")
    click.echo()
    click.echo("Usage Examples:")
    click.echo("  dfm <TAB>                    # Show available commands")
    click.echo("  dfm poc <TAB>               # Show poc subcommands")
    click.echo("  dfm fed submit <TAB>        # Show submit options")
    click.echo("  dfm --help                  # Show help")
