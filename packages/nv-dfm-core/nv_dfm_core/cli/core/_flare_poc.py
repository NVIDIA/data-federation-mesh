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
Flare POC (Proof of Concept) management module.

This module provides functionality for managing NVIDIA Flare POC environments, including:
- Workspace preparation and cleanup
- Process management (start/stop)
- Log management
- Connection handling
- Federation configuration management

The module integrates with the NVIDIA Flare CLI to provide a seamless experience
for setting up and managing federated learning environments.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import click
import sh

from ..config._federation import FederationConfig
from ._shell_runner import ShellRunner


class FlarePoc:
    """Manages NVIDIA Flare POC environment and operations.

    This class provides a high-level interface for managing NVIDIA Flare POC environments,
    including workspace setup, process management, and federation configuration.

    Attributes:
        LOG_FILE (str): Default name for the log file.
    """

    # Default name for the log file in the workspace
    LOG_FILE = "flare.log"

    def __init__(self, workspace: Path):
        """Initialize FlarePoc instance.

        Args:
            workspace: Path to the workspace directory where Flare will operate.
        """
        # Store the workspace path for all operations
        self._workspace = workspace
        # Federation configuration will be set later
        self._fed_cfg = None

    def set_federation_config(self, fed_cfg: FederationConfig):
        """Set the federation configuration for this POC instance.

        Args:
            fed_cfg: Federation configuration object containing all necessary settings.
        """
        self._fed_cfg = fed_cfg

    def show_logs(self):
        """Display logs from the Flare POC workspace.

        Reads and displays the contents of the log file. If the log file doesn't exist,
        displays an appropriate error message.

        Raises:
            click.Abort: If the log file doesn't exist.
        """
        # Construct the full path to the log file
        log_file = self._workspace.joinpath(self.LOG_FILE)
        try:
            # Read and display each line of the log file
            with open(log_file) as lf:
                for line in lf.readlines():
                    click.echo(line.rstrip())
        except FileNotFoundError:
            # Provide helpful error message if log file doesn't exist
            click.echo(f"Sorry, no logs available ({log_file} doesn't exist).")
            raise click.Abort()

    def prepare(self):
        """Prepare the Flare POC environment.

        Sets up the workspace directory and initializes the Flare POC environment
        using the provided federation configuration.

        Raises:
            RuntimeError: If federation configuration is not set.
        """
        # Ensure federation configuration is set before proceeding
        if self._fed_cfg is None:
            raise RuntimeError(
                "Federation config not set; use set_federation_config() to set it."
            )

        # Display configuration information
        click.echo(
            f"Preparing [configs: {self._fed_cfg.config_path}, {self._fed_cfg.project_path}]"
        )
        click.echo(f"Workspace: {self._fed_cfg.federation_workspace_dir}")

        # Clean up any existing workspace
        shutil.rmtree(self._fed_cfg.federation_workspace_dir, ignore_errors=True)
        click.echo("Removed old workspace directory.")

        # Set environment variable for NVFlare POC workspace
        os.environ["NVFLARE_POC_WORKSPACE"] = (
            self._fed_cfg.federation_workspace_dir.as_posix()
        )

        # Initialize the POC environment using NVFlare CLI
        sh.nvflare(["poc", "prepare", "-i", self._fed_cfg.project_path.as_posix()])  # pyright: ignore[reportAttributeAccessIssue]
        click.echo("NFLARE POC directory ready.")

    def start(self, debug: bool = False):
        """Start the Flare POC process in the background.

        Args:
            debug: Whether to run Flare in debug mode.

        Returns:
            subprocess.Popen: The started process object.

        Note:
            The process is started in the background and its output is redirected to a log file.
        """
        # Set up paths for PID and log files
        pid_file = self._workspace.joinpath("flare.pid")
        log_file = self._workspace.joinpath("flare.log")

        # Get the application name from federation config
        if self._fed_cfg is None:
            click.echo("No federation configuration set.")
            return None

        app_name = self._fed_cfg.app_name
        if app_name is None:
            click.echo("Invalid federation configuration: no app name found.")
            return None

        # Build the command to start Flare POC
        # -ex parameter prevents creation of admin console
        cmd = ["nvflare", "poc", "start", "-ex", app_name]
        if debug:
            cmd.append("--debug")

        # Start the process in the background
        runner = ShellRunner()
        return runner.start_in_background("NVFlare", cmd, pid_file, log_file)

    def stop(self, flare: subprocess.Popen[Any] | None = None):
        """Stop the Flare POC process.

        Args:
            flare: Process to stop. If None, will try to find and stop the process using the PID file.
        """
        click.echo("Stopping Flare...")
        # If no process object is provided, use the PID file
        if flare is None:
            flare_proc_info = self._workspace.joinpath("flare.pid")
        else:
            flare_proc_info = flare

        # Use ShellRunner to stop the process
        runner = ShellRunner()
        runner.stop_from_background("NVFlare", flare_proc_info)
        click.echo(click.style("Flare stopped.", fg="green"))

    def connect(self, debug: bool = False):
        """Establish a connection to the Flare POC environment.

        Args:
            debug: Whether to enable debug mode for the connection.

        Returns:
            The established Flare session.

        Note:
            This method creates a secure session using the admin package specified
            in the federation configuration.
        """
        if self._fed_cfg is None:
            click.echo("No federation configuration set.")
            return None

        if self._fed_cfg.app_name is None:
            click.echo("Invalid federation configuration: no app name found.")
            return None

        # Display connection information
        click.echo(
            f"Connecting to Flare with admin package: {self._fed_cfg.admin_package}"
        )

        # Import the Flare API for secure session creation
        from nvflare.fuel.flare_api.flare_api import new_secure_session

        # Create and return a new secure session
        flare_session = new_secure_session(
            self._fed_cfg.app_name,
            self._fed_cfg.admin_package.as_posix(),
            debug=debug,
        )

        return flare_session

    def wait(self, timeout: int = 30):
        """Wait for Flare to be ready and all expected clients to connect.

        Args:
            timeout: Maximum time to wait in seconds.

        Raises:
            click.Abort: If the timeout is reached before all clients connect.
        """
        # Get the expected number of client sites from federation config
        if self._fed_cfg is None:
            click.echo("No federation configuration set.")
            return
        expected_sites = self._fed_cfg.client_sites
        if expected_sites is None:
            click.echo("Invalid federation configuration: no client sites found.")
            return
        # Record start time for timeout calculation
        start_time = time.monotonic()
        flare_session = None

        # Poll for client connections until timeout
        while time.monotonic() - start_time <= timeout:
            try:
                # Create session if not already connected
                if not flare_session:
                    flare_session = self.connect()
                    assert flare_session is not None
                    click.echo("Connected to Flare.")

                # Get list of connected clients
                clients = flare_session.get_connected_client_list()
                click.echo(f"Connected clients: {len(clients)}/{len(expected_sites)}")

                # Check if we have all expected clients
                if len(clients) >= len(expected_sites):
                    click.echo("Flare ready.")
                    return

                # Wait before next check
                time.sleep(1)
            except Exception as e:
                # Handle connection errors with retry
                click.echo(f"Error waiting for Flare: {e}")
                time.sleep(5)

        # If we get here, we've timed out
        click.echo(click.style("Timeout waiting for Flare.", fg="red"))
        raise click.Abort()

    def cleanup(self):
        """Clean up the Flare POC workspace.

        Removes the federation workspace directory and all its contents.
        """
        # Display cleanup information
        if self._fed_cfg is None:
            click.echo("No federation configuration set.")
            return
        click.echo(
            f"Cleaning up POC workspace ({self._fed_cfg.federation_workspace_dir})..."
        )
        # Remove the workspace directory and all its contents
        shutil.rmtree(self._fed_cfg.federation_workspace_dir, ignore_errors=True)
        click.echo(click.style("POC workspace cleaned.", fg="green"))

    def status(self):
        """Show the current status of the POC environment."""

        pid_file = self._workspace.joinpath("flare.pid")

        runner = ShellRunner()
        is_running = runner.is_running_in_background("group_id", pid_file)
        any_flare_running = runner.is_running_in_background("in_cmdline", "nvflare")

        click.echo("POC environment status:")
        # click.echo(f"  Workspace: {self._fed_cfg.federation_workspace_dir}")
        if self._fed_cfg is None:
            click.echo("  No federation configuration set.")
            return
        click.echo(f"  Project: {self._fed_cfg.project_path}")
        click.echo(f"  Config: {self._fed_cfg.config_path}")
        click.echo(f"  Admin package: {self._fed_cfg.admin_package}")
        click.echo(f"  App name: {self._fed_cfg.app_name}")
        click.echo(f"  Client sites: {self._fed_cfg.client_sites}")
        if is_running:
            click.echo(click.style("  POC is running.", fg="green"))
        else:
            click.echo(click.style("  POC is not running.", fg="red"))
            if any_flare_running:
                click.echo(
                    click.style(
                        "  Some flare processes are still alive. Another POC instance might be running.",
                        fg="yellow",
                    )
                )
