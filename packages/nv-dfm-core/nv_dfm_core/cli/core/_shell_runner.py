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
Shell command execution utilities.

This module provides a robust interface for executing shell commands with various
options for process management, output capture, and error handling. It supports
both synchronous and asynchronous command execution, with special handling for
background processes and cross-platform compatibility.
"""

import os
import platform
import signal
import subprocess
from pathlib import Path
from typing import Any

import click
import psutil


class ShellRunner:
    """A class for running shell commands with a simple interface.

    This class provides a unified interface for executing shell commands with various
    options for process management, output capture, and error handling. It maintains
    state about the last executed command and its results.

    Attributes:
        cwd (Optional[str]): Current working directory for command execution.
        last_command (Optional[str]): The last command that was executed.
        last_output (Optional[str]): Output from the last command execution.
        last_error (Optional[str]): Error output from the last command execution.
        last_returncode (Optional[int]): Return code from the last command execution.
    """

    class LastCommandResult:
        def __init__(
            self, result: subprocess.CompletedProcess[Any], pid: int | None = None
        ):
            self._result: subprocess.CompletedProcess[Any] = result
            self._pid: int | None = pid

        @property
        def pid(self) -> int | None:
            return self._pid

        @property
        def result(self) -> subprocess.CompletedProcess[Any]:
            return self._result

    def __init__(self, cwd: str | Path | None = None):
        """Initialize the ShellRunner.

        Args:
            cwd: Working directory for commands. If None, uses current directory.
        """
        # Convert Path to string if provided, otherwise use None
        self.cwd = str(cwd) if cwd else None
        # Initialize state tracking variables
        self.last_command: str | None = None
        self.last_output: str | None = None
        self.last_error: str | None = None
        self.last_returncode: int | None = None

    def run(
        self,
        command: str | list[str],
        check: bool = True,
        capture_output: bool = True,
        env: dict[str, str] | None = None,
        detached: bool = False,
        **kwargs: Any,
    ) -> LastCommandResult | None:
        """Run a shell command.

        Args:
            command: Command to run. Can be a string or list of strings.
            check: If True, raises CalledProcessError if command fails.
            capture_output: If True, captures stdout and stderr.
            env: Environment variables to use.
            detached: If True, runs the process in detached mode (background).
            **kwargs: Additional arguments to pass to subprocess.run.

        Returns:
            CompletedProcess object containing command results.

        Raises:
            subprocess.CalledProcessError: If command fails and check=True.
            FileNotFoundError: If the command or script doesn't exist.
        """
        # Store command for reference and debugging
        self.last_command = command if isinstance(command, str) else " ".join(command)

        # Handle working directory parameter
        if "cwd" in kwargs:
            cwd = kwargs.pop("cwd")
        else:
            cwd = self.cwd

        try:
            if detached:
                # Special handling for detached (background) processes
                # Set up startup info for Windows to hide console window
                startupinfo = None
                if platform.system() == "Windows":
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE

                # Handle script execution differently from command execution
                if isinstance(command, str) and os.path.exists(command):
                    # For scripts, use shell=True to ensure proper execution
                    process = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                        shell=True,
                        startupinfo=startupinfo,
                        **kwargs,
                    )
                else:
                    # For regular commands, execute without shell
                    process = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                        startupinfo=startupinfo,
                        **kwargs,
                    )

                # Create a CompletedProcess with the process ID for consistency
                cp = subprocess.CompletedProcess(
                    args=command, returncode=0, stdout=None, stderr=None
                )
                result = ShellRunner.LastCommandResult(cp, process.pid)

                # Update state tracking variables
                self.last_output = None
                self.last_error = None
                self.last_returncode = 0

                return result
            else:
                # Execute command normally (synchronously)
                cp = subprocess.run(
                    command,
                    cwd=cwd,
                    check=check,
                    capture_output=capture_output,
                    text=True,
                    env=env,
                    **kwargs,
                )

                result = ShellRunner.LastCommandResult(result=cp, pid=None)

                # Update state tracking variables
                self.last_output = result.result.stdout
                self.last_error = result.result.stderr
                self.last_returncode = result.result.returncode

                return result
        except subprocess.CalledProcessError as e:
            # Store error information even when command fails
            self.last_output = e.stdout
            self.last_error = e.stderr
            self.last_returncode = e.returncode
            if check:
                raise
        except FileNotFoundError:
            # Handle case when script doesn't exist
            self.last_output = None
            self.last_error = f"No such file or directory: {command}"
            self.last_returncode = 1
            if check:
                raise

    def start_in_background(
        self,
        name: str,
        command: list[str],
        pid_file: Path,
        log_file: Path,
        **kwargs: Any,
    ) -> subprocess.Popen[Any]:
        """Start a process in the background and track its PID.

        Args:
            name: Name of the process (for logging).
            command: Command to run as a list of strings.
            pid_file: Path to store the process ID.
            log_file: Path to store the process output.
            **kwargs: Additional arguments to pass to subprocess.Popen.

        Returns:
            The started process object.

        Raises:
            click.Abort: If a PID file already exists or if the command is not found.
        """
        # Handle working directory parameter
        if "cwd" in kwargs:
            cwd = kwargs.pop("cwd")
        else:
            cwd = self.cwd

        # Check if process is already running
        if pid_file.exists():
            click.echo(f"Found {name} PID file: {pid_file}.")
            click.echo(
                f"Cowardly refusing to start {name} since another instance might be running."
            )
            click.echo(f"Stop {name} instance if necessary and remove {pid_file}.")
            raise click.Abort()

        try:
            # Start process with output redirected to log file
            with open(log_file, "w") as lf:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdout=lf,
                    stderr=lf,
                    preexec_fn=os.setpgrp,  # Start process group for proper cleanup
                )
            click.echo(f"{name} started with PID: {process.pid}")

            # Store PID in a file for future reference
            with open(pid_file, "w") as f:
                f.write(str(process.pid))
            return process
        except FileNotFoundError:
            # Handle case when command is not found
            click.echo(f"{name} is not available. Please install {name}!")
            click.echo(f"Tried command line: {' '.join(command)}")
            raise click.Abort()

    def stop_from_background(
        self,
        name: str,
        proc_info: subprocess.Popen[Any] | Path,
        stop_on_error: bool = False,
    ) -> None:
        """Stop a background process using its PID.

        Args:
            name: Name of the process (for logging).
            proc_info: Either a Popen object or a Path to the PID file.
            stop_on_error: Whether to raise an error if the process can't be stopped.

        Note:
            If proc_info is a Path, it will be removed after stopping the process.
        """
        if isinstance(proc_info, subprocess.Popen):
            # If we have a process object, terminate it directly
            click.echo(f"Stopping {name} stack (PID: {proc_info.pid})")
            proc_info.terminate()
        elif isinstance(proc_info, Path):
            # If we have a PID file, read it and terminate the process
            pid_file = proc_info
            try:
                with open(pid_file) as f:
                    pid = int(f.read())
            except FileNotFoundError:
                # Handle case when PID file doesn't exist
                click.echo(
                    f"No {name} PID file ({pid_file}) - no way to know which process we should terminate."
                )
                click.echo(
                    f"If you are sure that the {name} is running, use ps and kill to stop it."
                )
                if stop_on_error:
                    raise click.Abort()
                return

            # Terminate the process group
            click.echo(f"Stopping {name} (PID: {pid})")
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                # Handle case when process is already gone
                click.echo(
                    "WARNING: failed to send termination signal to process. It probably is not running any more."
                )
                pass

            # Clean up the PID file
            os.remove(pid_file)

    def is_running_in_background(self, attrib: str, value: Any) -> bool:
        """Check if a process is running in the background.

        Args:
            attrib: Attribute to check. Can be:
                - pid: int | Path
                - group_id: int | Path
                - in_cmdline: str
            value: Value of the attribute.

        Returns:
            True if the process is running in the background, False otherwise.
        """
        # Handle pid file case
        if isinstance(value, Path):
            if not value.exists():
                return False
            with open(value) as f:
                pid = int(f.read())
            if pid == 0:
                return False
        else:
            pid = value

        if attrib == "pid":
            # Simple case, just check if the process is running
            try:
                return psutil.Process(pid).is_running()
            except psutil.NoSuchProcess:
                return False
        elif attrib == "group_id" or attrib == "in_cmdline":
            # We need to iterate over all processes to check if the process is running
            # and if it is the one we are looking for
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if attrib == "in_cmdline":
                        has_cmdline = (
                            "cmdline" in proc.info and proc.info["cmdline"] is not None
                        )
                        if not has_cmdline:
                            continue
                        joined_cmdline = " ".join(proc.info["cmdline"])
                        if isinstance(value, Path):
                            value = value.as_posix()
                        if value in joined_cmdline:
                            return True
                    elif attrib == "group_id":
                        if os.getpgid(proc.info["pid"]) == pid:
                            return True
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    continue
            return False
        else:
            raise ValueError(f"Invalid attribute: {attrib}")
