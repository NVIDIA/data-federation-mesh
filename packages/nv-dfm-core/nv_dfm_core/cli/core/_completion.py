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
Core completion functionality for DFM CLI.

This module provides the core functionality for generating shell completion scripts
for the DFM CLI tool. It supports bash, zsh, and fish shells and provides utilities
for completion script generation and management.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional


class CompletionGenerator:
    """Generator for shell completion scripts."""

    SUPPORTED_SHELLS = ["bash", "zsh", "fish"]

    def __init__(self):
        """Initialize the completion generator."""
        self.env_var = "_DFM_COMPLETE"

    def generate_completion_script(
        self, shell: str, output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a shell completion script for the specified shell.

        Args:
            shell: The target shell type ('bash', 'zsh', or 'fish')
            output_path: Optional path to save the script. If None, returns the script content.

        Returns:
            The completion script content as a string.

        Raises:
            ValueError: If the shell type is not supported.
            RuntimeError: If the completion script generation fails.
        """
        if shell not in self.SUPPORTED_SHELLS:
            raise ValueError(
                f"Unsupported shell: {shell}. Supported shells: {self.SUPPORTED_SHELLS}"
            )

        try:
            # Run the dfm command with the completion environment variable
            result = subprocess.run(
                ["dfm"],
                env={**os.environ, self.env_var: f"{shell}_source"},
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to generate completion script: {result.stderr}"
                )

            completion_script = result.stdout

            if output_path:
                output_path.write_text(completion_script)

            return completion_script

        except FileNotFoundError:
            raise RuntimeError(
                "DFM CLI tool not found in PATH. Please install DFM first."
            )
        except Exception as e:
            raise RuntimeError(f"Error generating completion script: {e}")

    def get_completion_script_path(
        self, shell: str, home_dir: Optional[Path] = None
    ) -> Path:
        """
        Get the standard path for a completion script.

        Args:
            shell: The shell type
            home_dir: Optional home directory. Defaults to user's home directory.

        Returns:
            Path to the completion script file.
        """
        if home_dir is None:
            home_dir = Path.home()

        return home_dir / f".dfm-complete.{shell}"

    def get_shell_config_path(
        self, shell: str, home_dir: Optional[Path] = None
    ) -> Path:
        """
        Get the path to the shell configuration file.

        Args:
            shell: The shell type
            home_dir: Optional home directory. Defaults to user's home directory.

        Returns:
            Path to the shell configuration file.
        """
        if home_dir is None:
            home_dir = Path.home()

        config_files = {
            "bash": home_dir / ".bashrc",
            "zsh": home_dir / ".zshrc",
            "fish": home_dir / ".config" / "fish" / "config.fish",
        }

        return config_files.get(shell, home_dir / f".{shell}rc")

    def is_completion_configured(
        self, shell: str, home_dir: Optional[Path] = None
    ) -> bool:
        """
        Check if completion is already configured for the shell.

        Args:
            shell: The shell type
            home_dir: Optional home directory. Defaults to user's home directory.

        Returns:
            True if completion is configured, False otherwise.
        """
        config_path = self.get_shell_config_path(shell, home_dir)
        completion_path = self.get_completion_script_path(shell, home_dir)

        if not config_path.exists():
            return False

        try:
            config_content = config_path.read_text()
            return str(completion_path) in config_content
        except Exception:
            return False

    def setup_completion(
        self, shell: str, home_dir: Optional[Path] = None, force: bool = False
    ) -> tuple[bool, str]:
        """
        Set up shell completion for the specified shell.

        Args:
            shell: The shell type
            home_dir: Optional home directory. Defaults to user's home directory.
            force: If True, overwrite existing configuration.

        Returns:
            Tuple of (success, message).
        """
        if home_dir is None:
            home_dir = Path.home()

        completion_path = self.get_completion_script_path(shell, home_dir)
        config_path = self.get_shell_config_path(shell, home_dir)

        try:
            # Generate completion script
            self.generate_completion_script(shell, completion_path)

            # Check if already configured
            if not force and self.is_completion_configured(shell, home_dir):
                return True, f"Completion already configured in {config_path}"

            # Add to shell configuration
            if config_path.exists():
                config_content = config_path.read_text()
                source_line = f"source {completion_path}"

                if source_line not in config_content:
                    with open(config_path, "a") as f:
                        f.write(f"\n# DFM CLI completion\n{source_line}\n")
            else:
                # Create config file if it doesn't exist
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, "w") as f:
                    f.write(f"# DFM CLI completion\nsource {completion_path}\n")

            return True, f"Completion configured in {config_path}"

        except Exception as e:
            return False, f"Failed to setup completion: {e}"

    def detect_shell(self) -> str:
        """
        Detect the current shell type.

        Returns:
            The detected shell type ('bash', 'zsh', 'fish', or 'unknown').
        """
        # Check environment variables
        if os.environ.get("ZSH_VERSION"):
            return "zsh"
        elif os.environ.get("BASH_VERSION"):
            return "bash"
        elif os.environ.get("FISH_VERSION"):
            return "fish"

        # Try to detect from parent process
        try:
            import psutil

            parent = psutil.Process().parent()
            if parent:
                parent_name = parent.name().lower()
                if "zsh" in parent_name:
                    return "zsh"
                elif "bash" in parent_name:
                    return "bash"
                elif "fish" in parent_name:
                    return "fish"
        except ImportError:
            pass

        # Fallback to shell environment variable
        shell = os.environ.get("SHELL", "").lower()
        if "zsh" in shell:
            return "zsh"
        elif "bash" in shell:
            return "bash"
        elif "fish" in shell:
            return "fish"

        return "unknown"

    def get_shell_requirements(self, shell: str) -> dict[str, str]:
        """
        Get requirements and notes for shell completion.

        Args:
            shell: The shell type

        Returns:
            Dictionary with requirements information.
        """
        requirements = {
            "bash": {
                "min_version": "4.4",
                "note": "Bash 4.4 or later required for completion support",
            },
            "zsh": {"min_version": "any", "note": "Requires compinit to be loaded"},
            "fish": {
                "min_version": "any",
                "note": "Completion scripts should be placed in ~/.config/fish/completions/",
            },
        }

        return requirements.get(
            shell, {"min_version": "unknown", "note": "Unknown shell"}
        )
