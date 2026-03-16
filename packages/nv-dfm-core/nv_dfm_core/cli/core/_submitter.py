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

This module provides functionality for submitting and managing jobs in the Data Federation Mesh (DFM)
Proof of Concept environment. It includes different types of submitters for various job types
(scripts, pipelines) and handles the execution and monitoring of these jobs in the federated
learning environment.
"""

import importlib.util
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import click
from typing_extensions import override

from nv_dfm_core.api import Pipeline
from nv_dfm_core.exec import site_name_to_identifier

from ..config._federation import FederationConfig
from ..core._shell_runner import ShellRunner


@dataclass
class PreflightCheckResult:
    """Result of a preflight check for job submission.

    Attributes:
        success: Whether the check passed successfully.
        message: Description of the check result or error message.
    """

    success: bool
    message: str


@dataclass
class SubmitterConfig:
    """Configuration for job submission.

    Attributes:
        fed_cfg: Federation configuration containing workspace and app settings.
        poc_mode: Whether the federation is running in POC mode.
        timeout: Maximum time in seconds to wait for job completion.
        job_args: Optional list of arguments to pass to the job.
    """

    fed_cfg: FederationConfig
    poc_mode: bool = False
    timeout: int = 3600  # Default timeout: 1 hour
    job_args: list[str] | None = None


class SubmitterBase(ABC):
    """Base class for all job submitters.

    This abstract class defines the interface that all job submitters must implement.
    It provides common functionality for configuration management and preflight checks.

    Attributes:
        _config: Configuration for the submitter.
        _preflight_check_result: Result of the last preflight check.
    """

    def __init__(self, config: SubmitterConfig):
        """Initialize the submitter with configuration.

        Args:
            config: Configuration for the submitter.

        Raises:
            ValueError: If config is not a SubmitterConfig instance.
        """
        # Validate configuration type
        self._config: SubmitterConfig = config
        self._preflight_check_result: PreflightCheckResult | None = None

    @abstractmethod
    def submit(self, path: Path, target: str) -> None:
        """Submit a job for execution.

        Args:
            path: Path to the job file or script.
            target: Target environment for job execution.
        """
        pass

    @abstractmethod
    def preflight_check(self, path: Path) -> PreflightCheckResult:
        """Perform preflight checks before job submission.

        Args:
            path: Path to the job file or script.

        Returns:
            Result of the preflight check.
        """
        pass

    @staticmethod
    def type() -> str:
        """Get the type identifier for this submitter.

        Returns:
            String identifier for the submitter type.
        """
        return "__base__"


class ScriptSubmitter(SubmitterBase):
    """Submitter for executable script jobs.

    This submitter handles the execution of standalone scripts in the federated
    learning environment.
    """

    @staticmethod
    @override
    def type() -> str:
        """Get the type identifier for script submitter.

        Returns:
            String identifier "script".
        """
        return "script"

    @override
    def preflight_check(self, path: Path) -> PreflightCheckResult:
        """Check if the script is valid and executable.

        Args:
            path: Path to the script file.

        Returns:
            Result indicating if the script is valid and executable.
        """
        # Initialize result with default success state
        preflight_result = PreflightCheckResult(
            success=True, message=f"Path {path} is valid."
        )

        # Check if file exists
        if not path.exists():
            preflight_result.success = False
            preflight_result.message = f"Path {path} does not exist."
        # Check if it's a file
        elif not path.is_file():
            preflight_result.success = False
            preflight_result.message = f"Path {path} is not a file."
        # Check if it's executable
        elif not bool(path.stat().st_mode & 0o111):
            preflight_result.success = False
            preflight_result.message = f"Path {path} is not executable. Please ensure it has execute permissions."

        # Store result for future use
        self._preflight_check_result = preflight_result
        return preflight_result

    @override
    def submit(self, path: Path, target: str) -> None:
        """Submit a script for execution.

        Args:
            path: Path to the script file.
            _: Unused target parameter.

        Raises:
            RuntimeError: If preflight check fails.
        """
        # Perform preflight check if not already done
        if not self._preflight_check_result:
            _ = self.preflight_check(path)
        assert self._preflight_check_result is not None
        if not self._preflight_check_result.success:
            raise RuntimeError(self._preflight_check_result.message)

        # Get federation configuration
        fed_cfg = self._config.fed_cfg
        fed_cfg.poc_mode = self._config.poc_mode
        click.echo(
            f"Submitting script {path} to federation ({'POC mode' if fed_cfg.poc_mode else ''})..."
        )
        click.echo(f"Using federation path: {fed_cfg.federation_workspace_dir}")

        assert fed_cfg.flare_project_name
        # Set up workspace path and environment
        env = os.environ | {
            "DFM_FEDERATION_WORKSPACE_PATH": (
                fed_cfg.federation_workspace_dir / fed_cfg.flare_project_name
            ).as_posix()
        }

        # Execute the script
        runner = ShellRunner()
        cmd = [path.as_posix()]
        if self._config.job_args:
            cmd.extend(self._config.job_args)
        runner.run(
            cmd,
            check=False,
            capture_output=False,
            env=env,
            timeout=self._config.timeout,
        )


class PipelineSubmitter(SubmitterBase):
    """Submitter for pipeline jobs.

    This submitter handles the execution of pipeline-based jobs in the federated
    learning environment, including pipeline preparation and execution.
    """

    @staticmethod
    def type() -> str:
        """Get the type identifier for pipeline submitter.

        Returns:
            String identifier "pipeline".
        """
        return "pipeline"

    @override
    def preflight_check(self, path: Path) -> PreflightCheckResult:
        """Check if the pipeline file is valid and contains required functions.

        Args:
            path: Path to the pipeline file.

        Returns:
            Result indicating if the pipeline is valid.
        """
        # Initialize result with default success state
        preflight_result = PreflightCheckResult(
            success=True, message=f"Path {path} is a valid pipeline."
        )

        # Check if file exists
        if not path.exists():
            preflight_result.success = False
            preflight_result.message = f"Path {path} does not exist."
            return preflight_result
        # Check if it's a file
        elif not path.is_file():
            preflight_result.success = False
            preflight_result.message = f"Path {path} is not a file."
            return preflight_result
        # Check if it's a Python file
        elif not path.suffix == ".py":
            preflight_result.success = False
            preflight_result.message = f"Path {path} is not a Python file."
            return preflight_result

        # Try to import the file and check for required functions
        try:
            spec = importlib.util.spec_from_file_location("pipeline", path)
            if spec is None:
                raise RuntimeError(f"Failed to import pipeline from {path}")
            assert spec.loader is not None
            module = spec.loader.load_module(fullname="pipeline")

            if not hasattr(module, "get_pipeline"):
                preflight_result.success = False
                preflight_result.message = (
                    f"Path {path} does not implement get_pipeline() method."
                )
                return preflight_result
        except Exception as e:
            preflight_result.success = False
            preflight_result.message = (
                f"Failed to import pipeline from {path}: {str(e)}"
            )
            return preflight_result

        # Store result for future use
        self._preflight_check_result = preflight_result
        return preflight_result

    def _get_session_function(self, fed_cfg: FederationConfig) -> Callable[..., Any]:
        """Get the session function from the federation configuration.

        Args:
            fed_cfg: Federation configuration.

        Returns:
            Function to create a new session.

        Raises:
            RuntimeError: If session module or function is not found.
        """
        # Construct module path from app name
        fed_path = fed_cfg.federation_dir
        click.echo(f"App name: {fed_cfg.app_name}")
        assert fed_cfg.app_name is not None
        module_name = site_name_to_identifier(fed_cfg.app_name)
        module_path = fed_path / "fed" / "runtime" / module_name
        click.echo(f"Using session module: {module_path}")

        # Load and validate the session module
        module_path = module_path / "__init__.py"
        if not module_path.exists():
            raise RuntimeError(f"Session module not found at {module_path}")
        spec = importlib.util.spec_from_file_location("session", module_path)
        if spec is None:
            raise RuntimeError(f"Failed to import session from {module_path}")
        assert spec.loader is not None
        module = spec.loader.load_module(fullname="session")
        if not hasattr(module, "get_session"):
            raise RuntimeError(
                f"Path {fed_cfg.module} does not implement get_session() method."
            )
        return module.get_session

    def _get_pipeline(self, path: Path) -> Pipeline:
        """Get the pipeline from the specified file.

        Args:
            path: Path to the pipeline file.

        Returns:
            The pipeline object.

        Raises:
            RuntimeError: If pipeline function is not found.
        """
        # Load the pipeline module
        spec = importlib.util.spec_from_file_location("test", path)
        if spec is None:
            raise RuntimeError(f"Failed to import pipeline from {path}")
        assert spec.loader is not None
        module = spec.loader.load_module(fullname="pipeline")
        if not hasattr(module, "get_pipeline"):
            raise RuntimeError(f"Path {path} does not implement get_pipeline() method.")
        return module.get_pipeline(self._config.job_args)

    def _get_pipeline_parameters(self, path: Path) -> dict[str, Any]:
        """Get pipeline parameters from the file and command line arguments.

        Args:
            path: Path to the pipeline file.

        Returns:
            Dictionary of pipeline parameters.

        Raises:
            RuntimeError: If pipeline parameters function is not found.
        """
        # Load the pipeline module
        spec = importlib.util.spec_from_file_location("test", path)
        if spec is None:
            raise RuntimeError(f"Failed to import pipeline from {path}")
        assert spec.loader is not None
        module = spec.loader.load_module(fullname="pipeline")
        if not hasattr(module, "get_pipeline_parameters"):
            raise RuntimeError(
                f"Path {path} does not implement get_pipeline_parameters() method."
            )

        # Get base parameters from the module
        params = module.get_pipeline_parameters()

        # Merge with command line arguments if present
        if self._config.job_args:
            # Convert list of [key1, value1, key2, value2] into {key1: value1, key2: value2}
            job_args_dict = dict(
                zip(self._config.job_args[::2], self._config.job_args[1::2])
            )
            params.update(job_args_dict)
        return params

    def submit(self, path: Path, target: str) -> None:
        """Submit a pipeline for execution.

        Args:
            path: Path to the pipeline file.
            target: Target environment for execution.

        Raises:
            RuntimeError: If preflight check fails.
        """
        # Perform preflight check if not already done
        if not self._preflight_check_result:
            self.preflight_check(path)
        assert self._preflight_check_result is not None
        if not self._preflight_check_result.success:
            raise RuntimeError(self._preflight_check_result.message)

        # Get federation configuration
        fed_cfg = self._config.fed_cfg
        fed_cfg.poc_mode = self._config.poc_mode
        app_name = fed_cfg.app_name
        assert app_name
        job_workspace = fed_cfg.federation_workspace_dir / app_name
        flare_workspace = fed_cfg.federation_workspace_dir
        admin_package = fed_cfg.admin_package

        # Create and configure the session
        session_function = self._get_session_function(fed_cfg)
        session = session_function(
            user=app_name,
            flare_workspace=flare_workspace,
            job_workspace=job_workspace,
            admin_package=admin_package,
        )

        # Connect to Flare if target is "flare"
        if target == "flare":
            session.connect()

        # Prepare and execute the pipeline
        pipeline = self._get_pipeline(path)
        click.echo("Preparing pipeline...")
        prepared_pipeline = session.prepare(pipeline)
        click.echo("Pipeline prepared.")
        click.echo("Executing pipeline...")

        # Define callback for handling results
        def results_callback(
            from_site: str, node: int | str | None, target_place: str, data: Any
        ):
            node_str = f"node {node}" if node is not None else ""
            click.echo(
                f"New results found {node_str} from {from_site} to {target_place}:"
            )
            click.echo(f"  {data}")

        # Execute the pipeline with parameters and callbacks
        job = session.execute(
            prepared_pipeline,
            input_params=self._get_pipeline_parameters(path),
            place_callbacks={"yield": results_callback},
        )

        # Wait for job completion
        timeout = self._config.timeout
        click.echo(f"Pipeline started, Flare job id: {job.job_id()}")
        click.echo(f"Waiting for job to complete, timeout: {timeout} seconds...")
        job.wait_until_finished(timeout=timeout)
        click.echo("Job completed.")


# Dictionary to store all available submitters
SUBMITTERS = {}

# Scan for submitter classes and build type mapping
for name, obj in list(globals().items()):
    if name.endswith("Submitter") and isinstance(obj, type):
        if issubclass(obj, SubmitterBase):
            SUBMITTERS[obj.type()] = obj


def list_submitters():
    """Get a list of available submitter types.

    Returns:
        List of submitter type identifiers.
    """
    return list(SUBMITTERS.keys())


def get_submitter(job_type: str, config: SubmitterConfig) -> SubmitterBase:
    """Get a submitter instance for the specified job type.

    Args:
        job_type: Type of job to submit.
        config: Configuration for the submitter.

    Returns:
        An instance of the appropriate submitter.

    Raises:
        RuntimeError: If no submitter is found for the specified type.
    """
    if job_type in SUBMITTERS:
        return SUBMITTERS[job_type](config)
    else:
        raise RuntimeError(
            f"Couldn't find submitter for job type: {job_type}. Please use list_submitters() to see available submitters."
        )
