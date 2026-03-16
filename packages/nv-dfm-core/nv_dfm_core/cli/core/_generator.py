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
Data Federation Mesh (DFM) Generator Module.

This module provides the core Generator class for managing federation deployment
artifacts including Docker images, Helm charts, and code generation. The Generator
handles the provisioning, validation, and generation of various deployment assets
for federated learning environments.

The module supports:
- Federation project verification and provisioning
- Docker image and Dockerfile generation
- Helm chart generation for Kubernetes deployments
- API and runtime code generation
- Validation of federation configurations
"""

import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Any

import click
from jinja2 import Environment, FileSystemLoader, select_autoescape
from nvflare.lighter.utils import load_yaml

from nv_dfm_core import __version__ as dfm_version
from nv_dfm_core.gen.apigen import ApiGen

from ..config._federation import FederationConfig
from ._shell_runner import ShellRunner


def to_camel(input: str) -> str:
    """
    Convert a dash-separated string to camelCase.
    """
    words = input.split("-")
    return words[0] + "".join(word.capitalize() for word in words[1:])


def to_pascal(input: str) -> str:
    """
    Convert a dash-separated string to PascalCase.
    """
    words = input.split("-")
    return "".join(word.capitalize() for word in words)


def to_upper(input: str) -> str:
    """
    Convert a dash-separated string to UPPER_CASE.
    """
    return input.upper().replace("-", "_")


class Generator:
    """
    Core generator class for Data Federation Mesh deployment artifacts.

    This class manages the generation of various deployment assets including
    Dockerfiles, Helm charts, and API/runtime code for federated learning
    environments. It handles validation, provisioning, and template-based
    generation of deployment configurations.

    Attributes:
        _debug (bool): Enable debug logging
        _runner (ShellRunner): Shell command runner instance
        _fed_name (str): Name of the federation
        _fed_cfg (FederationConfig): Federation configuration object
        _project_path (Path): Path to the federation project file
        _workspace_path (Path): Path to the federation workspace
    """

    class CheckOperation(Enum):
        """
        Enumeration of supported generation and build operations.

        This enum defines the different types of operations that can be performed
        by the Generator, along with helper methods to categorize operations.
        """

        GENERATE_DOCKERFILE = "docker.generate"
        GENERATE_HELM_CHART = "helm.generate"
        GENERATE_CODE = "code.generate"
        BUILD_DOCKER_IMAGE = "docker.build"

        def is_docker(self) -> bool:
            """Check if this operation is Docker-related."""
            return self.value.startswith("docker.")

        def is_helm(self) -> bool:
            """Check if this operation is Helm-related."""
            return self.value.startswith("helm.")

        def is_code(self) -> bool:
            """Check if this operation is code generation-related."""
            return self.value.startswith("code.")

        def needs_output_dir(self) -> bool:
            """Check if this operation requires an output directory."""
            return self.value.endswith(".generate")

    def __init__(
        self, federation: str, federation_config: FederationConfig, debug: bool = False
    ):
        """
        Initialize the Generator with federation configuration.

        Args:
            federation (str): Name of the federation
            federation_config (FederationConfig): Configuration object for the federation
            debug (bool, optional): Enable debug logging. Defaults to False.
        """
        self._debug = debug
        self._runner = ShellRunner()
        self._fed_name = federation
        self._fed_cfg = federation_config
        self._project_path = federation_config.project_path
        self._workspace_path = federation_config.federation_workspace_dir
        self._log(f"Generator initialized for federation {self._fed_name}")

    def _log(self, message: str) -> None:
        """
        Log a debug message if debug mode is enabled.

        Args:
            message (str): Message to log
        """
        if self._debug:
            click.echo(f"DEBUG[{self.__class__.__name__}]: {message}", err=True)

    def verify_project(self) -> bool:
        """
        Verify that the federation project is properly configured.

        Checks if the project file contains the required WorkspaceArchiveBuilder
        in the builders section, which is necessary for proper federation deployment.

        Returns:
            bool: True if project is valid, False otherwise
        """
        click.echo(f"Verifying project {self._project_path}...")
        project_dict = load_yaml(self._project_path.as_posix())
        workspace_archive_builder_path = (
            "nv_dfm_core.targets.flare.builder.WorkspaceArchiveBuilder"
        )
        has_workspace_archive_builder = False

        # Check if the required builder is present in the project configuration
        for builder in project_dict.get("builders", []):
            cls = builder.get("path")
            if cls == workspace_archive_builder_path:
                click.echo(f"Found {cls}...")
                has_workspace_archive_builder = True

        if not has_workspace_archive_builder:
            click.echo("No workspace archive builder found in project.")
            click.echo(
                "Please add the following to 'builders' section of your project:"
            )
            click.echo(f"  - path: {workspace_archive_builder_path}")
            return False

        click.echo("Project verified.")
        return True

    def provision(self, clean: bool = False):
        """
        Provision the federation using NVFlare's provision command.

        This method validates the project configuration and then runs the
        NVFlare provision command to set up the federation workspace.

        Raises:
            click.Abort: If project verification fails or provisioning fails
        """
        if not self.verify_project():
            return

        click.echo(
            f"Provisioning federation {self._fed_name} from {self._project_path} to {self._workspace_path}..."
        )

        if clean:
            click.echo("Cleaning workspace...")
            shutil.rmtree(self._workspace_path, ignore_errors=True)

        # Construct the NVFlare provision command
        cmd = [
            "nvflare",
            "provision",
            "-p",
            self._project_path.as_posix(),
            "-w",
            self._workspace_path.as_posix(),  # Flare adds additional directory to the workspace path
        ]

        proc = self._runner.run(cmd, capture_output=False, check=False)
        if proc is not None and proc.result.returncode != 0:
            click.echo(f"Provisioning federation {self._fed_name} failed.")
            raise click.Abort()

        click.echo(f"Provisioning federation {self._fed_name} completed.")

    def _check_conditions_docker(
        self, operation: CheckOperation, force: bool
    ) -> Path | None:
        """
        Validate conditions for Docker-related operations.

        Args:
            operation (CheckOperation): The Docker operation to validate
            force (bool): Whether to force overwrite existing files

        Returns:
            Path: Output directory for the Docker operation

        Raises:
            click.Abort: If validation fails or required configuration is missing
        """
        # Docker operations require project_path and provision info
        if not self._fed_cfg.has_project_path:
            click.echo(
                f"Federation project file {self._fed_cfg.project_path} does not exist."
            )
            click.echo("Docker operations require a project.yaml file.")
            raise click.Abort()

        if (
            self._fed_cfg.provision_info is None
            or self._fed_cfg.provision_info.docker is None
        ):
            click.echo("No docker provision info found in federation configuration.")
            click.echo(
                "Please add a docker section to your federation configuration file."
            )
            raise click.Abort()

        # For now we only support one operation
        if operation == Generator.CheckOperation.GENERATE_DOCKERFILE:
            dockerfile_info = self._fed_cfg.provision_info.docker.dockerfile
            if dockerfile_info is None:
                click.echo("No dockerfile found in federation configuration.")
                click.echo(
                    "Please add a dockerfile section to your federation configuration file."
                )
                raise click.Abort()

            dockerfile_dir = dockerfile_info.dir or "."
            dockerfile_file = dockerfile_info.file or "Dockerfile"
            dockerfile = Path(dockerfile_dir) / dockerfile_file

            # Check if dockerfile already exists
            if dockerfile.exists() and not force:
                click.echo(f"Dockerfile already exists at {dockerfile}")
                click.echo("Use --force to overwrite.")
                raise click.Abort()

            output_dir = dockerfile.parent

            # Check if output directory already exists
            if output_dir.exists() and not force:
                click.echo(f"Output directory already exists at {output_dir}")
                click.echo("Use --force to overwrite.")
                raise click.Abort()

            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir

        return None

    def _check_conditions_helm(self, operation: CheckOperation, force: bool) -> Path:
        """
        Validate conditions for Helm-related operations.

        Args:
            operation (CheckOperation): The Helm operation to validate
            force (bool): Whether to force overwrite existing files

        Returns:
            Path: Output directory for the Helm operation

        Raises:
            click.Abort: If validation fails or required configuration is missing
        """
        # Helm operations require project_path and provision info
        if not self._fed_cfg.has_project_path:
            click.echo(
                f"Federation project file {self._fed_cfg.project_path} does not exist."
            )
            click.echo("Helm operations require a project.yaml file.")
            raise click.Abort()

        if (
            self._fed_cfg.provision_info is None
            or self._fed_cfg.provision_info.helm is None
        ):
            click.echo("No helm provision info found in federation configuration.")
            click.echo(
                "Please add a helm section to your federation configuration file."
            )
            raise click.Abort()

        info = self._fed_cfg.provision_info.helm

        # Validate required Helm configuration fields
        if info.path is None:
            click.echo("No helm path found in federation configuration.")
            click.echo(
                "Please add a helm path section to your federation configuration file."
            )
            raise click.Abort()

        if info.name is None:
            click.echo("No helm name found in federation configuration.")
            click.echo(
                "Please add a helm name section to your federation configuration file."
            )
            raise click.Abort()

        # Set default versions if not specified
        if info.chartVersion is None:
            click.echo(
                "No helm chart version found in federation configuration. Using DFM version."
            )
            info.chartVersion = dfm_version

        if info.appVersion is None:
            click.echo(
                "No helm app version found in federation configuration. Using DFM version."
            )
            info.appVersion = dfm_version

        output_dir = (Path(info.path) / info.name).absolute()

        # Check if output directory already exists
        if output_dir.exists() and not force:
            click.echo(f"Helm chart already exists at {output_dir}")
            click.echo("Use --force to overwrite.")
            raise click.Abort()

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _check_conditions_code(
        self, operation: CheckOperation, force: bool, **kwargs
    ) -> Path:
        """
        Validate conditions for code generation operations.

        Args:
            operation (CheckOperation): The code generation operation to validate
            force (bool): Whether to force overwrite existing files
            **kwargs: Additional keyword arguments, including 'output_dir'

        Returns:
            Path: Output directory for the code generation operation
        """
        output_dir = kwargs.get("output_dir")
        if output_dir is None:
            # Default to parent of federation directory
            output_dir = self._fed_cfg.federation_dir.parent
        else:
            output_dir = Path(output_dir)
        return output_dir

    def _check_conditions(
        self, operation: CheckOperation, force: bool, **kwargs
    ) -> Path | None:
        """
        Validate general conditions and delegate to operation-specific validators.

        This method performs common validation checks for all operations and then
        delegates to the appropriate operation-specific validation method.

        Args:
            operation (CheckOperation): The operation to validate
            force (bool): Whether to force overwrite existing files
            **kwargs: Additional keyword arguments passed to specific validators

        Returns:
            Path: Output directory for the operation

        Raises:
            click.Abort: If validation fails
            ValueError: If operation is not supported
        """
        fc = self._fed_cfg

        # Validate federation configuration paths
        if not fc.has_federation_dir:
            click.echo(f"Federation directory {fc.federation_dir} does not exist.")
            raise click.Abort()

        if not fc.has_config_path:
            click.echo(f"Federation config file {fc.config_path} does not exist.")
            raise click.Abort()

        # Note: project_path and provision_info are only required for docker/helm operations
        # They are validated in the operation-specific methods below

        # Delegate to operation-specific validation
        if operation.is_docker():
            return self._check_conditions_docker(operation, force, **kwargs)
        elif operation.is_helm():
            return self._check_conditions_helm(operation, force, **kwargs)
        elif operation.is_code():
            return self._check_conditions_code(operation, force, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def dockerfile(
        self, force: bool, templates_dir: Path | None = None, skip: str | None = None
    ):
        """
        Generate Dockerfile and related Docker assets from templates.

        This method uses Jinja2 templates to generate Docker-related files
        including Dockerfiles and shell scripts for containerized deployment.

        Args:
            force (bool): Whether to overwrite existing files
            templates_dir (Path, optional): Custom template directory.
                                          Defaults to built-in docker templates.
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates" / "docker"

        env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(),
        )

        output_dir = self._check_conditions(
            Generator.CheckOperation.GENERATE_DOCKERFILE, force
        )

        if output_dir is None:
            click.echo("No output directory specified.")
            raise click.Abort()

        click.echo("Generating Dockerfile...")

        # Process Jinja2 templates in the docker templates directory, skipping the ones specified in skip
        pattern = "*.jinja"
        if skip is not None:
            if skip == "dockerfile":
                pattern = "*.sh.jinja"
            elif skip == "scripts":
                pattern = "Dockerfile.jinja"
            else:
                raise ValueError(f"Unsupported skip value: {skip}")

        federation_dir = self._fed_cfg.federation_dir
        federation_dir_relative = federation_dir.relative_to(os.getcwd())

        for template_file in templates_dir.glob(pattern):
            template = env.get_template(template_file.name)
            content = template.render(
                name=self._fed_name,
                provision=self._fed_cfg.provision_info,
                federation_dir=federation_dir,
                federation_dir_relative=federation_dir_relative,
            )
            output_file = output_dir / template_file.stem
            output_file.write_text(content)

            # Make shell scripts executable
            if output_file.suffix == ".sh":
                mode = output_file.stat().st_mode
                output_file.chmod(mode | 0o111)

            click.echo(f"Generated {output_file}")

        click.echo("Dockerfile generated.")

    def _process_chart_template_dir(
        self, template_dir: Path, output_dir: Path, params: dict[str, Any] | None = None
    ) -> None:
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(),
            # Use custom field delimiters in chart template to avoid conflicts with Helm templates
            variable_start_string="[{",
            variable_end_string="}]",
            block_start_string="[%",
            block_end_string="%]",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for template_file in template_dir.glob("*"):
            if template_file.is_dir():
                # Ignore directories - we treat them in a special way
                continue
            if template_file.suffix == ".jinja":
                # Process Jinja2 templates
                template = env.get_template(template_file.name)
                content = template.render(
                    provision=self._fed_cfg.provision_info,
                    **params,
                )
                output_file = output_dir / template_file.stem
                output_file.write_text(content)

                # Make shell scripts executable
                if output_file.suffix == ".sh":
                    mode = output_file.stat().st_mode
                    output_file.chmod(mode | 0o111)

                click.echo(f"Generated {output_file}")
            else:
                # Copy regular files as-is
                output_file = output_dir / template_file.name
                shutil.copy(template_file, output_file)
                click.echo(f"Copied {template_file} to {output_file}")

    def helm_chart(self, force: bool, templates_dir: Path | None = None) -> None:
        """
        Generate Helm chart and deployment scripts from templates.

        This method generates Kubernetes Helm charts and deployment scripts
        for containerized federation deployment, including configuration files
        and deployment automation scripts.

        Args:
            force (bool): Whether to overwrite existing files
            templates_dir (Path, optional): Custom template directory.
                                          Defaults to built-in helm templates.
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates" / "helm"

        templates_chart_dir = templates_dir / "chart"

        env = Environment(
            loader=FileSystemLoader(templates_chart_dir),
            autoescape=select_autoescape(),
        )

        output_dir = self._check_conditions(
            Generator.CheckOperation.GENERATE_HELM_CHART, force
        )

        if output_dir is None:
            click.echo("No output directory specified.")
            raise click.Abort()

        click.echo(f"Generating Helm chart to {output_dir}...")

        # Process all templates and files in the chart directory and templates subdirectory
        server_sites = self._fed_cfg.server_sites or []
        client_sites = self._fed_cfg.client_sites or []
        params = {
            "server_sites": server_sites,
            "server_sites_camel": [to_pascal(site) for site in server_sites],
            "client_sites": client_sites,
            "client_sites_camel": [to_pascal(site) for site in client_sites],
        }
        click.echo(f"Processing chart templates with params: {params}")
        self._process_chart_template_dir(
            templates_chart_dir,
            output_dir,
            params=params,
        )
        self._process_chart_template_dir(
            templates_chart_dir / "templates",
            output_dir / "templates",
            params=params,
        )

        # Now, for every site, we need to create a subdirectory in the chart
        # Start with server sites
        templates_output_dir = output_dir / "templates"
        for server_site in server_sites or []:
            click.echo(f"Generating server site templates for: {server_site}")
            server_site_dir = templates_output_dir / server_site
            server_site_dir.mkdir(parents=True, exist_ok=True)
            template_dir = templates_chart_dir / "templates" / "server"
            self._process_chart_template_dir(
                template_dir,
                server_site_dir,
                params={
                    "site_name": server_site,
                    "site_name_camel": to_pascal(server_site),
                },
            )

        # Now, for every client site, we need to create a subdirectory in the chart
        for client_site in client_sites or []:
            click.echo(f"Generating client site templates for: {client_site}")
            client_site_dir = templates_output_dir / client_site
            client_site_dir.mkdir(parents=True, exist_ok=True)
            template_dir = templates_chart_dir / "templates" / "clients"
            self._process_chart_template_dir(
                template_dir,
                client_site_dir,
                params={
                    "site_name": client_site,
                    "site_name_camel": to_pascal(client_site),
                },
            )

        click.echo("Helm chart generated.")

        # Generate deployment script
        click.echo("Generating deployment script...")
        env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(),
        )
        template = env.get_template("deploy.sh.jinja")
        assert self._fed_cfg.flare_project_name is not None
        server_sites_upper = [to_upper(site) for site in server_sites]
        client_sites_upper = [to_upper(site) for site in client_sites]
        render_args = params | {
            "workspace_path": (
                self._workspace_path / self._fed_cfg.flare_project_name
            ).as_posix(),
            "provision": self._fed_cfg.provision_info,
            "server_sites_upper": server_sites_upper,
            "client_sites_upper": client_sites_upper,
        }

        content = template.render(**render_args)

        output_file = output_dir.parent / "deploy.sh"
        _ = output_file.write_text(content)

        # Make deployment script executable
        mode = output_file.stat().st_mode
        output_file.chmod(mode | 0o111)
        click.echo("Deployment script generated.")

    def code(
        self,
        output_dir: Path | None,
        cleanup: bool,
        no_api: bool,
        runtime_site: list[str] | None,
        skip_runtime: bool,
    ):
        """
        Generate API and runtime code for the federation.

        This method generates Python API and runtime code based on the federation
        configuration, allowing for selective generation of API code, runtime code,
        or both.

        Args:
            output_dir (Path, optional): Directory to output generated code.
                                       Defaults to parent of federation directory.
            cleanup (bool): Whether to clean up existing generated packages first
            no_api (bool): Skip API code generation
            runtime_site (list[str], optional): Specific sites to generate runtime code for.
                                              Defaults to all sites.
            skip_runtime (bool): Skip runtime code generation
        """
        output_dir = self._check_conditions(
            Generator.CheckOperation.GENERATE_CODE, True, output_dir=output_dir
        )
        if output_dir is None:
            click.echo("No output directory specified.")
            raise click.Abort()

        # Initialize the API generator from the federation configuration
        apigen = ApiGen.from_yaml_file(self._fed_cfg.config_path)

        # Generate API code if requested
        if not no_api:
            apigen.generate_api(
                language="python",
                outpath=Path(output_dir),
                delete_generated_packages_first=cleanup,
            )
            click.echo(f"Federation {self._fed_name} API code generated.")
        else:
            click.echo("Skipping API generation.")

        # Generate runtime code if requested
        if not skip_runtime:
            apigen.generate_runtime(
                language="python",
                outpath=Path(output_dir),
                delete_generated_packages_first=cleanup,
                generate_for_sites=runtime_site,
            )
            click.echo(
                f"Federation {self._fed_name} runtime code generated for site(s): {', '.join(runtime_site) if runtime_site else 'all'}."
            )
        else:
            click.echo("Skipping runtime code generation.")

        click.echo(f"Federation {self._fed_name} code generated.")
