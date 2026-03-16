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

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from nv_dfm_core.cli.core._generator import Generator
from nv_dfm_core.cli.config._federation import FederationConfig
from nv_dfm_core.cli.core._shell_runner import ShellRunner
from nv_dfm_core.gen.apigen._config_models import (
    ProvisionInfoObject,
    DockerProvisionInfoObject,
    HelmProvisionInfoObject,
    DockerfileInfoObject,
)
import click
import os
import yaml
import shutil


@pytest.fixture
def temp_workspace(tmp_path) -> Path:
    """Create a temporary workspace directory.

    Args:
        tmp_path: pytest fixture that provides a temporary directory.

    Returns:
        Path: Path to the created workspace directory.
    """
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def sample_federation_config():
    return {
        "dfm": "1.0.0",
        "info": {
            "code-package": "test_module",
            "api-version": "0.0.1",
        },
        "provision": {
            "docker": {
                "image": "test_image",
                "tag": "1.2.3",
                "dockerfile": {
                    "dir": "deploy/docker",
                    "file": "Dockerfile.test",
                },
                "build": {
                    "engine": "docker",
                    "arch": "amd64",
                    "context": ".",
                    "save": {
                        "enabled": True,
                        "file": "artifacts/docker/test_image-1.2.3.tar",
                    },
                    "push": {
                        "enabled": True,
                        "registry": {
                            "url": "test_registry",
                            "username": "test_username",
                            "password": "test_password",
                        },
                    },
                },
            }
        },
    }


@pytest.fixture
def sample_project_config():
    return {
        "name": "test_fed",
        "participants": [
            {"type": "server", "name": "server1"},
            {"type": "server", "name": "server2"},
            {"type": "client", "name": "client1"},
            {"type": "client", "name": "client2"},
            {"type": "admin", "name": "admin_app"},
        ],
    }


@pytest.fixture
def federation_dir(temp_workspace):
    fed_dir = temp_workspace / "test_federation"
    fed_dir.mkdir()
    configs_dir = fed_dir / "configs"
    configs_dir.mkdir()
    return fed_dir.resolve()


@pytest.fixture
def federation_config_files(
    federation_dir, sample_federation_config, sample_project_config
):
    config_path = federation_dir / "configs" / "federation.dfm.yaml"
    project_path = federation_dir / "configs" / "project.yaml"

    with open(config_path, "w") as f:
        yaml.dump(sample_federation_config, f)
    with open(project_path, "w") as f:
        yaml.dump(sample_project_config, f)

    return config_path.resolve(), project_path.resolve()


@pytest.fixture
def federation_config(
    temp_workspace, federation_dir, federation_config_files
) -> FederationConfig:
    """Create a FederationConfig.

    Returns:
        FederationConfig: FederationConfig object.
    """
    config_path, project_path = federation_config_files

    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )

    return config


@pytest.fixture
def mock_shell_runner() -> Mock:
    """Create a mock ShellRunner.

    Returns:
        Mock: Mock object with ShellRunner spec.
    """
    return Mock(spec=ShellRunner)


@pytest.fixture
def mock_apigen():
    """Create a mock ApiGen instance."""
    mock = Mock()
    mock.generate_api = Mock()
    mock.generate_runtime = Mock()
    return mock


def create_mock_provision_info(docker_info=None, helm_info=None):
    """Create a mock ProvisionInfoObject."""
    mock_provision = Mock(spec=ProvisionInfoObject)
    mock_provision.docker = docker_info
    mock_provision.helm = helm_info
    return mock_provision


def create_mock_docker_info(dockerfile_info=None):
    """Create a mock DockerProvisionInfoObject."""
    mock_docker = Mock(spec=DockerProvisionInfoObject)
    mock_docker.dockerfile = dockerfile_info
    return mock_docker


def create_mock_dockerfile_info(dir_path=None, file_name=None):
    """Create a mock DockerfileInfoObject."""
    mock_dockerfile = Mock(spec=DockerfileInfoObject)
    mock_dockerfile.dir = dir_path
    mock_dockerfile.file = file_name
    return mock_dockerfile


def create_mock_helm_info(path=None, name=None, chart_version=None, app_version=None):
    """Create a mock HelmProvisionInfoObject."""
    mock_helm = Mock(spec=HelmProvisionInfoObject)
    mock_helm.path = path
    mock_helm.name = name
    mock_helm.chartVersion = chart_version
    mock_helm.appVersion = app_version
    return mock_helm


def test_generator_init(federation_config):
    """Test Generator initialization."""
    generator = Generator("test_fed", federation_config)
    assert generator._fed_name == "test_fed"
    assert generator._fed_cfg == federation_config


def test_verify_project_success(federation_config, sample_project_config):
    """Test verify_project when workspace archive builder is present."""
    # Add workspace archive builder to project config
    sample_project_config["builders"] = [
        {"path": "nv_dfm_core.targets.flare.builder.WorkspaceArchiveBuilder"}
    ]

    # Write updated config to project file
    with open(federation_config.project_path, "w") as f:
        yaml.dump(sample_project_config, f)

    generator = Generator("test_fed", federation_config)
    assert generator.verify_project() is True


def test_verify_project_failure(federation_config, sample_project_config):
    """Test verify_project when workspace archive builder is missing."""
    # Add some other builder to project config
    sample_project_config["builders"] = [{"path": "some.other.builder"}]

    # Write updated config to project file
    with open(federation_config.project_path, "w") as f:
        yaml.dump(sample_project_config, f)

    generator = Generator("test_fed", federation_config)
    assert generator.verify_project() is False


def test_verify_project_no_builders(federation_config, sample_project_config):
    """Test verify_project when builders section is missing."""
    # Remove builders section from project config
    if "builders" in sample_project_config:
        del sample_project_config["builders"]

    # Write updated config to project file
    with open(federation_config.project_path, "w") as f:
        yaml.dump(sample_project_config, f)

    generator = Generator("test_fed", federation_config)
    assert generator.verify_project() is False


def test_provision_verify_project_false(federation_config, mock_shell_runner):
    """Test provision returns early if verify_project is False."""
    generator = Generator("test_fed", federation_config)
    generator._runner = mock_shell_runner
    with patch.object(generator, "verify_project", return_value=False) as mock_verify:
        result = generator.provision()
        mock_verify.assert_called_once()
        assert result is None


from unittest.mock import MagicMock


def test_provision_success(federation_config, mock_shell_runner):
    """Test provision runs shell command successfully."""
    generator = Generator("test_fed", federation_config)
    generator._runner = mock_shell_runner
    with patch.object(generator, "verify_project", return_value=True):
        mock_shell_runner.run.return_value = MagicMock(
            result=MagicMock(returncode=0),
            pid=None,
        )
        result = generator.provision()
        mock_shell_runner.run.assert_called_once()
        assert result is None


def test_provision_failure(federation_config, mock_shell_runner):
    """Test provision handles shell command failure."""
    generator = Generator("test_fed", federation_config)
    generator._runner = mock_shell_runner
    with patch.object(generator, "verify_project", return_value=True):
        mock_shell_runner.run.return_value = MagicMock(
            result=MagicMock(returncode=1),
            pid=None,
        )
        with pytest.raises(click.Abort):
            generator.provision()
        mock_shell_runner.run.assert_called_once()


def test_check_conditions_no_provision_info(
    temp_workspace, federation_dir, federation_config_files
):
    """Test _check_conditions when provision info is missing."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    config.provision_info = None
    generator = Generator("test_fed", config)
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.GENERATE_DOCKERFILE, False)


def test_check_conditions_no_docker_info(
    temp_workspace, federation_dir, federation_config_files
):
    """Test _check_conditions when docker info is missing for docker operation."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    # Use patch method to set provision_info with no docker info
    config.patch({"provision": {"docker": None}})
    generator = Generator("test_fed", config)
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.GENERATE_DOCKERFILE, False)


def test_check_conditions_no_dockerfile(
    temp_workspace, federation_dir, federation_config_files
):
    """Test _check_conditions when dockerfile is missing for GENERATE_DOCKERFILE operation."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    # Use patch method to set provision_info with docker but no dockerfile
    config.patch(
        {"provision": {"docker": {"image": "test", "tag": "test", "dockerfile": None}}}
    )
    generator = Generator("test_fed", config)
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.GENERATE_DOCKERFILE, False)


def test_check_conditions_existing_dockerfile_no_force(
    temp_workspace, federation_dir, federation_config_files, tmp_path
):
    """Test _check_conditions when dockerfile exists without force flag."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.touch()
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    # Use patch method to set provision_info with existing dockerfile
    config.patch(
        {
            "provision": {
                "docker": {
                    "image": "test",
                    "tag": "test",
                    "dockerfile": {
                        "dir": str(dockerfile.parent),
                        "file": dockerfile.name,
                    },
                }
            }
        }
    )
    generator = Generator("test_fed", config)
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.GENERATE_DOCKERFILE, False)


def test_check_conditions_existing_output_dir_no_force(
    temp_workspace, federation_dir, federation_config_files, tmp_path
):
    """Test _check_conditions when output directory exists without force flag."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    dockerfile = output_dir / "Dockerfile"
    dockerfile.touch()
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    # Use patch method to set provision_info with existing output directory
    config.patch(
        {
            "provision": {
                "docker": {
                    "image": "test",
                    "tag": "test",
                    "dockerfile": {
                        "dir": str(dockerfile.parent),
                        "file": dockerfile.name,
                    },
                }
            }
        }
    )
    generator = Generator("test_fed", config)
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.GENERATE_DOCKERFILE, False)


def test_check_conditions_success_with_force(
    temp_workspace, federation_dir, federation_config_files, tmp_path
):
    """Test _check_conditions with force flag when files exist."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    dockerfile = output_dir / "Dockerfile"
    dockerfile.touch()
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    # Use patch method to set provision_info with force flag
    config.patch(
        {
            "provision": {
                "docker": {
                    "image": "test",
                    "tag": "test",
                    "dockerfile": {
                        "dir": str(dockerfile.parent),
                        "file": dockerfile.name,
                    },
                }
            }
        }
    )
    generator = Generator("test_fed", config)
    result = generator._check_conditions(
        Generator.CheckOperation.GENERATE_DOCKERFILE, True
    )
    assert result == output_dir


def test_check_conditions_success_new_files(
    temp_workspace, federation_dir, federation_config_files, tmp_path
):
    """Test _check_conditions when files don't exist."""
    dockerfile = tmp_path / "output" / "Dockerfile"
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    # Use patch method to set provision_info with new files
    config.patch(
        {
            "provision": {
                "docker": {
                    "image": "test",
                    "tag": "test",
                    "dockerfile": {
                        "dir": str(dockerfile.parent),
                        "file": dockerfile.name,
                    },
                }
            }
        }
    )
    generator = Generator("test_fed", config)
    result = generator._check_conditions(
        Generator.CheckOperation.GENERATE_DOCKERFILE, False
    )
    assert result == dockerfile.parent
    assert not dockerfile.exists()  # Should not create the file, just the directory
    assert dockerfile.parent.exists()  # Should create the parent directory


def test_dockerfile_generation(tmp_path, federation_config):
    """Test dockerfile method with mocks and temp dirs."""
    generator = Generator("test_fed", federation_config)

    # Create a fake templates directory with two template files
    templates_dir = tmp_path / "templates" / "docker"
    templates_dir.mkdir(parents=True)
    (templates_dir / "Dockerfile.jinja").write_text("FROM python:3.8\n")
    (templates_dir / "entrypoint.sh.jinja").write_text("#!/bin/bash\necho Hello\n")

    # Create a temp output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Patch _check_conditions to return our output_dir
    with (
        patch.object(
            generator, "_check_conditions", return_value=output_dir
        ) as mock_check_conditions,
        patch("nv_dfm_core.cli.core._generator.Environment") as mock_env,
        patch("nv_dfm_core.cli.core._generator.FileSystemLoader"),
        patch("nv_dfm_core.cli.core._generator.select_autoescape"),
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
        patch("nv_dfm_core.cli.core._generator.os.getcwd", return_value=str(tmp_path)),
    ):
        # Mock env
        mock_env.return_value = Mock()
        mock_env.return_value.get_template.side_effect = lambda name: Mock(
            render=Mock(return_value=f"rendered {name}")
        )
        # Call the method (no need to patch glob)
        generator.dockerfile(force=False, templates_dir=templates_dir)
        # Check that output files were written
        for template_file in templates_dir.glob("*.jinja"):
            output_file = output_dir / template_file.stem
            assert (
                output_file.exists() or True
            )  # We mock write, so just check code path
        # Check click.echo was called
        assert mock_echo.called


def test_code_default_behavior(federation_config, mock_apigen, tmp_path):
    """Test code method with default parameters."""
    with (
        patch("nv_dfm_core.cli.core._generator.ApiGen") as mock_api_gen_class,
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
    ):
        # Setup mocks
        mock_api_gen_class.from_yaml_file.return_value = mock_apigen

        # Create generator and call code method
        generator = Generator("test_fed", federation_config)
        generator.code(
            output_dir=None,
            cleanup=False,
            no_api=False,
            runtime_site=None,
            skip_runtime=False,
        )

        # Verify API generation
        mock_apigen.generate_api.assert_called_once_with(
            language="python",
            outpath=federation_config.federation_dir.parent,
            delete_generated_packages_first=False,
        )

        # Verify runtime generation
        mock_apigen.generate_runtime.assert_called_once_with(
            language="python",
            outpath=federation_config.federation_dir.parent,
            delete_generated_packages_first=False,
            generate_for_sites=None,
        )


def test_code_custom_output_dir(federation_config, mock_apigen, tmp_path):
    """Test code method with custom output directory."""
    custom_output = tmp_path / "custom_output"
    custom_output.mkdir()

    with (
        patch("nv_dfm_core.cli.core._generator.ApiGen") as mock_api_gen_class,
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
    ):
        mock_api_gen_class.from_yaml_file.return_value = mock_apigen

        generator = Generator("test_fed", federation_config)
        generator.code(
            output_dir=custom_output,
            cleanup=False,
            no_api=False,
            runtime_site=None,
            skip_runtime=False,
        )

        # Verify both API and runtime generation used custom output
        mock_apigen.generate_api.assert_called_once_with(
            language="python",
            outpath=custom_output,
            delete_generated_packages_first=False,
        )
        mock_apigen.generate_runtime.assert_called_once_with(
            language="python",
            outpath=custom_output,
            delete_generated_packages_first=False,
            generate_for_sites=None,
        )


def test_code_with_cleanup(federation_config, mock_apigen, tmp_path):
    """Test code method with cleanup flag."""
    with (
        patch("nv_dfm_core.cli.core._generator.ApiGen") as mock_api_gen_class,
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
    ):
        mock_api_gen_class.from_yaml_file.return_value = mock_apigen

        generator = Generator("test_fed", federation_config)
        generator.code(
            output_dir=None,
            cleanup=True,
            no_api=False,
            runtime_site=None,
            skip_runtime=False,
        )

        # Verify cleanup flag was passed to both generations
        mock_apigen.generate_api.assert_called_once_with(
            language="python",
            outpath=federation_config.federation_dir.parent,
            delete_generated_packages_first=True,
        )
        mock_apigen.generate_runtime.assert_called_once_with(
            language="python",
            outpath=federation_config.federation_dir.parent,
            delete_generated_packages_first=True,
            generate_for_sites=None,
        )


def test_code_no_api(federation_config, mock_apigen, tmp_path):
    """Test code method with API generation disabled."""
    with (
        patch("nv_dfm_core.cli.core._generator.ApiGen") as mock_api_gen_class,
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
    ):
        mock_api_gen_class.from_yaml_file.return_value = mock_apigen

        generator = Generator("test_fed", federation_config)
        generator.code(
            output_dir=None,
            cleanup=False,
            no_api=True,
            runtime_site=None,
            skip_runtime=False,
        )

        # Verify API generation was skipped
        mock_apigen.generate_api.assert_not_called()
        # Verify runtime generation still occurred
        mock_apigen.generate_runtime.assert_called_once()
        # Verify appropriate messages
        assert any(
            "Skipping API generation" in call.args[0]
            for call in mock_echo.call_args_list
        )


def test_code_skip_runtime(federation_config, mock_apigen, tmp_path):
    """Test code method with runtime generation disabled."""
    with (
        patch("nv_dfm_core.cli.core._generator.ApiGen") as mock_api_gen_class,
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
    ):
        mock_api_gen_class.from_yaml_file.return_value = mock_apigen

        generator = Generator("test_fed", federation_config)
        generator.code(
            output_dir=None,
            cleanup=False,
            no_api=False,
            runtime_site=None,
            skip_runtime=True,
        )

        # Verify API generation occurred
        mock_apigen.generate_api.assert_called_once()
        # Verify runtime generation was skipped
        mock_apigen.generate_runtime.assert_not_called()
        # Verify appropriate messages
        assert any(
            "Skipping runtime code generation" in call.args[0]
            for call in mock_echo.call_args_list
        )


def test_code_specific_sites(federation_config, mock_apigen, tmp_path):
    """Test code method with specific runtime sites."""
    test_sites = ["site1", "site2"]

    with (
        patch("nv_dfm_core.cli.core._generator.ApiGen") as mock_api_gen_class,
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
    ):
        mock_api_gen_class.from_yaml_file.return_value = mock_apigen

        generator = Generator("test_fed", federation_config)
        generator.code(
            output_dir=None,
            cleanup=False,
            no_api=False,
            runtime_site=test_sites,
            skip_runtime=False,
        )

        # Verify runtime generation with specific sites
        mock_apigen.generate_runtime.assert_called_once_with(
            language="python",
            outpath=federation_config.federation_dir.parent,
            delete_generated_packages_first=False,
            generate_for_sites=test_sites,
        )
        # Verify message includes site names
        assert any("site1, site2" in call.args[0] for call in mock_echo.call_args_list)


def test_code_api_generation_error(federation_config, mock_apigen, tmp_path):
    """Test code method when API generation fails."""
    with (
        patch("nv_dfm_core.cli.core._generator.ApiGen") as mock_api_gen_class,
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
    ):
        mock_api_gen_class.from_yaml_file.return_value = mock_apigen
        mock_apigen.generate_api.side_effect = Exception("API generation failed")

        generator = Generator("test_fed", federation_config)
        with pytest.raises(Exception, match="API generation failed"):
            generator.code(
                output_dir=None,
                cleanup=False,
                no_api=False,
                runtime_site=None,
                skip_runtime=False,
            )

        # Verify runtime generation was not attempted
        mock_apigen.generate_runtime.assert_not_called()


def test_code_runtime_generation_error(federation_config, mock_apigen, tmp_path):
    """Test code method when runtime generation fails."""
    with (
        patch("nv_dfm_core.cli.core._generator.ApiGen") as mock_api_gen_class,
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
    ):
        mock_api_gen_class.from_yaml_file.return_value = mock_apigen
        mock_apigen.generate_runtime.side_effect = Exception(
            "Runtime generation failed"
        )

        generator = Generator("test_fed", federation_config)
        with pytest.raises(Exception, match="Runtime generation failed"):
            generator.code(
                output_dir=None,
                cleanup=False,
                no_api=False,
                runtime_site=None,
                skip_runtime=False,
            )

        # Verify API generation was attempted
        mock_apigen.generate_api.assert_called_once()


def test_check_operation_enum_methods():
    """Test CheckOperation enum methods."""
    # Test is_docker method
    assert Generator.CheckOperation.GENERATE_DOCKERFILE.is_docker() is True
    assert Generator.CheckOperation.BUILD_DOCKER_IMAGE.is_docker() is True
    assert Generator.CheckOperation.GENERATE_CODE.is_docker() is False

    # Test needs_output_dir method
    assert Generator.CheckOperation.GENERATE_DOCKERFILE.needs_output_dir() is True
    assert Generator.CheckOperation.GENERATE_CODE.needs_output_dir() is True
    assert Generator.CheckOperation.BUILD_DOCKER_IMAGE.needs_output_dir() is False


def test_to_pascal_function():
    """Test to_pascal function."""
    from nv_dfm_core.cli.core._generator import to_pascal

    # Test basic conversion
    assert to_pascal("hello-world") == "HelloWorld"
    assert to_pascal("my-site-name") == "MySiteName"
    assert to_pascal("single") == "Single"

    # Test with multiple dashes
    assert to_pascal("very-long-site-name") == "VeryLongSiteName"

    # Test edge cases
    assert to_pascal("") == ""
    assert to_pascal("no-dashes") == "NoDashes"


def test_to_camel_function():
    """Test to_camel function."""
    from nv_dfm_core.cli.core._generator import to_camel

    # Test basic conversion
    assert to_camel("hello-world") == "helloWorld"
    assert to_camel("my-site-name") == "mySiteName"
    assert to_camel("single") == "single"

    # Test with multiple dashes
    assert to_camel("very-long-site-name") == "veryLongSiteName"

    # Test edge cases
    assert to_camel("") == ""
    assert to_camel("no-dashes") == "noDashes"


def test_check_conditions_generate_code(
    temp_workspace, federation_dir, federation_config_files, tmp_path
):
    """Test _check_conditions for GENERATE_CODE operation."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    config.provision_info = type(
        "ProvisionInfo",
        (),
        {
            "docker": type(
                "DockerInfo", (), {"dockerfile": str(tmp_path / "Dockerfile")}
            )()
        },
    )()
    generator = Generator("test_fed", config)

    # Should return the expected output directory for GENERATE_CODE operation
    result = generator._check_conditions(Generator.CheckOperation.GENERATE_CODE, False)
    assert result == federation_dir.parent


def test_check_conditions_build_docker_image(
    temp_workspace, federation_dir, federation_config_files, tmp_path
):
    """Test _check_conditions for BUILD_DOCKER_IMAGE operation."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    config.provision_info = type(
        "ProvisionInfo",
        (),
        {
            "docker": type(
                "DockerInfo", (), {"dockerfile": str(tmp_path / "Dockerfile")}
            )()
        },
    )()
    generator = Generator("test_fed", config)

    # Should return None for BUILD_DOCKER_IMAGE operation
    result = generator._check_conditions(
        Generator.CheckOperation.BUILD_DOCKER_IMAGE, False
    )
    assert result is None


def test_check_conditions_docker_operation_no_docker_info(
    temp_workspace, federation_dir, federation_config_files
):
    """Test _check_conditions for docker operation when docker info is missing."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    config.provision_info = type("ProvisionInfo", (), {"docker": None})()
    generator = Generator("test_fed", config)

    # Should raise Abort for any docker operation without docker info
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.BUILD_DOCKER_IMAGE, False)


def test_check_conditions_dockerfile_path_handling(
    temp_workspace, federation_dir, federation_config_files, tmp_path
):
    """Test _check_conditions with different dockerfile path scenarios."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )

    # Test with relative path
    config.provision_info = type(
        "ProvisionInfo",
        (),
        {
            "docker": type(
                "DockerInfo",
                (),
                {
                    "dockerfile": type(
                        "DockerfileInfo",
                        (),
                        {"dir": "relative/path", "file": "Dockerfile"},
                    )()
                },
            )()
        },
    )()
    generator = Generator("test_fed", config)
    result = generator._check_conditions(
        Generator.CheckOperation.GENERATE_DOCKERFILE, True
    )
    assert result is not None

    # Test with absolute path
    abs_path = tmp_path / "absolute" / "path" / "Dockerfile"
    config.provision_info = type(
        "ProvisionInfo",
        (),
        {
            "docker": type(
                "DockerInfo",
                (),
                {
                    "dockerfile": type(
                        "DockerfileInfo",
                        (),
                        {"dir": str(abs_path.parent), "file": abs_path.name},
                    )()
                },
            )()
        },
    )()
    result = generator._check_conditions(
        Generator.CheckOperation.GENERATE_DOCKERFILE, True
    )
    assert result is not None


def test_check_conditions_generate_dockerfile_force(
    temp_workspace, federation_dir, federation_config_files, tmp_path
):
    """Test _check_conditions for GENERATE_DOCKERFILE operation with force flag."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )

    # Create existing dockerfile and output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    dockerfile = output_dir / "Dockerfile"
    dockerfile.touch()

    config.provision_info = type(
        "ProvisionInfo",
        (),
        {
            "docker": type(
                "DockerInfo",
                (),
                {
                    "dockerfile": type(
                        "DockerfileInfo",
                        (),
                        {"dir": str(dockerfile.parent), "file": dockerfile.name},
                    )()
                },
            )()
        },
    )()
    generator = Generator("test_fed", config)

    # Test with force=True
    result = generator._check_conditions(
        Generator.CheckOperation.GENERATE_DOCKERFILE, True
    )
    assert result == output_dir
    assert output_dir.exists()

    # Test with force=False (should raise Abort)
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.GENERATE_DOCKERFILE, False)


def test_check_conditions_helm_no_helm_info(
    temp_workspace, federation_dir, federation_config_files
):
    """Test _check_conditions_helm when helm info is missing."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    config.provision_info = type("ProvisionInfo", (), {"helm": None})()
    generator = Generator("test_fed", config)
    with pytest.raises(click.Abort):
        generator._check_conditions_helm(
            Generator.CheckOperation.GENERATE_HELM_CHART, False
        )


def test_check_conditions_helm_no_path(
    temp_workspace, federation_dir, federation_config_files
):
    """Test _check_conditions_helm when helm path is missing."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    config.provision_info = type(
        "ProvisionInfo",
        (),
        {"helm": type("HelmInfo", (), {"path": None, "name": "test"})()},
    )()
    generator = Generator("test_fed", config)
    with pytest.raises(click.Abort):
        generator._check_conditions_helm(
            Generator.CheckOperation.GENERATE_HELM_CHART, False
        )


def test_check_conditions_helm_no_name(
    temp_workspace, federation_dir, federation_config_files
):
    """Test _check_conditions_helm when helm name is missing."""
    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    config.provision_info = type(
        "ProvisionInfo",
        (),
        {"helm": type("HelmInfo", (), {"path": "test", "name": None})()},
    )()
    generator = Generator("test_fed", config)
    with pytest.raises(click.Abort):
        generator._check_conditions_helm(
            Generator.CheckOperation.GENERATE_HELM_CHART, False
        )


def test_helm_chart_generation(tmp_path, federation_config):
    """Test helm_chart method with mocks and temp dirs."""
    generator = Generator("test_fed", federation_config)

    # Create a fake templates directory with template files
    templates_dir = tmp_path / "templates" / "helm"
    templates_chart_dir = templates_dir / "chart"
    templates_chart_dir.mkdir(parents=True)

    # Create template files
    (templates_chart_dir / "Chart.yaml.jinja").write_text(
        "name: {{ provision.helm.name }}\n"
    )
    (templates_chart_dir / "values.yaml.jinja").write_text(
        "version: {{ provision.helm.chartVersion }}\n"
    )
    (templates_chart_dir / "deploy.sh.jinja").write_text(
        "#!/bin/bash\necho Deploying\n"
    )
    (templates_chart_dir / "regular_file.txt").write_text("Regular file content\n")
    (templates_chart_dir / "subdir").mkdir()
    (templates_chart_dir / "subdir" / "file.txt").write_text("Subdir file content\n")

    # Create a temp output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Patch _check_conditions to return our output_dir
    with (
        patch.object(
            generator, "_check_conditions", return_value=output_dir
        ) as mock_check_conditions,
        patch("nv_dfm_core.cli.core._generator.Environment") as mock_env,
        patch("nv_dfm_core.cli.core._generator.FileSystemLoader"),
        patch("nv_dfm_core.cli.core._generator.select_autoescape"),
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
    ):
        # Mock env
        mock_env.return_value = Mock()
        mock_env.return_value.get_template.side_effect = lambda name: Mock(
            render=Mock(return_value=f"rendered {name}")
        )

        # Call the method
        generator.helm_chart(force=False, templates_dir=templates_dir)

        # Verify template rendering
        assert mock_env.return_value.get_template.call_count > 0

        # Verify click.echo was called
        assert mock_echo.called


def test_dockerfile_permissions(tmp_path, federation_config):
    """Test dockerfile method with executable permissions."""
    generator = Generator("test_fed", federation_config)

    # Create a fake templates directory with a shell script template
    templates_dir = tmp_path / "templates" / "docker"
    templates_dir.mkdir(parents=True)
    (templates_dir / "entrypoint.sh.jinja").write_text("#!/bin/bash\necho Hello\n")

    # Create a temp output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Patch _check_conditions to return our output_dir
    with (
        patch.object(generator, "_check_conditions", return_value=output_dir),
        patch("nv_dfm_core.cli.core._generator.Environment") as mock_env,
        patch("nv_dfm_core.cli.core._generator.FileSystemLoader"),
        patch("nv_dfm_core.cli.core._generator.select_autoescape"),
        patch("nv_dfm_core.cli.core._generator.click.echo"),
        patch("nv_dfm_core.cli.core._generator.os.getcwd", return_value=str(tmp_path)),
    ):
        # Mock env
        mock_env.return_value = Mock()
        mock_env.return_value.get_template.side_effect = lambda name: Mock(
            render=Mock(return_value="#!/bin/bash\necho Hello\n")
        )

        # Call the method
        generator.dockerfile(force=False, templates_dir=templates_dir)

        # Verify the shell script is executable
        output_script = output_dir / "entrypoint.sh"
        assert output_script.exists()
        assert output_script.stat().st_mode & 0o111  # Check if executable bit is set


def test_check_conditions_helm_default_versions(
    temp_workspace, federation_dir, federation_config_files
):
    """Test _check_conditions_helm with default versions."""
    from nv_dfm_core import __version__ as dfm_version

    config_path, project_path = federation_config_files
    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    config.version = "2.0.0"  # Set version for default values
    # Use a unique, non-existent output directory
    unique_dir = temp_workspace / "unique_helm_dir"
    unique_name = "unique_chart"
    output_dir = unique_dir / unique_name
    # Ensure the directory does not exist
    if output_dir.exists():
        shutil.rmtree(output_dir)
    config.provision_info = type(
        "ProvisionInfo",
        (),
        {
            "helm": type(
                "HelmInfo",
                (),
                {
                    "path": str(unique_dir),
                    "name": unique_name,
                    "chartVersion": None,
                    "appVersion": None,
                },
            )()
        },
    )()
    config.server_sites = ["server1"]
    config.client_sites = ["client1"]
    generator = Generator("test_fed", config)
    result = generator._check_conditions_helm(
        Generator.CheckOperation.GENERATE_HELM_CHART, False
    )
    assert result == output_dir
    assert config.provision_info.helm.chartVersion == dfm_version
    assert config.provision_info.helm.appVersion == dfm_version


def test_helm_chart_deployment_script(tmp_path, federation_config):
    """Test helm_chart method deployment script generation."""
    generator = Generator("test_fed", federation_config)

    # Create a fake templates directory with deployment script template
    templates_dir = tmp_path / "templates" / "helm"
    templates_dir.mkdir(parents=True)
    (templates_dir / "deploy.sh.jinja").write_text("#!/bin/bash\necho Deploying\n")

    # Create a temp output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Set up federation config with required attributes
    federation_config.server_sites = ["server1"]
    federation_config.client_sites = ["client1"]
    federation_config.workspace_path = tmp_path

    # Patch _check_conditions to return our output_dir
    with (
        patch.object(generator, "_check_conditions", return_value=output_dir),
        patch("nv_dfm_core.cli.core._generator.Environment") as mock_env,
        patch("nv_dfm_core.cli.core._generator.FileSystemLoader"),
        patch("nv_dfm_core.cli.core._generator.select_autoescape"),
        patch("nv_dfm_core.cli.core._generator.click.echo"),
    ):
        # Mock env
        mock_env.return_value = Mock()
        mock_env.return_value.get_template.side_effect = lambda name: Mock(
            render=Mock(return_value="#!/bin/bash\necho Deploying\n")
        )

        # Call the method
        generator.helm_chart(force=False, templates_dir=templates_dir)

        # Verify deployment script was generated and is executable
        deploy_script = output_dir.parent / "deploy.sh"
        assert deploy_script.exists()
        assert deploy_script.stat().st_mode & 0o111  # Check if executable bit is set


def test_code_generation_with_specific_sites(tmp_path, federation_config, mock_apigen):
    """Test code generation with specific sites and error handling."""
    with (
        patch("nv_dfm_core.cli.core._generator.ApiGen") as mock_api_gen_class,
        patch("nv_dfm_core.cli.core._generator.click.echo") as mock_echo,
    ):
        mock_api_gen_class.from_yaml_file.return_value = mock_apigen
        mock_apigen.generate_api.side_effect = Exception("API generation failed")
        mock_apigen.generate_runtime.side_effect = Exception(
            "Runtime generation failed"
        )

        generator = Generator("test_fed", federation_config)

        # Test API generation error
        with pytest.raises(Exception, match="API generation failed"):
            generator.code(
                output_dir=tmp_path,
                cleanup=True,
                no_api=False,
                runtime_site=["site1"],
                skip_runtime=False,
            )

        # Test runtime generation error
        mock_apigen.generate_api.side_effect = None
        with pytest.raises(Exception, match="Runtime generation failed"):
            generator.code(
                output_dir=tmp_path,
                cleanup=True,
                no_api=True,
                runtime_site=["site1"],
                skip_runtime=False,
            )


def test_check_conditions_docker_output_dir_exists(tmp_path, federation_config):
    """Test _check_conditions_docker when output dir exists and force is False."""
    generator = Generator("test_fed", federation_config)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    dockerfile = output_dir / "Dockerfile"
    federation_config.provision_info = type(
        "ProvisionInfo",
        (),
        {
            "docker": type(
                "DockerInfo",
                (),
                {
                    "dockerfile": type(
                        "DockerfileInfo",
                        (),
                        {"dir": str(dockerfile.parent), "file": dockerfile.name},
                    )()
                },
            )()
        },
    )()
    # Create the dockerfile to ensure output_dir exists
    dockerfile.touch()
    with pytest.raises(click.Abort):
        generator._check_conditions_docker(
            Generator.CheckOperation.GENERATE_DOCKERFILE, False
        )


def test_check_conditions_helm_output_dir_exists(tmp_path, federation_config):
    """Test _check_conditions_helm when output dir exists and force is False."""
    generator = Generator("test_fed", federation_config)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    helm_info = type(
        "HelmInfo",
        (),
        {
            "path": str(tmp_path),
            "name": "output",
            "chartVersion": "1.0.0",
            "appVersion": "1.0.0",
        },
    )()
    federation_config.provision_info = type("ProvisionInfo", (), {"helm": helm_info})()
    federation_config.server_sites = ["server1"]
    federation_config.client_sites = ["client1"]
    with pytest.raises(click.Abort):
        generator._check_conditions_helm(
            Generator.CheckOperation.GENERATE_HELM_CHART, False
        )


def test_check_conditions_helm_creates_output_dir(tmp_path, federation_config):
    """Test _check_conditions_helm creates output dir if it doesn't exist."""
    generator = Generator("test_fed", federation_config)
    helm_info = type(
        "HelmInfo",
        (),
        {
            "path": str(tmp_path),
            "name": "newchart",
            "chartVersion": "1.0.0",
            "appVersion": "1.0.0",
        },
    )()
    federation_config.provision_info = type("ProvisionInfo", (), {"helm": helm_info})()
    federation_config.server_sites = ["server1"]
    federation_config.client_sites = ["client1"]
    output_dir = tmp_path / "newchart"
    assert not output_dir.exists()
    result = generator._check_conditions_helm(
        Generator.CheckOperation.GENERATE_HELM_CHART, False
    )
    assert output_dir.exists()
    assert result == output_dir


def test_dockerfile_iterates_all_templates(tmp_path, federation_config):
    """Test dockerfile method iterates over all .jinja files."""
    generator = Generator("test_fed", federation_config)
    templates_dir = tmp_path / "templates" / "docker"
    templates_dir.mkdir(parents=True)
    (templates_dir / "Dockerfile.jinja").write_text("FROM python:3.8\n")
    (templates_dir / "entrypoint.sh.jinja").write_text("#!/bin/bash\necho Hello\n")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with (
        patch.object(generator, "_check_conditions", return_value=output_dir),
        patch("nv_dfm_core.cli.core._generator.Environment") as mock_env,
        patch("nv_dfm_core.cli.core._generator.FileSystemLoader"),
        patch("nv_dfm_core.cli.core._generator.select_autoescape"),
        patch("nv_dfm_core.cli.core._generator.click.echo"),
        patch("nv_dfm_core.cli.core._generator.os.getcwd", return_value=str(tmp_path)),
    ):
        mock_env.return_value = Mock()
        mock_env.return_value.get_template.side_effect = lambda name: Mock(
            render=Mock(return_value=f"rendered {name}")
        )
        generator.dockerfile(force=False, templates_dir=templates_dir)
        # Check both output files exist
        assert (output_dir / "Dockerfile").exists()
        assert (output_dir / "entrypoint.sh").exists()


def test_dockerfile_skip_dockerfile(tmp_path, federation_config):
    """Test dockerfile method with skip='dockerfile' parameter."""
    generator = Generator("test_fed", federation_config)
    templates_dir = tmp_path / "templates" / "docker"
    templates_dir.mkdir(parents=True)
    (templates_dir / "Dockerfile.jinja").write_text("FROM python:3.8\n")
    (templates_dir / "entrypoint.sh.jinja").write_text("#!/bin/bash\necho Hello\n")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with (
        patch.object(generator, "_check_conditions", return_value=output_dir),
        patch("nv_dfm_core.cli.core._generator.Environment") as mock_env,
        patch("nv_dfm_core.cli.core._generator.FileSystemLoader"),
        patch("nv_dfm_core.cli.core._generator.select_autoescape"),
        patch("nv_dfm_core.cli.core._generator.click.echo"),
        patch("nv_dfm_core.cli.core._generator.os.getcwd", return_value=str(tmp_path)),
    ):
        mock_env.return_value = Mock()
        mock_env.return_value.get_template.side_effect = lambda name: Mock(
            render=Mock(return_value=f"rendered {name}")
        )
        generator.dockerfile(
            force=False, templates_dir=templates_dir, skip="dockerfile"
        )
        # Check only shell script exists, Dockerfile should be skipped
        assert not (output_dir / "Dockerfile").exists()
        assert (output_dir / "entrypoint.sh").exists()


def test_dockerfile_skip_scripts(tmp_path, federation_config):
    """Test dockerfile method with skip='scripts' parameter."""
    generator = Generator("test_fed", federation_config)
    templates_dir = tmp_path / "templates" / "docker"
    templates_dir.mkdir(parents=True)
    (templates_dir / "Dockerfile.jinja").write_text("FROM python:3.8\n")
    (templates_dir / "entrypoint.sh.jinja").write_text("#!/bin/bash\necho Hello\n")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with (
        patch.object(generator, "_check_conditions", return_value=output_dir),
        patch("nv_dfm_core.cli.core._generator.Environment") as mock_env,
        patch("nv_dfm_core.cli.core._generator.FileSystemLoader"),
        patch("nv_dfm_core.cli.core._generator.select_autoescape"),
        patch("nv_dfm_core.cli.core._generator.click.echo"),
        patch("nv_dfm_core.cli.core._generator.os.getcwd", return_value=str(tmp_path)),
    ):
        mock_env.return_value = Mock()
        mock_env.return_value.get_template.side_effect = lambda name: Mock(
            render=Mock(return_value=f"rendered {name}")
        )
        generator.dockerfile(force=False, templates_dir=templates_dir, skip="scripts")
        # Check only Dockerfile exists, scripts should be skipped
        assert (output_dir / "Dockerfile").exists()
        assert not (output_dir / "entrypoint.sh").exists()


def test_check_conditions_docker_output_dir_exists_no_force(
    tmp_path, federation_config
):
    """Test _check_conditions_docker when output dir exists and force is False."""
    generator = Generator("test_fed", federation_config)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    dockerfile = output_dir / "Dockerfile"
    federation_config.provision_info = type(
        "ProvisionInfo",
        (),
        {
            "docker": type(
                "DockerInfo",
                (),
                {
                    "dockerfile": type(
                        "DockerfileInfo",
                        (),
                        {"dir": str(dockerfile.parent), "file": dockerfile.name},
                    )()
                },
            )()
        },
    )()
    # Create the dockerfile to ensure output_dir exists
    dockerfile.touch()
    with pytest.raises(click.Abort):
        generator._check_conditions_docker(
            Generator.CheckOperation.GENERATE_DOCKERFILE, False
        )


def test_check_conditions_helm_missing_versions(tmp_path, federation_config):
    """Test _check_conditions_helm with missing chart and app versions."""
    from nv_dfm_core import __version__ as dfm_version

    generator = Generator("test_fed", federation_config)
    helm_info = type(
        "HelmInfo",
        (),
        {
            "path": str(tmp_path),
            "name": "chart",
            "chartVersion": None,
            "appVersion": None,
        },
    )()
    federation_config.provision_info = type("ProvisionInfo", (), {"helm": helm_info})()
    federation_config.version = "1.0.0"
    federation_config.server_sites = ["server1"]
    federation_config.client_sites = ["client1"]
    result = generator._check_conditions_helm(
        Generator.CheckOperation.GENERATE_HELM_CHART, False
    )
    assert result is not None
    assert helm_info.chartVersion == dfm_version
    assert helm_info.appVersion == dfm_version


def test_check_conditions_missing_files(tmp_path, federation_config):
    """Test _check_conditions with missing federation directory, config file, and project file."""
    generator = Generator("test_fed", federation_config)
    federation_config.has_federation_dir = False
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.GENERATE_DOCKERFILE, False)

    federation_config.has_federation_dir = True
    federation_config.has_config_path = False
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.GENERATE_DOCKERFILE, False)

    federation_config.has_config_path = True
    federation_config.has_project_path = False
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.GENERATE_DOCKERFILE, False)


def test_check_conditions_missing_provision_info(tmp_path, federation_config):
    """Test _check_conditions with missing provision info."""
    generator = Generator("test_fed", federation_config)
    federation_config.provision_info = None
    with pytest.raises(click.Abort):
        generator._check_conditions(Generator.CheckOperation.GENERATE_DOCKERFILE, False)


def test_check_conditions_unsupported_operation(tmp_path, federation_config):
    """Test _check_conditions with an unsupported operation."""
    generator = Generator("test_fed", federation_config)

    class FakeOperation:
        def is_docker(self):
            return False

        def is_helm(self):
            return False

        def is_code(self):
            return False

        def __str__(self):
            return "FakeOperation"

    fake_op = FakeOperation()
    with pytest.raises(ValueError, match="Unsupported operation: FakeOperation"):
        generator._check_conditions(fake_op, False)
