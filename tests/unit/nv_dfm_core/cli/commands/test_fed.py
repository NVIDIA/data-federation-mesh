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
import yaml

from click.testing import CliRunner
from unittest.mock import MagicMock, patch
from unittest.mock import call
import click

from nv_dfm_core.cli.commands._fed import fed
from nv_dfm_core.cli.core._context import CliContext
from nv_dfm_core.cli.core._shell_runner import ShellRunner
from nv_dfm_core.cli.core._generator import Generator

from nv_dfm_core.cli.main import cli


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def mock_context(tmp_path):
    with patch("nv_dfm_core.cli.main.CliContext") as mock:
        context = MagicMock(spec=CliContext)
        context.workspace_path = tmp_path
        context.debug = False
        context.add_config = MagicMock()  # Explicitly mock add_config
        mock.return_value = context
        yield context  # Return the context object, not the mock


@pytest.fixture
def mock_shell_runner():
    with patch("nv_dfm_core.cli.commands._dev.ShellRunner") as mock:
        runner = MagicMock(spec=ShellRunner)
        mock.return_value = runner
        yield runner


@pytest.fixture
def mock_federation_config_manager():
    with patch(
        "nv_dfm_core.cli.commands._fed.FederationConfigManager.add_to_context_and_get"
    ) as mock_mgr:
        manager = MagicMock()
        mock_mgr.return_value = manager
        yield manager


@pytest.fixture
def mock_federation_config():
    with patch("nv_dfm_core.cli.commands._fed.FederationConfig") as mock_config_class:
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        yield mock_config_class


@pytest.fixture
def mock_federation_config_getter():
    with patch(
        "nv_dfm_core.cli.commands._fed.FederationConfig.add_to_context_and_get"
    ) as mock_get_config:
        mock_config = MagicMock()
        mock_config.__str__.return_value = "Mock Federation Config"
        mock_get_config.return_value = mock_config
        yield mock_get_config


@pytest.fixture
def mock_path_exists():
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        yield mock_exists


@pytest.fixture
def mock_generator():
    with patch("nv_dfm_core.cli.commands._fed.Generator") as mock:
        generator = MagicMock()
        mock.return_value = generator
        yield mock


@pytest.fixture
def mock_submitter():
    with patch("nv_dfm_core.cli.commands._fed.get_submitter") as mock_get:
        submitter = MagicMock()
        # Set up default preflight check success
        submitter.preflight_check.return_value = MagicMock(
            success=True, message="Preflight check passed"
        )
        mock_get.return_value = submitter
        yield mock_get


def test_fed_list_all_command(cli_runner, mock_context, mock_federation_config_manager):
    # Mock the federation names to return expected values
    mock_federation_config_manager.get_federation_names.return_value = [
        "examplefed",
    ]

    result = cli_runner.invoke(cli, ["fed", "config", "list-all"])
    assert result.exit_code == 0
    assert "Configured federations: examplefed" in result.output


def test_fed_show_command_success(
    cli_runner, mock_context, mock_federation_config_getter
):
    result = cli_runner.invoke(cli, ["fed", "config", "show", "test_fed"])

    assert result.exit_code == 0
    assert "Federation config:" in result.output
    assert "Mock Federation Config" in result.output


def test_fed_show_command_not_found(
    cli_runner, mock_context, mock_federation_config_getter
):
    mock_federation_config_getter.side_effect = Exception("Federation not found")

    result = cli_runner.invoke(
        cli, ["fed", "config", "show", "nonexistent_fed"], catch_exceptions=True
    )

    assert result.exit_code != 0
    assert "Federation not found" in str(result.exception)


def test_fed_set_command_new_federation(
    cli_runner,
    mock_context,
    mock_federation_config_manager,
    mock_federation_config,
    mock_path_exists,
    tmp_path,
):
    # Mock that federation doesn't exist yet
    mock_federation_config_manager.get_federation_names.return_value = []

    # Create temporary directories for testing
    fed_dir = tmp_path / "fed"
    fed_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    project_path = tmp_path / "project"
    project_path.mkdir()

    # Mock the workspace path and debug flag
    mock_context.workspace_path = tmp_path
    mock_context.debug = False

    # Mock the FederationConfig initialization
    mock_config = mock_federation_config.return_value
    mock_config.initialize.return_value = None

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "config",
            "set",
            "newfed",
            "--federation-dir",
            str(fed_dir),
            "--config-path",
            str(config_path),
            "--project-path",
            str(project_path),
        ],
    )

    assert result.exit_code == 0, (
        f"Expected exit code 0, but got {result.exit_code}. Result: {result}"
    )
    assert "Federation newfed does not exist, creating it." in result.output
    assert "Federation newfed configured." in result.output

    # Verify the configuration was created with debug flag
    mock_federation_config.assert_called_once_with(False)

    # Verify the configuration was added with correct values
    mock_federation_config_manager.add_config.assert_called_once()
    args, kwargs = mock_federation_config_manager.add_config.call_args
    assert args[0] == "newfed"  # First arg should be federation name

    # Verify initialize was called with correct arguments
    mock_config.initialize.assert_called_once()
    _, kwargs = mock_config.initialize.call_args
    assert kwargs["federation_dir"] == str(fed_dir)
    assert kwargs["config_path"] == str(config_path)
    assert kwargs["project_path"] == str(project_path)
    assert kwargs["name"] == "newfed"
    assert kwargs["workspace_dir"] == tmp_path


def test_fed_set_command_existing_federation(
    cli_runner,
    mock_context,
    mock_federation_config_manager,
    mock_path_exists,
    tmp_path,
):
    # Mock that federation exists
    mock_federation_config_manager.get_federation_names.return_value = ["existingfed"]

    # Mock the existing config
    existing_config = MagicMock()
    existing_config.federation_dir = str(tmp_path / "old/fed/dir")
    existing_config.config_path = str(tmp_path / "old/config/path")
    existing_config.project_path = str(tmp_path / "old/project/path")
    mock_federation_config_manager.get_config.return_value = existing_config

    # Create paths for testing
    fed_dir = tmp_path / "new/fed/dir"
    fed_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "new/config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.touch()
    project_path = tmp_path / "new/project"
    project_path.mkdir(parents=True, exist_ok=True)

    # Mock the workspace path and debug flag
    mock_context.workspace_path = tmp_path
    mock_context.debug = False

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "config",
            "set",
            "existingfed",
            "--federation-dir",
            str(fed_dir),
            "--config-path",
            str(config_path),
            "--project-path",
            str(project_path),
        ],
    )

    assert result.exit_code == 0
    assert "Federation existingfed already exists, reconfiguring it." in result.output
    assert "Federation existingfed configured." in result.output

    # Verify the configuration was updated
    mock_federation_config_manager.add_config.assert_called_once()
    args, kwargs = mock_federation_config_manager.add_config.call_args
    assert args[0] == "existingfed"

    # Verify initialize was called with correct arguments
    existing_config.initialize.assert_called_once()
    _, kwargs = existing_config.initialize.call_args
    assert kwargs["federation_dir"] == str(fed_dir)
    assert kwargs["config_path"] == str(config_path)
    assert kwargs["project_path"] == str(project_path)
    assert kwargs["name"] == "existingfed"
    assert kwargs["workspace_dir"] == tmp_path


def test_fed_set_command_partial_update(
    cli_runner,
    mock_context,
    mock_federation_config_manager,
    mock_path_exists,
    tmp_path,
):
    # Mock that federation exists
    mock_federation_config_manager.get_federation_names.return_value = ["existingfed"]

    # Mock the existing config
    existing_config = MagicMock()
    existing_config.federation_dir = str(tmp_path / "old/fed/dir")
    existing_config.config_path = str(tmp_path / "old/config/path")
    existing_config.project_path = str(tmp_path / "old/project/path")
    mock_federation_config_manager.get_config.return_value = existing_config

    # Create path for testing
    fed_dir = tmp_path / "new/fed/dir"
    fed_dir.mkdir(parents=True, exist_ok=True)

    # Mock the workspace path and debug flag
    mock_context.workspace_path = tmp_path
    mock_context.debug = False

    result = cli_runner.invoke(
        cli, ["fed", "config", "set", "existingfed", "--federation-dir", str(fed_dir)]
    )

    assert result.exit_code == 0
    assert "Federation existingfed already exists, reconfiguring it." in result.output

    # Verify the configuration was updated with mix of new and old values
    mock_federation_config_manager.add_config.assert_called_once()

    # Verify initialize was called with correct arguments
    existing_config.initialize.assert_called_once()
    _, kwargs = existing_config.initialize.call_args
    assert kwargs["federation_dir"] == str(fed_dir)  # New value
    assert kwargs["config_path"] == existing_config.config_path  # Preserved old value
    assert kwargs["project_path"] == existing_config.project_path  # Preserved old value
    assert kwargs["name"] == "existingfed"
    assert kwargs["workspace_dir"] == tmp_path


def test_fed_delete_command_success(
    cli_runner,
    mock_context,
    mock_federation_config_manager,
    mock_federation_config_getter,
):
    # Mock that federation exists
    mock_federation_config_manager.get_federation_names.return_value = ["test_fed"]

    result = cli_runner.invoke(cli, ["fed", "config", "delete", "test_fed"])

    assert result.exit_code == 0
    assert "Federation test_fed deleted." in result.output

    # Verify the configuration was deleted
    mock_federation_config_manager.del_config.assert_called_once_with("test_fed")


def test_fed_delete_command_not_found(
    cli_runner,
    mock_context,
    mock_federation_config_manager,
    mock_federation_config_getter,
):
    # Mock that federation doesn't exist
    mock_federation_config_getter.side_effect = Exception("Federation not found")

    result = cli_runner.invoke(
        cli, ["fed", "config", "delete", "nonexistent_fed"], catch_exceptions=True
    )

    assert result.exit_code != 0
    assert "Federation not found" in str(result.exception)

    # Verify delete was not called
    mock_federation_config_manager.del_config.assert_not_called()


def test_fed_gen_code_command_full_generation(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "code",
            "test_fed",
            "--output-dir",
            str(output_dir),
            "--cleanup",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and code method was called with correct arguments
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.code.assert_called_once_with(
        str(output_dir),  # Positional argument
        True,  # cleanup
        False,  # no_api
        None,  # runtime_site
        False,  # skip_runtime
    )


def test_fed_gen_code_command_no_api(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "code",
            "test_fed",
            "--output-dir",
            str(output_dir),
            "--no-api",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and code method was called with correct arguments
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.code.assert_called_once_with(
        str(output_dir),  # Positional argument
        False,  # cleanup
        True,  # no_api
        None,  # runtime_site
        False,  # skip_runtime
    )


def test_fed_gen_code_command_specific_sites(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "code",
            "test_fed",
            "--output-dir",
            str(output_dir),
            "--runtime-site",
            "site1",
            "--runtime-site",
            "site2",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and code method was called with correct arguments
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.code.assert_called_once_with(
        str(output_dir),  # Positional argument
        False,  # cleanup
        False,  # no_api
        ("site1", "site2"),  # runtime_site as tuple
        False,  # skip_runtime
    )


def test_fed_gen_code_command_skip_runtime(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "code",
            "test_fed",
            "--output-dir",
            str(output_dir),
            "--runtime-site",
            "none",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and code method was called with correct arguments
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.code.assert_called_once_with(
        str(output_dir),  # Positional argument
        False,  # cleanup
        False,  # no_api
        ("none",),  # runtime_site as tuple
        True,  # skip_runtime
    )


def test_fed_gen_code_command_invalid_site_combination(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "code",
            "test_fed",
            "--output-dir",
            str(output_dir),
            "--runtime-site",
            "none",
            "--runtime-site",
            "site1",
        ],
    )

    assert result.exit_code != 0
    assert "None can only be specified alone, not with other sites." in result.output

    # Verify no generation happened
    mock_generator.assert_not_called()


def test_fed_gen_docker_command(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and dockerfile method was called
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.dockerfile.assert_called_once_with(
        False, skip=None
    )  # Positional argument


def test_fed_gen_docker_command_force(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--force",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and dockerfile method was called with force=True
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.dockerfile.assert_called_once_with(True, skip=None)  # Positional argument


def test_fed_gen_docker_command_with_patch(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    """Test docker command with patch file."""
    # Create a temporary patch file
    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text("""
    provision:
        docker:
            image: custom_image
            tag: 1.0.0
    """)

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--patch",
            str(patch_file),
        ],
    )

    assert result.exit_code == 0

    # Verify the federation config was patched with patch
    mock_federation_config_getter.return_value.patch.assert_called_once_with(
        str(patch_file)
    )

    # Verify the generator was called
    mock_generator.assert_called_once()
    mock_generator.return_value.dockerfile.assert_called_once_with(False, skip=None)


def test_fed_gen_docker_command_with_patch_and_force(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    """Test docker command with patch file and force flag."""
    # Create a temporary patch file
    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text("""
    provision:
        docker:
            image: custom_image
            tag: 1.0.0
    """)

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--patch",
            str(patch_file),
            "--force",
        ],
    )

    assert result.exit_code == 0

    # Verify the federation config was patched with patch
    mock_federation_config_getter.return_value.patch.assert_called_once_with(
        str(patch_file)
    )

    # Verify the generator was called with force=True
    mock_generator.assert_called_once()
    mock_generator.return_value.dockerfile.assert_called_once_with(True, skip=None)


def test_fed_gen_docker_command_skip_dockerfile(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    """Test docker command with --skip dockerfile."""
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--skip",
            "dockerfile",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and dockerfile method was called with skip=dockerfile
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.dockerfile.assert_called_once_with(False, skip="dockerfile")


def test_fed_gen_docker_command_skip_scripts(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    """Test docker command with --skip scripts."""
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--skip",
            "scripts",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and dockerfile method was called with skip=scripts
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.dockerfile.assert_called_once_with(False, skip="scripts")


def test_fed_gen_docker_command_skip_dockerfile_with_force(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    """Test docker command with --skip dockerfile and --force."""
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--skip",
            "dockerfile",
            "--force",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and dockerfile method was called with force=True and skip=dockerfile
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.dockerfile.assert_called_once_with(True, skip="dockerfile")


def test_fed_gen_docker_command_skip_scripts_with_force(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    """Test docker command with --skip scripts and --force."""
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--skip",
            "scripts",
            "--force",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and dockerfile method was called with force=True and skip=scripts
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.dockerfile.assert_called_once_with(True, skip="scripts")


def test_fed_gen_docker_command_skip_with_patch(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    """Test docker command with --skip and --patch."""
    # Create a temporary patch file
    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text("""
    provision:
        docker:
            image: custom_image
            tag: 1.0.0
    """)

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--skip",
            "dockerfile",
            "--patch",
            str(patch_file),
        ],
    )

    assert result.exit_code == 0

    # Verify the federation config was patched with patch
    mock_federation_config_getter.return_value.patch.assert_called_once_with(
        str(patch_file)
    )

    # Verify the generator was called with skip=dockerfile
    mock_generator.assert_called_once()
    mock_generator.return_value.dockerfile.assert_called_once_with(
        False, skip="dockerfile"
    )


def test_fed_gen_docker_command_skip_with_patch_and_force(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    """Test docker command with --skip, --patch, and --force."""
    # Create a temporary patch file
    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text("""
    provision:
        docker:
            image: custom_image
            tag: 1.0.0
    """)

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--skip",
            "scripts",
            "--patch",
            str(patch_file),
            "--force",
        ],
    )

    assert result.exit_code == 0

    # Verify the federation config was patched with patch
    mock_federation_config_getter.return_value.patch.assert_called_once_with(
        str(patch_file)
    )

    # Verify the generator was called with force=True and skip=scripts
    mock_generator.assert_called_once()
    mock_generator.return_value.dockerfile.assert_called_once_with(True, skip="scripts")


def test_fed_gen_docker_command_invalid_skip_value(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    """Test docker command with invalid --skip value."""
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--skip",
            "invalid_value",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--skip'" in result.output

    # Verify no generation happened
    mock_generator.assert_not_called()


def test_fed_gen_docker_command_with_nonexistent_patch(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    """Test docker command with non-existent patch file."""
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--patch",
            "nonexistent.yaml",
        ],
    )

    assert result.exit_code != 0
    assert "Path 'nonexistent.yaml' does not exist" in result.output

    # Verify the federation config was not patched
    mock_federation_config_getter.return_value.patch.assert_not_called()

    # Verify the generator was not called
    mock_generator.assert_not_called()


def test_fed_gen_docker_command_with_invalid_patch(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    """Test docker command with invalid patch file."""
    # Create a temporary patch file with invalid YAML
    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text("invalid: yaml: content: {")

    # Mock the patch method to raise an error for invalid YAML
    mock_federation_config_getter.return_value.patch.side_effect = yaml.YAMLError(
        "Invalid YAML"
    )

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "docker",
            "test_fed",
            "--patch",
            str(patch_file),
        ],
    )

    assert result.exit_code != 0
    assert "Invalid YAML" in str(result.exception)

    # Verify the federation config was not patched
    mock_federation_config_getter.return_value.patch.assert_called_once_with(
        str(patch_file)
    )

    # Verify the generator was not called
    mock_generator.assert_not_called()


def test_fed_gen_provision_command(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "provision",
            "test_fed",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and provision method was called
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.provision.assert_called_once()


def test_fed_submit_command_success(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_submitter,
    tmp_path,
):
    job_path = tmp_path / "job.py"
    job_path.touch()

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "submit",
            "test_fed",
            "script",
            str(job_path),
            "--target",
            "flare",
        ],
    )

    assert result.exit_code == 0
    assert "Preflight check passed" in result.output
    assert f"Submitting {job_path} to test_fed (flare)..." in result.output

    # Verify submitter was created with correct config
    mock_submitter.assert_called_once()
    args, _ = mock_submitter.call_args
    assert args[0] == "script"  # job_type
    assert args[1].fed_cfg == mock_federation_config_getter.return_value
    assert args[1].timeout == 3600  # Default timeout
    assert args[1].job_args == ()  # No job args

    # Verify submit was called with correct parameters
    submitter = mock_submitter.return_value
    submitter.submit.assert_called_once_with(job_path, "flare")


def test_fed_submit_command_with_args(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_submitter,
    tmp_path,
):
    job_path = tmp_path / "job.py"
    job_path.touch()

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "submit",
            "test_fed",
            "script",
            str(job_path),
            "--target",
            "local",
            "--timeout",
            "7200",
            "--",
            "--arg1",
            "value1",
            "--arg2",
            "value2",
        ],
    )

    assert result.exit_code == 0
    assert "Preflight check passed" in result.output
    assert f"Submitting {job_path} to test_fed (local)..." in result.output

    # Verify submitter was created with correct config including job args
    mock_submitter.assert_called_once()
    args, _ = mock_submitter.call_args
    assert args[0] == "script"  # job_type
    assert args[1].fed_cfg == mock_federation_config_getter.return_value
    assert args[1].timeout == 7200
    assert args[1].job_args == ("--arg1", "value1", "--arg2", "value2")

    # Verify submit was called with correct parameters
    submitter = mock_submitter.return_value
    submitter.submit.assert_called_once_with(job_path, "local")


def test_fed_submit_command_preflight_failure(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_submitter,
    tmp_path,
):
    job_path = tmp_path / "job.py"
    job_path.touch()

    # Set up preflight check failure
    submitter = mock_submitter.return_value
    submitter.preflight_check.return_value = MagicMock(
        success=False, message="Preflight check failed: invalid configuration"
    )

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "submit",
            "test_fed",
            "script",
            str(job_path),
        ],
    )

    assert result.exit_code != 0
    assert "Preflight check failed: invalid configuration" in result.output

    # Verify submit was not called
    submitter.submit.assert_not_called()


def test_fed_submit_command_invalid_target(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_submitter,
    tmp_path,
):
    job_path = tmp_path / "job.py"
    job_path.touch()

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "submit",
            "test_fed",
            "script",
            str(job_path),
            "--target",
            "invalid",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--target'" in result.output

    # Verify submitter was not even created
    mock_submitter.assert_not_called()


def test_fed_gen_helm_command(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "helm",
            "test_fed",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and helm_chart method was called
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.helm_chart.assert_called_once_with(False)  # Positional argument


def test_fed_gen_helm_command_force(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "helm",
            "test_fed",
            "--force",
        ],
    )

    assert result.exit_code == 0

    # Verify generator was created and helm_chart method was called with force=True
    mock_generator.assert_called_once()
    generator = mock_generator.return_value
    generator.helm_chart.assert_called_once_with(True)  # Positional argument


def test_fed_gen_helm_command_with_patch(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    """Test helm command with patch file."""
    # Create a temporary patch file
    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text("""
    provision:
        helm:
            chart_name: custom_chart
            version: 1.0.0
    """)

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "helm",
            "test_fed",
            "--patch",
            str(patch_file),
        ],
    )

    assert result.exit_code == 0

    # Verify the federation config was patched with patch
    mock_federation_config_getter.return_value.patch.assert_called_once_with(
        str(patch_file)
    )

    # Verify the generator was called
    mock_generator.assert_called_once()
    mock_generator.return_value.helm_chart.assert_called_once_with(False)


def test_fed_gen_helm_command_with_patch_and_force(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    """Test helm command with patch file and force flag."""
    # Create a temporary patch file
    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text("""
    provision:
        helm:
            chart_name: custom_chart
            version: 1.0.0
    """)

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "helm",
            "test_fed",
            "--patch",
            str(patch_file),
            "--force",
        ],
    )

    assert result.exit_code == 0

    # Verify the federation config was patched with patch
    mock_federation_config_getter.return_value.patch.assert_called_once_with(
        str(patch_file)
    )

    # Verify the generator was called with force=True
    mock_generator.assert_called_once()
    mock_generator.return_value.helm_chart.assert_called_once_with(True)


def test_fed_gen_helm_command_with_nonexistent_patch(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
):
    """Test helm command with non-existent patch file."""
    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "helm",
            "test_fed",
            "--patch",
            "nonexistent.yaml",
        ],
    )

    assert result.exit_code != 0
    assert "Path 'nonexistent.yaml' does not exist" in result.output

    # Verify the federation config was not patched
    mock_federation_config_getter.return_value.patch.assert_not_called()

    # Verify the generator was not called
    mock_generator.assert_not_called()


def test_fed_gen_helm_command_with_invalid_patch(
    cli_runner,
    mock_context,
    mock_federation_config_getter,
    mock_generator,
    tmp_path,
):
    """Test helm command with invalid patch file."""
    # Create a temporary patch file with invalid YAML
    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text("invalid: yaml: content: {")

    # Mock the patch method to raise an error for invalid YAML
    mock_federation_config_getter.return_value.patch.side_effect = yaml.YAMLError(
        "Invalid YAML"
    )

    result = cli_runner.invoke(
        cli,
        [
            "fed",
            "gen",
            "helm",
            "test_fed",
            "--patch",
            str(patch_file),
        ],
    )

    assert result.exit_code != 0
    assert "Invalid YAML" in str(result.exception)

    # Verify the federation config was not patched
    mock_federation_config_getter.return_value.patch.assert_called_once_with(
        str(patch_file)
    )

    # Verify the generator was not called
    mock_generator.assert_not_called()


def test_fed_set_command_interactive_prompt(
    cli_runner,
    mock_context,
    mock_federation_config_manager,
    mock_federation_config,
    mock_path_exists,
    tmp_path,
):
    """Test set command with interactive prompt for missing parameters."""
    # Mock that federation doesn't exist yet
    mock_federation_config_manager.get_federation_names.return_value = []

    # Create temporary directories for testing
    fed_dir = tmp_path / "fed"
    fed_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    project_path = tmp_path / "project"
    project_path.mkdir()

    # Mock the workspace path and debug flag
    mock_context.workspace_path = tmp_path
    mock_context.debug = False

    # Mock the FederationConfig initialization
    mock_config = mock_federation_config.return_value
    mock_config.initialize.return_value = None

    # Mock click.prompt to simulate user input
    with patch("click.prompt") as mock_prompt:
        # Set up mock prompt to return patch for each parameter
        mock_prompt.side_effect = [
            str(fed_dir),  # federation_dir
            str(config_path),  # config_path
            str(project_path),  # project_path
        ]

        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "config",
                "set",
                "newfed",
            ],
        )

        assert result.exit_code == 0
        assert "Federation newfed does not exist, creating it." in result.output
        assert "Federation newfed configured." in result.output

        # Verify prompt was called for each missing parameter
        assert mock_prompt.call_count == 3
        mock_prompt.assert_has_calls(
            [
                call("Please provide a value for federation_dir (default: None)"),
                call("Please provide a value for config_path (default: None)"),
                call("Please provide a value for project_path (default: None)"),
            ]
        )

        # Verify the configuration was created with debug flag
        mock_federation_config.assert_called_once_with(False)

        # Verify the configuration was added with correct values
        mock_federation_config_manager.add_config.assert_called_once()
        args, kwargs = mock_federation_config_manager.add_config.call_args
        assert args[0] == "newfed"  # First arg should be federation name

        # Verify initialize was called with correct arguments
        mock_config.initialize.assert_called_once()
        _, kwargs = mock_config.initialize.call_args
        assert kwargs["federation_dir"] == str(fed_dir)
        assert kwargs["config_path"] == str(config_path)
        assert kwargs["project_path"] == str(project_path)
        assert kwargs["name"] == "newfed"
        assert kwargs["workspace_dir"] == tmp_path


def test_fed_set_command_interactive_prompt_partial(
    cli_runner,
    mock_context,
    mock_federation_config_manager,
    mock_federation_config,
    mock_path_exists,
    tmp_path,
):
    """Test set command with interactive prompt for some missing parameters."""
    # Mock that federation doesn't exist yet
    mock_federation_config_manager.get_federation_names.return_value = []

    # Create temporary directories for testing
    fed_dir = tmp_path / "fed"
    fed_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    project_path = tmp_path / "project"
    project_path.mkdir()

    # Mock the workspace path and debug flag
    mock_context.workspace_path = tmp_path
    mock_context.debug = False

    # Mock the FederationConfig initialization
    mock_config = mock_federation_config.return_value
    mock_config.initialize.return_value = None

    # Mock click.prompt to simulate user input
    with patch("click.prompt") as mock_prompt:
        # Set up mock prompt to return values for missing parameters
        mock_prompt.side_effect = [
            str(config_path),  # config_path
            str(project_path),  # project_path
        ]

        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "config",
                "set",
                "newfed",
                "--federation-dir",
                str(fed_dir),
            ],
        )

        assert result.exit_code == 0
        assert "Federation newfed does not exist, creating it." in result.output
        assert "Federation newfed configured." in result.output

        # Verify prompt was called only for missing parameters
        assert mock_prompt.call_count == 2
        mock_prompt.assert_has_calls(
            [
                call("Please provide a value for config_path (default: None)"),
                call("Please provide a value for project_path (default: None)"),
            ]
        )

        # Verify the configuration was created with debug flag
        mock_federation_config.assert_called_once_with(False)

        # Verify the configuration was added with correct values
        mock_federation_config_manager.add_config.assert_called_once()
        args, kwargs = mock_federation_config_manager.add_config.call_args
        assert args[0] == "newfed"  # First arg should be federation name

        # Verify initialize was called with correct arguments
        mock_config.initialize.assert_called_once()
        _, kwargs = mock_config.initialize.call_args
        assert kwargs["federation_dir"] == str(fed_dir)
        assert kwargs["config_path"] == str(config_path)
        assert kwargs["project_path"] == str(project_path)
        assert kwargs["name"] == "newfed"
        assert kwargs["workspace_dir"] == tmp_path


def test_fed_set_command_interactive_prompt_cancel(
    cli_runner,
    mock_context,
    mock_federation_config_manager,
    mock_federation_config,
    mock_path_exists,
    tmp_path,
):
    """Test set command with interactive prompt when user cancels."""
    # Mock that federation doesn't exist yet
    mock_federation_config_manager.get_federation_names.return_value = []

    # Mock the workspace path and debug flag
    mock_context.workspace_path = tmp_path
    mock_context.debug = False

    # Mock click.prompt to simulate user cancellation
    with patch("click.prompt") as mock_prompt:
        # Set up mock prompt to raise Abort
        mock_prompt.side_effect = click.Abort()

        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "config",
                "set",
                "newfed",
            ],
            catch_exceptions=True,
        )

        assert result.exit_code != 0
        # Click converts Abort to SystemExit
        assert isinstance(result.exception, SystemExit)

        # Verify prompt was called
        mock_prompt.assert_called_once()

        # Verify no configuration was created (no initialize or add_config calls)
        mock_federation_config.return_value.initialize.assert_not_called()
        mock_federation_config_manager.add_config.assert_not_called()


def test_fed_config_create_default_command(
    cli_runner,
    mock_context,
    mock_federation_config_manager,
    tmp_path,
):
    """Test create-default command for federation configuration."""
    # Mock the CLI config: get_config("cli") returns the config object directly
    cli_config = MagicMock()
    default_path = tmp_path / "default_federations.yaml"
    cli_config.federations_config_path = default_path
    cli_config.federations_config_path_override = False
    mock_context.get_config.return_value = cli_config

    # Create a temporary path for the config
    config_path = tmp_path / "federations.yaml"

    # Mock the class method create_default_config
    with patch(
        "nv_dfm_core.cli.commands._fed.FederationConfigManager.create_default_config"
    ) as mock_create_default:
        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "config",
                "create-default",
                "--path",
                str(config_path),
            ],
        )

        assert result.exit_code == 0
        assert f"Default federation config created at {config_path}" in result.output

        # Verify FederationConfigManager.create_default_config was called
        mock_create_default.assert_called_once_with(config_path, mock_context.debug)

        # Verify the CLI config's federations_config_path was updated
        assert cli_config.federations_config_path == config_path


def test_fed_config_create_default_command_default_path(
    cli_runner,
    mock_context,
    mock_federation_config_manager,
    tmp_path,
):
    """Test create-default command with default path from CLI config."""
    # Mock the CLI config with a default path: get_config("cli") returns the config object directly
    cli_config = MagicMock()
    default_path = tmp_path / "default_federations.yaml"
    cli_config.federations_config_path = default_path
    mock_context.get_config.return_value = cli_config

    # Mock the class method create_default_config
    with patch(
        "nv_dfm_core.cli.commands._fed.FederationConfigManager.create_default_config"
    ) as mock_create_default:
        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "config",
                "create-default",
            ],
        )

        assert result.exit_code == 0
        assert f"Default federation config created at {default_path}" in result.output

        # Verify FederationConfigManager.create_default_config was called with default path
        called_args, called_kwargs = mock_create_default.call_args
        assert str(called_args[0]) == str(default_path)
        assert called_args[1] == mock_context.debug

        # Verify the CLI config's federations_config_path was updated
        assert str(cli_config.federations_config_path) == str(default_path)


def test_fed_command_with_config_parameter(
    cli_runner,
    mock_context,
    tmp_path,
):
    """Test fed command with --config parameter."""
    # Create a temporary config file
    config_path = tmp_path / "custom_config.yaml"
    config_path.touch()

    # Mock the CLI config: get_config("cli") returns the config object directly
    cli_config = MagicMock()
    cli_config.federations_config_path = None
    cli_config.federations_config_path_override = False
    mock_context.get_config.return_value = cli_config

    # Mock the FederationConfigManager to return a valid config
    with patch(
        "nv_dfm_core.cli.commands._fed.FederationConfigManager.add_to_context_and_get"
    ) as mock_federation_config_manager:
        mock_federation_config_manager.return_value = MagicMock()

        # Invoke the fed command with --config and a subcommand
        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "--config",
                str(config_path),
                "config",
                "list-all",
            ],
        )

        assert result.exit_code == 0
        assert cli_config.federations_config_path == config_path
        assert cli_config.federations_config_path_override is True


def test_create_default_with_config_and_path_same(
    cli_runner,
    mock_context,
    tmp_path,
):
    """Test create-default with --config and --path pointing to the same file."""
    config_path = tmp_path / "federations.yaml"
    config_path.touch()
    cli_config = MagicMock()
    cli_config.federations_config_path = config_path
    cli_config.federations_config_path_override = True
    mock_context.get_config.return_value = cli_config
    with patch(
        "nv_dfm_core.cli.commands._fed.FederationConfigManager.create_default_config"
    ) as mock_create_default:
        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "--config",
                str(config_path),
                "config",
                "create-default",
                "--path",
                str(config_path),
            ],
        )
        assert result.exit_code == 0
        assert f"Default federation config created at {config_path}" in result.output
        mock_create_default.assert_called_once_with(config_path, mock_context.debug)
        assert cli_config.federations_config_path == config_path


def test_create_default_with_config_and_path_different_override(
    cli_runner,
    mock_context,
    tmp_path,
):
    """Test create-default with --config and --path pointing to different files and override True."""
    config_path = tmp_path / "federations.yaml"
    path_arg = tmp_path / "other.yaml"
    config_path.touch()
    path_arg.touch()
    cli_config = MagicMock()
    cli_config.federations_config_path = config_path
    cli_config.federations_config_path_override = True
    mock_context.get_config.return_value = cli_config
    with patch(
        "nv_dfm_core.cli.commands._fed.FederationConfigManager.create_default_config"
    ) as mock_create_default:
        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "--config",
                str(config_path),
                "config",
                "create-default",
                "--path",
                str(path_arg),
            ],
        )
        assert result.exit_code != 0
        assert (
            "Arguments --config and --path point to different files." in result.output
        )
        mock_create_default.assert_not_called()


def test_create_default_with_only_path(
    cli_runner,
    mock_context,
    tmp_path,
):
    """Test create-default with only --path provided."""
    path_arg = tmp_path / "other.yaml"
    cli_config = MagicMock()
    cli_config.federations_config_path = None
    cli_config.federations_config_path_override = False
    mock_context.get_config.return_value = cli_config
    with patch(
        "nv_dfm_core.cli.commands._fed.FederationConfigManager.create_default_config"
    ) as mock_create_default:
        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "config",
                "create-default",
                "--path",
                str(path_arg),
            ],
        )
        assert result.exit_code == 0
        assert f"Default federation config created at {path_arg}" in result.output
        mock_create_default.assert_called_once_with(path_arg, mock_context.debug)
        assert cli_config.federations_config_path == path_arg


def test_create_default_with_only_config(
    cli_runner,
    mock_context,
    tmp_path,
):
    """Test create-default with only --config provided."""
    config_path = tmp_path / "federations.yaml"
    config_path.touch()
    cli_config = MagicMock()
    cli_config.federations_config_path = config_path
    cli_config.federations_config_path_override = True
    mock_context.get_config.return_value = cli_config
    with patch(
        "nv_dfm_core.cli.commands._fed.FederationConfigManager.create_default_config"
    ) as mock_create_default:
        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "--config",
                str(config_path),
                "config",
                "create-default",
            ],
        )
        assert result.exit_code == 0
        assert f"Default federation config created at {config_path}" in result.output
        mock_create_default.assert_called_once_with(config_path, mock_context.debug)
        assert cli_config.federations_config_path == config_path


def test_create_default_with_neither(
    cli_runner,
    mock_context,
    tmp_path,
):
    """Test create-default with neither --config nor --path provided."""
    default_path = tmp_path / "default.yaml"
    cli_config = MagicMock()
    cli_config.federations_config_path = default_path
    cli_config.federations_config_path_override = False
    mock_context.get_config.return_value = cli_config
    with patch(
        "nv_dfm_core.cli.commands._fed.FederationConfigManager.create_default_config"
    ) as mock_create_default:
        result = cli_runner.invoke(
            cli,
            [
                "fed",
                "config",
                "create-default",
            ],
        )
        assert result.exit_code == 0
        assert f"Default federation config created at {default_path}" in result.output
        mock_create_default.assert_called_once_with(default_path, mock_context.debug)
        assert cli_config.federations_config_path == default_path
