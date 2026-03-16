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
from click import Abort
from click.testing import CliRunner
from unittest.mock import MagicMock, patch
import click
import os
from pathlib import Path

from nv_dfm_core.cli.commands._dev import dev
from nv_dfm_core.cli.core._context import CliContext
from nv_dfm_core.cli.core._shell_runner import ShellRunner
from nv_dfm_core.cli.core._version import BumpType
from nv_dfm_core.cli.config._cli import CliConfig, Utils
from nv_dfm_core.cli.config.models._cli import (
    Cli,
    Workspace,
    Dev,
    DevTesting,
    DevLinting,
    DevFormatting,
    DevTypeChecking,
)
from nv_dfm_core.cli.main import cli


@pytest.fixture
def cli_runner():
    return CliRunner()


def get_good_cli_config(tmp_path):
    Utils._repo_dir_cache = tmp_path
    Utils.set_cli_config_path(tmp_path / "config.yaml")

    config = CliConfig()
    config.repo_dir = tmp_path  # Set the repo_dir property

    validated = Cli(
        federations_config_path=Path("./federations.yaml"),
        workspace=Workspace(path=Path("./workspace")),
        dev=Dev(
            testing=DevTesting(script=Path("ci", "scripts", "python-test-package.sh")),
            linting=DevLinting(script=Path("ci", "scripts", "python-run-lint.sh")),
            formatting=DevFormatting(
                script=Path("ci", "scripts", "python-run-format.sh")
            ),
            type_checking=DevTypeChecking(
                script=Path("ci", "scripts", "python-run-type-check.sh")
            ),
        ),
    )

    config._config = validated.model_dump()
    return config


def get_bad_cli_config(tmp_path):
    config = CliConfig(root_path=tmp_path)
    config._dev._stored = DevConfigStored(
        testing=DevTestingConfigStored(
            script=Path("ci", "scripts", "python-test-package.sh-not-found")
        ),
        linting=DevLintingConfigStored(
            script=Path("ci", "scripts", "python-run-lint.sh-not-found")
        ),
        formatting=DevFormattingConfigStored(
            script=Path("ci", "scripts", "python-run-format.sh-not-found")
        ),
    )
    return config


@pytest.fixture
def mock_context(tmp_path):
    with patch("nv_dfm_core.cli.main.CliContext") as mock:
        context = MagicMock(spec=CliContext)
        context.workspace_path = tmp_path
        context.debug = False
        context.add_config = MagicMock()  # Explicitly mock add_config
        context.get_config = MagicMock()  # Mock get_config method
        mock.return_value = context
        yield context  # Return the context object, not the mock


def test_dev_version_command(cli_runner, mock_context):
    result = cli_runner.invoke(cli, ["dev", "version"])
    print(result.output)
    assert result.exit_code == 0
    assert "DFM version" in result.output


@pytest.fixture
def mock_shell_runner():
    with patch("nv_dfm_core.cli.commands._dev.ShellRunner") as mock:
        runner = MagicMock(spec=ShellRunner)
        mock.return_value = runner
        yield runner


def test_dev_test_command_success(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock test script
    script_path = config.dev.testing.script
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock successful test run
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=0),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "test"])

    assert result.exit_code == 0
    assert "Tests completed successfully" in result.output
    mock_shell_runner.run.assert_called_once_with(
        [str(script_path)],
        env=(os.environ),
        capture_output=False,
        check=False,
    )


def test_dev_test_command_failure(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock test script
    script_path = config.dev.testing.script
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock failed test run
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=1),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "test"])

    assert result.exit_code == 1
    assert "Tests failed" in result.output
    mock_shell_runner.run.assert_called_once_with(
        [str(script_path)],
        env=(os.environ),
        capture_output=False,
        check=False,
    )


def test_dev_test_command_script_not_found(cli_runner, mock_context, tmp_path):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Do not create the script file
    result = cli_runner.invoke(cli, ["dev", "test"])

    assert result.exit_code == 1
    assert "Error: Test script not found" in result.output


def test_dev_lint_command_success(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock lint script
    script_path = config.dev.linting.script
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock successful lint run
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=0),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "lint"])

    assert result.exit_code == 0
    assert "Linting completed successfully" in result.output
    mock_shell_runner.run.assert_called_once_with(
        [str(script_path)],
        env=(os.environ),
        capture_output=False,
        check=False,
    )


def test_dev_lint_command_with_fix(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock lint script
    script_path = config.dev.linting.script
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock successful lint run with fix
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=0),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "lint", "--fix"])

    assert result.exit_code == 0
    assert "Linting completed successfully" in result.output
    mock_shell_runner.run.assert_called_once_with(
        [str(script_path), "--fix"],
        env=(os.environ),
        capture_output=False,
        check=False,
    )


def test_dev_lint_command_failure(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock lint script
    script_path = config.dev.linting.script
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock failed lint run
    mock_shell_runner.run.return_value = MagicMock(returncode=1)

    result = cli_runner.invoke(cli, ["dev", "lint"])

    assert result.exit_code == 1
    assert "Linting failed" in result.output
    mock_shell_runner.run.assert_called_once_with(
        [str(script_path)],
        env=(os.environ),
        capture_output=False,
        check=False,
    )


def test_dev_lint_command_script_not_found(cli_runner, mock_context, tmp_path):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Do not create the script file
    result = cli_runner.invoke(cli, ["dev", "lint"])

    assert result.exit_code == 1
    assert "Error: Lint script not found" in result.output


def test_dev_format_command_success(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock format script
    script_path = config.dev.formatting.script
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock successful format run
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=0),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "format"])

    assert result.exit_code == 0
    assert "Formatting completed successfully" in result.output
    mock_shell_runner.run.assert_called_once_with(
        str(script_path),
        env=(os.environ),
        capture_output=False,
        check=False,
    )


def test_dev_format_command_failure(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock format script
    script_path = config.dev.formatting.script
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock failed format run
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=1),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "format"])

    assert result.exit_code == 1
    assert "Formatting failed" in result.output
    mock_shell_runner.run.assert_called_once_with(
        str(script_path),
        env=(os.environ),
        capture_output=False,
        check=False,
    )


def test_dev_format_command_script_not_found(cli_runner, mock_context, tmp_path):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Do not create the script file
    result = cli_runner.invoke(cli, ["dev", "format"])

    assert result.exit_code == 1
    assert "Error: Format script not found" in result.output


def test_dev_type_check_command_success(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock type check script
    script_path = config.dev.type_checking.script
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock successful type check run
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=0),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "type-check"])

    assert result.exit_code == 0
    assert "Type checking completed successfully" in result.output
    mock_shell_runner.run.assert_called_once_with(
        [str(script_path)],
        env=(os.environ),
        capture_output=False,
        check=False,
    )


def test_dev_type_check_command_with_path(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock type check script
    script_path = config.dev.type_checking.script
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Create a test path
    test_path = tmp_path / "test_file.py"
    test_path.touch()

    # Mock successful type check run with path
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=0),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "type-check", "--path", str(test_path)])

    assert result.exit_code == 0
    assert "Type checking completed successfully" in result.output
    mock_shell_runner.run.assert_called_once_with(
        [str(script_path), str(test_path)],
        env=(os.environ),
        capture_output=False,
        check=False,
    )


def test_dev_type_check_command_failure(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock type check script
    script_path = config.dev.type_checking.script
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock failed type check run
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=1),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "type-check"])

    assert result.exit_code == 1
    assert "Type checking failed" in result.output
    mock_shell_runner.run.assert_called_once_with(
        [str(script_path)],
        env=(os.environ),
        capture_output=False,
        check=False,
    )


def test_dev_type_check_command_script_not_found(cli_runner, mock_context, tmp_path):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Do not create the script file
    result = cli_runner.invoke(cli, ["dev", "type-check"])

    assert result.exit_code == 1
    assert "Error: Type check script not found at" in result.output


def test_dev_bump_command_patch(cli_runner, mock_context):
    with patch("nv_dfm_core.cli.commands._dev.bump_version") as mock_bump:
        result = cli_runner.invoke(cli, ["dev", "bump", "patch"])

        assert result.exit_code == 0
        mock_bump.assert_called_once_with(
            type=BumpType.PATCH,
            new_version="",
            force=False,
            dry_run=False,
        )


def test_dev_bump_command_minor(cli_runner, mock_context):
    with patch("nv_dfm_core.cli.commands._dev.bump_version") as mock_bump:
        result = cli_runner.invoke(cli, ["dev", "bump", "minor"])

        assert result.exit_code == 0
        mock_bump.assert_called_once_with(
            type=BumpType.MINOR,
            new_version="",
            force=False,
            dry_run=False,
        )


def test_dev_bump_command_major(cli_runner, mock_context):
    with patch("nv_dfm_core.cli.commands._dev.bump_version") as mock_bump:
        result = cli_runner.invoke(cli, ["dev", "bump", "major"])

        assert result.exit_code == 0
        mock_bump.assert_called_once_with(
            type=BumpType.MAJOR,
            new_version="",
            force=False,
            dry_run=False,
        )


def test_dev_bump_command_literal_version(cli_runner, mock_context):
    with patch("nv_dfm_core.cli.commands._dev.bump_version") as mock_bump:
        result = cli_runner.invoke(cli, ["dev", "bump", "1.2.3"])

        assert result.exit_code == 0
        mock_bump.assert_called_once_with(
            type=BumpType.LITERAL,
            new_version="1.2.3",
            force=False,
            dry_run=False,
        )


def test_dev_bump_command_with_force(cli_runner, mock_context):
    with patch("nv_dfm_core.cli.commands._dev.bump_version") as mock_bump:
        result = cli_runner.invoke(cli, ["dev", "bump", "patch", "--force"])

        assert result.exit_code == 0
        mock_bump.assert_called_once_with(
            type=BumpType.PATCH,
            new_version="",
            force=True,
            dry_run=False,
        )


def test_dev_bump_command_with_dry_run(cli_runner, mock_context):
    with patch("nv_dfm_core.cli.commands._dev.bump_version") as mock_bump:
        result = cli_runner.invoke(cli, ["dev", "bump", "patch", "--dry-run"])

        assert result.exit_code == 0
        mock_bump.assert_called_once_with(
            type=BumpType.PATCH,
            new_version="",
            force=False,
            dry_run=True,
        )


def test_dev_bump_command_with_all_options(cli_runner, mock_context):
    with patch("nv_dfm_core.cli.commands._dev.bump_version") as mock_bump:
        result = cli_runner.invoke(
            cli, ["dev", "bump", "1.2.3", "--force", "--dry-run"]
        )

        assert result.exit_code == 0
        mock_bump.assert_called_once_with(
            type=BumpType.LITERAL,
            new_version="1.2.3",
            force=True,
            dry_run=True,
        )


def test_dev_package_command_success(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock build script
    script_path = tmp_path / "ci" / "scripts" / "python-build-wheels.sh"
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock successful build run
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=0),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "package"])

    assert result.exit_code == 0
    assert "Build completed successfully" in result.output
    mock_shell_runner.run.assert_called_once()


def test_dev_package_command_with_output_dir(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock build script
    script_path = tmp_path / "ci" / "scripts" / "python-build-wheels.sh"
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Create custom output directory
    output_dir = tmp_path / "custom_output"
    output_dir.mkdir()

    # Mock successful build run
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=0),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "package", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert "Build completed successfully" in result.output
    mock_shell_runner.run.assert_called_once()


def test_dev_package_command_failure(
    cli_runner, mock_context, mock_shell_runner, tmp_path
):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Create a mock build script
    script_path = tmp_path / "ci" / "scripts" / "python-build-wheels.sh"
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    # Mock failed build run
    mock_shell_runner.run.return_value = MagicMock(
        result=MagicMock(returncode=1),
        pid=None,
    )

    result = cli_runner.invoke(cli, ["dev", "package"])

    assert result.exit_code == 1
    assert "Build failed" in result.output
    mock_shell_runner.run.assert_called_once()


def test_dev_package_command_script_not_found(cli_runner, mock_context, tmp_path):
    config = get_good_cli_config(tmp_path)
    mock_context.get_config.return_value = config

    # Do not create the script file
    result = cli_runner.invoke(cli, ["dev", "package"])

    assert result.exit_code == 1
    assert "Error: Build script not found" in result.output
