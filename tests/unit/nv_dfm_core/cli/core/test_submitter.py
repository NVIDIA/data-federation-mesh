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

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from nv_dfm_core.cli.core._submitter import (
    ScriptSubmitter,
    PipelineSubmitter,
    SubmitterConfig,
    PreflightCheckResult,
    get_submitter,
    list_submitters,
)
from nv_dfm_core.cli.config._federation import FederationConfig


@pytest.fixture
def mock_fed_config():
    """Fixture that provides a mock FederationConfig."""
    config = MagicMock(spec=FederationConfig)
    config.federation_workspace_path = Path("/mock/federation/workspace")
    config.module = "test_module"
    config.federation_dir = Path("/mock/federation/dir")
    config.app_name = "test_app"
    config.admin_package = "test_admin"
    config.flare_project_name = "test_fed"
    return config


@pytest.fixture
def submitter_config(mock_fed_config):
    """Fixture that provides a SubmitterConfig."""
    return SubmitterConfig(fed_cfg=mock_fed_config)


def test_submitter_base_initialization(submitter_config):
    """Test that SubmitterBase initializes correctly."""
    submitter = ScriptSubmitter(submitter_config)
    assert submitter._config == submitter_config
    assert submitter._preflight_check_result is None


def test_script_submitter_type():
    """Test that ScriptSubmitter returns correct type."""
    assert ScriptSubmitter.type() == "script"


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
def test_script_submitter_preflight_check(
    mock_stat, mock_is_file, mock_exists, tmp_path
):
    """Test ScriptSubmitter preflight checks."""
    submitter = ScriptSubmitter(SubmitterConfig(fed_cfg=MagicMock()))

    # Test non-existent file
    non_existent_file = tmp_path / "nonexistent.sh"
    mock_exists.return_value = False
    mock_is_file.return_value = False
    result = submitter.preflight_check(non_existent_file)
    assert not result.success
    assert "does not exist" in result.message

    # Test non-file
    test_file = tmp_path / "test.sh"
    mock_exists.return_value = True
    mock_is_file.return_value = False
    result = submitter.preflight_check(test_file)
    assert not result.success
    assert "not a file" in result.message

    # Test non-executable file
    mock_is_file.return_value = True
    mock_stat.return_value.st_mode = 0o644  # Regular file without execute permissions
    result = submitter.preflight_check(test_file)
    assert not result.success
    assert "not executable" in result.message

    # Test valid executable file
    mock_stat.return_value.st_mode = 0o755  # Regular file with execute permissions
    result = submitter.preflight_check(test_file)
    assert result.success
    assert "is valid" in result.message


@patch("nv_dfm_core.cli.core._submitter.ShellRunner")
def test_script_submitter_submit(mock_runner, submitter_config, tmp_path):
    """Test ScriptSubmitter submit functionality."""
    # Create a test script
    test_script = tmp_path / "test.sh"
    test_script.write_text("echo 'test'")
    test_script.chmod(0o755)

    submitter = ScriptSubmitter(submitter_config)
    mock_runner_instance = MagicMock()
    mock_runner.return_value = mock_runner_instance

    # Test submission
    submitter.submit(test_script, "target")

    # Verify runner was called with correct arguments
    mock_runner_instance.run.assert_called_once()
    call_args = mock_runner_instance.run.call_args[1]
    assert call_args["check"] is False
    assert call_args["capture_output"] is False
    assert call_args["timeout"] == submitter_config.timeout
    assert "DFM_FEDERATION_WORKSPACE_PATH" in call_args["env"]


def test_pipeline_submitter_type():
    """Test that PipelineSubmitter returns correct type."""
    assert PipelineSubmitter.type() == "pipeline"


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("nv_dfm_core.cli.core._submitter.importlib.util.spec_from_file_location")
def test_pipeline_submitter_preflight_check(
    mock_import, mock_is_file, mock_exists, tmp_path
):
    """Test PipelineSubmitter preflight checks."""
    submitter = PipelineSubmitter(SubmitterConfig(fed_cfg=MagicMock()))

    # Test non-existent file
    non_existent_file = tmp_path / "nonexistent.py"
    mock_exists.return_value = False
    mock_is_file.return_value = False
    result = submitter.preflight_check(non_existent_file)
    assert not result.success
    assert "does not exist" in result.message

    # Test non-file
    test_file = tmp_path / "test.txt"
    mock_exists.return_value = True
    mock_is_file.return_value = False
    result = submitter.preflight_check(test_file)
    assert not result.success
    assert "not a file" in result.message

    # Test non-python file
    mock_is_file.return_value = True
    result = submitter.preflight_check(test_file)
    assert not result.success
    assert "not a Python file" in result.message

    # Test python file without get_pipeline
    test_file = tmp_path / "test.py"
    mock_module = MagicMock(spec={})  # Module without get_pipeline
    mock_import.return_value.loader.load_module.return_value = mock_module
    result = submitter.preflight_check(test_file)
    assert not result.success
    assert "does not implement get_pipeline()" in result.message


@patch("pathlib.Path.exists")
def test_pipeline_submitter_submit(mock_exists, submitter_config, tmp_path):
    """Test PipelineSubmitter submit functionality."""
    # Mock file existence for all paths
    mock_exists.return_value = True

    # Create a mock pipeline and session
    mock_pipeline = MagicMock()
    mock_session = MagicMock()
    mock_job = MagicMock()
    mock_job.job_id.return_value = "test_job_id"
    mock_session.execute.return_value = mock_job

    # Create the submitter and mock its methods
    submitter = PipelineSubmitter(submitter_config)
    submitter._get_session_function = MagicMock(
        return_value=lambda **kwargs: mock_session
    )
    submitter._get_pipeline = MagicMock(return_value=mock_pipeline)
    submitter._get_pipeline_parameters = MagicMock(return_value={"param": "value"})

    # Set up preflight check result
    submitter._preflight_check_result = PreflightCheckResult(
        success=True, message="Path is valid"
    )

    test_file = tmp_path / "test.py"

    # Test submission with flare target
    submitter.submit(test_file, "flare")

    # Verify session was created with correct arguments
    submitter._get_session_function.assert_called_once_with(submitter_config.fed_cfg)
    mock_session.connect.assert_called_once()

    # Verify pipeline was prepared and executed
    submitter._get_pipeline.assert_called_once_with(test_file)
    mock_session.prepare.assert_called_once_with(mock_pipeline)
    mock_session.execute.assert_called_once()
    assert mock_session.execute.call_args[1]["input_params"] == {"param": "value"}
    assert "yield" in mock_session.execute.call_args[1]["place_callbacks"]
    mock_job.wait_until_finished.assert_called_once_with(
        timeout=submitter_config.timeout
    )

    # Test submission with non-flare target
    mock_session.reset_mock()
    submitter.submit(test_file, "other")
    mock_session.connect.assert_not_called()  # connect should not be called for non-flare targets


def test_get_submitter(submitter_config):
    """Test get_submitter function."""
    # Test getting script submitter
    script_submitter = get_submitter("script", submitter_config)
    assert isinstance(script_submitter, ScriptSubmitter)

    # Test getting pipeline submitter
    pipeline_submitter = get_submitter("pipeline", submitter_config)
    assert isinstance(pipeline_submitter, PipelineSubmitter)

    # Test getting invalid submitter
    with pytest.raises(RuntimeError, match="Couldn't find submitter for job type"):
        get_submitter("invalid", submitter_config)


def test_list_submitters():
    """Test list_submitters function."""
    submitters = list_submitters()
    assert "script" in submitters
    assert "pipeline" in submitters


@patch("pathlib.Path.exists")
@patch("nv_dfm_core.cli.core._submitter.importlib.util.spec_from_file_location")
def test_pipeline_submitter_get_session_function(
    mock_import, mock_exists, submitter_config
):
    """Test PipelineSubmitter _get_session_function method."""
    # Mock file existence
    mock_exists.return_value = True

    # Create mock session module with get_session function
    mock_session_module = MagicMock()
    mock_get_session = MagicMock()
    mock_session_module.get_session = mock_get_session

    # Set up the mock to return our session module
    mock_import.return_value.loader.load_module.return_value = mock_session_module

    submitter = PipelineSubmitter(submitter_config)

    # Get the session function
    session_function = submitter._get_session_function(submitter_config.fed_cfg)

    # Verify the function was returned
    assert session_function is mock_get_session

    # Verify module was loaded with correct path
    mock_import.assert_called_once_with(
        "session",
        submitter_config.fed_cfg.federation_dir
        / "fed"
        / "runtime"
        / "test_app"
        / "__init__.py",
    )


@patch("nv_dfm_core.cli.core._submitter.importlib.util.spec_from_file_location")
def test_pipeline_submitter_get_pipeline_parameters(
    mock_import, submitter_config, tmp_path
):
    """Test PipelineSubmitter _get_pipeline_parameters method."""
    # Create a test pipeline file
    test_file = tmp_path / "test.py"
    test_file.write_text("""
def get_pipeline_parameters(args=None):
    params = {"base_param": "value"}
    if args:
        params.update(dict(zip(args[::2], args[1::2])))
    return params
""")

    # Mock the module loading
    mock_module = MagicMock()
    mock_module.get_pipeline_parameters = MagicMock()
    mock_module.get_pipeline_parameters.return_value = {"base_param": "value"}
    mock_import.return_value.loader.load_module.return_value = mock_module

    submitter = PipelineSubmitter(submitter_config)

    # Test without job args
    submitter._config.job_args = None
    params = submitter._get_pipeline_parameters(test_file)
    assert params == {"base_param": "value"}

    # Test with job args
    submitter._config.job_args = ["arg1", "value1", "arg2", "value2"]
    mock_module.get_pipeline_parameters.return_value = {
        "base_param": "value",
        "arg1": "value1",
        "arg2": "value2",
    }
    params = submitter._get_pipeline_parameters(test_file)
    assert params == {"base_param": "value", "arg1": "value1", "arg2": "value2"}

    # Verify module was loaded with correct path
    mock_import.assert_called_with("test", test_file)


def test_script_submitter_submit_preflight_failure(submitter_config, tmp_path):
    """Test ScriptSubmitter submit when preflight check fails."""
    submitter = ScriptSubmitter(submitter_config)
    test_file = tmp_path / "test.sh"
    test_file.touch()  # Create file without execute permissions

    # Mock preflight check to fail
    submitter._preflight_check_result = PreflightCheckResult(
        success=False, message="Preflight check failed"
    )

    with pytest.raises(RuntimeError, match="Preflight check failed"):
        submitter.submit(test_file, "target")


def test_script_submitter_submit_execution_failure(submitter_config, tmp_path):
    """Test ScriptSubmitter submit when script execution fails."""
    # Create a test script that will fail
    test_script = tmp_path / "test.sh"
    test_script.write_text("exit 1")  # Script that exits with error
    test_script.chmod(0o755)

    submitter = ScriptSubmitter(submitter_config)
    mock_runner = MagicMock()
    mock_runner.run.return_value.returncode = 1

    with patch("nv_dfm_core.cli.core._submitter.ShellRunner", return_value=mock_runner):
        submitter.submit(test_script, "target")
        mock_runner.run.assert_called_once()


def test_pipeline_submitter_session_module_not_found(submitter_config):
    """Test PipelineSubmitter when session module is not found."""
    submitter = PipelineSubmitter(submitter_config)
    submitter_config.fed_cfg.federation_dir = Path("/nonexistent/path")

    with pytest.raises(RuntimeError, match="Session module not found"):
        submitter._get_session_function(submitter_config.fed_cfg)


def test_pipeline_submitter_session_no_get_session(submitter_config, tmp_path):
    """Test PipelineSubmitter when session module doesn't implement get_session."""
    mock_module = MagicMock()
    if hasattr(mock_module, "get_session"):
        delattr(mock_module, "get_session")
    submitter = PipelineSubmitter(submitter_config)
    with (
        patch("importlib.util.spec_from_file_location") as mock_import,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_import.return_value.loader.load_module.return_value = mock_module
        with pytest.raises(RuntimeError, match="does not implement get_session"):
            submitter._get_session_function(submitter_config.fed_cfg)


def test_pipeline_submitter_no_get_pipeline(submitter_config, tmp_path):
    """Test PipelineSubmitter when pipeline module doesn't implement get_pipeline."""
    mock_module = MagicMock()
    if hasattr(mock_module, "get_pipeline"):
        delattr(mock_module, "get_pipeline")
    submitter = PipelineSubmitter(submitter_config)
    test_file = tmp_path / "test.py"
    test_file.touch()
    with patch("importlib.util.spec_from_file_location") as mock_import:
        mock_import.return_value.loader.load_module.return_value = mock_module
        with pytest.raises(RuntimeError, match="does not implement get_pipeline"):
            submitter._get_pipeline(test_file)


def test_pipeline_submitter_no_get_pipeline_parameters(submitter_config, tmp_path):
    """Test PipelineSubmitter when pipeline module doesn't implement get_pipeline_parameters."""
    mock_module = MagicMock()
    if hasattr(mock_module, "get_pipeline_parameters"):
        delattr(mock_module, "get_pipeline_parameters")
    submitter = PipelineSubmitter(submitter_config)
    test_file = tmp_path / "test.py"
    test_file.touch()
    with patch("importlib.util.spec_from_file_location") as mock_import:
        mock_import.return_value.loader.load_module.return_value = mock_module
        with pytest.raises(
            RuntimeError, match="does not implement get_pipeline_parameters"
        ):
            submitter._get_pipeline_parameters(test_file)


def test_pipeline_submitter_submit_preflight_failure(submitter_config, tmp_path):
    """Test PipelineSubmitter submit when preflight check fails."""
    submitter = PipelineSubmitter(submitter_config)
    test_file = tmp_path / "test.py"
    test_file.touch()

    # Mock preflight check to fail
    submitter._preflight_check_result = PreflightCheckResult(
        success=False, message="Preflight check failed"
    )

    with pytest.raises(RuntimeError, match="Preflight check failed"):
        submitter.submit(test_file, "target")


def test_pipeline_submitter_submit_callback_and_completion(submitter_config, tmp_path):
    """Test PipelineSubmitter submit covers results_callback and job completion logic."""
    # Create a test pipeline file
    test_file = tmp_path / "test.py"
    test_file.write_text("""
def get_pipeline():
    return None
def get_pipeline_parameters():
    return {}
""")

    # Mock session and job
    mock_session = MagicMock()
    mock_job = MagicMock()
    mock_job.job_id.return_value = "test_job_id"
    mock_session.execute.return_value = mock_job
    mock_session.prepare.return_value = "prepared_pipeline"

    submitter = PipelineSubmitter(submitter_config)
    submitter._get_session_function = MagicMock(
        return_value=lambda **kwargs: mock_session
    )
    submitter._get_pipeline = MagicMock(return_value=None)
    submitter._get_pipeline_parameters = MagicMock(return_value={})
    submitter._preflight_check_result = PreflightCheckResult(
        success=True, message="Success"
    )

    with patch("nv_dfm_core.cli.core._submitter.click.echo") as mock_echo:
        submitter.submit(test_file, "flare")
        # Simulate the callback invocation
        callback = mock_session.execute.call_args[1]["place_callbacks"]["yield"]
        callback("siteA", 1, "targetB", {"result": 42})
        # Check that click.echo was called for callback and job completion
        assert any(
            "New results found" in str(call) for call in mock_echo.call_args_list
        )
        assert any("Job completed." in str(call) for call in mock_echo.call_args_list)


def test_get_submitter_invalid_type(submitter_config):
    """Test get_submitter with invalid job type."""
    with pytest.raises(RuntimeError, match="Couldn't find submitter for job type"):
        get_submitter("invalid_type", submitter_config)
