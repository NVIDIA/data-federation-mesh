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

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nvflare.lighter.entity import Participant, Project

from nv_dfm_core.targets.flare.builder._wheel_builder import WheelBuilder


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def mock_project():
    project = MagicMock(spec=Project)
    participant = MagicMock(spec=Participant)
    participant.name = "test_participant"
    participant.type = "client"
    project.participants = [participant]
    return project


@pytest.fixture
def wheel_builder(temp_dir):
    py_project = temp_dir / "pyproject.toml"
    py_project.touch()
    test_module = temp_dir / "test_module"
    test_module.mkdir()
    (test_module / "test_file.py").touch()
    with patch.dict(os.environ, {"DFM_WHEEL_BUILDER_PROJECT_ROOT": str(temp_dir)}):
        return WheelBuilder(py_project=str(py_project), include=["test_module"])


def test_wheel_builder_initialization(temp_dir):
    py_project = temp_dir / "pyproject.toml"
    py_project.touch()
    include = ["test_module"]

    with patch.dict(os.environ, {"DFM_WHEEL_BUILDER_PROJECT_ROOT": str(temp_dir)}):
        builder = WheelBuilder(py_project=str(py_project), include=include)

        assert builder._py_project == str(py_project)
        assert builder._include == include
        assert isinstance(builder._my_dir, Path)
        assert isinstance(builder._dfm_dir, Path)
        assert isinstance(builder._project_root, Path)
        assert builder._project_root == temp_dir


def test_initialize_with_invalid_pyproject():
    builder = WheelBuilder(py_project=None)
    with pytest.raises(RuntimeError, match="py_project is required"):
        builder.initialize({})


def test_initialize_with_nonexistent_pyproject(temp_dir):
    non_existent = temp_dir / "nonexistent.toml"
    builder = WheelBuilder(py_project=str(non_existent))
    with pytest.raises(
        RuntimeError, match=f"Project file {non_existent} does not exist"
    ):
        builder.initialize({})


def test_copy_includes(temp_dir):
    with patch.dict(os.environ, {"DFM_WHEEL_BUILDER_PROJECT_ROOT": str(temp_dir)}):
        builder = WheelBuilder()
        src_dir = temp_dir / "src"
        dst_dir = temp_dir / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        # Create test files
        (src_dir / "test_file.py").touch()
        (src_dir / "test_dir").mkdir()
        (src_dir / "test_dir" / "nested_file.py").touch()

        includes = ["test_file.py", "test_dir"]
        builder._copy_includes(src_dir, dst_dir, includes)

        assert (dst_dir / "test_file.py").exists()
        assert (dst_dir / "test_dir").exists()
        assert (dst_dir / "test_dir" / "nested_file.py").exists()


def test_copy_includes_nonexistent_file(temp_dir):
    with patch.dict(os.environ, {"DFM_WHEEL_BUILDER_PROJECT_ROOT": str(temp_dir)}):
        builder = WheelBuilder()
        src_dir = temp_dir / "src"
        dst_dir = temp_dir / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        with pytest.raises(RuntimeError, match="Include file .* does not exist"):
            builder._copy_includes(src_dir, dst_dir, ["nonexistent_file"])


@pytest.mark.skip(reason="[TODO] Fix this test")
@patch("nv_dfm_core.targets.flare.builder._wheel_builder.WheelBuilder.get_ws_dir")
def test_prepare_participant(mock_get_ws_dir, temp_dir, wheel_builder):
    mock_get_ws_dir.return_value = str(temp_dir)

    participant = MagicMock(spec=Participant)
    participant.name = "test_participant"
    participant.type = "client"

    result = wheel_builder._prepare_participant(participant, {})
    assert result is None

    participant.type = "other"
    result = wheel_builder._prepare_participant(participant, {})
    assert result is not None
    build_dir, dst_dir = result
    assert build_dir.exists()
    assert dst_dir.exists()


@patch("subprocess.run")
@patch("os.chdir")
def test_build_participant(mock_chdir, mock_run, temp_dir, wheel_builder):
    build_dir = temp_dir / "build"
    dst_dir = temp_dir / "dst"
    build_dir.mkdir()
    dst_dir.mkdir()

    # Create mock wheel and tar files
    dist_dir = build_dir / "dist"
    dist_dir.mkdir()
    (dist_dir / "test-0.1.0-py3-none-any.whl").touch()
    (dist_dir / "test-0.1.0.tar.gz").touch()

    wheel_builder._build_participant(build_dir, dst_dir, {})

    assert mock_run.called
    assert (dst_dir / "test-0.1.0-py3-none-any.whl").exists()
    assert (dst_dir / "test-0.1.0.tar.gz").exists()


def test_cleanup(temp_dir):
    builder = WheelBuilder()
    build_dir = temp_dir / "build"
    build_dir.mkdir()
    (build_dir / "test_file").touch()

    builder._cleanup(build_dir)
    assert not build_dir.exists()


@patch(
    "nv_dfm_core.targets.flare.builder._wheel_builder.WheelBuilder._prepare_participant"
)
@patch(
    "nv_dfm_core.targets.flare.builder._wheel_builder.WheelBuilder._build_participant"
)
@patch("nv_dfm_core.targets.flare.builder._wheel_builder.WheelBuilder._cleanup")
def test_build(mock_cleanup, mock_build, mock_prepare, wheel_builder, mock_project):
    mock_prepare.return_value = (Path("build"), Path("dst"))

    wheel_builder.build(mock_project, {})

    assert mock_prepare.called
    assert mock_build.called
    assert mock_cleanup.called
