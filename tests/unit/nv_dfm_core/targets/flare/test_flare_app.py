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

# pyright: reportPrivateUsage=false
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nv_dfm_core.api._prepared_pipeline import PreparedPipeline
from nv_dfm_core.targets.flare._flare_app import FlareApp
from nv_dfm_core.targets.flare._defs import Constant
from nv_dfm_core.targets.flare._flare_options import FlareOptions
from nvflare.fuel.f3.cellnet.fqcn import FQCN  # pyright: ignore[reportMissingImports]
from nv_dfm_core.exec import Frame


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock(spec=PreparedPipeline)
    pipeline.pipeline_name = "test"
    return pipeline


@pytest.fixture
def flare_app(mock_pipeline: PreparedPipeline):
    return FlareApp(pipeline=mock_pipeline, input_params=[], options=FlareOptions())


def test_flare_app_initialization(mock_pipeline: PreparedPipeline):
    app = FlareApp(pipeline=mock_pipeline, input_params=[], options=FlareOptions())

    assert app._pipeline == mock_pipeline
    assert app._job is None


def test_simulate_with_default_workspace(mock_pipeline: PreparedPipeline):
    app = FlareApp(pipeline=mock_pipeline, input_params=[], options=FlareOptions())

    with patch("nv_dfm_core.targets.flare._flare_app.FedJob") as mock_fed_job:
        mock_job = MagicMock()
        mock_fed_job.return_value = mock_job

        workspace = app.simulate()

        assert workspace == "/tmp/dfm_workspace"
        mock_job.simulator_run.assert_called_once_with(
            "/tmp/dfm_workspace", n_clients=1
        )


def test_simulate_with_custom_workspace(mock_pipeline: PreparedPipeline):
    app = FlareApp(pipeline=mock_pipeline, input_params=[], options=FlareOptions())

    custom_workspace = "/custom/workspace"

    with patch("nv_dfm_core.targets.flare._flare_app.FedJob") as mock_fed_job:
        mock_job = MagicMock()
        mock_fed_job.return_value = mock_job

        workspace = app.simulate(custom_workspace)

        assert workspace == custom_workspace
        mock_job.simulator_run.assert_called_once_with(custom_workspace, n_clients=1)


def test_export(mock_pipeline: PreparedPipeline):
    app = FlareApp(pipeline=mock_pipeline, input_params=[], options=FlareOptions())

    custom_workspace = Path("/custom/workspace")

    with patch("nv_dfm_core.targets.flare._flare_app.FedJob") as mock_fed_job:
        mock_job = MagicMock()
        mock_fed_job.return_value = mock_job

        workspace = app.export(custom_workspace)

        assert workspace.as_posix().startswith(str(custom_workspace))
        mock_job.export_job.assert_called_once_with(str(custom_workspace))


def _make_minimal_pipeline_mock() -> PreparedPipeline:
    pipeline = MagicMock(spec=PreparedPipeline)
    pipeline.pipeline_name = "test"
    pipeline.federation_module = "fedmod"
    pipeline.homesite = "server"
    pipeline.get_participating_sites.return_value = [FQCN.ROOT_SERVER]
    # bound_net_irs mapping with server entry only
    server_bound = MagicMock()
    server_bound.site = FQCN.ROOT_SERVER
    server_bound.model_dump.return_value = {}
    pipeline.bind_net_irs.return_value = {FQCN.ROOT_SERVER: server_bound}
    return pipeline


def test_timeout_precedence_env_overrides(monkeypatch):
    pipeline = _make_minimal_pipeline_mock()
    with (
        patch("nv_dfm_core.targets.flare._flare_app.FedJob") as mock_fed_job,
        patch("nv_dfm_core.targets.flare._flare_app.Controller") as mock_controller,
    ):
        mock_fed_job.return_value = MagicMock()
        monkeypatch.setenv("DFM_FLARE_TASK_TIMEOUT", "321")
        app = FlareApp(
            pipeline=pipeline,
            input_params=[(Frame.start_frame(0), {})],
            options=FlareOptions(),
        )
        app._prepare()
        kwargs = mock_controller.call_args.kwargs
        assert kwargs["options"].task_timeout_s == 321


def test_timeout_precedence_explicit_over_env(monkeypatch):
    pipeline = _make_minimal_pipeline_mock()
    with (
        patch("nv_dfm_core.targets.flare._flare_app.FedJob") as mock_fed_job,
        patch("nv_dfm_core.targets.flare._flare_app.Controller") as mock_controller,
    ):
        mock_fed_job.return_value = MagicMock()
        monkeypatch.setenv("DFM_FLARE_TASK_TIMEOUT", "456")
        app = FlareApp(
            pipeline=pipeline,
            input_params=[(Frame.start_frame(0), {})],
            options=FlareOptions(task_timeout_s=999),
        )
        app._prepare()
        kwargs = mock_controller.call_args.kwargs
        assert kwargs["options"].task_timeout_s == 999


def test_timeout_explicit_when_no_env(monkeypatch):
    pipeline = _make_minimal_pipeline_mock()
    with (
        patch("nv_dfm_core.targets.flare._flare_app.FedJob") as mock_fed_job,
        patch("nv_dfm_core.targets.flare._flare_app.Controller") as mock_controller,
    ):
        mock_fed_job.return_value = MagicMock()
        monkeypatch.delenv("DFM_FLARE_TASK_TIMEOUT", raising=False)
        app = FlareApp(
            pipeline=pipeline,
            input_params=[(Frame.start_frame(0), {})],
            options=FlareOptions(task_timeout_s=777),
        )
        app._prepare()
        kwargs = mock_controller.call_args.kwargs
        assert kwargs["options"].task_timeout_s == 777


def test_timeout_default_when_no_overrides(monkeypatch):
    pipeline = _make_minimal_pipeline_mock()
    with (
        patch("nv_dfm_core.targets.flare._flare_app.FedJob") as mock_fed_job,
        patch("nv_dfm_core.targets.flare._flare_app.Controller") as mock_controller,
    ):
        mock_fed_job.return_value = MagicMock()
        monkeypatch.delenv("DFM_FLARE_TASK_TIMEOUT", raising=False)
        app = FlareApp(
            pipeline=pipeline,
            input_params=[(Frame.start_frame(0), {})],
            options=FlareOptions(),
        )
        app._prepare()
        kwargs = mock_controller.call_args.kwargs
        assert kwargs["options"].task_timeout_s == int(Constant.TASK_TIMEOUT)
