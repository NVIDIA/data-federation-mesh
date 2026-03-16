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

# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false
"""Tests for the Executor class."""

from unittest.mock import MagicMock, patch

import pytest
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from nv_dfm_core.gen.modgen.ir import BoundNetIR, NetIR
from nv_dfm_core.targets.flare._defs import Constant
from nv_dfm_core.targets.flare._executor import Executor


@pytest.fixture
def executor() -> Executor:
    """Create an Executor instance."""
    net_ir = NetIR(
        pipeline_name="test",
        site="test-app",
        transitions=[],
    )
    bound_net_ir = BoundNetIR.bind_netir(net_ir, input_params=[])
    return Executor(
        submitted_api_version="0.1.0",
        federation_name="test-federation",
        homesite="test-app",
        bound_net_ir=bound_net_ir.model_dump(),
    )


@pytest.fixture
def mock_fl_ctx() -> FLContext:
    """Create a mock FLContext for testing."""
    ctx = MagicMock(spec=FLContext)
    ctx.get_identity_name.return_value = "testclient"
    ctx.get_job_id.return_value = "test-job-123"
    ctx.get_peer_context.return_value = None
    ctx.get_prop.return_value = None
    return ctx


@pytest.fixture
def mock_abort_signal() -> Signal:
    """Create a mock abort signal for testing."""
    signal = MagicMock(spec=Signal)
    signal.triggered = False
    return signal


@pytest.fixture
def mock_shareable() -> Shareable:
    """Create a mock Shareable for testing."""
    shareable = MagicMock(spec=Shareable)
    return shareable


@pytest.fixture
def mock_flare_router():
    """Create a mock FlareRouter."""
    router = MagicMock()
    router.route_token_package_sync = MagicMock()
    return router


@pytest.fixture
def mock_job_controller():
    """Create a mock JobController."""
    job_controller = MagicMock()
    job_controller.start = MagicMock()
    job_controller.submit_initial_tokens = MagicMock()
    job_controller.wait_for_done = MagicMock()
    job_controller.shutdown = MagicMock()
    return job_controller


class TestExecutor:
    """Tests for the Executor class."""

    def test_init(self, executor: Executor):
        """Test Executor initialization."""
        assert executor is not None
        assert executor._submitted_api_version == "0.1.0"
        assert executor._federation_name == "test-federation"
        assert executor._homesite == "test-app"
        assert isinstance(executor._bound_net_ir, dict)
        assert executor._bound_net_ir_model is None
        assert executor._job_controller is None
        assert executor._abort_signal is None

    def test_execute_abort_signal_triggered(
        self, executor: Executor, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test execute method when abort signal is triggered."""
        mock_abort_signal.triggered = True
        mock_shareable = MagicMock(spec=Shareable)

        result = executor.execute(
            "any_task", mock_shareable, mock_fl_ctx, mock_abort_signal
        )

        assert result.get_return_code() == ReturnCode.TASK_ABORTED
        assert executor._abort_signal == mock_abort_signal

    def test_execute_start_execution_task(
        self, executor: Executor, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test execute method with start execution task."""
        mock_shareable = MagicMock(spec=Shareable)

        with (
            patch("nv_dfm_core.targets.flare._executor.FlareRouter"),
            patch("nv_dfm_core.targets.flare._executor.JobController"),
            patch("nv_dfm_core.targets.flare._executor.ReliableMessage"),
        ):
            result = executor.execute(
                Constant.TASK_START_EXECUTION,
                mock_shareable,
                mock_fl_ctx,
                mock_abort_signal,
            )

            assert result.get_return_code() == ReturnCode.OK
            assert executor._abort_signal == mock_abort_signal

    def test_execute_start_execution_task_no_job_controller(
        self, executor: Executor, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test execute method with start execution task when JobController doesn't exist."""
        mock_shareable = MagicMock(spec=Shareable)
        executor._abort_signal = MagicMock()
        executor.log_error = MagicMock()

        with (
            patch("nv_dfm_core.targets.flare._executor.FlareRouter"),
            patch("nv_dfm_core.targets.flare._executor.JobController"),
            patch("nv_dfm_core.targets.flare._executor.ReliableMessage"),
        ):
            result = executor._on_start_execution_task(
                mock_fl_ctx,
            )

            # The method logs an error but still returns OK since it doesn't check for job_controller in execute
            assert result
            assert result.get_return_code() == ReturnCode.BAD_TASK_DATA
            executor.log_error.assert_called_once_with(
                mock_fl_ctx, "JobController not started"
            )

    def test_execute_unsupported_task(
        self, executor: Executor, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test execute method with unsupported task."""
        mock_shareable = MagicMock(spec=Shareable)
        executor.log_error = MagicMock()

        result = executor.execute(
            "unsupported_task", mock_shareable, mock_fl_ctx, mock_abort_signal
        )

        assert result.get_return_code() == ReturnCode.TASK_UNSUPPORTED
        executor.log_error.assert_called_once_with(
            mock_fl_ctx, "ignored unsupported unsupported_task"
        )

    @patch("nv_dfm_core.targets.flare._executor.ReliableMessage")
    def test_on_prepare_execution_task(
        self,
        mock_reliable_message,
        executor: Executor,
        mock_fl_ctx: FLContext,
        mock_abort_signal: Signal,
    ):
        """Test _on_prepare_execution_task method."""
        executor._abort_signal = mock_abort_signal
        executor.log_info = MagicMock()
        executor.logger = MagicMock()

        with (
            patch(
                "nv_dfm_core.targets.flare._executor.FlareRouter"
            ) as mock_router_class,
            patch(
                "nv_dfm_core.targets.flare._executor.JobController"
            ) as mock_job_controller_class,
        ):
            mock_router = MagicMock()
            mock_router_class.return_value = mock_router

            mock_job_controller = MagicMock()
            mock_job_controller_class.return_value = mock_job_controller

            executor._on_prepare_execution_task(mock_fl_ctx)

            # Verify BoundNetIR was validated
            assert executor._bound_net_ir_model is not None

            # Verify FlareRouter was created
            mock_router_class.assert_called_once_with(
                fl_ctx=mock_fl_ctx,
                abort_signal=mock_abort_signal,
                app_io_manager=None,
                client_names=None,
                logger=executor.logger,
            )

            # Verify JobController was created (trace_id is now derived from job_id)
            mock_job_controller_class.assert_called_once_with(
                router=mock_router,
                pipeline_api_version=executor._submitted_api_version,
                federation_name=executor._federation_name,
                homesite=executor._homesite,
                this_site="testclient",
                job_id="test-job-123",
                netir=executor._bound_net_ir_model,
                logger=executor.logger,
                force_modgen=False,
            )

            # Verify ReliableMessage handler was registered
            mock_reliable_message.register_request_handler.assert_called_once()

    def test_on_prepare_execution_task_missing_site_name(
        self, executor: Executor, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test _on_prepare_execution_task with missing site name."""
        executor._abort_signal = mock_abort_signal
        mock_fl_ctx.get_identity_name.return_value = None

        with pytest.raises(ValueError, match="Site name not found"):
            executor._on_prepare_execution_task(mock_fl_ctx)

    def test_on_prepare_execution_task_missing_job_id(
        self, executor: Executor, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test _on_prepare_execution_task with missing job ID."""
        executor._abort_signal = mock_abort_signal
        mock_fl_ctx.get_job_id.return_value = None

        with pytest.raises(RuntimeError, match="job ID is missing from FL context"):
            executor._on_prepare_execution_task(mock_fl_ctx)

    def test_on_start_execution_task(
        self, executor: Executor, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test _on_start_execution_task method."""
        executor._abort_signal = mock_abort_signal
        executor._job_controller = MagicMock()
        executor._job_controller.error_occurred.return_value = (
            False  # Mock error_occurred to return False
        )
        executor._bound_net_ir_model = MagicMock()
        executor.log_info = MagicMock()

        executor._on_start_execution_task(mock_fl_ctx)

        executor._job_controller.start.assert_called_once()
        executor._job_controller.submit_initial_tokens.assert_called_once()
        executor._job_controller.wait_for_done.assert_called_once_with(
            abort_signal=mock_abort_signal
        )

    def test_on_start_execution_task_no_job_controller(
        self, executor: Executor, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test _on_start_execution_task when JobController doesn't exist."""
        executor._abort_signal = mock_abort_signal
        executor._job_controller = None
        executor.log_error = MagicMock()

        executor._on_start_execution_task(mock_fl_ctx)

        executor.log_error.assert_called_once_with(
            mock_fl_ctx, "JobController not started"
        )

    def test_on_send_to_place_topic(self, executor: Executor, mock_fl_ctx: FLContext):
        """Test _on_send_to_place_topic method."""
        executor._router = MagicMock()
        executor.log_debug = MagicMock()

        # Create a test token package
        from nv_dfm_core.exec import Frame

        frame = Frame.start_frame(1)
        token_package_dict = {
            "source_site": "client1",
            "target_site": "client2",
            "source_job": "test-job",
            "target_job": "test-job",
            "target_place": "place2",
            "frame": frame.model_dump(),
            "tagged_data": ("json", "test_data"),
            "source_node": None,
            "is_yield": False,
        }

        request = {Constant.MSG_KEY_TOKEN_PACKAGE_DICT: token_package_dict}

        result = executor._on_send_to_place_topic(
            Constant.TOPIC_SEND_TO_PLACE, request, mock_fl_ctx
        )

        executor._router.route_token_package_sync.assert_called_once()
        assert result.get_return_code() == ReturnCode.OK

    def test_on_send_to_place_topic_wrong_topic(
        self, executor: Executor, mock_fl_ctx: FLContext
    ):
        """Test _on_send_to_place_topic with wrong topic."""
        with pytest.raises(AssertionError):
            executor._on_send_to_place_topic("wrong_topic", {}, mock_fl_ctx)

    def test_on_send_to_place_topic_missing_token_package(
        self, executor: Executor, mock_fl_ctx: FLContext
    ):
        """Test _on_send_to_place_topic with missing token package in request."""
        request = {}  # Missing token package

        with pytest.raises(KeyError):
            executor._on_send_to_place_topic(
                Constant.TOPIC_SEND_TO_PLACE, request, mock_fl_ctx
            )

    def test_execute_flow_prepare_then_start(
        self, executor: Executor, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test the complete flow: prepare execution then start execution."""
        mock_shareable = MagicMock(spec=Shareable)
        executor.log_info = MagicMock()
        executor.logger = MagicMock()

        # First, prepare execution
        with (
            patch(
                "nv_dfm_core.targets.flare._executor.FlareRouter"
            ) as mock_router_class,
            patch(
                "nv_dfm_core.targets.flare._executor.JobController"
            ) as mock_job_controller_class,
            patch(
                "nv_dfm_core.targets.flare._executor.ReliableMessage"
            ) as mock_reliable_message,
        ):
            mock_router = MagicMock()
            mock_router_class.return_value = mock_router

            mock_job_controller = MagicMock()
            mock_job_controller_class.return_value = mock_job_controller

            # Start execution
            result2 = executor.execute(
                Constant.TASK_START_EXECUTION,
                mock_shareable,
                mock_fl_ctx,
                mock_abort_signal,
            )
            assert result2.get_return_code() == ReturnCode.OK
            assert executor._job_controller is not None

            # Verify JobController methods were called
            mock_job_controller.start.assert_called_once()
            mock_job_controller.submit_initial_tokens.assert_called_once()
            mock_job_controller.wait_for_done.assert_called_once_with(
                abort_signal=mock_abort_signal
            )

    def test_execute_with_different_task_names(
        self, executor: Executor, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test execute method with various task names."""
        mock_shareable = MagicMock(spec=Shareable)
        executor.log_error = MagicMock()

        # Test unsupported task names
        unsupported_tasks = ["task1", "task2", "unknown_task", ""]
        for task_name in unsupported_tasks:
            result = executor.execute(
                task_name, mock_shareable, mock_fl_ctx, mock_abort_signal
            )
            assert result.get_return_code() == ReturnCode.TASK_UNSUPPORTED

        # Verify error logging for each unsupported task
        assert executor.log_error.call_count == len(unsupported_tasks)

    def test_execute_abort_signal_handling(
        self, executor: Executor, mock_fl_ctx: FLContext
    ):
        """Test execute method abort signal handling."""
        mock_shareable = MagicMock(spec=Shareable)
        executor.log_info = MagicMock()

        # Test with triggered abort signal
        triggered_signal = MagicMock(spec=Signal)
        triggered_signal.triggered = True

        result = executor.execute(
            "any_task", mock_shareable, mock_fl_ctx, triggered_signal
        )
        assert result.get_return_code() == ReturnCode.TASK_ABORTED

        # Test with non-triggered abort signal
        non_triggered_signal = MagicMock(spec=Signal)
        non_triggered_signal.triggered = False

        # This should proceed to task handling and return TASK_UNSUPPORTED for unknown task
        result = executor.execute(
            "any_task", mock_shareable, mock_fl_ctx, non_triggered_signal
        )
        assert result.get_return_code() == ReturnCode.TASK_UNSUPPORTED
