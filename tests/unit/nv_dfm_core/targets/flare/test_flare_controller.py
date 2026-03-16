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
"""Tests for the Controller class."""

import os
from unittest.mock import MagicMock, patch

import pytest
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from nv_dfm_core.gen.modgen.ir import BoundNetIR, NetIR
from nv_dfm_core.targets.flare._controller import ClientTaskResultLogger, Controller
from nv_dfm_core.targets.flare._defs import Constant


@pytest.fixture
def controller() -> Controller:
    """Create a Controller instance with test settings."""
    net_ir = NetIR(
        pipeline_name="test",
        site="test-app",
        transitions=[],
    )
    bound_net_ir = BoundNetIR.bind_netir(net_ir, input_params=[])
    return Controller(
        submitted_api_version="0.1.0",
        federation_name="test-federation",
        homesite="test-app",
        bound_net_ir=bound_net_ir.model_dump(),
    )


@pytest.fixture
def mock_fl_ctx() -> FLContext:
    """Create a mock FLContext for testing."""
    ctx = MagicMock(spec=FLContext)
    ctx.get_prop.side_effect = lambda key, default=None: {
        FLContextKey.WORKFLOW: "test-workflow-123",
        FLContextKey.APP_ROOT: "/tmp/test_workspace",
        FLContextKey.TASK_NAME: "test_task",
        FLContextKey.TASK_ID: "test_task_id",
    }.get(key, default)
    ctx.get_job_id.return_value = "test-job-123"
    ctx.get_identity_name.return_value = "test-client"
    ctx.get_peer_context.return_value = None
    return ctx


@pytest.fixture
def mock_abort_signal() -> Signal:
    """Create a mock abort signal for testing."""
    signal = MagicMock(spec=Signal)
    signal.triggered = False
    return signal


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


class TestController:
    """Tests for the Controller class."""

    def test_init(self, controller: Controller):
        """Test Controller initialization."""
        assert controller is not None
        assert controller._submitted_api_version == "0.1.0"
        assert controller._federation_name == "test-federation"
        assert controller._homesite == "test-app"
        assert isinstance(controller._bound_net_ir, dict)
        assert controller._bound_net_ir_model is None
        assert controller._job_controller is None
        assert controller._app_io_manager is not None
        assert controller._debug_results is False

    def test_init_with_debug_results(self, controller: Controller):
        """Test Controller initialization with debug results enabled."""
        with patch.dict(os.environ, {"DFM_FLARE_DEBUG_RESULTS": "true"}):
            controller = Controller(
                submitted_api_version="0.1.0",
                federation_name="test-federation",
                homesite="test-app",
                bound_net_ir=controller._bound_net_ir,
            )
            assert controller._debug_results is True

    def test_start_controller(self, controller: Controller, mock_fl_ctx: FLContext):
        """Test start_controller method."""
        # Mock engine and clients
        mock_client = MagicMock()
        mock_client.name = "test-client-1"
        controller._engine = MagicMock()
        controller._engine.get_clients.return_value = [mock_client]
        controller._engine.register_app_command = MagicMock()

        controller.start_controller(mock_fl_ctx)

        assert controller.workflow_id == "test-workflow-123"
        assert controller._client_names == ["test-client-1"]
        assert controller._engine.register_app_command.call_count == 2

    def test_start_controller_missing_workflow_id(
        self, controller: Controller, mock_fl_ctx: FLContext
    ):
        """Test start_controller with missing workflow ID."""
        # Mock engine to avoid AttributeError
        controller._engine = MagicMock()
        controller._engine.get_clients.return_value = []
        controller._engine.register_app_command = MagicMock()

        # Override the side_effect to return None for workflow
        mock_fl_ctx.get_prop.side_effect = lambda key, default=None: {
            FLContextKey.WORKFLOW: None,
            FLContextKey.APP_ROOT: "/tmp/test_workspace",
            FLContextKey.TASK_NAME: "test_task",
            FLContextKey.TASK_ID: "test_task_id",
        }.get(key, default)

        with pytest.raises(
            RuntimeError, match="workflow ID is missing from FL context"
        ):
            controller.start_controller(mock_fl_ctx)

    def test_stop_controller(self, controller: Controller, mock_fl_ctx: FLContext):
        """Test stop_controller method."""
        controller._app_io_manager.save_all_remaining = MagicMock()

        with patch("nv_dfm_core.targets.flare._controller.super") as mock_super:
            controller.stop_controller(mock_fl_ctx)

            controller._app_io_manager.save_all_remaining.assert_called_once_with(
                mock_fl_ctx
            )
            mock_super.assert_called_once()

    def test_stop_controller_missing_app_root(
        self, controller: Controller, mock_fl_ctx: FLContext
    ):
        """Test stop_controller with missing app root."""
        # Override the side_effect to return None for app root
        mock_fl_ctx.get_prop.side_effect = lambda key, default=None: {
            FLContextKey.WORKFLOW: "test-workflow-123",
            FLContextKey.APP_ROOT: None,
            FLContextKey.TASK_NAME: "test_task",
            FLContextKey.TASK_ID: "test_task_id",
        }.get(key, default)

        with pytest.raises(RuntimeError, match="app root is missing from FL context"):
            controller.stop_controller(mock_fl_ctx)

    def test_handle_event(self, controller: Controller, mock_fl_ctx: FLContext):
        """Test handle_event method."""
        # Should not raise any exception
        controller.handle_event("test_event", mock_fl_ctx)

    @patch("nv_dfm_core.targets.flare._controller.ReliableMessage")
    def test_do_prepare(
        self,
        mock_reliable_message: MagicMock,
        controller: Controller,
        mock_fl_ctx: FLContext,
        mock_abort_signal: Signal,
    ):
        """Test _do_prepare method."""
        controller._abort_signal = mock_abort_signal
        controller._client_names = ["client1", "client2"]
        controller.logger = MagicMock()

        with (
            patch(
                "nv_dfm_core.targets.flare._controller.FlareRouter"
            ) as mock_router_class,
            patch(
                "nv_dfm_core.targets.flare._controller.JobController"
            ) as mock_job_controller_class,
        ):
            mock_router = MagicMock()
            mock_router_class.return_value = mock_router

            mock_job_controller = MagicMock()
            mock_job_controller_class.return_value = mock_job_controller

            controller._do_prepare(mock_fl_ctx)

            # Verify BoundNetIR was validated
            assert controller._bound_net_ir_model is not None

            # Verify FlareRouter was created
            mock_router_class.assert_called_once()

            # Verify JobController was created
            mock_job_controller_class.assert_called_once()

            # Verify ReliableMessage handler was registered
            mock_reliable_message.register_request_handler.assert_called_once()

    def test_do_prepare_missing_job_id(
        self, controller: Controller, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test _do_prepare with missing job ID."""
        controller._abort_signal = mock_abort_signal
        mock_fl_ctx.get_job_id.return_value = None

        with pytest.raises(RuntimeError, match="job ID is missing from FL context"):
            controller._do_prepare(mock_fl_ctx)

    def test_do_start_netrunner(
        self, controller: Controller, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test _do_start_netrunner method."""
        controller._bound_net_ir_model = MagicMock()
        controller._job_controller = MagicMock()
        controller._abort_signal = mock_abort_signal
        controller.log_info = MagicMock()

        controller._do_start_netrunner(mock_fl_ctx)

        controller._job_controller.start.assert_called_once()
        controller._job_controller.submit_initial_tokens.assert_called_once()

    def test_make_clients_do_start_execute_task(
        self, controller: Controller, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test _make_clients_do_start_execute_task method."""
        controller._abort_signal = mock_abort_signal
        controller._client_names = ["client1", "client2"]
        controller._job_controller = MagicMock()
        controller.workflow_id = "test-workflow-123"
        controller.log_error = MagicMock()
        controller.broadcast_and_wait = MagicMock()

        _result = controller._make_clients_do_start_execute_task(mock_fl_ctx)

        # Verify send_and_wait was called
        controller.broadcast_and_wait.assert_called_once()
        call_args = controller.broadcast_and_wait.call_args
        assert call_args[1]["targets"] == ["client1", "client2"]
        assert call_args[1]["fl_ctx"] == mock_fl_ctx
        assert call_args[1]["abort_signal"] == mock_abort_signal
        controller._job_controller.wait_for_done.assert_called_once_with(
            abort_signal=mock_abort_signal
        )

    def test_do_handle_finalization(
        self, controller: Controller, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test _do_handle_finalization method."""
        controller._abort_signal = mock_abort_signal
        controller._job_controller = MagicMock()
        controller._app_io_manager.save_all_remaining = MagicMock()

        controller._do_handle_finalization(mock_fl_ctx)

        controller._job_controller.wait_for_done.assert_called_once_with(
            mock_abort_signal
        )
        controller._app_io_manager.save_all_remaining.assert_called_once_with(
            mock_fl_ctx
        )

    def test_do_handle_finalization_no_job_controller(
        self, controller: Controller, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test _do_handle_finalization method when job_controller is None."""
        controller._abort_signal = mock_abort_signal
        controller._job_controller = None
        controller._app_io_manager.save_all_remaining = MagicMock()

        controller._do_handle_finalization(mock_fl_ctx)

        controller._app_io_manager.save_all_remaining.assert_called_once_with(
            mock_fl_ctx
        )

    def test_on_send_to_place_topic(
        self, controller: Controller, mock_fl_ctx: FLContext
    ):
        """Test _on_send_to_place_topic method."""
        controller._router = MagicMock()
        controller.log_debug = MagicMock()

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

        result = controller._on_send_to_place_topic(
            Constant.TOPIC_SEND_TO_PLACE, request, mock_fl_ctx
        )

        controller._router.route_token_package_sync.assert_called_once()
        assert result.get_return_code() == ReturnCode.OK

    def test_on_send_to_place_topic_wrong_topic(
        self, controller: Controller, mock_fl_ctx: FLContext
    ):
        """Test _on_send_to_place_topic with wrong topic."""
        with pytest.raises(AssertionError):
            controller._on_send_to_place_topic("wrong_topic", {}, mock_fl_ctx)

    def test_app_command_retrieve_tokens_handler(
        self, controller: Controller, mock_fl_ctx: FLContext
    ):
        """Test _app_command_retrieve_tokens_handler method."""
        controller._app_io_manager.get_all = MagicMock(return_value=[])
        controller.log_debug = MagicMock()
        controller._fl_ctx = mock_fl_ctx

        result = controller._app_command_retrieve_tokens_handler(
            Constant.CMD_RETRIEVE_TOKENS, {}, mock_fl_ctx
        )

        assert "tokens" in result
        assert result["tokens"] == []

    def test_app_command_retrieve_tokens_handler_wrong_command(
        self, controller: Controller, mock_fl_ctx: FLContext
    ):
        """Test _app_command_retrieve_tokens_handler with wrong command."""
        controller._fl_ctx = mock_fl_ctx
        with pytest.raises(RuntimeError, match="Invalid command"):
            controller._app_command_retrieve_tokens_handler(
                "wrong_command", {}, mock_fl_ctx
            )

    def test_app_command_send_to_place_handler(
        self, controller: Controller, mock_fl_ctx: FLContext
    ):
        """Test _app_command_send_to_place_handler method."""
        controller._router = MagicMock()
        controller.log_debug = MagicMock()
        controller._fl_ctx = mock_fl_ctx

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

        result = controller._app_command_send_to_place_handler(
            Constant.CMD_SEND_TO_PLACE, token_package_dict, mock_fl_ctx
        )

        controller._router.route_token_package_sync.assert_called_once()
        assert result == {"status": "OK"}

    def test_app_command_send_to_place_handler_wrong_command(
        self, controller: Controller, mock_fl_ctx: FLContext
    ):
        """Test _app_command_send_to_place_handler with wrong command."""
        controller._fl_ctx = mock_fl_ctx
        with pytest.raises(RuntimeError, match="Invalid command"):
            controller._app_command_send_to_place_handler(
                "wrong_command", {}, mock_fl_ctx
            )

    @patch("nv_dfm_core.targets.flare._controller.time")
    def test_control_flow_success(
        self,
        mock_time,
        controller: Controller,
        mock_fl_ctx: FLContext,
        mock_abort_signal: Signal,
    ):
        """Test control_flow method with successful execution."""
        mock_time.time.return_value = 100.0

        controller._abort_signal = mock_abort_signal
        controller._do_prepare = MagicMock()
        controller._make_clients_do_prepare_execute_task = MagicMock(
            return_value=ReturnCode.OK
        )
        controller._do_start_netrunner = MagicMock()
        controller._make_clients_do_start_execute_task = MagicMock(
            return_value=ReturnCode.OK
        )
        controller._do_handle_finalization = MagicMock()
        controller.log_info = MagicMock()

        result = controller.control_flow(mock_abort_signal, mock_fl_ctx)

        assert result is None
        controller._do_prepare.assert_called_once_with(mock_fl_ctx)

        controller._do_start_netrunner.assert_called_once_with(mock_fl_ctx)
        controller._make_clients_do_start_execute_task.assert_called_once_with(
            mock_fl_ctx
        )
        controller._do_handle_finalization.assert_called_once_with(mock_fl_ctx)

    def test_control_flow_abort_signal_triggered(
        self, controller: Controller, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test control_flow method when abort signal is triggered."""
        mock_abort_signal.triggered = True
        controller._abort_signal = mock_abort_signal
        controller.log_info = MagicMock()

        result = controller.control_flow(mock_abort_signal, mock_fl_ctx)

        assert result is not None
        assert result.get_return_code() == ReturnCode.TASK_ABORTED

    def test_control_flow_client_execution_failed(
        self, controller: Controller, mock_fl_ctx: FLContext, mock_abort_signal: Signal
    ):
        """Test control_flow method when client execution fails."""
        controller._abort_signal = mock_abort_signal
        controller._do_prepare = MagicMock()
        controller._make_clients_do_prepare_execute_task = MagicMock(
            return_value=ReturnCode.OK
        )
        controller._do_start_netrunner = MagicMock()
        controller._make_clients_do_start_execute_task = MagicMock(
            return_value=ReturnCode.EXECUTION_EXCEPTION
        )
        controller._job_controller = MagicMock()
        controller.log_info = MagicMock()

        result = controller.control_flow(mock_abort_signal, mock_fl_ctx)

        assert result is not None
        assert result.get_return_code() == ReturnCode.TASK_ABORTED
        controller._job_controller.shutdown.assert_called_once()


class TestClientTaskResultLogger:
    """Tests for the ClientTaskResultLogger class."""

    def test_init(self):
        """Test ClientTaskResultLogger initialization."""
        logger = MagicMock()
        fl_ctx = MagicMock()

        result_logger = ClientTaskResultLogger(
            logger=logger, task_name="test_task", expected_num_clients=2, fl_ctx=fl_ctx
        )

        assert result_logger._logger == logger
        assert result_logger._task_name == "test_task"
        assert result_logger._expected_num_clients == 2
        assert result_logger._fl_ctx == fl_ctx
        assert result_logger._client_results == []

    def test_result_received_callback_success(self):
        """Test result_received_callback with successful result."""
        logger = MagicMock()
        fl_ctx = MagicMock()

        result_logger = ClientTaskResultLogger(
            logger=logger, task_name="test_task", expected_num_clients=1, fl_ctx=fl_ctx
        )

        client_task = MagicMock()
        client_task.result = MagicMock(spec=Shareable)
        client_task.result.get_return_code.return_value = ReturnCode.OK
        client_task.client.name = "test_client"

        result_logger.result_received_callback(client_task, fl_ctx)

        assert len(result_logger._client_results) == 1
        rc, client_name, message = result_logger._client_results[0]
        assert rc == ReturnCode.OK
        assert client_name == "test_client"
        assert message == "OK"

    def test_result_received_callback_error(self):
        """Test result_received_callback with error result."""
        logger = MagicMock()
        fl_ctx = MagicMock()

        result_logger = ClientTaskResultLogger(
            logger=logger, task_name="test_task", expected_num_clients=1, fl_ctx=fl_ctx
        )

        client_task = MagicMock()
        client_task.result = MagicMock(spec=Shareable)
        client_task.result.get_return_code.return_value = ReturnCode.EXECUTION_EXCEPTION
        client_task.result.get.return_value = "Test error"
        client_task.client.name = "test_client"

        result_logger.result_received_callback(client_task, fl_ctx)

        assert len(result_logger._client_results) == 1
        rc, client_name, message = result_logger._client_results[0]
        assert rc == ReturnCode.EXECUTION_EXCEPTION
        assert client_name == "test_client"
        assert message == "Test error"

    def test_log_results(self):
        """Test log_results method."""
        logger = MagicMock()
        fl_ctx = MagicMock()

        result_logger = ClientTaskResultLogger(
            logger=logger, task_name="test_task", expected_num_clients=2, fl_ctx=fl_ctx
        )

        result_logger._client_results = [
            (ReturnCode.OK, "client1", "OK"),
            (ReturnCode.EXECUTION_EXCEPTION, "client2", "Error"),
        ]

        result_logger.log_results()

        assert logger.call_count == 2

    def test_get_overall_return_code_all_success(self):
        """Test get_overall_return_code when all clients succeed."""
        logger = MagicMock()
        fl_ctx = MagicMock()

        result_logger = ClientTaskResultLogger(
            logger=logger, task_name="test_task", expected_num_clients=2, fl_ctx=fl_ctx
        )

        result_logger._client_results = [
            (ReturnCode.OK, "client1", "OK"),
            (ReturnCode.OK, "client2", "OK"),
        ]

        result = result_logger.get_overall_client_return_code()
        assert result == ReturnCode.OK

    def test_get_overall_return_code_with_error(self):
        """Test get_overall_return_code when some clients fail."""
        logger = MagicMock()
        fl_ctx = MagicMock()

        result_logger = ClientTaskResultLogger(
            logger=logger, task_name="test_task", expected_num_clients=2, fl_ctx=fl_ctx
        )

        result_logger._client_results = [
            (ReturnCode.OK, "client1", "OK"),
            (ReturnCode.EXECUTION_EXCEPTION, "client2", "Error"),
        ]

        result = result_logger.get_overall_client_return_code()
        assert result == ReturnCode.EXECUTION_EXCEPTION

    def test_get_overall_return_code_missing_clients(self):
        """Test get_overall_return_code when not all clients responded."""
        logger = MagicMock()
        fl_ctx = MagicMock()

        result_logger = ClientTaskResultLogger(
            logger=logger, task_name="test_task", expected_num_clients=2, fl_ctx=fl_ctx
        )

        result_logger._client_results = [(ReturnCode.OK, "client1", "OK")]

        result = result_logger.get_overall_client_return_code()
        assert result == ReturnCode.EXECUTION_EXCEPTION
