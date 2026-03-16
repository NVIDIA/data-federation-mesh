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

# pyright: reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

import os
import time
from pathlib import Path
from typing import Any, Literal

from nvflare.apis.controller_spec import (  # pyright: ignore[reportMissingImports]
    ClientTask,
    Task,
)
from nvflare.apis.event_type import EventType  # pyright: ignore[reportMissingImports]
from nvflare.apis.fl_constant import (
    FLContextKey,  # pyright: ignore[reportMissingImports]
)
from nvflare.apis.fl_context import FLContext  # pyright: ignore[reportMissingImports]
from nvflare.apis.impl.controller import (
    Controller as FlareController,  # pyright: ignore[reportMissingImports]
)
from nvflare.apis.shareable import (  # pyright: ignore[reportMissingImports]
    ReturnCode,
    Shareable,
    make_reply,
)
from nvflare.apis.signal import Signal  # pyright: ignore[reportMissingImports]
from nvflare.apis.utils.reliable_message import (
    ReliableMessage,  # pyright: ignore[reportMissingImports]
)
from nvflare.fuel.f3.cellnet.fqcn import FQCN  # pyright: ignore[reportMissingImports]

from nv_dfm_core.exec import (
    Frame,
    JobController,
    TokenPackage,
)
from nv_dfm_core.gen.modgen.ir import BoundNetIR
from nv_dfm_core.telemetry import (
    TELEMETRY_PLACE_NAME,
    SiteTelemetryCollector,
    telemetry_enabled,
)

from ._app_io_manager import AppIOManager
from ._defs import Constant
from ._flare_options import FlareOptions
from ._flare_router import FlareRouter
from ._receive_tokens_cmd_response import ReceiveTokensCmdResponse


class ClientTaskResultLogger:
    def __init__(
        self, logger, task_name: str, expected_num_clients: int, fl_ctx: FLContext
    ):
        self._logger = logger
        self._task_name: str = task_name
        self._fl_ctx: FLContext = fl_ctx
        self._client_results: list[tuple[str, str, Any]] = []
        self._expected_num_clients: int = expected_num_clients

    def result_received_callback(self, client_task: ClientTask, fl_ctx: FLContext):
        assert isinstance(client_task.result, Shareable)
        rc = client_task.result.get_return_code()
        client_name = client_task.client.name
        if rc == ReturnCode.OK:
            message = "OK"
        else:
            message = client_task.result.get(Constant.ERROR, "?")
        self._client_results.append((rc, client_name, message))

    def log_results(self):
        for rc, client_name, error in self._client_results:
            self._logger(
                self._fl_ctx,
                f"Task {self._task_name}, client {client_name} returned return code {rc}: {error}",
            )

    def get_overall_client_return_code(self) -> str:
        if len(self._client_results) != self._expected_num_clients:
            return ReturnCode.EXECUTION_EXCEPTION
        for rc, _, _ in self._client_results:
            if rc != ReturnCode.OK:
                return rc
        return ReturnCode.OK


class Controller(FlareController):
    """DFM Controller that manages the execution of jobs on federated clients.

    The Controller is responsible for:
    1. Managing the lifecycle of jobs
    2. Coordinating execution across clients
    3. Collecting and managing results
    4. Handling client communication and task distribution
    """

    def __init__(
        self,
        submitted_api_version: str,
        federation_name: str,
        homesite: str,
        bound_net_ir: dict[str, Any] | Literal[False],
        options: FlareOptions = FlareOptions(),
        force_modgen: bool = False,
    ):
        super().__init__()
        self._submitted_api_version: str = submitted_api_version
        self._federation_name: str = federation_name
        self._homesite: str = homesite
        self._bound_net_ir: dict[str, Any] | Literal[False] = bound_net_ir
        self._bound_net_ir_model: BoundNetIR | None = None
        self._force_modgen: bool = force_modgen
        self._options: FlareOptions = options

        self._cw_started: bool = False
        self._asked_to_stop: bool = False
        self._app_io_manager: AppIOManager = AppIOManager()
        # Enable debug mode if environment variable is set
        dr: str = os.environ.get("DFM_FLARE_DEBUG_RESULTS", "false")
        self._debug_results: bool = dr == "true"

        self._job_controller: JobController | None = None

    def _app_command_retrieve_tokens_handler(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """App wants to retrieve tokens from the server.

        Result is a list of dumped TokenPackage dicts

        This method is registered as a handler for the CMD_RETRIEVE_TOKENS app command.
        """
        command, _payload, _ = args
        self.log_debug(self._fl_ctx, f"Received app command: {command}")
        if command != Constant.CMD_RETRIEVE_TOKENS:
            raise RuntimeError(
                f"Invalid command: {command} (expected {Constant.CMD_RETRIEVE_TOKENS})"
            )
        self.log_debug(self._fl_ctx, "Getting results from AppIOManager")
        tokens = self._app_io_manager.get_all()

        json_dicts = [token.model_dump() for token in tokens]
        response = ReceiveTokensCmdResponse(tokens=json_dicts)
        self.log_debug(
            self._fl_ctx, f"Sending {len(json_dicts)} tokens from AppIOManager to app"
        )
        return response.model_dump()

    def _app_command_send_to_place_handler(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Handle requests from the app to send a token to a place.

        Command data is a single TokenPackage dumped as a dict

        This method is registered as a handler for the CMD_SEND_TO_PLACE app command.
        """
        command, payload, _ = args
        self.log_debug(self._fl_ctx, f"Received app command: {command}")
        if command != Constant.CMD_SEND_TO_PLACE:
            raise RuntimeError(
                f"Invalid command: {command} (expected {Constant.CMD_SEND_TO_PLACE})"
            )
        token_package = TokenPackage.model_validate(payload)
        self._router.route_token_package_sync(token_package)

        return {"status": "OK"}

    def start_controller(self, fl_ctx: FLContext):
        self._fl_ctx = fl_ctx
        wf_id = fl_ctx.get_prop(FLContextKey.WORKFLOW)
        if not wf_id:
            raise RuntimeError("workflow ID is missing from FL context")

        full_path = os.path.abspath(__file__)
        self.log_info(
            fl_ctx, f"starting controller for workflow {wf_id} from {full_path}"
        )

        self.workflow_id = wf_id

        # Get list of available clients
        all_clients = self._engine.get_clients()
        self._client_names: list[str] = [t.name for t in all_clients]
        self.log_info(fl_ctx, f"Available clients: {self._client_names}")

        # Register handlers for communication
        self._engine.register_app_command(
            Constant.CMD_RETRIEVE_TOKENS, self._app_command_retrieve_tokens_handler
        )
        self._engine.register_app_command(
            Constant.CMD_SEND_TO_PLACE, self._app_command_send_to_place_handler
        )
        self.log_info(
            fl_ctx,
            f"Registered app command handlers for {Constant.CMD_RETRIEVE_TOKENS}"
            + f" and {Constant.CMD_SEND_TO_PLACE}",
        )

    def stop_controller(self, fl_ctx: FLContext):
        """Stop the controller and save all results.

        This method:
        1. Saves all results to the workspace directory
        2. Calls the parent class's stop_controller method
        """
        dirname = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        if not dirname:
            raise RuntimeError("app root is missing from FL context")
        self._app_io_manager.save_all_remaining(fl_ctx)
        super().stop_controller(fl_ctx)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle NVFlare system events for telemetry.

        Creates spans for key Flare-level events to provide visibility
        into the Flare infrastructure layer.
        """
        # Get or create collector so we can capture early events
        collector = self._get_or_create_collector(fl_ctx)
        if collector is None:
            return

        # Track paired events (start/end) as spans
        if event_type == EventType.START_RUN:
            # Store both the context manager and the yielded span
            self._flare_run_span_cm = collector.span(
                "flare.run",
                attributes={"job_id": fl_ctx.get_job_id() or "unknown"},
            )
            self._flare_run_span = self._flare_run_span_cm.__enter__()
        elif event_type == EventType.END_RUN:
            if hasattr(self, "_flare_run_span") and self._flare_run_span:
                self._flare_run_span.set_ok()
                if hasattr(self, "_flare_run_span_cm"):
                    self._flare_run_span_cm.__exit__(None, None, None)
                self._flare_run_span = None

    def _get_or_create_collector(
        self, fl_ctx: FLContext
    ) -> SiteTelemetryCollector | None:
        """Get or create a telemetry collector for this job.

        Unlike _get_telemetry_collector which only returns an existing collector,
        this method will create a temporary collector if needed (e.g., before
        JobController exists) to ensure early spans like flare.run are captured.
        """
        # First try to get existing collector
        collector = self._get_telemetry_collector()
        if collector:
            return collector

        if not telemetry_enabled():
            return None

        # Create a temporary collector with job_id from fl_ctx
        # This ensures trace_id matches what NetRunner will use later
        job_id = fl_ctx.get_job_id()
        if job_id:
            if not hasattr(self, "_temp_collector") or self._temp_collector is None:
                self._temp_collector = SiteTelemetryCollector(
                    site="server",
                    job_id=job_id,
                )
            return self._temp_collector

        return None

    def _get_telemetry_collector(self) -> SiteTelemetryCollector | None:
        """Get the telemetry collector if available."""
        if not telemetry_enabled():
            return None

        # Get collector from job controller's context if available
        if self._job_controller and hasattr(self._job_controller, "_netrunner"):
            return self._job_controller._netrunner.dfm_context.telemetry_collector

        # Fall back to temp collector if it exists
        if hasattr(self, "_temp_collector") and self._temp_collector:
            return self._temp_collector

        return None

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> Shareable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Main control flow for job execution."""

        self._abort_signal = abort_signal
        if self._abort_signal.triggered:
            self.log_info(fl_ctx, "Abort signal triggered, aborting")
            return make_reply(ReturnCode.TASK_ABORTED)

        start_time = time.time()
        # Get collector - may be None early, but we'll get it again after prepare
        collector = self._get_or_create_collector(fl_ctx)

        ################################
        # Prepare the controller
        if collector:
            with collector.span("flare.controller.prepare") as span:
                self._do_prepare(fl_ctx)
                span.set_ok()
        else:
            self._do_prepare(fl_ctx)
        if self._abort_signal.triggered:
            self.log_info(fl_ctx, "Abort signal triggered, aborting")
            return make_reply(ReturnCode.TASK_ABORTED)

        ################################
        # Start the NetRunner
        if collector:
            with collector.span("flare.controller.start_netrunner") as span:
                self._do_start_netrunner(fl_ctx)
                span.set_ok()
        else:
            self._do_start_netrunner(fl_ctx)
        if self._abort_signal.triggered:
            self.log_info(fl_ctx, "Abort signal triggered, aborting")
            if self._job_controller is not None:
                self._job_controller.shutdown()
            return make_reply(ReturnCode.TASK_ABORTED)

        ################################
        # Make all clients do the execute task
        if collector:
            with collector.span(
                "flare.controller.broadcast_task",
                attributes={"client_count": len(self._client_names)},
            ) as span:
                rc = self._make_clients_do_start_execute_task(fl_ctx)
                if rc == ReturnCode.OK:
                    span.set_ok()
                else:
                    span.set_error(f"Client execution failed: {rc}")
        else:
            rc = self._make_clients_do_start_execute_task(fl_ctx)
        ################################
        # Flush controller-level telemetry ALWAYS - even on failure
        # This ensures we capture whatever telemetry was collected
        self._flush_controller_telemetry(fl_ctx)

        if self._abort_signal.triggered or rc != ReturnCode.OK:
            self.log_info(fl_ctx, "Client execution failed or was aborted, aborting")
            if self._job_controller is not None:
                self._job_controller.shutdown()
            return make_reply(ReturnCode.TASK_ABORTED)

        ################################
        # all clients are done,
        # check if there are any results left
        # that the app didn't pick up
        self._do_handle_finalization(fl_ctx)
        if self._abort_signal.triggered:
            self.log_info(fl_ctx, "Abort signal triggered, aborting")
            return make_reply(ReturnCode.TASK_ABORTED)

        # done
        time_taken = time.time() - start_time
        self.log_info(fl_ctx, f"Total pipeline ran for {time_taken} seconds")

    def _flush_controller_telemetry(self, fl_ctx: FLContext) -> None:
        """Flush any telemetry collected by the controller itself.

        Uses the queue mechanism - telemetry will be saved by save_all_remaining.
        We don't create the results directory here to avoid conflicts.
        """
        if not hasattr(self, "_temp_collector") or self._temp_collector is None:
            return

        try:
            batch = self._temp_collector.flush()
            if batch.is_empty:
                return

            self.logger.info(
                f"Sending controller telemetry batch to homesite: {len(batch.spans)} spans"
            )

            # Get job_id from fl_ctx
            job_id = fl_ctx.get_job_id() or "unknown"

            token_package = TokenPackage.wrap_data(
                source_site="server",
                source_node="__controller__",
                source_job=job_id,
                target_site=self._homesite,
                target_place=TELEMETRY_PLACE_NAME,
                target_job=job_id,
                is_yield=True,
                frame=Frame.start_frame(num=0),
                data=batch.model_dump(),
                trace_context=None,
            )

            # Check if results directory already exists (created by save_all_remaining)
            # If so, save directly; otherwise use the queue
            app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
            if app_root:
                results_dir = Path(app_root) / "results"
                if results_dir.exists():
                    # Results directory exists, safe to save there
                    existing = list(results_dir.glob("token_*.json"))
                    next_idx = len(existing)
                    token_file = results_dir / f"token_{next_idx}.json"
                    with open(token_file, "w") as f:
                        f.write(token_package.model_dump_json(indent=2))
                    self.logger.info(f"Saved controller telemetry to {token_file}")
                else:
                    # Results directory doesn't exist yet, use queue
                    # Telemetry will be saved by save_all_remaining
                    self._app_io_manager.receive_token_package(token_package)
            else:
                # No app_root, use queue
                self._app_io_manager.receive_token_package(token_package)

        except Exception as e:
            self.logger.warning(f"Failed to flush controller telemetry: {e}")

    def _do_prepare(self, fl_ctx: FLContext):
        assert self._abort_signal is not None

        self._bound_net_ir_model = BoundNetIR.model_validate(self._bound_net_ir)

        # create the execution environment
        job_id = fl_ctx.get_job_id()
        if job_id is None:
            raise RuntimeError("job ID is missing from FL context")
        assert isinstance(job_id, str)

        self._router: FlareRouter = FlareRouter(
            fl_ctx=fl_ctx,
            abort_signal=self._abort_signal,
            app_io_manager=self._app_io_manager,
            client_names=self._client_names,
            logger=self.logger,
        )

        self._job_controller = JobController(
            router=self._router,
            pipeline_api_version=self._submitted_api_version,
            federation_name=self._federation_name,
            homesite=self._homesite,
            this_site=FQCN.ROOT_SERVER,
            job_id=job_id,
            netir=self._bound_net_ir_model,
            logger=self.logger,
            force_modgen=self._force_modgen,
        )

        ReliableMessage.register_request_handler(
            topic=Constant.TOPIC_SEND_TO_PLACE,
            handler_f=self._on_send_to_place_topic,
            fl_ctx=fl_ctx,
        )

    def _do_start_netrunner(self, fl_ctx: FLContext):
        self.log_info(
            fl_ctx, f"Server controller starts executing job {fl_ctx.get_job_id()}"
        )
        # if there's no NetIR for the server controller to execute, we won't have a JobController

        if self._job_controller:
            self._job_controller.start()
            self._job_controller.submit_initial_tokens()
        else:
            self.log_info(fl_ctx, "Not NetIR for the server, just running empty")

    def _make_clients_do_start_execute_task(self, fl_ctx: FLContext) -> str:
        """ "
        This method should block untill all clients are done executing
        their nets.
        Returns ReturnCode.OK if all clients finished successfully,
        ReturnCode.EXECUTION_EXCEPTION otherwise.
        """
        self.log_info(
            fl_ctx,
            f"Sending {Constant.TASK_START_EXECUTION} task to clients {self._client_names} for workflow {self.workflow_id}",
        )

        start_time = time.time()
        result_logger = ClientTaskResultLogger(
            logger=self.log_error,
            task_name=Constant.TASK_START_EXECUTION,
            expected_num_clients=len(self._client_names),
            fl_ctx=fl_ctx,
        )
        task = Task(
            name=Constant.TASK_START_EXECUTION,
            data=Shareable(),
            timeout=self._options.task_timeout_s,
            result_received_cb=result_logger.result_received_callback,
        )
        self.broadcast_and_wait(
            task=task,
            targets=self._client_names,
            min_responses=len(self._client_names),
            fl_ctx=fl_ctx,
            abort_signal=self._abort_signal,
        )
        if result_logger.get_overall_client_return_code() != ReturnCode.OK:
            result_logger.log_results()

        # wait for the own net runner to finish
        if self._job_controller:
            self._job_controller.wait_for_done(abort_signal=self._abort_signal)

        time_taken = time.time() - start_time
        self.log_info(
            fl_ctx, f"{Constant.TASK_START_EXECUTION} task took {time_taken} seconds"
        )

        if self._job_controller and self._job_controller.error_occurred():
            self.log_error(
                fl_ctx,
                f"JobController error occurred: {self._job_controller.get_panic_error()}",
            )
            return ReturnCode.EXECUTION_EXCEPTION

        return result_logger.get_overall_client_return_code()

    def _do_handle_finalization(self, fl_ctx: FLContext):
        """All the clients finished successfully. Wait for our own
        NetRunner to finish and then save any results that the app didn't pick up."""
        if self._job_controller:
            self._job_controller.wait_for_done(self._abort_signal)
        self._app_io_manager.save_all_remaining(fl_ctx)

    def _on_send_to_place_topic(self, topic: str, request, fl_ctx):
        """Called when a client sends a token to the server."""
        assert topic == Constant.TOPIC_SEND_TO_PLACE

        payload_dict: dict[str, Any] = request[Constant.MSG_KEY_TOKEN_PACKAGE_DICT]
        token_package = TokenPackage.model_validate(payload_dict)
        self.log_debug(
            fl_ctx,
            f"server received token for site {token_package.target_place},"
            + f" source site is {token_package.source_site}",
        )
        self._router.route_token_package_sync(token_package)

        return make_reply(ReturnCode.OK)
