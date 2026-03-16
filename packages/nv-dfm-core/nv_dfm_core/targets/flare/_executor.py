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

# pyright: reportMissingTypeStubs=false
#!/usr/bin/env python3

from typing import Any

from nvflare.apis.executor import Executor as FlareExecutor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import (
    Shareable,
    make_reply,
)
from nvflare.apis.signal import Signal
from nvflare.apis.utils.reliable_message import ReliableMessage
from typing_extensions import override

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

from ._defs import Constant
from ._flare_router import FlareRouter


class Executor(FlareExecutor):
    """DFM Executor that runs on federated client side.

    The Executor is responsible for:
    1. Receiving and processing tasks from the Controller
    2. Executing pipeline steps on the client side
    3. Sending results back to the Controller

    """

    def __init__(
        self,
        submitted_api_version: str,
        federation_name: str,
        homesite: str,
        bound_net_ir: dict[str, Any],
        force_modgen: bool = False,
    ):
        """Initialize the Executor with task names from Constants."""
        FlareExecutor.__init__(self)
        self._submitted_api_version: str = submitted_api_version
        self._federation_name: str = federation_name
        self._homesite: str = homesite
        self._bound_net_ir: dict[str, Any] = bound_net_ir
        self._bound_net_ir_model: BoundNetIR | None = None
        self._force_modgen: bool = force_modgen

        self._job_controller: JobController | None = None
        self._abort_signal: Signal | None = None
        self._telemetry_collector: SiteTelemetryCollector | None = None

    def _get_telemetry_collector(self) -> SiteTelemetryCollector | None:
        """Get the telemetry collector if available.

        Returns the collector from the job controller's context, or None if
        telemetry is disabled or not yet initialized.
        """
        if self._telemetry_collector:
            return self._telemetry_collector

        if not telemetry_enabled():
            return None

        # Get collector from job controller's context if available
        if self._job_controller and hasattr(self._job_controller, "_netrunner"):
            collector = self._job_controller._netrunner.dfm_context.telemetry_collector
            if collector:
                self._telemetry_collector = collector
                return collector

        return None

    def _get_or_create_collector(
        self, fl_ctx: FLContext
    ) -> SiteTelemetryCollector | None:
        """Get or create a telemetry collector for this execution.

        Unlike _get_telemetry_collector which only returns an existing collector,
        this method will create a temporary collector if needed (e.g., before
        JobController exists) to ensure early spans are captured.
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
        site_name = fl_ctx.get_identity_name()
        if job_id and site_name:
            self._telemetry_collector = SiteTelemetryCollector(
                site=site_name,
                job_id=job_id,
            )
            return self._telemetry_collector

        return None

    @override
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        self._abort_signal = abort_signal

        if self._abort_signal.triggered:
            self.log_info(fl_ctx, "Abort signal triggered, aborting")
            return make_reply(ReturnCode.TASK_ABORTED)
        elif task_name == Constant.TASK_START_EXECUTION:
            # Get or create collector early so we can capture all spans
            collector = self._get_or_create_collector(fl_ctx)
            site_name = fl_ctx.get_identity_name() or "unknown"

            if collector:
                # Wrap entire task execution with a span
                with collector.span(
                    "flare.executor.task",
                    attributes={
                        "task_name": task_name,
                        "site": site_name,
                    },
                ) as task_span:
                    with collector.span("flare.executor.prepare") as prep_span:
                        self._on_prepare_execution_task(fl_ctx=fl_ctx)
                        prep_span.set_ok()
                    with collector.span("flare.executor.run") as run_span:
                        self._on_start_execution_task(fl_ctx=fl_ctx)
                        run_span.set_ok()
                    task_span.set_ok()
                # Flush executor-level telemetry
                self._flush_executor_telemetry(site_name)
            else:
                self._on_prepare_execution_task(fl_ctx=fl_ctx)
                self._on_start_execution_task(fl_ctx=fl_ctx)
            return make_reply(ReturnCode.OK)
        else:
            self.log_error(fl_ctx, f"ignored unsupported {task_name}")
            return make_reply(ReturnCode.TASK_UNSUPPORTED)

    def _on_prepare_execution_task(self, fl_ctx: FLContext):
        assert self._abort_signal, "Abort signal not set"
        self.log_info(fl_ctx, "DfmExecutor preparing execution")

        site_name = fl_ctx.get_identity_name()
        if not site_name:
            raise ValueError("Site name not found")
        assert isinstance(site_name, str)

        job_id = fl_ctx.get_job_id()
        if job_id is None:
            raise RuntimeError("job ID is missing from FL context")
        assert isinstance(job_id, str)

        self._bound_net_ir_model = BoundNetIR.model_validate(self._bound_net_ir)

        self._router: FlareRouter = FlareRouter(
            fl_ctx=fl_ctx,
            abort_signal=self._abort_signal,
            app_io_manager=None,
            client_names=None,
            logger=self.logger,
        )

        self._job_controller = JobController(
            router=self._router,
            pipeline_api_version=self._submitted_api_version,
            federation_name=self._federation_name,
            homesite=self._homesite,
            this_site=site_name,
            job_id=job_id,
            netir=self._bound_net_ir_model,
            logger=self.logger,
            force_modgen=self._force_modgen,
        )

        # after this we do full asynchronous message handling
        ReliableMessage.register_request_handler(
            topic=Constant.TOPIC_SEND_TO_PLACE,
            handler_f=self._on_send_to_place_topic,
            fl_ctx=fl_ctx,
        )

    def _on_start_execution_task(self, fl_ctx: FLContext):
        assert self._abort_signal, "Abort signal not set"
        if not self._job_controller:
            self.log_error(fl_ctx, "JobController not started")
            return make_reply(ReturnCode.BAD_TASK_DATA)
        self.log_info(fl_ctx, "Received start execute event")
        assert self._bound_net_ir is not None

        self._job_controller.start()
        self._job_controller.submit_initial_tokens()
        # and wait for the net runner to finish
        self._job_controller.wait_for_done(abort_signal=self._abort_signal)
        if self._job_controller.error_occurred():
            self.log_error(
                fl_ctx,
                f"JobController error occurred during NetRunner execution for job {self._job_controller.job_id}: {self._job_controller.get_panic_error()}",
            )
            return make_reply(ReturnCode.TASK_ABORTED)

    def _on_send_to_place_topic(self, topic: str, request, fl_ctx):
        """Called when a client sends a token to the server."""
        assert topic == Constant.TOPIC_SEND_TO_PLACE

        payload_dict: dict[str, Any] = request[Constant.MSG_KEY_TOKEN_PACKAGE_DICT]
        token_package = TokenPackage.model_validate(payload_dict)
        self.log_debug(
            fl_ctx,
            f"Client {fl_ctx.get_identity_name()} received token for site {token_package.target_place},"
            + f" source site is {token_package.source_site}",
        )
        self._router.route_token_package_sync(token_package)
        return make_reply(ReturnCode.OK)

    def _flush_executor_telemetry(self, site_name: str) -> None:
        """Flush any telemetry collected by the executor itself."""
        if self._telemetry_collector is None:
            return

        try:
            batch = self._telemetry_collector.flush()
            if batch.is_empty:
                return

            # Send telemetry batch to homesite via the router
            if self._router is not None and self._job_controller is not None:
                self.logger.info(
                    f"Sending executor telemetry batch to homesite: {len(batch.spans)} spans"
                )
                self._router.route_token_sync(
                    to_job=None,
                    to_site=self._homesite,
                    to_place=TELEMETRY_PLACE_NAME,
                    is_yield=True,
                    frame=Frame.start_frame(num=0),
                    data=batch.model_dump(),
                    node_id="__executor__",
                )
        except Exception as e:
            self.logger.warning(f"Failed to flush executor telemetry: {e}")
