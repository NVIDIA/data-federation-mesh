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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from nv_dfm_core.api import PreparedPipeline
from nv_dfm_core.exec import FlowInfo, Frame
from nv_dfm_core.exec._token_package import TokenPackage
from nv_dfm_core.gen.modgen.ir import START_PLACE_NAME
from nv_dfm_core.telemetry import job_id_to_trace_id, telemetry_enabled

# Import callback types from central location (for backward compatibility)
from ._callback_dispatcher import CallbackRunner, DfmDataCallback  # noqa: F401


class JobStatus(Enum):
    """Enumeration of possible job execution states."""

    SUBMITTED = 0
    RUNNING = 1
    FINISHED = 2
    ABORTED = 3
    UNKNOWN = 4


class Job(ABC):
    """
    A Job object is returned by Session.execute() and provides
    a way to interact with the running job.

    Depending on the type of execution, different Job implementations
    are used.

    Note: Data callbacks are registered directly within Session.execute()
    because we want to make sure that no messages get lost.
    Currently it's not supported to register callbacks later on.
    """

    def __init__(
        self,
        homesite: str,
        job_id: str,
        pipeline: PreparedPipeline | None = None,
        next_frame: Frame | None = None,
        was_found: bool = True,
        callback_runner: CallbackRunner | None = None,
    ):
        self._homesite: str = homesite
        self._job_id: str = job_id
        self._pipeline: PreparedPipeline | None = pipeline
        self._next_frame: Frame | None = next_frame
        self._was_found: bool = was_found
        self._callback_runner: CallbackRunner | None = callback_runner

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def was_found(self) -> bool:
        """Whether this Job object is attached to a known/real job.

        For execute(), this is always True.
        For reattach(), this may be False if the job_id was not recognized by the target.
        """
        return self._was_found

    @property
    def callback_runner(self) -> CallbackRunner | None:
        """Access the callback runner for manual dispatch control.

        When using ManualDispatcher, call `job.callback_runner.process_pending()`
        from your desired thread to process pending callbacks.

        Returns:
            The callback runner, or None if no callbacks were registered.
        """
        return self._callback_runner

    @abstractmethod
    def get_status(self) -> JobStatus: ...

    @abstractmethod
    def wait_until_finished(self, timeout: float | None = None) -> bool: ...

    @abstractmethod
    def cancel(self): ...

    @abstractmethod
    def detach(self) -> None:
        """Detach from the job, stopping result polling and releasing callbacks.

        After calling detach(), this Job object becomes inert and should not
        be used. The job execution continues, but results will not be delivered
        until reattachment with session.reattach().

        This allows the Job object to be garbage collected while the job
        continues running.

        This is idempotent - calling detach() multiple times is safe.
        """
        ...

    def send_input_params(
        self,
        input_params: dict[str, Any] | list[dict[str, Any]],
    ):
        """Sends more frames for the given input parameter sets, if the pipeline hasn't been stopped yet."""
        if self._pipeline is None or self._next_frame is None:
            raise RuntimeError(
                "Cannot send input params to a reattached job. "
                + "Reattached jobs can only query status and receive results, not send new input."
            )

        if self._next_frame.is_stop_frame():
            raise ValueError("Cannot send input params after the stop frame")

        if isinstance(input_params, dict):
            input_params = [dict(input_params)]

        # add a frame to each parameter set
        input_params_with_frame: list[tuple[Frame, dict[str, Any]]] = []
        for paramset in input_params:
            # add a start signal to the paramset
            paramset = paramset | {START_PLACE_NAME: FlowInfo()}
            input_params_with_frame.append((self._next_frame, paramset))
            self._next_frame = self._next_frame.with_loop_inc()

        self._send_input_params_internal(input_params_with_frame)

    def send_stop_frame(self):
        """Sends the stop frame, if the pipeline hasn't been stopped yet."""
        if self._pipeline is None or self._next_frame is None:
            raise RuntimeError(
                "Cannot send stop frame to a reattached job. "
                + "Reattached jobs can only query status and receive results, not send frames."
            )

        self._next_frame = Frame.stop_frame()
        input_params_with_frame: list[tuple[Frame, dict[str, Any]]] = [
            (self._next_frame, {START_PLACE_NAME: FlowInfo()})
        ]
        self._send_input_params_internal(input_params_with_frame)

    def _send_input_params_internal(
        self, input_params_with_frame: list[tuple[Frame, dict[str, Any]]]
    ) -> None:
        """Send input parameters to all participating net IRs by creating and sending token packages."""
        if self._pipeline is None:
            raise RuntimeError(
                "Cannot send input params to a reattached job without pipeline information."
            )

        # Build trace context for propagation (derived from job_id)
        trace_context: dict[str, str] | None = None
        if telemetry_enabled():
            trace_context = {
                "trace_id": job_id_to_trace_id(self._job_id),
                "span_id": "0" * 16,  # Placeholder - sites will create their own spans
            }

        for net_ir in self._pipeline.net_irs().values():
            # for each net, pick the params it receives
            for frame, params in net_ir.pick_input_params(input_params_with_frame):
                # and send each param individually
                for place, tagged_data in params.items():
                    package = TokenPackage(
                        source_site=self._homesite,
                        source_node=None,
                        source_job=self.job_id,
                        target_site=net_ir.site,
                        target_place=place,
                        target_job=self.job_id,
                        is_yield=False,
                        frame=frame,
                        tagged_data=tagged_data,
                        trace_context=trace_context,
                    )
                    self._send_token_package_internal(package)

    @abstractmethod
    def _send_token_package_internal(self, token_package: TokenPackage) -> None:
        """Sends a single token package to a place in the running pipeline."""
