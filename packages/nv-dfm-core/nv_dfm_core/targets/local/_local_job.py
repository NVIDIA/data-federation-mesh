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

import threading
import time
from logging import Logger
from queue import Empty
from queue import Queue as ThreadQueue
from typing import TYPE_CHECKING

from typing_extensions import override

from nv_dfm_core.api import PreparedPipeline
from nv_dfm_core.exec import Frame, TokenPackage
from nv_dfm_core.gen.modgen.ir._bound_net_ir import BoundNetIR
from nv_dfm_core.session import CallbackRunner, JobStatus
from nv_dfm_core.session import Job as SessionJob
from nv_dfm_core.targets.local import FederationRunner, JobSubmission
from nv_dfm_core.targets.local._federation_runner import LOCAL_COMPLETION_PLACE
from nv_dfm_core.telemetry import TelemetryAggregator, TelemetryBatch, create_exporter

if TYPE_CHECKING:
    from ._job_execution import JobExecution
    from ._job_handle import JobHandle
else:
    JobExecution = object
    JobHandle = object  # type: ignore


class LocalJob(SessionJob):
    """
    A Job is a request to run a pipeline.
    """

    def __init__(
        self,
        homesite: str,
        job_id: str,
        pipeline: PreparedPipeline,
        next_frame: Frame,
        federation: FederationRunner,
        federation_name: str,
        logger: Logger,
        force_modgen: bool = False,
        callback_runner: CallbackRunner | None = None,
    ):
        super().__init__(
            homesite=homesite,
            job_id=job_id,
            pipeline=pipeline,
            next_frame=next_frame,
            callback_runner=callback_runner,
        )
        self._federation: FederationRunner | None = federation
        self._pipeline_api_version: str = pipeline.api_version
        self._federation_name: str = federation_name
        self._logger: Logger = logger
        self._force_modgen: bool = force_modgen

        # the execution is ours until we must release it
        self._execution: JobExecution | None = None

        # Client-local consumer for yields (federation drains the multiprocessing yield queue)
        self._consumer_id: str | None = None
        self._consumer_queue: ThreadQueue[TokenPackage] | None = None
        self._consumer_thread: threading.Thread | None = None
        self._stop_consumer_event: threading.Event = threading.Event()
        self._completion_event: threading.Event = threading.Event()

        # detached flag
        self._detached: bool = False

    @classmethod
    def create_stub(
        cls,
        job_id: str,
        homesite: str,
        logger: Logger,
        callback_runner: CallbackRunner | None = None,
    ) -> "LocalJob":
        """Create a stub LocalJob for unknown job_id.

        This job will always report UNKNOWN status and cannot perform operations.
        Used when reattaching to a job_id that doesn't exist.
        """
        stub_job = cls.__new__(cls)
        stub_job._homesite = homesite
        stub_job._job_id = job_id
        stub_job._pipeline = None
        stub_job._next_frame = None
        stub_job._was_found = False
        stub_job._callback_runner = callback_runner
        stub_job._federation = None  # type: ignore
        stub_job._pipeline_api_version = ""
        stub_job._federation_name = ""
        stub_job._logger = logger
        stub_job._force_modgen = False
        # Keep legacy attribute for stub jobs only (no federation to query)
        stub_job._status = JobStatus.UNKNOWN  # type: ignore[attr-defined]
        stub_job._execution = None
        stub_job._consumer_id = None
        stub_job._consumer_queue = None
        stub_job._consumer_thread = None
        stub_job._stop_consumer_event = threading.Event()
        stub_job._completion_event = threading.Event()
        stub_job._completion_event.set()  # Already "done"
        stub_job._detached = False  # Not detached, just doesn't exist

        logger.info(f"Created stub job for unknown job_id: {job_id}")
        return stub_job

    @classmethod
    def create_from_handle(
        cls,
        handle: "JobHandle",
        federation: FederationRunner,
        logger: Logger,
        callback_runner: CallbackRunner | None = None,
    ) -> "LocalJob":
        """Create a LocalJob from a JobHandle.

        This is the primary way to create LocalJob objects, both for
        initial execution and for reattachment.

        Args:
            handle: JobHandle with job metadata and execution reference
            federation: FederationRunner managing this job
            logger: Logger for this job
            callback_runner: Optional callback runner for dispatch control

        Returns:
            New LocalJob connected to the execution tracked by handle
        """
        logger.info(f"Creating LocalJob from handle for job_id: {handle.job_id}")

        job = cls(
            homesite=handle.homesite,
            job_id=handle.job_id,
            pipeline=handle.pipeline,
            next_frame=handle.next_frame,
            federation=federation,
            federation_name=handle.federation_name,
            logger=logger,
            force_modgen=handle.force_modgen,
            callback_runner=callback_runner,
        )

        # Link to the execution from the handle
        job._execution = handle.execution
        job._pipeline_api_version = handle.pipeline_api_version

        # Attach consumer if callback runner was provided
        if callback_runner is not None and federation:
            attached = federation.attach_client(handle.job_id)
            if attached is not None:
                consumer_id, q = attached
                job._consumer_id = consumer_id
                job._consumer_queue = q
                job._start_consumer_thread()

        return job

    @override
    def detach(self):
        """Stop polling for results and release callbacks.

        After detach, this LocalJob becomes inert and can be GC'd.
        The job execution continues running in FederationRunner, but
        results will not be delivered until another LocalJob reattaches.

        This is idempotent - calling detach() multiple times is safe.
        """
        if self._detached:
            return

        self._logger.info(f"Detaching from job {self._job_id}")

        # Stop the local consumer thread
        self._cleanup_consumer_thread()

        # Unregister consumer with federation (does NOT abort job)
        try:
            if self._federation is not None and self._consumer_id is not None:
                self._federation.detach_client(self._job_id, self._consumer_id)
        except Exception as e:
            self._logger.warning(
                f"Error detaching consumer for job {self._job_id}: {e}"
            )

        # Clear callback runner to break references and allow GC
        self._callback_runner = None

        # Clear federation and execution references
        self._federation = None
        self._execution = None
        self._consumer_id = None
        self._consumer_queue = None

        # Mark as detached
        self._detached = True

        self._logger.info(f"Successfully detached from job {self._job_id}")

    def _check_not_detached(self):
        """Raise error if job has been detached."""
        if self._detached:
            raise RuntimeError(
                f"Cannot perform operation on detached job {self._job_id}. "
                + "Use session.reattach() to create a new job object."
            )

    def _start_consumer_thread(self) -> None:
        """Start the client-local consumer thread (reads from federation-provided queue)."""
        if self._consumer_queue is None:
            return
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            return
        self._stop_consumer_event.clear()
        self._consumer_thread = threading.Thread(target=self._consume_tokens_task)
        self._consumer_thread.daemon = True
        self._consumer_thread.start()
        self._completion_event.clear()

        # Notify callback runner that the consumer thread has started
        if self._callback_runner:
            self._callback_runner.start()

    def _cleanup_consumer_thread(self) -> None:
        """Stop the client-local consumer thread."""
        if self._consumer_thread is None:
            return
        self._stop_consumer_event.set()
        if (
            self._consumer_thread != threading.current_thread()
            and self._consumer_thread.is_alive()
        ):
            self._consumer_thread.join(timeout=1.0)
        self._consumer_thread = None

        # Notify callback runner that the consumer thread has stopped
        if self._callback_runner:
            self._callback_runner.stop()

    @property
    @override
    def job_id(self) -> str:
        return self._job_id

    @override
    def get_status(self) -> JobStatus:
        self._check_not_detached()
        # Stub jobs have no federation
        if self._federation is None:
            return getattr(self, "_status", JobStatus.UNKNOWN)
        return self._federation.get_job_status(self._job_id)

    def _handle_token(self, token: TokenPackage) -> None:
        """Process a token: handle telemetry separately, dispatch rest via callback runner.

        Args:
            token: The token to process.
        """
        # Handle telemetry tokens specially
        if self._handle_telemetry_token(token):
            return

        # Dispatch user token via callback runner
        if self._callback_runner:
            self._callback_runner.dispatch(token)
        else:
            self._logger.warning(
                f"Job {self._job_id}: No callback runner, dropping token for place '{token.target_place}'"
            )

    def _handle_telemetry_token(self, token: TokenPackage) -> bool:
        """Handle telemetry tokens from sites.

        Returns True if this was a telemetry token (and was handled), False otherwise.
        """
        # Check for telemetry place BEFORE processing
        if token.target_place != "__telemetry__":
            return False

        try:
            # This is a telemetry token - process it
            batch_data = token.unwrap_data()

            # Convert dict to TelemetryBatch if needed (happens when sent via model_dump())
            if isinstance(batch_data, dict):
                batch = TelemetryBatch.model_validate(batch_data)
            elif isinstance(batch_data, TelemetryBatch):
                batch = batch_data
            else:
                self._logger.warning(
                    f"Received telemetry token with unexpected data type: {type(batch_data)}"
                )
                return True

            # Get or create the aggregator
            if not hasattr(self, "_telemetry_aggregator"):
                exporter = create_exporter(logger=self._logger)
                self._telemetry_aggregator = TelemetryAggregator(
                    exporter=exporter, logger=self._logger
                )

            # Add batch to aggregator (which exports immediately)
            self._telemetry_aggregator.add_batch(batch)
            return True

        except Exception as e:
            self._logger.warning(f"Error handling telemetry token: {e}")
            return True  # Still mark as handled to avoid user callback confusion

    def _consume_tokens_task(self) -> None:
        """Consume tokens from the federation-provided consumer queue and dispatch via callback runner."""
        assert self._consumer_queue is not None
        while not self._stop_consumer_event.is_set():
            try:
                token = self._consumer_queue.get(timeout=0.1)
            except Empty:
                continue

            # Federation completion sentinel: not user-visible, just end-of-stream.
            if token.target_place == LOCAL_COMPLETION_PLACE:
                self._completion_event.set()
                break

            try:
                self._handle_token(token)
            except Exception as e:
                self._logger.warning(
                    f"Error in token callback for job {self._job_id}: {e}"
                )

    @override
    def wait_until_finished(self, timeout: float | None = None) -> bool:
        self._check_not_detached()
        # Add a reasonable default timeout if none provided
        if timeout is None:
            timeout = 300.0  # 5 minutes default timeout

        # Get telemetry collector if passed from Session
        collector = getattr(self, "_telemetry_collector", None)

        if collector:
            with collector.span(
                "job.wait_until_finished",
                attributes={
                    "job_id": self._job_id,
                    "timeout": timeout,
                },
            ) as span:
                res = self._wait_until_finished_internal(timeout)
                span.set_attribute("completed", res)
                span.set_attribute("final_status", self.get_status().name)
                if res:
                    span.set_ok()
                else:
                    span.set_error("Timeout waiting for job completion")
                return res
        else:
            return self._wait_until_finished_internal(timeout)

    def _wait_until_finished_internal(self, timeout: float) -> bool:
        """Internal wait logic without telemetry wrapper."""
        # Prefer federation authoritative completion if available
        if self._federation is not None:
            started = time.monotonic()
            res = self._federation.wait_until_finished(self._job_id, timeout=timeout)
            if not res:
                return False

            # Ensure callbacks have had a chance to receive/drain yields before returning.
            remaining = max(0.0, timeout - (time.monotonic() - started))
            if self._consumer_thread is not None:
                if (
                    self._consumer_thread.is_alive()
                    and self._consumer_thread != threading.current_thread()
                ):
                    self._consumer_thread.join(timeout=remaining)
            return True

        return self._completion_event.wait(timeout=timeout)

    @override
    def cancel(self):
        self._check_not_detached()
        try:
            if self._federation is not None:
                self._federation.cancel_job(self._job_id)
        finally:
            self._cleanup_consumer_thread()

        # Cleanup telemetry aggregator if it exists
        try:
            if hasattr(self, "_telemetry_aggregator"):
                self._telemetry_aggregator.shutdown()
        except Exception as e:
            self._logger.warning(f"Error during telemetry cleanup: {e}")

    def job_execution(self) -> JobExecution:
        if self._execution is None:
            raise RuntimeError("Job is not running")
        return self._execution

    def execution_was_assigned(self, execution: JobExecution | None):
        """Called by the FederationRunner to set or remove the execution."""
        self._execution = execution

    def execution_was_released(self):
        """Called by the FederationRunner to remove the execution."""
        self._execution = None

    def start(self, netirs: dict[str, BoundNetIR]):
        """Called by the FederationRunner to start the job."""
        try:
            assert self._execution is not None, "Execution is not owned by a job"
            if not self._execution.is_alive():
                dead_sites = self._execution.get_dead_sites()
                raise RuntimeError(
                    f"JobRunner {self._execution.id} is not alive. Workers for sites {dead_sites} have died."
                )

            self._participating_sites: set[str] = set(netirs.keys())
            self._logger.info(
                f"Job {self._job_id} is taking job {self._job_id} with participating sites: {self._participating_sites}"
            )

            # send the netirs to the workers
            for site in self._participating_sites:
                netir = netirs[site]
                js = JobSubmission(
                    pipeline_api_version=self._pipeline_api_version,
                    federation_name=self._federation_name,
                    job_id=self._job_id,
                    homesite=self._homesite,
                    netir=netir,
                    force_modgen=self._force_modgen,
                )
                self._execution.submit(site=site, js=js)

        except Exception as e:
            self._logger.error(f"Job {self._job_id} error in start: {e}")
            raise e

    @override
    def _send_token_package_internal(self, token_package: TokenPackage) -> None:
        """Sends a single token package to a place in the running pipeline."""
        if self._federation is None:
            raise RuntimeError(
                "Cannot send token package: job has no federation (stub job or invalid state)"
            )
        self._federation.send_token_package(token_package=token_package)
