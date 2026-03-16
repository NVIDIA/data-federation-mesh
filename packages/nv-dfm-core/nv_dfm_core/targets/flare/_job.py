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
import json
import logging
import os
import threading
import time
from pathlib import Path
from types import ModuleType

from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.fuel.flare_api.flare_api import (
    InternalError,
    JobMetaKey,
    NoReply,
)
from nvflare.fuel.flare_api.flare_api import (
    JobNotRunning as FlareJobNotRunning,
)
from nvflare.fuel.flare_api.flare_api import (
    Session as FlareSession,
)
from typing_extensions import override

from nv_dfm_core.api import PreparedPipeline
from nv_dfm_core.exec import (
    Frame,
    TokenPackage,
)
from nv_dfm_core.session import CallbackRunner, DfmDataCallback, JobStatus
from nv_dfm_core.session import Job as BaseJob
from nv_dfm_core.telemetry import TelemetryAggregator, TelemetryBatch, create_exporter

from ._defs import Constant
from ._receive_tokens_cmd_response import ReceiveTokensCmdResponse


def job_status_from_flare_string(status: str) -> JobStatus:
    """Convert a Flare string status to a JobStatus enum value.

    Args:
        status: String representation of job status (e.g., "running", "finished:completed")

    Returns:
        Corresponding JobStatus enum value, or UNKNOWN if status is unrecognized

    Note:
        Flare uses colon-separated status strings (e.g., "finished:completed").
        This method handles both simple statuses and compound statuses.
    """
    status = status.lower()
    # Special case: Flare uses "finished:aborted" to indicate an aborted job
    if status == "finished:aborted":
        return JobStatus.ABORTED
    if status in ("approved", "dispatched"):
        return JobStatus.SUBMITTED
    # For other statuses, we only care about the base status before the colon
    status = status.split(":")[0]
    try:
        return JobStatus[status.upper()]
    except KeyError:
        return JobStatus.UNKNOWN


class Job(BaseJob):
    def __init__(
        self,
        homesite: str,
        job_id: str,
        flare_session: FlareSession,
        runtime_module: ModuleType,
        pipeline: PreparedPipeline | None = None,
        next_frame: Frame | None = None,
        was_found: bool = True,
        logger: logging.Logger | None = None,
        callback_runner: CallbackRunner | None = None,
    ):
        super().__init__(
            homesite=homesite,
            job_id=job_id,
            pipeline=pipeline,
            next_frame=next_frame,
            was_found=was_found,
            callback_runner=callback_runner,
        )
        self._flare_session: FlareSession = flare_session
        self._job_workspace: Path = Path(
            os.environ.get("FLARE_JOB_WORKSPACE", "/tmp/dfm_workspace")
        )
        self._job_name: str | None = None
        self._runtime_module: ModuleType = runtime_module
        # Legacy callback fields kept for detach() cleanup (not used for dispatching)
        self._default_callback: DfmDataCallback | None = None
        self._place_callbacks: dict[str, DfmDataCallback] = {}
        self._logger: logging.Logger = logger or logging.getLogger(
            f"flare_job.{self._job_id}"
        )

        self._logger.info(
            "Initializing Job with job_id=%s, homesite=%s, workspace=%s",
            self._job_id,
            self._homesite,
            self._job_workspace,
        )

        # Background thread setup
        self._stop_token_receiving_thread_event: threading.Event = threading.Event()
        self._token_receiving_thread: threading.Thread = threading.Thread(
            target=self._receive_tokens_task, daemon=True
        )
        self._token_receiving_thread.start()
        self._logger.info("Started background token receiving thread")

        # detached flag
        self._detached: bool = False

    @property
    @override
    def job_id(self) -> str:
        return self._job_id

    @override
    def get_status(self) -> JobStatus:
        if not self._job_id:
            self._logger.error("Cannot get status: Job wasn't submitted")
            raise RuntimeError("Job wasn't submitted.")

        self._logger.debug("Getting job status for job_id=%s", self._job_id)
        job_meta = self._flare_session.get_job_meta(self._job_id)
        job_status = job_meta.get(JobMetaKey.STATUS.value, None)
        if job_status is None:
            self._logger.error("Job status is None for job_id=%s", self._job_id)
            raise RuntimeError("No job status found for job_id=%s", self._job_id)
        status = job_status_from_flare_string(job_status)
        self._logger.debug("Job status: %s (raw: %s)", status, job_status)
        return status

    @override
    def wait_until_finished(self, timeout: float | None = None) -> bool:
        """
        Monitor the job until it's finished or the timeout is reached.
        The caller should then check the status to see if it's finished or not.
        """
        self._check_not_detached()

        if not self._job_id:
            self._logger.error("Cannot wait for job: Job wasn't submitted")
            raise RuntimeError("Job wasn't submitted.")

        if timeout is None:
            timeout = 0

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
                result = self._wait_until_finished_internal(timeout)
                span.set_attribute("completed", result)
                span.set_attribute("final_status", self.get_status().name)
                if result:
                    span.set_ok()
                else:
                    span.set_error("Timeout or job not finished")
                return result
        else:
            return self._wait_until_finished_internal(timeout)

    def _wait_until_finished_internal(self, timeout: float) -> bool:
        """Internal wait logic without telemetry wrapper."""
        self._logger.info(
            "Starting to monitor job %s with timeout=%s", self._job_id, timeout
        )
        # flare does that already
        monitor_return = self._flare_session.monitor_job(self._job_id, timeout)
        self._logger.info(
            "Job monitoring completed for job %s with code %s",
            self._job_id,
            monitor_return,
        )
        self._stop_receive_tokens_background_thread()
        # The background thread handles downloading remaining tokens with proper delay,
        # but do a final check here to catch any stragglers
        tokens = self._download_remaining_tokens()
        if tokens:
            # Only warn if there are non-telemetry tokens left
            non_telemetry_tokens = [
                t for t in tokens if t.target_place != "__telemetry__"
            ]
            if non_telemetry_tokens:
                self._logger.warning(
                    "Job finished but had remaining tokens left, distributing them, but this should usually not happen with a controlled shutdown. Maybe the session wasn't properly closed?"
                )
            self._handle_tokens(tokens)
        return monitor_return == MonitorReturnCode.JOB_FINISHED

    @override
    def cancel(self):
        if not self._job_id:
            self._logger.error("Cannot cancel job: Job wasn't submitted")
            raise RuntimeError("Job wasn't submitted.")

        self._logger.info("Cancelling job %s", self._job_id)
        self._flare_session.abort_job(self._job_id)
        self._logger.info("Job %s cancellation requested", self._job_id)
        self._stop_receive_tokens_background_thread()

    @override
    def detach(self):
        """Stop polling for results and release callbacks.

        After detach, this FlareJob becomes inert and can be GC'd.
        The job continues running on the Flare server, but results
        will not be retrieved until another FlareJob reattaches.

        This is idempotent - calling detach() multiple times is safe.
        """
        if self._detached:
            return

        self._logger.info(f"Detaching from Flare job {self._job_id}")

        # Stop background polling thread
        self._stop_receive_tokens_background_thread()

        # Clear callback runner and legacy callbacks to break references and allow GC
        self._callback_runner = None
        self._default_callback = None
        self._place_callbacks = {}

        # Keep _flare_session for status queries

        # Mark as detached
        self._detached = True

        self._logger.info(f"Successfully detached from Flare job {self._job_id}")

    def _check_not_detached(self):
        """Raise error if job has been detached."""
        if self._detached:
            raise RuntimeError(
                f"Cannot perform operation on detached job {self._job_id}. "
                + "Use session.reattach() to create a new job object."
            )

    def _handle_tokens(self, tokens: list[TokenPackage] | None) -> None:
        """Process tokens: handle telemetry separately, dispatch rest via callback runner.

        Args:
            tokens: List of tokens to process, or None.
        """
        if not tokens:
            return

        # Filter out telemetry tokens and handle them
        user_tokens: list[TokenPackage] = []
        for token in tokens:
            if self._handle_telemetry_token(token):
                continue
            user_tokens.append(token)

        # Dispatch user tokens via callback runner
        if user_tokens and self._callback_runner:
            self._logger.debug(
                "Dispatching %d tokens via callback runner", len(user_tokens)
            )
            self._callback_runner.dispatch(user_tokens)
        elif user_tokens:
            self._logger.warning(
                "No callback runner, dropping %d tokens", len(user_tokens)
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

            # Track processed batches to avoid duplicates (can happen when tokens
            # are received via polling AND downloaded from job results)
            if not hasattr(self, "_processed_telemetry_batches"):
                self._processed_telemetry_batches: set[tuple[str, int]] = set()

            batch_key = (batch.site, batch.sequence_num)
            if batch_key in self._processed_telemetry_batches:
                self._logger.debug(
                    f"Skipping duplicate telemetry batch from {batch.site} (seq={batch.sequence_num})"
                )
                return True

            self._processed_telemetry_batches.add(batch_key)

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

    def _receive_tokens_task(self):
        """Background task that runs until stop_event is set."""
        self._logger.info("Token receiving background task started")

        # Notify callback runner that the background thread has started
        if self._callback_runner:
            self._callback_runner.start()

        try:
            while not self._stop_token_receiving_thread_event.is_set():
                time.sleep(0.1)
                status = self.get_status()
                if status in [JobStatus.SUBMITTED, JobStatus.UNKNOWN]:
                    continue
                if status != JobStatus.RUNNING:
                    self._logger.debug(
                        "Job status is %s, stopping token receiving thread", status
                    )
                    self._stop_token_receiving_thread_event.set()
                    break
                try:
                    self._logger.debug(
                        "Receive tokens thread is calling app command %s",
                        Constant.CMD_RETRIEVE_TOKENS,
                    )
                    json_dict = self._flare_session.do_app_command(
                        self._job_id, Constant.CMD_RETRIEVE_TOKENS, cmd_data=None
                    )
                    response = ReceiveTokensCmdResponse.model_validate(json_dict)
                    tokens = [
                        TokenPackage.model_validate(token) for token in response.tokens
                    ]
                    self._logger.debug(
                        "Receive tokens thread received %s tokens", len(tokens)
                    )
                    self._handle_tokens(tokens)
                except (FlareJobNotRunning, NoReply, InternalError) as e:
                    self._logger.debug("Exception in token receiving thread: %s", e)
                    continue
            # if there are any tokens we haven't handled yet, handle them
            status = self.get_status()
            self._logger.info("Finishing receive tokens thread. Job status: %s", status)
            if status == JobStatus.FINISHED:
                self._logger.info("Job finished, downloading remaining tokens")
                tokens = self._download_remaining_tokens()
                self._handle_tokens(tokens)
            self._logger.info("Token receiving background task completed")
        finally:
            # Notify callback runner that the background thread has stopped
            if self._callback_runner:
                self._callback_runner.stop()

    def _stop_receive_tokens_background_thread(self):
        """Cleanup when the object is destroyed."""
        self._logger.info("Stopping token receiving background thread")
        self._stop_token_receiving_thread_event.set()
        if self._token_receiving_thread.is_alive():
            self._logger.debug("Waiting for token receiving thread to join")
            self._token_receiving_thread.join(timeout=5.0)
            if self._token_receiving_thread.is_alive():
                self._logger.warning(
                    "Token receiving thread did not join within timeout"
                )
            else:
                self._logger.info("Token receiving thread stopped successfully")

        # Cleanup telemetry aggregator if it exists
        if hasattr(self, "_telemetry_aggregator"):
            try:
                self._telemetry_aggregator.shutdown()
            except Exception as e:
                self._logger.warning(f"Error during telemetry cleanup: {e}")

    def _download_remaining_tokens(
        self,
    ) -> list[TokenPackage] | None:
        """
        After the job is finished, if there were any remaining tokens left in the AppIOManager,
        the server will have stored them in the job workspace. Before we end this job on the
        app, we download the remaining tokens and return them just as if they were retrieved
        at runtime.

        Returns:
            None if the job is not finished yet.
            Path to results directory, if finished.
            Raises RuntimeError if the job was aborted
        """
        assert self.get_status() == JobStatus.FINISHED

        self._logger.info(
            "Downloading remaining tokens for finished job %s", self._job_id
        )
        # For finished jobs, download the complete results package
        loc = self._flare_session.download_job_result(self._job_id)
        self._logger.info("Downloaded job results to %s", loc)

        # Flare can only download results along with job metadata etc.,
        # so we need to repackage the results.
        results_dir = Path(loc) / "workspace" / "app_server" / "results"
        self._logger.debug("Looking for token files in %s", results_dir)

        results: list[TokenPackage] = []
        # sort the files by the number in the filename
        token_files = sorted(
            results_dir.glob("token_*.json"), key=lambda x: int(x.stem.split("_")[1])
        )
        self._logger.info("Found %s token files to process", len(token_files))

        for file in token_files:
            self._logger.debug("Processing token file %s", file)
            with open(file, "r", encoding="utf-8") as f:
                token_package = TokenPackage.model_validate(json.loads(f.read()))
                results.append(token_package)

        self._logger.info("Successfully downloaded %s remaining tokens", len(results))
        return results

    @override
    def _send_token_package_internal(self, token_package: TokenPackage) -> None:
        """Sends more frames for the given input parameter sets, if the pipeline hasn't been stopped yet."""
        self._logger.info(
            "Sending token to %s.%s. Calling app command %s",
            token_package.target_site,
            token_package.target_place,
            Constant.TOPIC_SEND_TO_PLACE,
        )
        _ = self._flare_session.do_app_command(
            job_id=self._job_id,
            topic=Constant.TOPIC_SEND_TO_PLACE,
            cmd_data=token_package.model_dump(),
        )
