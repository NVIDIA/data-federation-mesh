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

# pyright: reportImportCycles=false
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from logging import Logger
from multiprocessing import Queue
from queue import Empty, Full
from queue import Queue as ThreadQueue
from typing import TYPE_CHECKING

from nv_dfm_core.api._prepared_pipeline import PreparedPipeline
from nv_dfm_core.exec._frame import Frame
from nv_dfm_core.exec._token_package import TokenPackage
from nv_dfm_core.gen.modgen.ir import BoundNetIR
from nv_dfm_core.session import (
    CallbackRunner,
    JobStatus,
    configure_session_logging,
)

from ._context import get_spawn_context

if TYPE_CHECKING:
    from ._job_execution import JobExecution
    from ._job_handle import JobHandle
    from ._local_job import LocalJob
else:
    JobExecution = object
    JobHandle = object  # type: ignore
    LocalJob = object


def _get_int_env(name: str, default: int) -> int:
    val = os.environ.get(name, "")
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


DFM_LOCAL_REATTACH_BUFFER_MAX_TOKENS: int = _get_int_env(
    "DFM_LOCAL_REATTACH_BUFFER_MAX_TOKENS", 10_000
)
DFM_LOCAL_REATTACH_FINISH_GRACE_SECONDS: float = float(
    os.environ.get("DFM_LOCAL_REATTACH_FINISH_GRACE_SECONDS", "5.0")
)
DFM_LOCAL_REATTACH_DRAIN_MAX_TOKENS_PER_JOB_PER_TICK: int = _get_int_env(
    "DFM_LOCAL_REATTACH_DRAIN_MAX_TOKENS_PER_JOB_PER_TICK", 200
)

LOCAL_COMPLETION_PLACE: str = "__local_completion__"


@dataclass
class JobRecord:
    job_id: str
    handle: "JobHandle"
    created_at: float
    status: JobStatus = JobStatus.SUBMITTED
    finished_at: float | None = None

    # Set when we observe completion from ALL participating sites, or abort.
    completion_event: threading.Event = field(default_factory=threading.Event)

    # Participating sites for this job
    participating_sites: set[str] = field(default_factory=set)

    # Sites that have sent their LOCAL_COMPLETION_PLACE marker
    completed_sites: set[str] = field(default_factory=set)

    # Consumer queues for attached clients (local threads only, not multiprocessing).
    consumers: dict[str, ThreadQueue[TokenPackage]] = field(default_factory=dict)

    # Buffer tokens while detached (no consumers). Bounded to avoid unbounded memory growth.
    buffer: deque[TokenPackage] = field(
        default_factory=lambda: deque(maxlen=DFM_LOCAL_REATTACH_BUFFER_MAX_TOKENS)
    )

    # Completion handling
    finish_observed_at: float | None = None


class FederationRunner:
    """
    Manages a pool of N JobRunner instances, each of which has site-many processes.
    Comes with health monitoring and recovery.
    """

    def __init__(
        self,
        max_concurrent_jobs: int,
        sites: list[str],
        federation_name: str,
        homesite: str,
        logger: Logger | None = None,
    ):
        if max_concurrent_jobs <= 0:
            raise ValueError("max_concurrent_jobs must be a positive integer.")

        self._max_concurrent_jobs: int = max_concurrent_jobs
        self._sites: list[str] = sites
        self._federation_name: str = federation_name
        self._homesite: str = homesite
        self._logger: Logger = logger or configure_session_logging()

        # use explicit spawn context for all multiprocessing objects
        ctx = get_spawn_context()
        self._inter_job_queue: Queue[TokenPackage] = ctx.Queue()
        self._running_jobs: dict[str, "JobHandle"] = {}
        # Persist job IDs for the life of the process (reattach should only be UNKNOWN if id never existed)
        self._jobs_by_id: dict[str, JobRecord] = {}

        # A lock to protect execution/job tracking structures.
        #
        # IMPORTANT: this lock is acquired in multiple helper methods. Some call paths
        # (e.g. releasing finished executions from within the drain loop) can re-enter
        # methods that also acquire this lock on the same thread. Using an RLock avoids
        # self-deadlock while preserving mutual exclusion across threads.
        self._executions_lock: threading.RLock = threading.RLock()

        self._all_executions: dict[int, JobExecution] = {}
        self._free_executions_q: Queue[int] = ctx.Queue(
            maxsize=self._max_concurrent_jobs
        )
        self._next_execution_id: int = 0

        for _ in range(self._max_concurrent_jobs):
            self._create_and_register_execution()

        self._running: bool = True
        self._inter_job_comm_and_health_check_thread: threading.Thread = (
            threading.Thread(target=self._inter_job_comm_and_health_check_loop)
        )
        self._inter_job_comm_and_health_check_thread.daemon = True
        self._inter_job_comm_and_health_check_thread.start()

        self._logger.info(
            f"Local Federation is ready with {self._max_concurrent_jobs} concurrent JobRunner objects."
        )

    def _create_and_register_execution(self):
        """Creates a new execution with a unique ID, starts it, and registers it."""
        from ._job_execution import JobExecution

        execution: JobExecution | None = None

        try:
            with self._executions_lock:
                execution_id = self._next_execution_id
                self._next_execution_id += 1

            execution_logger = self._logger.getChild("JobExecution")

            execution = JobExecution(
                id=execution_id,
                federation=self,
                sites=self._sites,
                inter_job_queue=self._inter_job_queue,
                logger=execution_logger,
            )

            with self._executions_lock:
                self._all_executions[execution_id] = execution
                self._free_executions_q.put(execution_id)
        except Exception as e:
            # Ensure cleanup if start fails
            if execution:
                self._logger.error(
                    f"Failed to create and start execution {execution.id}. Cleaning up after it: {e}"
                )
                execution.shutdown()
                # Clean up from executions dict if it was added
                with self._executions_lock:
                    if execution.id in self._all_executions:
                        del self._all_executions[execution.id]
            else:
                self._logger.error(f"Failed to create and start execution: {e}")

    def _process_inter_job_token(self, token: TokenPackage) -> bool:
        """Process an inter-job token and forward it to the correct job.

        Args:
            token: The token to process

        Returns:
            True if token was processed successfully, False if it should be skipped
        """
        with self._executions_lock:
            if token.target_job not in self._running_jobs:
                self._logger.warning(
                    f"Job {token.target_job} is not running, but we received a token for it. Dropping."
                )
                return False
            handle = self._running_jobs[token.target_job]
            # Get the execution while still holding the lock
            execution = handle.execution
            if execution is None:
                self._logger.warning(
                    f"Job {token.target_job} handle has no execution, dropping token."
                )
                return False
        # Release lock before calling receive_inter_job_token to avoid deadlock
        execution.receive_inter_job_token(token)
        return True

    def _get_executions_snapshot(self) -> dict[int, "JobExecution"]:
        """Get a thread-safe snapshot of all executions.

        Returns:
            A copy of the all_executions dictionary
        """
        with self._executions_lock:
            # Create a copy for safe iteration
            return dict(self._all_executions)

    def _cleanup_dead_executions(self, dead_execution_ids: set[int]) -> None:
        """Clean up dead executions and their associated jobs.

        Args:
            dead_execution_ids: Set of execution IDs that are dead
        """
        with self._executions_lock:
            living_ids_in_queue: list[int] = []
            try:
                while True:
                    execution_id = self._free_executions_q.get_nowait()
                    if execution_id not in dead_execution_ids:
                        living_ids_in_queue.append(execution_id)
            except Empty:
                pass

            for execution_id in living_ids_in_queue:
                self._free_executions_q.put(execution_id)

            for execution_id in dead_execution_ids:
                if execution_id in self._all_executions:
                    execution = self._all_executions[execution_id]
                    # check if this execution is currently running a job
                    job_id = execution.job_id()
                    if job_id is not None and job_id in self._running_jobs:
                        # Remove the job handle from tracking
                        del self._running_jobs[job_id]
                        # Mark record aborted and completed
                        rec = self._jobs_by_id.get(job_id)
                        if rec and rec.status not in (
                            JobStatus.FINISHED,
                            JobStatus.ABORTED,
                        ):
                            rec.status = JobStatus.ABORTED
                            rec.finished_at = time.time()
                            rec.completion_event.set()

                    # Ensure resources are cleaned up even if processes are dead
                    try:
                        execution.shutdown()
                    except Exception as e:
                        self._logger.warning(
                            f"[HealthCheck] Error during shutdown of dead execution {execution_id}: {e}"
                        )

                    # remove the execution from the all executions dict
                    del self._all_executions[execution_id]
                else:
                    # Execution was already removed, but we still need to clean up any associated job
                    # This could happen if the execution was removed by another thread
                    self._logger.warning(
                        f"[HealthCheck] Dead execution {execution_id} not found in _all_executions during cleanup"
                    )

    def _inter_job_comm_and_health_check_loop(self):
        """
        Receive tokens for inter-job communication and forward them to the correct job.
        Also periodically checks for dead workers and replaces them."""
        last_health_check_time: float = time.time()
        while self._running:
            try:
                token: TokenPackage = self._inter_job_queue.get(timeout=0.1)
                assert token
                # Process the token and forward to the correct job
                if not self._process_inter_job_token(token):
                    continue
            except Empty:
                pass
            except OSError as e:
                # this can happen if the session isn't properly closed and the process simply ends
                # then we may still be in self._inter_job_queue.get(...) above. This raises an OSError
                # We add a bit of additional info
                raise RuntimeError(
                    f"OSError in health check loop; forced shutdown. Maybe the session wasn't properly closed? {e}"
                ) from e

            # Drain yield queues for running jobs and update job records.
            self._drain_yields_and_update_jobs()

            # Get snapshot of executions for health check
            all_executions_copy = self._get_executions_snapshot()

            # health check
            current_time = time.time()
            if current_time - last_health_check_time <= 2.0:
                continue
            last_health_check_time = current_time

            dead_execution_ids: set[int] = {
                id
                for id, execution in all_executions_copy.items()
                if not execution.is_alive()
            }

            if not dead_execution_ids:
                continue

            self._logger.info(
                f"[HealthCheck] Detected dead execution IDs: {dead_execution_ids}. Cleaning up..."
            )

            # Clean up dead executions
            self._cleanup_dead_executions(dead_execution_ids)

            self._logger.info(
                f"[HealthCheck] Replacing {len(dead_execution_ids)} dead executions..."
            )
            for _ in range(len(dead_execution_ids)):
                self._create_and_register_execution()

    def _assign_execution_to_handle(self, handle: "JobHandle") -> "JobExecution":
        """
        Checks out an execution from the free pool and assigns it to a job handle.

        Returns:
            JobExecution assigned to this handle
        """
        with self._executions_lock:
            try:
                execution_id = self._free_executions_q.get(timeout=30.0)
            except Empty:
                raise TimeoutError(
                    "Could not reserve an execution space within the timeout period."
                )
            if execution_id not in self._all_executions:
                raise RuntimeError(
                    f"Execution {execution_id} not found in all executions"
                )
            reserved_execution = self._all_executions[execution_id]

            reserved_execution.handle_was_assigned(handle)
            return reserved_execution

    def release_execution(self, execution: JobExecution, *, abort: bool = True):
        """
        Return the execution to the free pool
        """
        self._logger.info(
            f"[Federation] Releasing execution {execution.id} back to the pool."
        )
        if abort:
            execution.abort_job()
        with self._executions_lock:
            handle = execution.handle()
            if handle:
                # Remove from tracking - NO callback to LocalJob needed
                del self._running_jobs[handle.job_id]
            execution.handle_was_released()

            if execution.id in self._all_executions:
                try:
                    self._free_executions_q.put_nowait(execution.id)
                except Full:
                    # Queue is full, which shouldn't happen in normal operation
                    self._logger.warning(
                        f"Warning: Free executions queue is full, dropping execution {execution.id}"
                    )

    def _mark_job_finished(self, rec: JobRecord) -> None:
        if rec.status in (JobStatus.FINISHED, JobStatus.ABORTED):
            return
        rec.status = JobStatus.FINISHED
        if rec.finished_at is None:
            rec.finished_at = time.time()
        rec.completion_event.set()

    def _mark_job_aborted(self, rec: JobRecord) -> None:
        if rec.status in (JobStatus.FINISHED, JobStatus.ABORTED):
            return
        rec.status = JobStatus.ABORTED
        if rec.finished_at is None:
            rec.finished_at = time.time()
        rec.completion_event.set()

    def _drain_yields_and_update_jobs(self) -> None:
        # Snapshot running jobs to avoid holding lock while draining queues
        with self._executions_lock:
            running_items = list(self._running_jobs.items())
            records = dict(self._jobs_by_id)

        now = time.time()

        for job_id, handle in running_items:
            rec = records.get(job_id)
            if rec is None:
                continue

            execution = handle.execution
            if execution is None:
                continue

            # Detect hard abort (workers died)
            if not execution.is_alive() and rec.status not in (
                JobStatus.FINISHED,
                JobStatus.ABORTED,
            ):
                with self._executions_lock:
                    rec2 = self._jobs_by_id.get(job_id)
                    if rec2:
                        self._mark_job_aborted(rec2)
                # Release execution back to pool (abort)
                self.release_execution(execution, abort=True)
                continue

            drained = 0
            while drained < DFM_LOCAL_REATTACH_DRAIN_MAX_TOKENS_PER_JOB_PER_TICK:
                try:
                    token: TokenPackage = execution.yield_queue.get_nowait()
                except Empty:
                    break

                drained += 1

                with self._executions_lock:
                    rec2 = self._jobs_by_id.get(job_id)
                    if rec2 is None:
                        continue

                    # First yield implies running.
                    if rec2.status == JobStatus.SUBMITTED:
                        rec2.status = JobStatus.RUNNING

                    # Completion marker is federation-internal: deliver to consumers/buffer as an
                    # end-of-stream sentinel, and use it as the authoritative completion signal.
                    if token.target_place == LOCAL_COMPLETION_PLACE:
                        # Track completion for this site
                        rec2.completed_sites.add(token.source_site)

                        # Authoritative job completion only after ALL participating sites are done
                        if rec2.completed_sites.issuperset(rec2.participating_sites):
                            self._logger.info(
                                f"Job {job_id} finished: All sites {rec2.participating_sites} completed. Sending final completion sentinel to consumers."
                            )
                            if rec2.finish_observed_at is None:
                                rec2.finish_observed_at = now
                            self._mark_job_finished(rec2)

                            # NOW send the sentinel to consumers, after everything is drained
                            if rec2.consumers:
                                for q in rec2.consumers.values():
                                    q.put(token)
                            else:
                                rec2.buffer.append(token)
                        else:
                            self._logger.debug(
                                f"Job {job_id} site {token.source_site} completed. "
                                + f"Waiting for {rec2.participating_sites - rec2.completed_sites}"
                            )
                        continue

                    if rec2.consumers:
                        for q in rec2.consumers.values():
                            q.put(token)
                    else:
                        rec2.buffer.append(token)

                    # Do NOT mark job finished on arbitrary stop-frame tokens.
                    # In local mode, stop-frame yields (e.g., StopToken on _dfm_status_) can race
                    # with data yields from other transitions/threads. We only treat the dedicated
                    # LOCAL_COMPLETION_PLACE marker as completion.

            # If finished and grace time elapsed, release execution without aborting.
            with self._executions_lock:
                rec3 = self._jobs_by_id.get(job_id)
                if (
                    rec3
                    and rec3.status == JobStatus.FINISHED
                    and rec3.finish_observed_at is not None
                    and (now - rec3.finish_observed_at)
                    >= DFM_LOCAL_REATTACH_FINISH_GRACE_SECONDS
                ):
                    # We can release the execution back to the pool; keep JobRecord forever.
                    self.release_execution(execution, abort=False)
                    # Ensure handle no longer points at the released execution
                    rec3.handle.execution = None

    def is_known_job_id(self, job_id: str) -> bool:
        with self._executions_lock:
            return job_id in self._jobs_by_id

    def get_job_handle_any(self, job_id: str) -> "JobHandle | None":
        """Return the JobHandle for a known job_id, even if it is no longer running."""
        with self._executions_lock:
            rec = self._jobs_by_id.get(job_id)
            if rec is None:
                return None
            return rec.handle

    def get_job_status(self, job_id: str) -> JobStatus:
        with self._executions_lock:
            rec = self._jobs_by_id.get(job_id)
            if rec is None:
                return JobStatus.UNKNOWN
            return rec.status

    def wait_until_finished(self, job_id: str, timeout: float | None = None) -> bool:
        if timeout is None:
            timeout = 300.0
        with self._executions_lock:
            rec = self._jobs_by_id.get(job_id)
            if rec is None:
                return False
            ev = rec.completion_event
        return ev.wait(timeout=timeout)

    def cancel_job(self, job_id: str) -> None:
        with self._executions_lock:
            rec = self._jobs_by_id.get(job_id)
            handle = self._running_jobs.get(job_id)
        if rec is None:
            return
        if handle is None or handle.execution is None:
            self._mark_job_aborted(rec)
            return
        self._mark_job_aborted(rec)
        self.release_execution(handle.execution, abort=True)

    def attach_client(
        self, job_id: str
    ) -> tuple[str, ThreadQueue[TokenPackage]] | None:
        """Attach a consumer for buffered/live tokens.\n+\n+        Returns (consumer_id, queue) if job_id is known, else None.\n+"""
        consumer_id = str(uuid.uuid4())
        q: ThreadQueue[TokenPackage] = ThreadQueue()
        with self._executions_lock:
            rec = self._jobs_by_id.get(job_id)
            if rec is None:
                return None
            if rec.consumers:
                self._logger.warning(
                    f"Multiple consumers attached to local job {job_id}. Result delivery is undefined."
                )
            rec.consumers[consumer_id] = q
            # Flush buffered tokens (buffer only exists when there were no consumers)
            if rec.buffer:
                flushed = 0
                for token in rec.buffer:
                    q.put(token)
                    flushed += 1
                rec.buffer.clear()
            else:
                flushed = 0
        return consumer_id, q

    def detach_client(self, job_id: str, consumer_id: str) -> None:
        with self._executions_lock:
            rec = self._jobs_by_id.get(job_id)
            if rec is None:
                return
            _ = rec.consumers.pop(consumer_id, None)

    def find_job_handle(self, job_id: str):
        """Find job execution info by ID.

        Returns JobHandle (not LocalJob) to avoid keeping client objects alive.

        Returns:
            JobHandle if found in running jobs, None otherwise.

        Note:
            Only returns jobs currently running. Finished jobs are not tracked.
        """
        with self._executions_lock:
            return self._running_jobs.get(job_id)

    def submit(
        self,
        pipeline: PreparedPipeline,
        next_frame: Frame,
        netirs: dict[str, BoundNetIR],
        force_modgen: bool = False,
        callback_runner: CallbackRunner | None = None,
    ) -> LocalJob:
        """
        Executes a job by finding a free JobExecution object (which has one JobRunner process per site).

        Args:
            pipeline: The prepared pipeline to execute.
            next_frame: The next frame for the job.
            netirs: Dictionary of bound net IRs for each site.
            force_modgen: Whether to force module regeneration.
            callback_runner: The callback runner for dispatching results.

        Returns:
            LocalJob: The created and started job.
        """
        participating_sites = set(netirs.keys())
        if not participating_sites.issubset(set(self._sites)):
            raise ValueError(
                f"Job requires sites {participating_sites}, but the federation only has {self._sites}."
            )

        job_id = str(uuid.uuid4())[:8]
        self._logger.info(
            f"--- Submitting new async job {job_id} with {len(participating_sites)} tasks ---"
        )

        from ._job_handle import JobHandle
        from ._local_job import LocalJob

        # Create JobHandle for tracking (lightweight, no callbacks/threads)
        handle = JobHandle(
            job_id=job_id,
            homesite=self._homesite,
            pipeline=pipeline,
            next_frame=next_frame,
            federation_name=self._federation_name,
            pipeline_api_version=pipeline.api_version,
            force_modgen=force_modgen,
            execution=None,  # Set after execution assigned
        )

        # Assign execution to handle
        execution = self._assign_execution_to_handle(handle)
        handle.execution = execution

        # Track by handle, not full job
        with self._executions_lock:
            self._running_jobs[job_id] = handle
            # Create or overwrite job record (job_id is unique per process)
            self._jobs_by_id[job_id] = JobRecord(
                job_id=job_id,
                handle=handle,
                created_at=time.time(),
                status=JobStatus.SUBMITTED,
                participating_sites=participating_sites,
            )

        # Create LocalJob (client interface with callback runner for dispatching results)
        job = LocalJob.create_from_handle(
            handle=handle,
            federation=self,
            logger=self._logger,
            callback_runner=callback_runner,
        )

        # Start the job
        job.start(netirs=netirs)
        with self._executions_lock:
            rec = self._jobs_by_id.get(job_id)
            if rec and rec.status == JobStatus.SUBMITTED:
                rec.status = JobStatus.RUNNING

        return job

    def shutdown(self):
        """
        Shuts down all worker processes. Gracefully, hopefully.
        """
        self._logger.info("--- Shutting local Federation down ---")
        self._running = False

        # Clean up all running jobs first
        self._cleanup_all_jobs()

        # Wait a bit longer for the health check to finish its last loop
        self._inter_job_comm_and_health_check_thread.join(timeout=3.0)

        self._logger.info("Sending termination signal to all JobRunner objects...")
        with self._executions_lock:
            for execution in self._all_executions.values():
                try:
                    execution.shutdown()
                except Exception as e:
                    self._logger.error(
                        f"Error shutting down execution {execution.id}: {e}"
                    )

        # Close federation-level queues
        try:
            self._inter_job_queue.close()
            self._inter_job_queue.join_thread()
        except Exception:
            pass
        try:
            self._free_executions_q.close()
            self._free_executions_q.join_thread()
        except Exception:
            pass

        self._logger.info("Local Federation has been shut down.")

    def _cleanup_all_jobs(self):
        """Robust cleanup method to ensure all running jobs are properly cancelled."""
        with self._executions_lock:
            running_jobs_copy = dict(self._running_jobs)

        for job_id, handle in running_jobs_copy.items():
            try:
                rec = self._jobs_by_id.get(job_id)
                if rec and rec.status == JobStatus.FINISHED:
                    self._logger.info(
                        f"Job {job_id} is already finished, releasing execution gracefully during shutdown"
                    )
                    if handle.execution:
                        self.release_execution(handle.execution, abort=False)
                    continue

                self._logger.info(f"Aborting job {job_id} during shutdown")
                if handle.execution:
                    # Mark aborted and release immediately to ensure _running_jobs drains.
                    if rec:
                        self._mark_job_aborted(rec)
                    self.release_execution(handle.execution, abort=True)
            except Exception as e:
                self._logger.error(f"Error aborting job {job_id} during shutdown: {e}")

        # Wait a bit for jobs to finish cancelling (with timeout)
        max_wait_time = (
            2.0  # Reduced from 10.0s now that we handle finished jobs gracefully
        )
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            with self._executions_lock:
                if not self._running_jobs:
                    break
            time.sleep(0.05)  # Check every 50ms instead of blocking
        else:
            self._logger.warning(
                f"Some jobs did not finish cancelling within {max_wait_time} seconds"
            )

    def send_token_package(self, token_package: TokenPackage) -> None:
        self._inter_job_queue.put(token_package)
