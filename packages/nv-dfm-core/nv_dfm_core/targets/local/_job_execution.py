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

import multiprocessing
from logging import Logger
from typing import TYPE_CHECKING

from nv_dfm_core.exec import TokenPackage

from ._context import get_spawn_context

if TYPE_CHECKING:
    from ._federation_runner import FederationRunner
    from ._job_handle import JobHandle
    from ._job_runner import JobSubmission
else:
    FederationRunner = object
    JobHandle = object  # type: ignore
    JobSubmission = object

# Use explicit spawn context for all multiprocessing objects.
_spawn_ctx = get_spawn_context()


class JobExecution:
    """
    A JobExecution object has a JobRunner process for every site to execute a pipeline. We reuse executions in a pool.
    We do this because we want to a) start all the background processes in a pool and b) we cannot send
    (normal) queues to processes after they have been created. Therefore, we pre-create JobExecution objects which are all
    essentially full federations with a process for every existing site. For those sites we can create all the comm
    queues at startup.
    """

    def __init__(
        self,
        id: int,
        federation: "FederationRunner",
        sites: list[str],
        inter_job_queue: multiprocessing.Queue,
        logger: Logger,
    ):
        self.id: int = id
        self._federation: "FederationRunner" = federation
        self._inter_job_queue: multiprocessing.Queue = inter_job_queue
        self._logger: Logger = logger
        self._ctx = _spawn_ctx
        # create the input queues for each site
        self._channels: dict[str, multiprocessing.Queue[TokenPackage]] = {
            site: self._ctx.Queue() for site in sites
        }
        self.yield_queue: multiprocessing.Queue[TokenPackage] = self._ctx.Queue()

        # Create worker processes
        from ._job_runner import JobRunner

        self._workers: dict[str, JobRunner] = {}
        for site in sites:
            worker_logger = self._logger.getChild("JobRunner_" + site)
            self._workers[site] = JobRunner(
                site=site,
                channels=self._channels,
                yield_queue=self.yield_queue,
                inter_job_queue=self._inter_job_queue,  # pyright: ignore[reportUnknownMemberType]
                logger=worker_logger,
            )
        # start the workers
        for worker in self._workers.values():
            worker.start()
        self._logger.info(
            f"JobRunner {self.id} started {len(self._workers)} worker processes"
        )

        # the job handle this execution is assigned to
        self._job_handle: "JobHandle | None" = None

    def is_alive(self) -> bool:
        """Returns True if the execution is alive."""
        return all(worker.is_alive() for worker in self._workers.values())

    def get_dead_sites(self) -> list[str]:
        """Returns a list of sites whose worker processes have died."""
        return [site for site, worker in self._workers.items() if not worker.is_alive()]

    def abort_job(self):
        if self._job_handle is not None:
            for worker in self._workers.values():
                worker.abort_netrunner_event.set()

    def shutdown(self):
        """Shuts down the execution. Note that workers are generally waiting for new data on the command queue
        and only have a timeout of 0.5 seconds on the command queue. So we give them 5 seconds to finish."""
        for worker in self._workers.values():
            # try to shut down gracefully
            worker.abort_netrunner_event.set()
            worker.shutdown_event.set()
        for worker in self._workers.values():
            # NetRunner shutdown can take up to 5 seconds per site, so we give it some time.
            worker.join(timeout=5.0)
        for worker in self._workers.values():
            if worker.is_alive():
                self._logger.warning(
                    f"JobRunner {self.id} shutting down worker {worker.pid} but it is still alive. Terminating."
                )
                worker.terminate()
                worker.join(timeout=1.0)

        # Close all multiprocessing queues to avoid leaked semaphores
        for q in self._channels.values():
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
        try:
            self.yield_queue.close()
            self.yield_queue.join_thread()
        except Exception:
            pass
        for worker in self._workers.values():
            try:
                worker.command_queue.close()
                worker.command_queue.join_thread()
                worker.ack_command_queue.close()
                worker.ack_command_queue.join_thread()
            except Exception:
                pass

    def job_id(self) -> str | None:
        """Returns the job ID if this execution is assigned to a job. Otherwise, returns None."""
        if self._job_handle is None:
            return None
        return self._job_handle.job_id

    def handle(self):
        """Returns the job handle if this execution is assigned. Otherwise, returns None."""
        return self._job_handle

    def receive_inter_job_token(self, token: TokenPackage):
        """Called by the FederationRunner background thread that polls the inter job queue.
        Receives an inter-job token and forwards it to the correct queue."""
        assert token.target_job == self.job_id(), (
            f"Token target job {token.target_job} does not match execution job {self.job_id()}"
        )
        self._channels[token.target_site].put(token)

    def handle_was_assigned(self, handle: "JobHandle"):
        """Assign a job handle to this execution."""
        if self._job_handle is not None:
            raise RuntimeError(
                f"JobExecution {self.id} is already assigned to job {self._job_handle.job_id}"
            )
        self._job_handle = handle

    def handle_was_released(self):
        """Called when the job execution completes."""
        self._job_handle = None

    def submit(
        self,
        site: str,
        js: JobSubmission,
    ):
        """Called by the job object when it is started."""
        if not self.is_alive():
            dead_sites = self.get_dead_sites()
            raise RuntimeError(
                f"JobRunner {self.id} is not alive. Workers for sites {dead_sites} have died."
            )

        worker = self._workers[site]
        worker.command_queue.put(js)
        ack = worker.ack_command_queue.get()
        assert ack == js.job_id, f"JobRunner {self.id} received ack for wrong job {ack}"
