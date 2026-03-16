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
import threading
from dataclasses import dataclass
from logging import Logger
from queue import Empty
from typing import Any

from typing_extensions import override

from nv_dfm_core.exec import TokenPackage
from nv_dfm_core.exec._frame import Frame
from nv_dfm_core.exec._job_controller import JobController
from nv_dfm_core.gen.modgen.ir._bound_net_ir import BoundNetIR

from ._context import get_spawn_context
from ._federation_runner import LOCAL_COMPLETION_PLACE
from ._local_router import LocalRouter

# Use explicit spawn context for all multiprocessing objects to avoid fork on Linux.
# Forking is problematic in multi-threaded environments like Omniverse Kit.
# We use a cooperative getter that respects global 'spawn' settings if present.
_spawn_ctx = get_spawn_context()


@dataclass
class JobSubmission:
    pipeline_api_version: str
    federation_name: str
    job_id: str
    homesite: str
    netir: BoundNetIR
    force_modgen: bool


class JobRunner(_spawn_ctx.Process):
    """
    The JobRunner is a worker process that represents the logic running on a dfm site.
    It receives tasks from an input queue, instantiates a target-specific router (passing
    it all the input queues for the other sites so it can route between them), and creates and
    starts the controller.
    """

    def __init__(
        self,
        site: str,
        channels: dict[str, multiprocessing.Queue],  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
        yield_queue: multiprocessing.Queue,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
        inter_job_queue: multiprocessing.Queue,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
        logger: Logger,
    ):
        super().__init__()
        self._ctx = _spawn_ctx
        self._site: str = site
        self.command_queue: multiprocessing.Queue[JobSubmission] = self._ctx.Queue()
        self.ack_command_queue: multiprocessing.Queue[str] = self._ctx.Queue()
        # input queues for all sites, so we can send tokens to them
        self._channels: dict[str, multiprocessing.Queue[TokenPackage]] = channels
        assert self._site in self._channels, f"Site {self._site} not found in channels"

        # special queue for all Yields
        self._yield_queue: multiprocessing.Queue[TokenPackage] = yield_queue
        self._inter_job_queue: multiprocessing.Queue[TokenPackage] = inter_job_queue
        self._logger: Logger = logger

        self._router: LocalRouter | None = None

        self.daemon: bool = True
        self.shutdown_event: Any = self._ctx.Event()
        self.abort_netrunner_event: Any = self._ctx.Event()

        # Background thread for monitoring the input queue
        self._routing_thread: threading.Thread | None = None
        self._router_thread_shutdown_event: threading.Event = threading.Event()

    @override
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Exclude the Event. Otherwise we will get pickle error, since the
        # Event contains a lock.
        del state["_router_thread_shutdown_event"]
        return state

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)
        # Re-create the event on restore
        self._router_thread_shutdown_event = threading.Event()

    def _start_routing_thread(self, router: LocalRouter):
        """Start the background routing thread."""
        if self._routing_thread is None or not self._routing_thread.is_alive():
            self._router_thread_shutdown_event.clear()
            self._routing_thread = threading.Thread(
                target=self._monitor_and_route_input_queue, args=(router,)
            )
            self._routing_thread.daemon = True
            self._routing_thread.start()

    def _stop_routing_thread(self):
        """Stop the background routing thread."""
        if self._routing_thread is not None and self._routing_thread.is_alive():
            self._router_thread_shutdown_event.set()
            self._routing_thread.join(timeout=1.0)
            self._routing_thread = None

    def _monitor_and_route_input_queue(self, router: LocalRouter):
        """Background thread monitors the input queue for this site and passes
        token packages to the router."""
        input_channel = self._channels[self._site]
        assert input_channel is not None, f"Site {self._site} not found in channels"

        while not self._router_thread_shutdown_event.is_set():
            try:
                try:
                    token_package = input_channel.get(timeout=0.1)
                except Empty:
                    continue
                if token_package is None:  # pyright: ignore[reportUnnecessaryComparison]
                    continue
                router.route_token_package_sync(token_package)
            except Exception as e:
                self._logger.error(
                    f"Background monitor thread error in JobRunner for site {self._site}: {e}"
                )
                break

    @override
    def run(self):
        """The main loop of the worker process."""
        self.shutdown_event.clear()
        while not self.shutdown_event.is_set():
            try:
                try:
                    new_job = self.command_queue.get(timeout=0.1)
                except Empty:
                    continue
                if new_job is None:  # pyright: ignore[reportUnnecessaryComparison]
                    continue
                assert new_job.netir.site == self._site, (
                    f"JobRunner for site {self._site} received a netir for wrong site {new_job.netir.site}"
                )

                self._router = LocalRouter(
                    channels=self._channels,
                    inter_job_queue=self._inter_job_queue,
                    yield_queue=self._yield_queue,
                )
                controller = JobController(
                    router=self._router,
                    pipeline_api_version=new_job.pipeline_api_version,
                    federation_name=new_job.federation_name,
                    homesite=new_job.homesite,
                    this_site=self._site,
                    job_id=new_job.job_id,
                    netir=new_job.netir,
                    logger=self._logger,
                    force_modgen=new_job.force_modgen,
                )
                # start the routing thread and the netrunner
                self._start_routing_thread(self._router)
                self.abort_netrunner_event.clear()
                controller.start()
                controller.submit_initial_tokens()
                self.ack_command_queue.put(new_job.job_id)
                controller.wait_for_done(abort_signal=self.abort_netrunner_event)
                controller.shutdown()
                if controller.error_occurred():
                    self._logger.error(
                        f"JobController error occurred during NetRunner execution for job {new_job.job_id}: {controller.get_panic_error()}"
                    )
                    break
                # Always emit a completion marker to the yield queue so the federation can
                # mark job completion even for pipelines with no user yields.
                completion_token = TokenPackage.wrap_data(
                    source_site=self._site,
                    source_node=None,
                    source_job=new_job.job_id,
                    target_site=new_job.homesite,
                    target_place=LOCAL_COMPLETION_PLACE,
                    target_job=new_job.job_id,
                    is_yield=True,
                    frame=Frame.stop_frame(),
                    data=None,
                )
                self._yield_queue.put(completion_token)

            except (KeyboardInterrupt, SystemExit):
                # In local mode, Ctrl+C is commonly used to stop a run. When the signal reaches
                # worker processes, we want to exit cleanly without spewing a traceback.
                self._logger.info(
                    f"JobRunner process for site {self._site} received interrupt; shutting down."
                )
                self.shutdown_event.set()
                break
            except Exception as e:
                import traceback

                tb_str = traceback.format_exc()
                self._logger.error(
                    f"Worker Process PID: {self.pid} for site {self._site}: An error occurred during job execution. I keep running. Error was: {e}\n{tb_str}"
                )
            finally:
                self.abort_netrunner_event.set()
                self._stop_routing_thread()

        # Final cleanup of multiprocessing resources in the site process
        try:
            self.command_queue.close()
            self.command_queue.join_thread()
            self.ack_command_queue.close()
            self.ack_command_queue.join_thread()
            self.shutdown_event.set()
            self.abort_netrunner_event.set()
        except Exception:
            pass
