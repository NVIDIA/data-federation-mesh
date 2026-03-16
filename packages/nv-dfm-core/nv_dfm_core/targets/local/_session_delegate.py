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

import atexit
import logging
from typing import Any

from typing_extensions import override

from nv_dfm_core.api import PreparedPipeline
from nv_dfm_core.exec import Frame
from nv_dfm_core.gen.modgen.ir._bound_net_ir import BoundNetIR
from nv_dfm_core.session import CallbackRunner, Job, SessionDelegate

from ._federation_runner import FederationRunner


class LocalSessionDelegate(SessionDelegate):
    def __init__(
        self,
        session: Any,
        federation_name: str,
        homesite: str,
        logger: logging.Logger,
        max_concurrent_jobs: int = 4,
        sites: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(session, federation_name, homesite, logger)
        self._federation: FederationRunner | None = None
        self._max_concurrent_jobs: int = max_concurrent_jobs
        self._sites: list[str] | None = sites
        self._cleanup_registered: bool = False

    @override
    def connect(self, debug: bool) -> None:
        if self._federation:
            return

        sites = self._sites
        # if we didn't get an explicit list of sites (makes it easier to test),
        # we load the fed_info.json
        if sites is None:
            fed_info = self.load_fed_info()
            sites = list(fed_info.sites.keys())

        self._logger.info(f"Creating local federation with sites: {sites}")

        federation_logger = self._logger.getChild("FederationRunner")
        self._federation = FederationRunner(
            max_concurrent_jobs=self._max_concurrent_jobs,
            sites=sites,
            federation_name=self._federation_name,
            homesite=self._homesite,
            logger=federation_logger,
        )

        # Register cleanup on first connect
        if not self._cleanup_registered:
            _ = atexit.register(self._cleanup_on_exit)
            self._cleanup_registered = True
            self._logger.debug("Registered atexit cleanup handler")

    def _cleanup_on_exit(self) -> None:
        """Ensure cleanup of federation processes even if close() wasn't called.

        This is called automatically by atexit and helps prevent zombie processes
        when the application is terminated via Ctrl-C or other signals.
        """
        if self._federation:
            try:
                self._logger.info(
                    "atexit: Cleaning up local federation processes to prevent zombies"
                )
                self._federation.shutdown()
            except Exception as e:
                # Use print as logger might not be available during shutdown
                print(f"Warning: Error during atexit cleanup of local federation: {e}")
            finally:
                self._federation = None

    @override
    def close(self, debug: bool) -> None:
        if self._federation:
            self._federation.shutdown()
        self._federation = None

    @override
    def reattach(
        self,
        job_id: str,
        callback_runner: CallbackRunner | None,
    ) -> Job:
        """Reattach to an existing local job (within the same process + same connected Session only).

        Args:
            job_id: The ID of the job to reattach to
            callback_runner: The callback runner for receiving results, or None for
                            informational-only attachment (status/abort only).

        Returns:
            Job object for the existing local job. If job_id is not known to this
            local federation process, returns a job with UNKNOWN status.

        Note:
            In local mode, jobs are tracked in-memory by the `FederationRunner` instance created by
            this Session's `connect()`. Reattachment therefore requires:
              - the same OS process, AND
              - the same connected Session (i.e., the same `FederationRunner` instance).
            Reattachment works for both running and finished jobs for the lifetime of that runner.
        """
        if not self._federation:
            raise ValueError(
                "Local federation not started. call Session.connect() first."
            )

        self._logger.info(f"Attempting to reattach to local job {job_id}")

        # Get handle from the federation registry (works for running and finished jobs)
        handle = self._federation.get_job_handle_any(job_id)

        from ._local_job import LocalJob

        if handle is None:
            self._logger.warning(
                f"Job {job_id} not found in this local federation instance. "
                + "Local jobs only persist within the same process AND the same connected Session. "
                + "Returning stub job with UNKNOWN status."
            )
            # Create stub job with UNKNOWN status
            stub_job = LocalJob.create_stub(
                job_id=job_id,
                homesite=self._homesite,
                logger=self._logger,
                callback_runner=callback_runner,
            )
            return stub_job

        # Create fresh LocalJob from handle
        job = LocalJob.create_from_handle(
            handle=handle,
            federation=self._federation,
            logger=self._logger,
            callback_runner=callback_runner,
        )

        self._logger.info(f"Successfully reattached to local job {job_id}")
        return job

    @override
    def execute(
        self,
        pipeline: PreparedPipeline,
        next_frame: Frame,
        input_params: list[tuple[Frame, dict[str, Any]]],
        callback_runner: CallbackRunner,
        debug: bool,
        options: Any | None = None,
        force_modgen: bool = False,
    ) -> Job:
        if not self._federation:
            raise ValueError(
                "Local federation not started. call Session.connect() first."
            )

        bound_net_irs: dict[str, BoundNetIR] = pipeline.bind_net_irs(
            input_params=input_params
        )

        self._logger.info("Submitting local job")
        job = self._federation.submit(
            pipeline=pipeline,
            next_frame=next_frame,
            netirs=bound_net_irs,
            force_modgen=force_modgen,
            callback_runner=callback_runner,
        )
        self._logger.info("Successfully submitted local job")
        return job
