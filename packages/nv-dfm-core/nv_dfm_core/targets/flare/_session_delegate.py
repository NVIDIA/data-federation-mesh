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

# -*- coding: utf-8 -*-

import atexit
import logging
import os
from pathlib import Path
from typing import Any

import appdirs
from nvflare.fuel.flare_api.flare_api import Session as FlareSession
from typing_extensions import override

from nv_dfm_core.api import PreparedPipeline
from nv_dfm_core.exec import Frame
from nv_dfm_core.gen.modgen.ir._net_ir import NetIR
from nv_dfm_core.session import CallbackRunner, Job, SessionDelegate

from ._flare_options import FlareOptions


class FlareSessionDelegate(SessionDelegate):
    def __init__(
        self,
        session: Any,
        federation_name: str,
        homesite: str,
        logger: logging.Logger,
        user: str | None = None,
        flare_workspace: Path | None = None,
        job_workspace: Path | None = None,
        admin_package: Path | None = None,
    ):
        super().__init__(session, federation_name, homesite, logger)

        # Use a path without spaces: Flare submit_job sends path as unquoted string, so
        # spaces break the protocol (APIStatus.ERROR_SYNTAX). Prefer appdirs; fallback
        # to ~/.nv_dfm_core when it contains a space (e.g. macOS "Application Support").
        _state_dir_raw: str = appdirs.user_state_dir("nv_dfm_core") or "~/.nv_dfm_core"
        state_dir = (
            _state_dir_raw
            if " " not in _state_dir_raw
            else str(Path("~/.nv_dfm_core").expanduser())
        )

        self._user: str = user or os.environ.get("DFM_FLARE_USER", "admin@nvidia.com")
        self._flare_workspace: Path = (
            flare_workspace
            or Path(os.environ.get("DFM_FLARE_WORKSPACE", f"{state_dir}/nvflare/poc"))
        ).absolute()
        self._job_workspace: Path = (
            job_workspace
            or Path(
                os.environ.get("DFM_FLARE_JOB_WORKSPACE", f"{state_dir}/dfm_workspace")
            )
        ).absolute()
        self._admin_package: Path = (
            admin_package
            or Path(
                os.environ.get(
                    "DFM_FLARE_ADMIN_PACKAGE", f"{state_dir}/nvflare/poc/{self._user}"
                )
            )
        ).absolute()
        self._flare_session: FlareSession | None = None
        self._cleanup_registered: bool = False

        self._logger.debug(
            f"Session configuration: user={self._user}, flare_workspace={self._flare_workspace},"
            + f" job_workspace={self._job_workspace}, admin_package={self._admin_package}"
        )

    @override
    def connect(self, debug: bool) -> None:
        if self._flare_session is not None:
            self._logger.debug("Flare session already connected")
            return

        self._logger.info(f"Connecting to Flare session for user '{self._user}'")
        self._logger.debug(f"Using admin package: {self._admin_package}")

        try:
            from nvflare.fuel.flare_api.flare_api import new_secure_session

            self._flare_session = new_secure_session(
                self._user,
                self._admin_package.as_posix(),
                debug=debug,
            )
            self._logger.info("Successfully connected to Flare session")

            # Register cleanup on first connect
            if not self._cleanup_registered:
                _ = atexit.register(self._cleanup_on_exit)
                self._cleanup_registered = True
                self._logger.debug("Registered atexit cleanup handler")
        except Exception as e:
            self._logger.error(f"Failed to connect to Flare session: {e}")
            raise

    def _cleanup_on_exit(self) -> None:
        """Ensure cleanup of Flare session even if close() wasn't called.

        This is called automatically by atexit and helps ensure Flare sessions
        are properly closed when the application is terminated via Ctrl-C or
        other signals.
        """
        if self._flare_session is not None:
            try:
                self._logger.info("atexit: Closing Flare session")
                self._flare_session.close()
            except Exception as e:
                # Use print as logger might not be available during shutdown
                print(f"Warning: Error during atexit cleanup of Flare session: {e}")
            finally:
                self._flare_session = None

    @override
    def close(self, debug: bool) -> None:
        if self._flare_session is not None:
            self._flare_session.close()
            self._flare_session = None
            self._logger.info("Successfully closed Flare session")

    @override
    def reattach(
        self,
        job_id: str,
        callback_runner: CallbackRunner | None,
    ) -> Job:
        """Reattach to an existing FLARE job.

        Args:
            job_id: The ID of the job to reattach to
            callback_runner: The callback runner for receiving results, or None for
                            informational-only attachment (status/abort only).

        Returns:
            Job object for the existing FLARE job. If job_id is not found,
            returns a job that will report UNKNOWN status.
        """
        self._logger.info(f"Reattaching to FLARE job {job_id}")

        if not self._flare_session:
            msg = "Flare session not connected"
            self._logger.error(msg)
            raise ValueError(msg)

        # Check if job exists by querying job metadata
        was_found: bool = True
        try:
            job_meta = self._flare_session.get_job_meta(job_id)
            if job_meta is None:
                self._logger.warning(
                    f"Job {job_id} not found in FLARE. Job will report UNKNOWN status."
                )
                was_found = False
        except Exception as e:
            self._logger.warning(
                f"Error checking job {job_id} in FLARE: {e}. Job may report UNKNOWN status."
            )
            was_found = False

        try:
            from ._job import Job as FlareJob

            # Create job object - FlareJob will handle querying status and polling results
            # We don't need the original pipeline/next_frame for reattachment
            job = FlareJob(
                pipeline=None,  # Not needed for reattachment
                next_frame=None,  # Not sending new data
                flare_session=self._flare_session,
                job_id=job_id,
                runtime_module=self.session.runtime_module,
                homesite=self._homesite,
                was_found=was_found,
                logger=self._logger,
                callback_runner=callback_runner,
            )

            self._logger.info(f"Successfully reattached to FLARE job {job_id}")
            return job

        except Exception as e:
            self._logger.error(f"Failed to reattach to FLARE job {job_id}: {e}")
            raise

    def provision_only(
        self,
        pipeline: PreparedPipeline,
        input_params: list[tuple[Frame, dict[str, Any]]],
        options: FlareOptions,
        force_modgen: bool = False,
    ) -> Path:
        self._logger.info("Provisioning pipeline for target 'flare'")
        if len(input_params) > 0:
            self._logger.debug(f"Input parameters: {list(input_params[0][1].keys())}")

        try:
            from nv_dfm_core.targets.flare import FlareApp

            pipeline.check_input_params(input_params)
            app = FlareApp(
                pipeline=pipeline,
                input_params=input_params,
                options=options,
                force_modgen=force_modgen,
            )
            job_path = app.export(self._job_workspace)

            self._logger.info(f"Successfully provisioned Flare job to: {job_path}")
            return job_path

        except Exception as e:
            self._logger.error(f"Failed to provision Flare job: {e}")
            raise

    @override
    def execute(
        self,
        pipeline: PreparedPipeline,
        next_frame: Frame,
        input_params: list[tuple[Frame, dict[str, Any]]],
        callback_runner: CallbackRunner,
        debug: bool,
        options: FlareOptions | None = None,
        force_modgen: bool = False,
    ) -> Job:
        """
        Execute a pipeline.

        Args:
            pipeline: The pipeline to execute.
            input_params: The input parameters for the pipeline. Can be a single set or parameters
                          or a sequence of parameters which will be submitted in sequence.
            callback_runner: The callback runner for receiving results.

        Returns:
            The job object.
        """
        self._logger.info("Starting Flare job execution")

        if not self._flare_session:
            msg = "Flare session not connected"
            self._logger.error(msg)
            raise ValueError(msg)

        if options is None:
            options = FlareOptions()

        if not isinstance(options, FlareOptions):
            self._logger.error("Options are not of type FlareOptions")
            raise ValueError("Options are not of type FlareOptions")

        if "server" not in pipeline.net_irs():
            self._logger.info(
                "No net IR for the flare server found in the pipeline. Creating an empty one."
            )
            pipeline.net_irs()["server"] = NetIR(
                pipeline_name=pipeline.pipeline_name,
                site="server",
                transitions=[],
            )

        try:
            from ._job import Job as FlareJob

            if pipeline.homesite in pipeline.net_irs():
                raise ValueError(
                    f"Homesite '{pipeline.homesite}' is in the net_irs of the pipeline with a flare target. Mixed homesite-flare pipelines are not yet implemented."
                )

            job_dir = self.provision_only(
                pipeline=pipeline,
                input_params=input_params,
                options=options,
                force_modgen=force_modgen,
            )
            self._logger.debug(f"Submitting job from directory: {job_dir}")

            flare_job_id = self._flare_session.submit_job(job_dir.as_posix())
            self._logger.info(f"Submitted Flare job with ID: {flare_job_id}")

            job = FlareJob(
                pipeline=pipeline,
                next_frame=next_frame,
                flare_session=self._flare_session,
                job_id=flare_job_id,
                runtime_module=self.session.runtime_module,
                homesite=self._homesite,
                logger=self._logger,
                callback_runner=callback_runner,
            )

            self._logger.info("Successfully created Flare job")
            return job

        except Exception as e:
            self._logger.error(f"Failed to execute Flare job: {e}")
            raise
