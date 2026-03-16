#!/usr/bin/env python3
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

# -*- coding: utf-8 -*-

import logging
import os
from types import ModuleType
from typing import Any, Literal

from nv_dfm_core.api import Pipeline, PreparedPipeline
from nv_dfm_core.exec import FlowInfo, Frame, load_site_runtime_module
from nv_dfm_core.gen.irgen import IRGen
from nv_dfm_core.gen.modgen.ir import START_PLACE_NAME
from nv_dfm_core.telemetry import (
    SiteTelemetryCollector,
    TelemetryAggregator,
    create_exporter,
    job_id_to_trace_id,
    telemetry_enabled,
)

from ._callback_dispatcher import (
    CallbackDispatcher,
    CallbackRunner,
    DfmDataCallback,
    DirectDispatcher,
)
from ._job import Job
from ._session_delegate import SessionDelegate


def configure_session_logging(
    force_enable: bool = False,
) -> logging.Logger:
    """Configure logging for DFM Session based on environment variables.

    Args:
        force_enable: If True, logging is enabled regardless of the environment variable.
        add_console_handler: If True, a console handler is added to the logger.
    """
    logger = logging.getLogger(__name__)

    # Check if DFM_SESSION_LOG is set to enable detailed logging
    session_log_enabled = force_enable or os.environ.get(
        "DFM_SESSION_LOG", ""
    ).lower() in (
        "true",
        "1",
        "yes",
    )

    if session_log_enabled and not logger.handlers:
        # Configure logging format and level
        log_level = os.environ.get("DFM_SESSION_LOG_LEVEL", "INFO").upper()

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        logger.info(f"DFM Session logging enabled with level: {log_level}")

    return logger


class Session:
    """Main entry point for executing DFM pipelines in a federation.

    A Session manages the lifecycle of pipeline execution, including preparation,
    execution, and communication with the federation sites.
    """

    def __init__(
        self,
        federation_name: str,
        homesite: str,
        target: Literal["flare", "local"] = "flare",
        logger: logging.Logger | None = None,
        callback_dispatcher: CallbackDispatcher | None = None,
        **kwargs: Any,
    ):
        if not federation_name or not homesite:
            raise ValueError("Federation module and homesite are required")
        self._lazy_runtime_module: ModuleType | None = None
        self._target: Literal["flare", "local"] = target

        self._logger: logging.Logger = logger or configure_session_logging()
        self._delegate: SessionDelegate

        # Default to DirectDispatcher if not specified
        self._callback_dispatcher: CallbackDispatcher = (
            callback_dispatcher or DirectDispatcher()
        )

        if target == "flare":
            from nv_dfm_core.targets.flare import FlareSessionDelegate

            delegate_logger = self._logger.getChild("FlareDelegate")

            self._delegate = FlareSessionDelegate(
                session=self,
                federation_name=federation_name,
                homesite=homesite,
                logger=delegate_logger,
                **kwargs,
            )
        else:
            assert target == "local"
            from nv_dfm_core.targets.local import LocalSessionDelegate

            delegate_logger = self._logger.getChild("LocalDelegate")
            self._delegate = LocalSessionDelegate(
                session=self,
                federation_name=federation_name,
                homesite=homesite,
                logger=delegate_logger,
                **kwargs,
            )

        # Log session initialization
        self._logger.info(
            f"Initializing DFM Session for federation '{federation_name}' with homesite '{homesite}' and target '{target}'"
        )

        # Initialize app-side telemetry collector
        self._telemetry_collector: "SiteTelemetryCollector | None" = None
        self._init_telemetry(homesite)

    def _init_telemetry(self, homesite: str) -> None:
        """Initialize telemetry for app-side tracing."""
        if telemetry_enabled():
            # App-side collector uses "app" as site identifier
            self._telemetry_collector = SiteTelemetryCollector(
                site=f"{homesite}.app",
                job_id="session",  # Updated per-job
            )
            self._logger.debug("App-side telemetry initialized")

    @property
    def callback_dispatcher(self) -> CallbackDispatcher:
        """The callback dispatcher used for all jobs created by this session."""
        return self._callback_dispatcher

    @property
    def runtime_module(self) -> ModuleType:
        """Loading the runtime module can fail, so we do it lazily when we need it
        instead of raising an error in the constructor."""
        if not self._lazy_runtime_module:
            self._logger.debug(
                f"Loading runtime module for federation '{self._delegate.federation_name}' and homesite '{self._delegate.homesite}'"
            )
            try:
                self._lazy_runtime_module = load_site_runtime_module(
                    self._delegate.federation_name, self._delegate.homesite
                )
                self._logger.info(
                    f"Successfully loaded runtime module with API version: {getattr(self._lazy_runtime_module, 'API_VERSION', 'unknown')}"
                )
            except Exception as e:
                self._logger.error(f"Failed to load runtime module: {e}")
                raise
        return self._lazy_runtime_module

    def connect(self, debug: bool = False) -> None:
        self._delegate.connect(debug)

    def close(self, debug: bool = False) -> None:
        # Flush app-side telemetry before closing
        if self._telemetry_collector:
            try:
                # Export app-side spans directly (no need to send to homesite - we ARE the app)
                batch = self._telemetry_collector.flush()
                if not batch.is_empty:
                    exporter = create_exporter()
                    aggregator = TelemetryAggregator(exporter)
                    aggregator.add_batch(batch)
                    aggregator.shutdown()
            except Exception as e:
                self._logger.debug(f"Failed to flush app-side telemetry: {e}")

        self._delegate.close(debug)

    def __enter__(self) -> "Session":
        """Enter the context manager and optionally auto-connect."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
        """Exit the context manager and close the session."""
        self.close()

    def prepare(
        self,
        pipeline: Pipeline,
        restrict_to_sites: Literal["homesite"] | list[str] | None = None,
        debug: bool = False,
    ) -> PreparedPipeline:
        """Prepare a pipeline for execution. This method may optimize the pipeline
        for execution in the whole federation, which can take a bit. Prepared
        pipelines can be reused for multiple executions."""

        if restrict_to_sites == "homesite":
            restrict_to_sites = [self._delegate.homesite]

        self._logger.info(
            f"Preparing pipeline for candidate sites: {restrict_to_sites}"
        )

        # if pipeline doesn't have api_version, use the federation package version
        if pipeline.api_version == "":
            self._logger.debug(
                "Pipeline has no API version, using federation module API version"
            )
            pipeline = pipeline.model_copy(
                update={
                    "api_version": self.runtime_module.API_VERSION,
                }
            )

        # Wrap preparation with telemetry span
        if self._telemetry_collector:
            with self._telemetry_collector.span(
                "session.prepare",
                attributes={
                    "restrict_to_sites": str(restrict_to_sites),
                },
            ) as span:
                try:
                    result = self._prepare_internal(pipeline, restrict_to_sites, debug)
                    span.set_attribute("sites_count", len(result.net_irs()))
                    span.set_ok()
                    return result
                except Exception as e:
                    span.set_error(str(e))
                    raise
        else:
            return self._prepare_internal(pipeline, restrict_to_sites, debug)

    def _prepare_internal(
        self,
        pipeline: Pipeline,
        restrict_to_sites: list[str] | None,
        debug: bool,
    ) -> PreparedPipeline:
        """Internal preparation logic."""
        try:
            fed_info = self._delegate.load_fed_info()

            irgen = IRGen()
            prepared_pipeline = irgen.prepare(
                pipeline=pipeline,
                candidate_sites=restrict_to_sites,
                federation_module_name=self._delegate.federation_name,
                fed_info=fed_info,
                homesite=self._delegate.homesite,
                logger=self._logger,
                debug=debug,
            )

            self._logger.info("Successfully prepared pipeline for execution")
            return prepared_pipeline

        except Exception as e:
            self._logger.error(f"Failed to prepare pipeline: {e}")
            raise

    def debug_show_code(self, pipeline: PreparedPipeline) -> dict[str, str]:
        """Generate and return the Python code for each site in the prepared pipeline for debugging purposes."""
        from nv_dfm_core.gen.modgen import ModGen

        modgen = ModGen()

        result: dict[str, str] = {}

        for site, net_ir in pipeline.net_irs().items():
            mock_module = type(
                "ModuleType",
                (),
                {"__name__": f"{self._delegate.federation_name}.fed.runtime.{site}"},
            )()
            code = modgen._generate_python_code(  # pyright: ignore[reportPrivateUsage]
                this_site_runtime_module=mock_module,  # pyright: ignore[reportArgumentType]
                netir=net_ir,
            )
            result[site] = code

        return result

    def reattach(
        self,
        job_id: str,
        default_callback: DfmDataCallback | None = None,
        place_callbacks: dict[str, DfmDataCallback] | None = None,
    ) -> Job:
        """Reattach to an existing job using its job_id.

        This allows reconnecting to long-running jobs, retrieving status,
        and optionally receiving results that haven't been consumed yet.

        Args:
            job_id: The ID of the job to reattach to.
            default_callback: Optional callback for receiving results from any place.
                             Required if place_callbacks is provided.
            place_callbacks: Optional dict mapping place names to specific callbacks.
                            Can only be provided if default_callback is also provided.

        Returns:
            Job object connected to the existing job. If the job_id is not found,
            returns a job with UNKNOWN status.

        Raises:
            ValueError: If place_callbacks is provided without default_callback.

        Note:
            - If both callbacks are None, returns "informational" job (status/abort only).
            - Multiple clients can reattach to same job (undefined which receives results).
            - For finished jobs, may retrieve buffered results not yet consumed.
            - FLARE: Can reattach to any job (even old/finished ones).
            - Local: Can only reattach to jobs still running in same process.
        """
        if place_callbacks is not None and default_callback is None:
            raise ValueError(
                "default_callback must be provided when place_callbacks is specified"
            )

        self._logger.info(f"Reattaching to job {job_id}")

        # Create callback runner from dispatcher (or None if no callbacks)
        callback_runner: CallbackRunner | None = None
        if default_callback is not None or place_callbacks is not None:
            callback_runner = self._callback_dispatcher.create_runner(
                default_callback=default_callback,
                place_callbacks=place_callbacks,
            )

        return self._delegate.reattach(
            job_id=job_id,
            callback_runner=callback_runner,
        )

    def execute(
        self,
        pipeline: PreparedPipeline,
        input_params: dict[str, Any] | list[dict[str, Any]] | None = {},
        autostop: bool = True,
        default_callback: DfmDataCallback | None = None,
        place_callbacks: dict[str, DfmDataCallback] | None = None,
        debug: bool = False,
        options: Any = None,
        force_modgen: bool = False,
    ) -> Job:
        """
        Execute a pipeline.

        Args:
            pipeline: The pipeline to execute.
            input_params: Default: empty dict.
                          The input parameters for the pipeline. Can be None, a single dict, or
                          a list of dicts.
                          If input_params is None, the pipeline is started but no start signal is sent, the
                          pipeline is waiting for a call to job.send_input_params(). In that case, autostop
                          should probably be False because otherwise the pipeline will be stopped immediately.
                          If input_params is a single dict, it's treated as a list of one dict.
                          The pipeline gets started once for each parameter dict in the input_params list.
                          Each paramset must contain values for all param places in the pipeline.
                          The dict or dicts can be empty, if the pipeline has no PlaceParams.
            autostop: Default True. Whether to automatically send the stop frame to the pipeline after the given input parameters.
                      If False, the pipeline is kept running and additional params can be sent through the Job object. In this
                      case, the stop frame must be sent manually, also through the Job object.
            default_callback: The default callback for the pipeline.
            place_callbacks: The place callbacks for the pipeline.
            force_modgen: If True, modgen will ignore the cache and generate the code again.

        Returns:
            The job object.
        """
        # make sure we have a list of input params
        if isinstance(input_params, dict):
            # single dict, execute pipeline once
            input_params = [dict(input_params)]
        elif input_params is None:
            # no input params, use empty list (waits for later calls to send_input_params())
            input_params = []
            if autostop:
                self._logger.warning(
                    "Autostop is True but input_params is None. The pipeline will be stopped immediately."
                )

        # we want to start the pipeline n times, once for each parameter set in input_params
        # if input_params is an empty list, we don't start the pipeline at all (but may wait
        # for deferred execution through job.send_input_params())

        next_frame = Frame.start_frame(num=0)
        input_params_with_frame: list[tuple[Frame, dict[str, Any]]] = []
        for paramset in input_params:
            # add a start signal to the paramset
            paramset = paramset | {START_PLACE_NAME: FlowInfo()}
            input_params_with_frame.append((next_frame, paramset))
            # multiple parameter sets are essentially an implicit outer loop
            next_frame = next_frame.with_loop_inc()

        # if autostop is true, we also append a final stop frame
        if autostop:
            next_frame = Frame.stop_frame()
            input_params_with_frame.append((next_frame, {START_PLACE_NAME: FlowInfo()}))

        self._logger.info(f"Executing pipeline with target '{self._target}'")
        self._logger.debug(f"Input parameters: {list(input_params_with_frame)}")
        self._logger.debug(f"Default callback?: {'Yes' if default_callback else 'No'}")
        self._logger.debug(
            f"Place callbacks: {list(place_callbacks.keys()) if place_callbacks else 'None'}"
        )

        # first make sure we have all the required input params for each parameter set
        try:
            pipeline.check_input_params(input_params_with_frame)
        except Exception as e:
            self._logger.error(f"Input parameter validation failed: {e}")
            raise

        if default_callback is None:
            # if there's a default callback it will catch all anyways. If not, let's
            # check that all yield places have a callback. Only logs warnings, doesn't raise.
            pipeline.check_callbacks(
                callback_places=list(place_callbacks.keys() if place_callbacks else []),
                logger=self._logger,
            )

        # Create callback runner from dispatcher
        callback_runner = self._callback_dispatcher.create_runner(
            default_callback=default_callback,
            place_callbacks=place_callbacks,
        )

        # Execute the job
        job = self._delegate.execute(
            pipeline=pipeline,
            next_frame=next_frame,
            input_params=input_params_with_frame,
            callback_runner=callback_runner,
            debug=debug,
            options=options,
            force_modgen=force_modgen,
        )

        # Update app-side telemetry to use job_id for trace correlation
        # trace_id is derived from job_id, so app and site spans will match
        if self._telemetry_collector:
            self._telemetry_collector._job_id = job.job_id
            self._telemetry_collector._trace_id = job_id_to_trace_id(job.job_id)
            # Record a span for the job execution
            # (we do this after getting job_id so trace_id matches sites)
            with self._telemetry_collector.span(
                "session.execute",
                attributes={
                    "target": self._target,
                    "job_id": job.job_id,
                    "input_param_count": len(input_params_with_frame),
                    "autostop": autostop,
                    "sites": str(list(pipeline.net_irs().keys())),
                },
            ) as span:
                span.set_ok()
            # Pass collector to job for wait_until_finished tracking
            job._telemetry_collector = self._telemetry_collector  # type: ignore[attr-defined]

        return job
