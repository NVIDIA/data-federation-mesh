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

import logging
from abc import ABC, abstractmethod
from typing import Any

from nv_dfm_core.api import PreparedPipeline
from nv_dfm_core.exec import Frame
from nv_dfm_core.exec._helpers import load_fed_runtime_module
from nv_dfm_core.gen.irgen._fed_info import FedInfo
from nv_dfm_core.gen.irgen._helpers import load_fed_info_json

from ._callback_dispatcher import CallbackRunner
from ._job import Job


class SessionDelegate(ABC):
    """Abstract base class for session delegates that handle different execution targets.

    A delegate implements the specific logic for executing pipelines on different
    backends (e.g., FLARE, local execution).
    """

    def __init__(
        self,
        session: Any,
        federation_name: str,
        homesite: str,
        logger: logging.Logger,
    ):
        self._session: Any = session
        self._federation_name: str = federation_name
        self._homesite: str = homesite
        self._logger: logging.Logger = logger

    @property
    def session(self) -> Any:
        return self._session

    @property
    def federation_name(self) -> str:
        return self._federation_name

    @property
    def homesite(self) -> str:
        return self._homesite

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def load_fed_info(self) -> FedInfo:
        """Load federation information from the federation runtime module."""
        # by default we load the fed_info.json from the fed runtime module
        fed_runtime_module = load_fed_runtime_module(self._federation_name)
        self._logger.info(
            f"Loading fed_info.json as resource from fed runtime module: {fed_runtime_module.__name__}"
        )
        fed_info = load_fed_info_json(fed_runtime_module)
        return fed_info

    @abstractmethod
    def connect(self, debug: bool) -> None: ...

    @abstractmethod
    def close(self, debug: bool) -> None: ...

    @abstractmethod
    def reattach(
        self,
        job_id: str,
        callback_runner: CallbackRunner | None,
    ) -> Job:
        """Reattach to an existing job.

        Args:
            job_id: The ID of the job to reattach to.
            callback_runner: The callback runner for receiving results, or None for
                            informational-only attachment (status/abort only).

        Returns:
            Job object connected to the existing job.
        """
        ...

    @abstractmethod
    def execute(
        self,
        pipeline: PreparedPipeline,
        next_frame: Frame,
        input_params: list[tuple[Frame, dict[str, Any]]],
        callback_runner: CallbackRunner,
        debug: bool,
        options: Any = None,
        force_modgen: bool = False,
    ) -> Job:
        """Execute a prepared pipeline.

        Args:
            pipeline: The prepared pipeline to execute.
            next_frame: The next frame for the pipeline.
            input_params: List of (frame, params) tuples for input.
            callback_runner: The callback runner for receiving results.
            debug: Whether to run in debug mode.
            options: Target-specific options.
            force_modgen: If True, force code regeneration.

        Returns:
            Job object for the execution.
        """
        ...
