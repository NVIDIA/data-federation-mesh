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

from __future__ import annotations

from logging import Logger
from types import ModuleType
from typing import TYPE_CHECKING, Any

from nv_dfm_core.api import NodeId
from nv_dfm_core.api._yield import DISCOVERY_PLACE_NAME
from nv_dfm_core.api.discovery import (
    BranchFieldAdvice,
    SingleFieldAdvice,
)

from ._frame import Frame
from ._helpers import load_site_runtime_module
from ._router import Router

if TYPE_CHECKING:
    from nv_dfm_core.telemetry import SiteTelemetryCollector


class DfmContext:
    def __init__(
        self,
        pipeline_api_version: str,
        federation_name: str,
        homesite: str,
        this_site: str,
        job_id: str,
        router: Router,
        logger: Logger,
    ):
        self._federation_name: str = federation_name
        self._this_site_runtime_module: ModuleType = load_site_runtime_module(
            federation_name, this_site
        )

        # test expected api version
        if (
            not self._this_site_runtime_module.API_VERSION.split(".")[0]
            == pipeline_api_version.split(".")[0]
        ):
            raise ValueError(
                f"API major version mismatch: Execution environment is {self._this_site_runtime_module.API_VERSION} but app sent pipeline with version {pipeline_api_version}"
            )

        self._homesite: str = homesite
        self._this_site: str = this_site
        self._job_id: str = job_id
        self._router: Router = router
        self._logger: Logger = logger

        # for when/if this request is a discovery request
        self._advice: list[
            tuple[NodeId, BranchFieldAdvice | SingleFieldAdvice | None]
        ] = []

        # Telemetry collector (None if telemetry is disabled)
        self._telemetry_collector: SiteTelemetryCollector | None = None

    @property
    def api_version(self) -> str:
        return self._this_site_runtime_module.API_VERSION

    @property
    def federation_name(self) -> str:
        """The name of the federation, must be equal to the name of the federation's root package."""
        return self._federation_name

    @property
    def this_site_runtime_module(self) -> ModuleType:
        """The runtime module is the <federation_name>.fed.runtime.<this_site> module."""
        return self._this_site_runtime_module

    @property
    def this_site(self) -> str:
        return self._this_site

    @property
    def homesite(self) -> str:
        """The site (app) which started the job."""
        return self._homesite

    @property
    def router(self) -> Router:
        return self._router

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def telemetry_collector(self) -> "SiteTelemetryCollector | None":
        """Get the telemetry collector for this context.

        Returns None if telemetry is disabled or not yet initialized.
        """
        return self._telemetry_collector

    @telemetry_collector.setter
    def telemetry_collector(self, value: "SiteTelemetryCollector | None") -> None:
        """Set the telemetry collector for this context."""
        self._telemetry_collector = value

    async def _send_to_place_internal(
        self,
        to_job: str | None,
        to_site: str,
        to_place: str,
        is_yield: bool,
        frame: Frame,
        data: Any,
        node_id: int | str | None,
    ):
        self.logger.info(
            f"Sending data of type {type(data)} to {to_site=} {to_place=}, homesite={self.homesite}"
        )

        await self._router.route_token_async(
            to_job=to_job,
            to_site=to_site,
            to_place=to_place,
            is_yield=is_yield,
            frame=frame,
            data=data,
            node_id=node_id,
        )

    def send_to_place_sync(
        self,
        to_job: str | None,
        to_site: str,
        to_place: str,
        is_yield: bool,
        frame: Frame,
        data: Any,
        node_id: int | str | None = None,
    ):
        self.logger.info(f"Sending to {to_site=} {to_place=}, homesite={self.homesite}")

        self._router.route_token_sync(
            to_job=to_job,
            to_site=to_site,
            to_place=to_place,
            is_yield=is_yield,
            frame=frame,
            data=data,
            node_id=node_id,
        )

    async def send_to_place(
        self,
        to_job: str | None,
        to_site: str,
        to_place: str,
        is_yield: bool,
        frame: Frame,
        data: Any,
        node_id: int | str | None = None,
    ):
        import collections.abc
        import types

        # Helper to check for sync/async generators
        def _is_sync_iterator(obj: Any) -> bool:
            return isinstance(obj, collections.abc.Iterator)

        def _is_async_iterator(obj: Any) -> bool:
            return isinstance(obj, collections.abc.AsyncIterator)

        def _is_sync_generator(obj: Any) -> bool:
            return isinstance(obj, types.GeneratorType)

        def _is_async_generator(obj: Any) -> bool:
            return isinstance(obj, types.AsyncGeneratorType)

        # If data is a sync or async iterator/generator, iterate and send each element
        if _is_async_iterator(data) or _is_async_generator(data):
            self.logger.info(
                f"DfmContext.send_to_place received async iterable or generator: {type(data)}. Sending individual items"
            )
            # for a yield we pretend that the values are returned
            if not is_yield:
                # TODO: think about how to support this.
                # If an adapter returns an iterator and it's not followed by a for-each expression
                # but should be sent, we cannot pickle and send the iterator.
                # Therefore, we would probably need to "drain" the iterator and send an array with the values
                self.logger.error(
                    f"DfmContext.send_to_place received async iterable or generator: {type(data)} for a non-yield place. This is not supported."
                )
            # for a yield, we simply pretend that it's yielded inside a loop
            frame = frame.with_pushed_scope()
            async for item in data:
                await self._send_to_place_internal(
                    to_job=to_job,
                    to_site=to_site,
                    to_place=to_place,
                    is_yield=is_yield,
                    frame=frame,
                    data=item,
                    node_id=node_id,
                )
                frame = frame.with_loop_inc()
            frame = frame.with_popped_scope()
        elif _is_sync_iterator(data) or _is_sync_generator(data):
            self.logger.info(
                f"DfmContext.send_to_place received sync iterable or generator: {type(data)}. Sending individual items"
            )
            if not is_yield:
                # TODO: think about how to support this.
                # If an adapter returns an iterator and it's not followed by a for-each expression
                # but should be sent, we cannot pickle and send the iterator.
                # Therefore, we would probably need to "drain" the iterator and send an array with the values
                self.logger.error(
                    f"DfmContext.send_to_place received sync iterable or generator: {type(data)} for a non-yield place. This is not supported."
                )
            # for a yield, we simply pretend that it's yielded inside a loop
            frame = frame.with_pushed_scope()
            for item in data:
                await self._send_to_place_internal(
                    to_job=to_job,
                    to_site=to_site,
                    to_place=to_place,
                    is_yield=is_yield,
                    frame=frame,
                    data=item,
                    node_id=node_id,
                )
                frame = frame.with_loop_inc()
            frame = frame.with_popped_scope()
        else:
            # send it directly
            await self._send_to_place_internal(
                to_job=to_job,
                to_site=to_site,
                to_place=to_place,
                is_yield=is_yield,
                frame=frame,
                data=data,
                node_id=node_id,
            )

    def add_discovery(
        self, node_id: int | str, advice: BranchFieldAdvice | SingleFieldAdvice | None
    ):
        self._advice.append((NodeId(ident=node_id), advice))

    async def send_discovery(self, frame: Frame):
        advice = self._advice
        self._advice = []
        await self.send_to_place(
            to_job=None,
            to_site=self.homesite,
            to_place=DISCOVERY_PLACE_NAME,
            is_yield=True,
            frame=frame,
            data=advice,
            node_id=None,
        )
