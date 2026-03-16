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

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

from ._frame import Frame
from ._token_package import TokenPackage

if TYPE_CHECKING:
    from nv_dfm_core.telemetry import SiteTelemetryCollector, SpanBuilder

    from ._net_runner import NetRunner
else:
    NetRunner = object


@contextmanager
def _routing_span(
    collector: "SiteTelemetryCollector | None",
    from_site: str,
    to_site: str,
    to_place: str,
    is_yield: bool,
    is_cross_site: bool,
) -> Generator["SpanBuilder | None", None, None]:
    """Context manager for routing telemetry span."""
    if collector is None:
        yield None
        return

    span_name = "route.cross_site" if is_cross_site else "route.yield"
    attributes = {
        "route.from_site": from_site,
        "route.to_site": to_site,
        "route.to_place": to_place,
        "route.is_yield": is_yield,
        "route.is_cross_site": is_cross_site,
    }

    with collector.span(span_name, attributes=attributes) as span:
        yield span
        span.set_ok()


def _get_trace_context_for_injection() -> dict[str, str] | None:
    """Get trace context for injection into outgoing tokens.

    Note: Since trace_id is derived from job_id, we don't need to propagate
    trace context through tokens anymore. This function returns None.
    """
    return None


def _restore_trace_context(
    trace_context: dict[str, str] | None,
) -> None:
    """Restore trace context from incoming token.

    Note: Since trace_id is derived from job_id, we don't need to restore
    trace context from tokens. This is a no-op.
    """
    pass


class Router(ABC):
    def __init__(self):
        self._netrunner: NetRunner | None = None

    def set_netrunner(self, netrunner: NetRunner):
        self._netrunner = netrunner

    @property
    def netrunner(self) -> NetRunner:
        assert self._netrunner is not None, "NetRunner must be set before routing"
        return self._netrunner

    @property
    def job_id(self) -> str:
        assert self._netrunner is not None, "NetRunner must be set before routing"
        return self._netrunner.dfm_context.job_id

    @property
    def this_site(self) -> str:
        assert self._netrunner is not None, "NetRunner must be set before routing"
        return self._netrunner.dfm_context.this_site

    @property
    def homesite(self) -> str:
        assert self._netrunner is not None, "NetRunner must be set before routing"
        return self._netrunner.dfm_context.homesite

    def _get_collector(self) -> "SiteTelemetryCollector | None":
        """Get telemetry collector from the netrunner's context."""
        if self._netrunner is None:
            return None
        return self._netrunner.dfm_context.telemetry_collector

    async def route_token_async(
        self,
        to_job: str | None,
        to_site: str,
        to_place: str,
        is_yield: bool,
        frame: Frame,
        data: Any,
        node_id: int | str | None,
    ):
        """A routing method that only creates a TokenPackage when needed, avoiding serialization for local tokens."""

        if to_job is None:
            to_job = self.job_id

        if not is_yield and to_job == self.job_id and to_site == self.this_site:
            # this is a local token, we don't need to serialize it
            with self.netrunner.receive_token_transaction() as transaction:
                transaction.receive_token(place=to_place, frame=frame, data=data)
        else:
            # this is a remote token, we need to serialize it
            is_cross_site = to_site != self.this_site

            # Record routing span for cross-site or yield tokens
            collector = self._get_collector()
            with _routing_span(
                collector,
                from_site=self.this_site,
                to_site=to_site,
                to_place=to_place,
                is_yield=is_yield,
                is_cross_site=is_cross_site,
            ):
                # Inject trace context for distributed tracing
                trace_context = _get_trace_context_for_injection()
                token_package = TokenPackage.wrap_data(
                    source_site=self.this_site,
                    source_node=node_id,
                    source_job=self.job_id,
                    target_site=to_site,
                    target_place=to_place,
                    target_job=to_job,
                    is_yield=is_yield,
                    frame=frame,
                    data=data,
                    trace_context=trace_context,
                )
                if to_job != self.job_id:
                    await self._send_other_job_remote_token_package_async(token_package)
                elif is_yield:
                    await self._send_this_job_yield_token_package_async(token_package)
                else:
                    await self._send_this_job_remote_token_package_async(token_package)

    def route_token_sync(
        self,
        to_job: str | None,
        to_site: str,
        to_place: str,
        is_yield: bool,
        frame: Frame,
        data: Any,
        node_id: int | str | None,
    ):
        """A routing method that only creates a TokenPackage when needed, avoiding serialization for local tokens."""

        if to_job is None:
            to_job = self.job_id

        if not is_yield and to_job == self.job_id and to_site == self.this_site:
            # this is a local token, we don't need to serialize it
            with self.netrunner.receive_token_transaction() as transaction:
                transaction.receive_token(place=to_place, frame=frame, data=data)
        else:
            # this is a remote token, we need to serialize it
            is_cross_site = to_site != self.this_site

            # Record routing span for cross-site or yield tokens
            collector = self._get_collector()
            with _routing_span(
                collector,
                from_site=self.this_site,
                to_site=to_site,
                to_place=to_place,
                is_yield=is_yield,
                is_cross_site=is_cross_site,
            ):
                # Inject trace context for distributed tracing
                trace_context = _get_trace_context_for_injection()
                token_package = TokenPackage.wrap_data(
                    source_site=self.this_site,
                    source_node=node_id,
                    source_job=self.job_id,
                    target_site=to_site,
                    target_place=to_place,
                    target_job=to_job,
                    is_yield=is_yield,
                    frame=frame,
                    data=data,
                    trace_context=trace_context,
                )
                if to_job != self.job_id:
                    self._send_other_job_remote_token_package_sync(token_package)
                elif is_yield:
                    self._send_this_job_yield_token_package_sync(token_package)
                else:
                    self._send_this_job_remote_token_package_sync(token_package)

    async def route_token_package_async(self, token_package: TokenPackage):
        # Extract trace context from incoming token for distributed tracing
        _restore_trace_context(token_package.trace_context)

        if (
            not token_package.is_yield
            and token_package.target_job == self.job_id
            and token_package.target_site == self.this_site
        ):
            # this token is for us
            with self.netrunner.receive_token_transaction() as transaction:
                transaction.receive_token(
                    place=token_package.target_place,
                    frame=token_package.frame,
                    data=token_package.unwrap_data(),
                )
        elif token_package.target_job != self.job_id:
            await self._send_other_job_remote_token_package_async(token_package)
        elif token_package.is_yield:
            await self._send_this_job_yield_token_package_async(token_package)
        else:
            # this token is for another job, we need to send it to the other job
            await self._send_this_job_remote_token_package_async(token_package)

    def route_token_package_sync(self, token_package: TokenPackage):
        # Extract trace context from incoming token for distributed tracing
        _restore_trace_context(token_package.trace_context)

        if (
            not token_package.is_yield
            and token_package.target_job == self.job_id
            and token_package.target_site == self.this_site
        ):
            # this token is for us
            with self.netrunner.receive_token_transaction() as transaction:
                transaction.receive_token(
                    place=token_package.target_place,
                    frame=token_package.frame,
                    data=token_package.unwrap_data(),
                )
        elif token_package.target_job != self.job_id:
            self._send_other_job_remote_token_package_sync(token_package)
        elif token_package.is_yield:
            self._send_this_job_yield_token_package_sync(token_package)
        else:
            # this token is for another job, we need to send it to the other job
            self._send_this_job_remote_token_package_sync(token_package)

    async def _send_this_job_yield_token_package_async(
        self, token_package: TokenPackage
    ):
        """Override this method too if there's a real async implementation. Otherwise it will just call the sync version."""
        self._send_this_job_yield_token_package_sync(token_package)

    async def _send_this_job_remote_token_package_async(
        self, token_package: TokenPackage
    ):
        """Override this method too if there's a real async implementation. Otherwise it will just call the sync version."""
        self._send_this_job_remote_token_package_sync(token_package)

    async def _send_other_job_remote_token_package_async(
        self, token_package: TokenPackage
    ):
        """Override this method too if there's a real async implementation. Otherwise it will just call the sync version."""
        self._send_other_job_remote_token_package_sync(token_package)

    @abstractmethod
    def _send_this_job_yield_token_package_sync(self, token_package: TokenPackage): ...

    @abstractmethod
    def _send_this_job_remote_token_package_sync(self, token_package: TokenPackage): ...

    @abstractmethod
    def _send_other_job_remote_token_package_sync(
        self, token_package: TokenPackage
    ): ...
