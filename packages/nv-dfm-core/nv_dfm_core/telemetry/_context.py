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

"""Trace context management.

Provides utilities for:
- Generating trace IDs and span IDs
- Deriving trace IDs from job IDs
- Building and recording spans
"""

import hashlib
import secrets
import time
from typing import TYPE_CHECKING, Any

from ._models import TraceContext

if TYPE_CHECKING:
    from ._collector import SiteTelemetryCollector


def generate_trace_id() -> str:
    """Generate a new 128-bit trace ID as a 32-character hex string."""
    return secrets.token_hex(16)


def generate_span_id() -> str:
    """Generate a new 64-bit span ID as a 16-character hex string."""
    return secrets.token_hex(8)


def job_id_to_trace_id(job_id: str) -> str:
    """Derive a valid 32-character trace ID from a job ID.

    This allows using DFM's job_id as the correlation key for distributed
    tracing without needing to propagate a separate trace_id through the system.

    The same job_id always produces the same trace_id (deterministic).

    Args:
        job_id: The DFM job identifier (any string)

    Returns:
        A 32-character hex string suitable for use as an OpenTelemetry trace ID
    """
    return hashlib.md5(job_id.encode()).hexdigest()


def get_current_time_ns() -> int:
    """Get the current time in nanoseconds since epoch."""
    return time.time_ns()


class SpanBuilder:
    """Helper for building and recording spans.

    This is a convenience class that handles timing and context management.

    Example:
        with SpanBuilder("my_operation", collector) as span:
            span.set_attribute("key", "value")
            do_work()
            if error:
                span.set_error("Something went wrong")
    """

    def __init__(
        self,
        name: str,
        collector: "SiteTelemetryCollector",
        trace_id: str,
        site: str = "",
        job_id: str = "",
        parent_span_id: str | None = None,
    ):
        """Initialize a span builder.

        Args:
            name: Name of the span
            collector: The SiteTelemetryCollector to record to
            trace_id: The trace ID for this span (from collector)
            site: Site name for the span
            job_id: Job ID for the span
            parent_span_id: Optional parent span ID for nested spans
        """
        self._name = name
        self._collector = collector
        self._trace_id = trace_id
        self._site = site
        self._job_id = job_id
        self._parent_span_id = parent_span_id
        self._span_id = generate_span_id()
        self._attributes: dict[str, str | int | float | bool] = {}
        self._status = "UNSET"
        self._status_message: str | None = None
        self._start_time_ns: int = 0
        self._end_time_ns: int = 0

        self._context = TraceContext(
            trace_id=self._trace_id,
            span_id=self._span_id,
        )

    @property
    def context(self) -> TraceContext:
        """Get the trace context for this span."""
        return self._context

    @property
    def span_id(self) -> str:
        """Get the span ID."""
        return self._span_id

    def set_attribute(self, key: str, value: str | int | float | bool) -> "SpanBuilder":
        """Set an attribute on the span.

        Args:
            key: Attribute name
            value: Attribute value

        Returns:
            self for chaining
        """
        self._attributes[key] = value
        return self

    def set_attributes(
        self, attributes: dict[str, str | int | float | bool]
    ) -> "SpanBuilder":
        """Set multiple attributes on the span.

        Args:
            attributes: Dict of attribute name to value

        Returns:
            self for chaining
        """
        self._attributes.update(attributes)
        return self

    def set_ok(self) -> "SpanBuilder":
        """Mark the span as successful.

        Returns:
            self for chaining
        """
        self._status = "OK"
        return self

    def set_error(self, message: str | None = None) -> "SpanBuilder":
        """Mark the span as errored.

        Args:
            message: Optional error message

        Returns:
            self for chaining
        """
        self._status = "ERROR"
        self._status_message = message
        return self

    def __enter__(self) -> "SpanBuilder":
        """Start the span."""
        self._start_time_ns = get_current_time_ns()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End the span and record it."""
        self._end_time_ns = get_current_time_ns()

        # Auto-set error status if exception occurred
        if exc_type is not None and self._status == "UNSET":
            self._status = "ERROR"
            self._status_message = str(exc_val) if exc_val else exc_type.__name__

        # Record the span
        if self._collector is not None:
            from ._models import SpanData, SpanStatus

            span = SpanData(
                trace_id=self._trace_id,
                span_id=self._span_id,
                parent_span_id=self._parent_span_id,
                name=self._name,
                start_time_ns=self._start_time_ns,
                end_time_ns=self._end_time_ns,
                status=SpanStatus(self._status),
                status_message=self._status_message,
                attributes=self._attributes,
                site=self._site,
                job_id=self._job_id,
            )
            self._collector.record_span(span)
