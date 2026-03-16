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

"""Telemetry data models.

These Pydantic models define the structure for telemetry data (spans and metrics).
They are designed to be:
- Serializable via Pydantic's JSON support (for TokenPackage transport)
- Compatible with OpenTelemetry concepts (for optional OTel export)
- Lightweight with no external dependencies beyond Pydantic
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SpanStatus(str, Enum):
    """Status of a span execution."""

    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class SpanKind(str, Enum):
    """Kind of span, following OpenTelemetry conventions."""

    INTERNAL = "INTERNAL"
    CLIENT = "CLIENT"
    SERVER = "SERVER"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class SpanData(BaseModel):
    """A single span representing a unit of work.

    This model is compatible with OpenTelemetry span concepts but
    does not require the OpenTelemetry SDK.

    Attributes:
        trace_id: 32-character hex string identifying the trace
        span_id: 16-character hex string identifying this span
        parent_span_id: 16-character hex string of parent span (None for root)
        name: Human-readable name describing the operation
        kind: Type of span (internal, client, server, etc.)
        start_time_ns: Start time in nanoseconds since epoch
        end_time_ns: End time in nanoseconds since epoch
        status: Execution status (OK, ERROR, UNSET)
        status_message: Optional message describing the status (especially for errors)
        attributes: Key-value pairs with additional context
        site: The DFM site where this span was recorded
        job_id: The DFM job ID this span belongs to
    """

    trace_id: str = Field(..., min_length=32, max_length=32)
    span_id: str = Field(..., min_length=16, max_length=16)
    parent_span_id: str | None = Field(default=None, min_length=16, max_length=16)
    name: str
    kind: SpanKind = SpanKind.INTERNAL
    start_time_ns: int
    end_time_ns: int
    status: SpanStatus = SpanStatus.UNSET
    status_message: str | None = None
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)
    site: str
    job_id: str

    @property
    def duration_ns(self) -> int:
        """Duration of the span in nanoseconds."""
        return self.end_time_ns - self.start_time_ns

    @property
    def duration_ms(self) -> float:
        """Duration of the span in milliseconds."""
        return self.duration_ns / 1_000_000

    @property
    def duration_s(self) -> float:
        """Duration of the span in seconds."""
        return self.duration_ns / 1_000_000_000


class MetricType(str, Enum):
    """Type of metric."""

    COUNTER = "COUNTER"
    GAUGE = "GAUGE"
    HISTOGRAM = "HISTOGRAM"


class MetricData(BaseModel):
    """A single metric data point.

    Attributes:
        name: Metric name (e.g., "dfm.tokens.routed.total")
        type: Type of metric (counter, gauge, histogram)
        value: The metric value
        timestamp_ns: When the metric was recorded (nanoseconds since epoch)
        attributes: Labels/dimensions for the metric
        site: The DFM site where this metric was recorded
        job_id: The DFM job ID this metric belongs to
    """

    name: str
    type: MetricType
    value: float
    timestamp_ns: int
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)
    site: str
    job_id: str


class TelemetryBatch(BaseModel):
    """A batch of telemetry data from a single site.

    This is the unit of telemetry that gets sent from sites
    to the homesite for aggregation and export.

    Attributes:
        site: The site that generated this batch
        job_id: The job ID this batch belongs to
        spans: List of spans recorded at this site
        metrics: List of metrics recorded at this site
        sequence_num: Sequence number for ordering batches
    """

    site: str
    job_id: str
    spans: list[SpanData] = Field(default_factory=list)
    metrics: list[MetricData] = Field(default_factory=list)
    sequence_num: int = 0

    @property
    def is_empty(self) -> bool:
        """Check if this batch has no data."""
        return len(self.spans) == 0 and len(self.metrics) == 0

    @property
    def span_count(self) -> int:
        """Number of spans in this batch."""
        return len(self.spans)

    @property
    def metric_count(self) -> int:
        """Number of metrics in this batch."""
        return len(self.metrics)


class TraceContext(BaseModel):
    """Trace context for propagation across sites.

    This is a minimal W3C Trace Context compatible structure
    that gets serialized into TokenPackage for cross-site tracing.

    Attributes:
        trace_id: 32-character hex string identifying the trace
        span_id: 16-character hex string of the current span
        trace_flags: Trace flags (01 = sampled)
    """

    trace_id: str = Field(..., min_length=32, max_length=32)
    span_id: str = Field(..., min_length=16, max_length=16)
    trace_flags: str = "01"  # 01 = sampled

    def to_dict(self) -> dict[str, str]:
        """Convert to a dict suitable for TokenPackage.trace_context field."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_flags": self.trace_flags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceContext":
        """Create from a dict (e.g., from TokenPackage.trace_context)."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            trace_flags=data.get("trace_flags", "01"),
        )
