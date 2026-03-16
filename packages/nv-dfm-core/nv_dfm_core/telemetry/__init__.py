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

"""DFM Telemetry Module.

This module provides observability for DFM pipeline execution through
distributed tracing and metrics collection. Telemetry is:

- **Opt-in**: Disabled by default, enabled via DFM_TELEMETRY_ENABLED=true
- **Target-agnostic**: Works with both Flare and Local execution targets
- **Lightweight**: No external dependencies beyond Pydantic
- **Federated**: Telemetry flows through existing token routing infrastructure

Quick Start:
    # Enable telemetry via environment variable
    export DFM_TELEMETRY_ENABLED=true

    # Or programmatically check status
    from nv_dfm_core.telemetry import telemetry_enabled
    if telemetry_enabled():
        print("Telemetry is active")

Configuration (Environment Variables):
    DFM_TELEMETRY_ENABLED: Enable telemetry (default: false)
    DFM_TELEMETRY_SERVICE_NAME: Service name for spans (default: dfm)
    DFM_TELEMETRY_EXPORTER: Output type - console, file, none (default: console)
    DFM_TELEMETRY_FILE_PATH: File path for file exporter
    DFM_TELEMETRY_INCLUDE_ATTRIBUTES: Include detailed attributes (default: true)
    DFM_TELEMETRY_SAMPLE_RATE: Trace sampling rate 0.0-1.0 (default: 1.0)

Example Usage (for DFM internals):
    from nv_dfm_core.telemetry import create_collector, create_exporter, TelemetryAggregator

    # At a site (e.g., in NetRunner)
    collector = create_collector(site="datacenter", job_id="job123")

    with collector.span("process_operation") as span:
        span.set_attribute("input_size", 1000)
        result = do_work()
        span.set_ok()

    # Flush and send to homesite
    batch = collector.flush()

Note: TYPE_CHECKING imports are used to avoid circular imports while
maintaining type safety for DfmContext and SpanBuilder references.

    # At homesite (e.g., in Session)
    exporter = create_exporter()
    aggregator = TelemetryAggregator(exporter)
    aggregator.add_batch(batch)  # Exports to configured backend
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from nv_dfm_core.exec import DfmContext

# Configuration
# Collection
from ._collector import (
    FlushCallback,
    NoOpCollector,
    SiteTelemetryCollector,
    create_collector,
)
from ._config import (
    ExporterType,
    TelemetryConfig,
    get_telemetry_config,
    reset_config_cache,
    telemetry_enabled,
)

# Context management
from ._context import (
    SpanBuilder,
    generate_span_id,
    generate_trace_id,
    get_current_time_ns,
    job_id_to_trace_id,
)

# Export
from ._exporter import (
    ConsoleExporter,
    FileExporter,
    NoOpExporter,
    TelemetryAggregator,
    TelemetryExporter,
    create_exporter,
)

# Data models
from ._models import (
    MetricData,
    MetricType,
    SpanData,
    SpanKind,
    SpanStatus,
    TelemetryBatch,
    TraceContext,
)

# Reserved place name for telemetry tokens
TELEMETRY_PLACE_NAME = "__telemetry__"


@contextmanager
def adapter_span(
    dfm_context: "DfmContext", adapter_name: str, operation_name: str
) -> Generator[SpanBuilder | None, None, None]:
    """Context manager for instrumenting adapter calls in generated code.

    This is used by the generated ThisSite code to wrap adapter calls
    with telemetry spans.

    Args:
        dfm_context: The DfmContext object (has telemetry_collector property)
        adapter_name: The adapter class name (e.g., "GreetMeAdapter")
        operation_name: The operation name (e.g., "users.GreetMe")

    Yields:
        SpanBuilder if telemetry is enabled, None otherwise

    Example (in generated code):
        with dfm.telemetry.adapter_span(self.dfm_context, "GreetMeAdapter", "users.GreetMe") as span:
            result = self.adapter_GreetMeAdapter.body(**kwargs)
            if span:
                span.set_ok()
            return result
    """
    if not telemetry_enabled():
        yield None
        return

    collector = dfm_context.telemetry_collector
    if collector is None:
        yield None
        return

    with collector.span(
        f"adapter.{operation_name}",
        attributes={
            "adapter.name": adapter_name,
            "adapter.operation": operation_name,
        },
    ) as span:
        yield span


__all__ = [
    # Configuration
    "ExporterType",
    "TelemetryConfig",
    "get_telemetry_config",
    "reset_config_cache",
    "telemetry_enabled",
    # Data models
    "MetricData",
    "MetricType",
    "SpanData",
    "SpanKind",
    "SpanStatus",
    "TelemetryBatch",
    "TraceContext",
    # Context management
    "SpanBuilder",
    "generate_span_id",
    "generate_trace_id",
    "get_current_time_ns",
    "job_id_to_trace_id",
    # Collection
    "FlushCallback",
    "NoOpCollector",
    "SiteTelemetryCollector",
    "create_collector",
    # Export
    "ConsoleExporter",
    "FileExporter",
    "NoOpExporter",
    "TelemetryAggregator",
    "TelemetryExporter",
    "create_exporter",
    # Constants
    "TELEMETRY_PLACE_NAME",
    # Helper for generated code
    "adapter_span",
]
