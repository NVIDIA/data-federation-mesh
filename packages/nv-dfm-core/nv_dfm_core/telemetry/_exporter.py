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

"""Telemetry exporters for outputting collected data.

Exporters are responsible for outputting telemetry data to various backends.
The default implementation exports to console (stdout) in a human-readable format.
"""

import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import TextIO

from ._config import ExporterType, get_telemetry_config
from ._models import MetricData, SpanData, TelemetryBatch


class TelemetryExporter(ABC):
    """Abstract base class for telemetry exporters.

    Subclasses must implement the export_batch method to output
    telemetry data to their specific backend.
    """

    @abstractmethod
    def export_batch(self, batch: TelemetryBatch) -> None:
        """Export a batch of telemetry data.

        Args:
            batch: The telemetry batch to export
        """
        pass

    @abstractmethod
    def export_span(self, span: SpanData) -> None:
        """Export a single span.

        Args:
            span: The span to export
        """
        pass

    @abstractmethod
    def export_metric(self, metric: MetricData) -> None:
        """Export a single metric.

        Args:
            metric: The metric to export
        """
        pass

    def shutdown(self) -> None:
        """Clean up any resources used by the exporter.

        Called when the exporter is no longer needed.
        """
        pass


class ConsoleExporter(TelemetryExporter):
    """Exports telemetry to console (stdout) in a human-readable format.

    This is the default exporter and requires no external dependencies.
    """

    def __init__(
        self,
        output: TextIO | None = None,
        include_attributes: bool = True,
        colorize: bool = True,
    ):
        """Initialize the console exporter.

        Args:
            output: Output stream (defaults to sys.stdout)
            include_attributes: Whether to include span/metric attributes
            colorize: Whether to use ANSI colors (auto-detected if None)
        """
        self._output = output or sys.stdout
        self._include_attributes = include_attributes
        self._colorize = (
            colorize and hasattr(self._output, "isatty") and self._output.isatty()
        )

    def _format_timestamp(self, ns: int) -> str:
        """Format a nanosecond timestamp to human-readable."""
        dt = datetime.fromtimestamp(ns / 1_000_000_000)
        return dt.strftime("%H:%M:%S.%f")[:-3]  # Trim to milliseconds

    def _format_duration(self, ns: int) -> str:
        """Format duration in a human-readable way."""
        if ns < 1_000:
            return f"{ns}ns"
        elif ns < 1_000_000:
            return f"{ns / 1_000:.2f}μs"
        elif ns < 1_000_000_000:
            return f"{ns / 1_000_000:.2f}ms"
        else:
            return f"{ns / 1_000_000_000:.2f}s"

    def _color(self, text: str, color_code: str) -> str:
        """Apply ANSI color if colorization is enabled."""
        if not self._colorize:
            return text
        return f"\033[{color_code}m{text}\033[0m"

    def _status_color(self, status: str) -> str:
        """Get color code for a span status."""
        if status == "OK":
            return "32"  # Green
        elif status == "ERROR":
            return "31"  # Red
        return "33"  # Yellow for UNSET

    def export_span(self, span: SpanData) -> None:
        """Export a single span to console."""
        status_color = self._status_color(span.status)
        status_text = self._color(f"[{span.status}]", status_color)

        duration = self._format_duration(span.duration_ns)
        timestamp = self._format_timestamp(span.start_time_ns)

        # Build the main line
        trace_short = span.trace_id[:8]
        parent_info = (
            f" parent={span.parent_span_id[:8]}" if span.parent_span_id else ""
        )
        line = (
            f"{timestamp} {self._color('SPAN', '36')} "
            f"trace={trace_short} span={span.span_id[:8]}{parent_info} "
            f"site={self._color(span.site, '35')} "
            f"{self._color(span.name, '1')} "
            f"{status_text} {duration}"
        )

        self._output.write(line + "\n")

        # Optionally include attributes
        if self._include_attributes and span.attributes:
            attrs_str = " ".join(f"{k}={v}" for k, v in span.attributes.items())
            self._output.write(f"         attrs: {attrs_str}\n")

        # Include error message if present
        if span.status_message:
            self._output.write(
                f"         {self._color('error:', '31')} {span.status_message}\n"
            )

    def export_metric(self, metric: MetricData) -> None:
        """Export a single metric to console."""
        timestamp = self._format_timestamp(metric.timestamp_ns)

        # Format value based on type
        type_str = metric.type.value.lower()
        value_str = (
            f"{metric.value:.2f}"
            if isinstance(metric.value, float)
            else str(metric.value)
        )

        line = (
            f"{timestamp} {self._color('METRIC', '33')} "
            f"site={self._color(metric.site, '35')} "
            f"{metric.name} ({type_str}) = {value_str}"
        )

        self._output.write(line + "\n")

        # Optionally include attributes
        if self._include_attributes and metric.attributes:
            attrs_str = " ".join(f"{k}={v}" for k, v in metric.attributes.items())
            self._output.write(f"         labels: {attrs_str}\n")

    def export_batch(self, batch: TelemetryBatch) -> None:
        """Export a batch of telemetry to console."""
        if batch.is_empty:
            return

        # Calculate summary statistics
        summary = self._calculate_batch_summary(batch)

        # Header for batch with summary
        self._output.write(
            f"\n{self._color('=== Telemetry Batch ===', '1')} "
            f"site={self._color(batch.site, '35')} job={batch.job_id[:8]}\n"
        )

        # Summary line
        self._output.write(
            f"    Duration: {self._color(summary['total_duration'], '36')} | "
            f"Transitions: {summary['transition_count']} | "
            f"Adapters: {summary['adapter_count']} | "
            f"Cross-site: {summary['cross_site_count']}\n"
        )

        # Show adapters executed
        if summary["adapters"]:
            adapter_summary = ", ".join(
                f"{name}({count})" for name, count in summary["adapters"].items()
            )
            self._output.write(f"    Adapters: {adapter_summary}\n")

        # Show transitions grouped by type
        if summary["transitions"]:
            trans_summary = ", ".join(
                f"{name}({count})" for name, count in summary["transitions"].items()
            )
            self._output.write(f"    Transitions: {trans_summary}\n")

        self._output.write("\n")

        # Export individual spans
        for span in batch.spans:
            self.export_span(span)

        # Export metrics
        for metric in batch.metrics:
            self.export_metric(metric)

        self._output.write("\n")
        self._output.flush()

    def _calculate_batch_summary(self, batch: TelemetryBatch) -> dict:
        """Calculate summary statistics from a batch of spans."""
        summary: dict = {
            "total_duration": "N/A",
            "transition_count": 0,
            "adapter_count": 0,
            "cross_site_count": 0,
            "yield_count": 0,
            "transitions": {},
            "adapters": {},
        }

        if not batch.spans:
            return summary

        # Find time range
        min_start = min(s.start_time_ns for s in batch.spans)
        max_end = max(s.end_time_ns for s in batch.spans)
        total_ns = max_end - min_start
        summary["total_duration"] = self._format_duration(total_ns)

        # Count by type
        for span in batch.spans:
            if span.name.startswith("transition."):
                summary["transition_count"] += 1
                # Extract transition name (remove t1_, t2_ prefixes for grouping)
                trans_name = span.name.replace("transition.", "")
                # Group by base name (fire, signal_stop)
                if "_fire" in trans_name:
                    base_name = "fire"
                elif "_signal_stop" in trans_name:
                    base_name = "signal_stop"
                else:
                    base_name = trans_name
                summary["transitions"][base_name] = (
                    summary["transitions"].get(base_name, 0) + 1
                )
            elif span.name.startswith("adapter."):
                summary["adapter_count"] += 1
                # Extract adapter operation name
                adapter_op = span.name.replace("adapter.", "")
                summary["adapters"][adapter_op] = (
                    summary["adapters"].get(adapter_op, 0) + 1
                )
            elif span.name == "route.cross_site":
                summary["cross_site_count"] += 1
            elif span.name == "route.yield":
                summary["yield_count"] += 1

        return summary


class FileExporter(TelemetryExporter):
    """Exports telemetry to a file in JSON Lines format.

    Each span/metric is written as a single JSON line for easy parsing.
    """

    def __init__(self, file_path: str | Path):
        """Initialize the file exporter.

        Args:
            file_path: Path to the output file
        """
        self._file_path = Path(file_path)
        self._file: TextIO | None = None

    def _ensure_file_open(self) -> TextIO:
        """Ensure the output file is open."""
        if self._file is None:
            # Create parent directories if needed
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._file_path, "a", encoding="utf-8")
        return self._file

    def export_span(self, span: SpanData) -> None:
        """Export a single span to file."""
        f = self._ensure_file_open()
        record = {"type": "span", "data": span.model_dump()}
        f.write(json.dumps(record) + "\n")
        f.flush()

    def export_metric(self, metric: MetricData) -> None:
        """Export a single metric to file."""
        f = self._ensure_file_open()
        record = {"type": "metric", "data": metric.model_dump()}
        f.write(json.dumps(record) + "\n")
        f.flush()

    def export_batch(self, batch: TelemetryBatch) -> None:
        """Export a batch of telemetry to file."""
        if batch.is_empty:
            return

        f = self._ensure_file_open()
        record = {"type": "batch", "data": batch.model_dump()}
        f.write(json.dumps(record) + "\n")
        f.flush()

    def shutdown(self) -> None:
        """Close the output file."""
        if self._file is not None:
            self._file.close()
            self._file = None


class NoOpExporter(TelemetryExporter):
    """An exporter that does nothing.

    Used when telemetry export is disabled.
    """

    def export_span(self, span: SpanData) -> None:
        pass

    def export_metric(self, metric: MetricData) -> None:
        pass

    def export_batch(self, batch: TelemetryBatch) -> None:
        pass


class OTLPExporter(TelemetryExporter):
    """Exports telemetry to an OTLP-compatible endpoint (e.g., Jaeger, Zipkin).

    This exporter sends traces to OTLP HTTP endpoints using the standard
    OpenTelemetry Protocol JSON format. Compatible with:
    - Jaeger (with OTLP enabled)
    - Zipkin (with OTLP collector)
    - OpenTelemetry Collector
    - Grafana Tempo

    Example:
        # Start Jaeger with OTLP support:
        docker run -d --name jaeger \\
            -p 16686:16686 \\
            -p 4318:4318 \\
            jaegertracing/all-in-one:latest

        # Configure DFM:
        export DFM_TELEMETRY_ENABLED=true
        export DFM_TELEMETRY_EXPORTER=otlp
        export DFM_TELEMETRY_OTLP_ENDPOINT=http://localhost:4318
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        service_name: str = "dfm",
        timeout: float = 10.0,
    ):
        """Initialize the OTLP exporter.

        Args:
            endpoint: OTLP HTTP endpoint (default: http://localhost:4318)
            service_name: Service name for traces (default: dfm)
            timeout: HTTP request timeout in seconds
        """
        self._endpoint = endpoint.rstrip("/")
        self._traces_url = f"{self._endpoint}/v1/traces"
        self._service_name = service_name
        self._timeout = timeout

    def _span_to_otlp(self, span: SpanData) -> dict:
        """Convert a SpanData to OTLP span format."""
        # Convert status
        status_code = 1  # OK
        if span.status == "ERROR":
            status_code = 2  # ERROR
        elif span.status == "UNSET":
            status_code = 0  # UNSET

        # Build attributes list
        attributes = []
        for key, value in span.attributes.items():
            attr = {"key": key}
            if isinstance(value, bool):
                attr["value"] = {"boolValue": value}
            elif isinstance(value, int):
                attr["value"] = {"intValue": str(value)}
            elif isinstance(value, float):
                attr["value"] = {"doubleValue": value}
            else:
                attr["value"] = {"stringValue": str(value)}
            attributes.append(attr)

        # Build span
        otlp_span: dict = {
            "traceId": span.trace_id,
            "spanId": span.span_id,
            "name": span.name,
            "kind": 1,  # INTERNAL
            "startTimeUnixNano": str(span.start_time_ns),
            "endTimeUnixNano": str(span.end_time_ns),
            "attributes": attributes,
            "status": {"code": status_code},
        }

        if span.parent_span_id:
            otlp_span["parentSpanId"] = span.parent_span_id

        if span.status_message:
            otlp_span["status"]["message"] = span.status_message

        return otlp_span

    def _batch_to_otlp(self, batch: TelemetryBatch) -> dict:
        """Convert a TelemetryBatch to OTLP format."""
        spans = [self._span_to_otlp(span) for span in batch.spans]

        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": self._service_name},
                            },
                            {
                                "key": "nv_dfm_core.site",
                                "value": {"stringValue": batch.site},
                            },
                            {
                                "key": "nv_dfm_core.job_id",
                                "value": {"stringValue": batch.job_id},
                            },
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "nv_dfm_core.telemetry",
                                "version": "1.0.0",
                            },
                            "spans": spans,
                        }
                    ],
                }
            ]
        }

    def export_span(self, span: SpanData) -> None:
        """Export a single span (batched export is preferred)."""
        # For OTLP, we prefer batch export
        pass

    def export_metric(self, metric: MetricData) -> None:
        """Export a single metric (not yet implemented for OTLP)."""
        # For now, we don't export metrics to OTLP
        pass

    def export_batch(self, batch: TelemetryBatch) -> None:
        """Export a batch of telemetry to the OTLP endpoint."""
        if batch.is_empty:
            return

        import json
        import urllib.error
        import urllib.request

        otlp_data = self._batch_to_otlp(batch)

        try:
            request = urllib.request.Request(
                self._traces_url,
                data=json.dumps(otlp_data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                _ = response.read()  # Consume response

        except urllib.error.URLError as e:
            # Log but don't fail - telemetry should not break the application
            import sys

            print(
                f"[DFM Telemetry] Failed to export to OTLP endpoint {self._traces_url}: {e}",
                file=sys.stderr,
            )
        except Exception as e:
            import sys

            print(f"[DFM Telemetry] OTLP export error: {e}", file=sys.stderr)


class TelemetryAggregator:
    """Aggregates telemetry from multiple sites at the homesite.

    This class receives TelemetryBatch objects from remote sites
    and can export them to a configured backend.
    """

    def __init__(self, exporter: TelemetryExporter, logger: Logger | None = None):
        """Initialize the aggregator.

        Args:
            exporter: The exporter to use for output
            logger: Optional logger for debug output
        """
        self._exporter = exporter
        self._logger = logger
        self._batches: list[TelemetryBatch] = []
        self._total_spans = 0
        self._total_metrics = 0

    def add_batch(self, batch: TelemetryBatch) -> None:
        """Add a batch from a remote site.

        The batch is immediately exported to the configured backend.

        Args:
            batch: The telemetry batch from a site
        """
        if batch.is_empty:
            return

        self._batches.append(batch)
        self._total_spans += batch.span_count
        self._total_metrics += batch.metric_count

        if self._logger:
            self._logger.debug(
                f"Received telemetry batch from site {batch.site}: "
                f"{batch.span_count} spans, {batch.metric_count} metrics"
            )

        # Export immediately
        self._exporter.export_batch(batch)

    def get_stats(self) -> dict[str, int]:
        """Get aggregation statistics.

        Returns:
            Dict with total_batches, total_spans, total_metrics
        """
        return {
            "total_batches": len(self._batches),
            "total_spans": self._total_spans,
            "total_metrics": self._total_metrics,
        }

    def shutdown(self) -> None:
        """Shutdown the aggregator and its exporter."""
        self._exporter.shutdown()


def create_exporter(logger: Logger | None = None) -> TelemetryExporter:
    """Create an exporter based on configuration.

    Args:
        logger: Optional logger for debug output

    Returns:
        Configured TelemetryExporter instance
    """
    config = get_telemetry_config()

    if not config.enabled:
        return NoOpExporter()

    if config.exporter == ExporterType.CONSOLE:
        return ConsoleExporter(include_attributes=config.include_attributes)
    elif config.exporter == ExporterType.FILE:
        return FileExporter(file_path=config.file_path)
    elif config.exporter == ExporterType.OTLP:
        return OTLPExporter(
            endpoint=config.otlp_endpoint,
            service_name=config.service_name,
            timeout=config.otlp_timeout,
        )
    elif config.exporter == ExporterType.NONE:
        return NoOpExporter()
    else:
        # Default to console
        if logger:
            logger.warning(
                f"Unknown exporter type '{config.exporter}', defaulting to console"
            )
        return ConsoleExporter(include_attributes=config.include_attributes)
