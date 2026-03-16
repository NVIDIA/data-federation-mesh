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

"""Telemetry collector for individual sites.

The SiteTelemetryCollector is responsible for:
- Recording spans and metrics at a single site
- Batching telemetry data for efficient transport
- Providing a context manager for easy span creation
- Streaming to local file for long-running jobs (prevents data loss)
- Periodic flushing to homesite
"""

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Generator, TextIO

from ._config import get_telemetry_config, telemetry_enabled
from ._context import (
    SpanBuilder,
    get_current_time_ns,
)
from ._models import (
    MetricData,
    MetricType,
    SpanData,
    TelemetryBatch,
)

# Type alias for flush callback
FlushCallback = Callable[[TelemetryBatch], None]


class SiteTelemetryCollector:
    """Collects telemetry data for a single DFM site.

    This class is thread-safe and can be used from multiple transitions
    executing concurrently.

    Features for long-running jobs:
    - Streams spans to local file immediately (no data loss on crash)
    - Bounded memory buffer (oldest spans dropped if full, but saved to file)
    - Optional periodic flush callback for sending to homesite
    - Configurable via environment variables

    Example:
        collector = SiteTelemetryCollector(site="datacenter", job_id="job123")

        # Optionally set a callback for periodic flushing to homesite
        collector.set_flush_callback(lambda batch: send_to_homesite(batch))

        # Record a span
        with collector.span("my_operation") as span:
            span.set_attribute("key", "value")
            do_work()

        # Record a metric
        collector.record_counter("tokens_processed", 1, {"type": "data"})

        # Final flush on job complete
        batch = collector.flush()
        collector.shutdown()
    """

    def __init__(self, site: str, job_id: str):
        """Initialize the collector.

        Args:
            site: The DFM site name where this collector runs
            job_id: The DFM job ID for this collection session
        """
        from ._context import job_id_to_trace_id

        self._site = site
        self._job_id = job_id
        # Derive trace_id from job_id - same job always gets same trace_id
        self._trace_id = job_id_to_trace_id(job_id)
        self._spans: list[SpanData] = []
        self._metrics: list[MetricData] = []
        self._lock = threading.Lock()
        self._sequence_num = 0
        self._config = get_telemetry_config()

        # Streaming to local file
        self._streaming_file: TextIO | None = None
        self._streaming_file_path: Path | None = None
        if self._config.streaming_enabled:
            self._init_streaming_file()

        # Periodic flush support
        self._flush_callback: FlushCallback | None = None
        self._last_flush_time: float = time.time()
        self._total_spans_recorded: int = 0
        self._spans_since_last_flush: int = 0

    def _init_streaming_file(self) -> None:
        """Initialize the local streaming file."""
        try:
            file_path_str = self._config.streaming_file_template.format(
                job_id=self._job_id, site=self._site
            )
            self._streaming_file_path = Path(file_path_str)
            self._streaming_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._streaming_file = open(
                self._streaming_file_path, "a", encoding="utf-8"
            )
        except Exception:
            # Silently disable streaming if we can't open the file
            self._streaming_file = None
            self._streaming_file_path = None

    def set_flush_callback(self, callback: FlushCallback | None) -> None:
        """Set a callback for periodic flushing to homesite.

        The callback is invoked when:
        - flush_threshold_spans is reached
        - flush_interval_seconds has passed since last flush

        Args:
            callback: Function that receives TelemetryBatch, or None to disable
        """
        self._flush_callback = callback

    def _stream_to_file(self, span: SpanData) -> None:
        """Write a span to the local streaming file."""
        if self._streaming_file is not None:
            try:
                record = {"type": "span", "data": span.model_dump()}
                self._streaming_file.write(json.dumps(record) + "\n")
                self._streaming_file.flush()
            except Exception:
                # Don't let streaming errors affect operation
                pass

    def _maybe_trigger_periodic_flush(self) -> None:
        """Check if we should trigger a periodic flush to homesite.

        If the flush callback fails, the spans are restored to the buffer
        so they can be sent in the final flush.
        """
        if self._flush_callback is None:
            return

        should_flush = False
        current_time = time.time()

        # Check span threshold
        if (
            self._config.flush_threshold_spans > 0
            and self._spans_since_last_flush >= self._config.flush_threshold_spans
        ):
            should_flush = True

        # Check time threshold
        if (
            self._config.flush_interval_seconds > 0
            and (current_time - self._last_flush_time)
            >= self._config.flush_interval_seconds
        ):
            should_flush = True

        if should_flush:
            batch = self.flush()
            if not batch.is_empty:
                try:
                    self._flush_callback(batch)
                except Exception:
                    # Callback failed - restore spans to buffer so they're not lost
                    # They'll be sent in the final flush
                    with self._lock:
                        # Prepend the failed batch's spans (they're older)
                        self._spans = list(batch.spans) + self._spans
                        self._metrics = list(batch.metrics) + self._metrics

    def _enforce_buffer_limit(self) -> None:
        """Drop oldest spans if buffer is full (they're already saved to streaming file)."""
        if len(self._spans) > self._config.buffer_max_spans:
            # Drop oldest spans (they're already in the streaming file)
            excess = len(self._spans) - self._config.buffer_max_spans
            self._spans = self._spans[excess:]

    @property
    def site(self) -> str:
        """The site this collector belongs to."""
        return self._site

    @property
    def job_id(self) -> str:
        """The job ID this collector belongs to."""
        return self._job_id

    def record_span(self, span: SpanData) -> None:
        """Record a completed span.

        This is typically called automatically by SpanBuilder.__exit__,
        but can also be called directly for pre-built spans.

        The span is:
        1. Immediately written to the streaming file (if enabled)
        2. Added to the in-memory buffer
        3. May trigger a periodic flush if thresholds are met

        Args:
            span: The completed span data to record
        """
        if not telemetry_enabled():
            return

        # Stream to local file immediately (outside lock for performance)
        self._stream_to_file(span)

        with self._lock:
            self._spans.append(span)
            self._total_spans_recorded += 1
            self._spans_since_last_flush += 1
            self._enforce_buffer_limit()

        # Check for periodic flush (outside lock)
        self._maybe_trigger_periodic_flush()

    def record_metric(self, metric: MetricData) -> None:
        """Record a metric data point.

        Args:
            metric: The metric data to record
        """
        if not telemetry_enabled():
            return

        with self._lock:
            self._metrics.append(metric)

    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        """Record a counter metric increment.

        Args:
            name: Metric name (e.g., "dfm.tokens.routed")
            value: Amount to increment (default: 1)
            attributes: Optional labels/dimensions
        """
        if not telemetry_enabled():
            return

        metric = MetricData(
            name=name,
            type=MetricType.COUNTER,
            value=value,
            timestamp_ns=get_current_time_ns(),
            attributes=attributes or {},
            site=self._site,
            job_id=self._job_id,
        )
        self.record_metric(metric)

    def record_gauge(
        self,
        name: str,
        value: float,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        """Record a gauge metric value.

        Args:
            name: Metric name (e.g., "dfm.transitions.active")
            value: Current value
            attributes: Optional labels/dimensions
        """
        if not telemetry_enabled():
            return

        metric = MetricData(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            timestamp_ns=get_current_time_ns(),
            attributes=attributes or {},
            site=self._site,
            job_id=self._job_id,
        )
        self.record_metric(metric)

    def record_histogram(
        self,
        name: str,
        value: float,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        """Record a histogram metric observation.

        Args:
            name: Metric name (e.g., "dfm.transition.duration_ms")
            value: Observed value
            attributes: Optional labels/dimensions
        """
        if not telemetry_enabled():
            return

        metric = MetricData(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            timestamp_ns=get_current_time_ns(),
            attributes=attributes or {},
            site=self._site,
            job_id=self._job_id,
        )
        self.record_metric(metric)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> Generator[SpanBuilder, None, None]:
        """Create and record a span using a context manager.

        The span is automatically timed and recorded when exiting the context.
        If an exception occurs, the span is marked as errored.

        Args:
            name: Name of the operation being traced
            attributes: Optional initial attributes

        Yields:
            SpanBuilder that can be used to add attributes or set status

        Example:
            with collector.span("process_data") as span:
                span.set_attribute("record_count", 100)
                result = process(data)
                if result.warnings:
                    span.set_attribute("warnings", len(result.warnings))
        """
        if not telemetry_enabled():
            # Yield a no-op span builder when disabled
            yield SpanBuilder(
                name=name,
                collector=None,  # Won't record
                trace_id=self._trace_id,
                site=self._site,
                job_id=self._job_id,
            )
            return

        builder = SpanBuilder(
            name=name,
            collector=self,
            trace_id=self._trace_id,
            site=self._site,
            job_id=self._job_id,
        )
        if attributes:
            builder.set_attributes(attributes)

        with builder as span:
            yield span

    def flush(self) -> TelemetryBatch:
        """Flush collected telemetry and return a batch.

        This clears the internal buffers and returns all collected data.
        The batch can then be sent to the homesite for aggregation.

        Returns:
            TelemetryBatch containing all collected spans and metrics
        """
        with self._lock:
            batch = TelemetryBatch(
                site=self._site,
                job_id=self._job_id,
                spans=self._spans.copy(),
                metrics=self._metrics.copy(),
                sequence_num=self._sequence_num,
            )
            self._spans.clear()
            self._metrics.clear()
            self._sequence_num += 1
            self._last_flush_time = time.time()
            self._spans_since_last_flush = 0
            return batch

    def get_pending_count(self) -> tuple[int, int]:
        """Get the count of pending spans and metrics.

        Returns:
            Tuple of (span_count, metric_count)
        """
        with self._lock:
            return len(self._spans), len(self._metrics)

    def is_empty(self) -> bool:
        """Check if there's no pending telemetry data."""
        span_count, metric_count = self.get_pending_count()
        return span_count == 0 and metric_count == 0

    def get_stats(self) -> dict[str, int | float | str | None]:
        """Get collector statistics.

        Returns:
            Dict with total_spans_recorded, buffer_size, streaming_file, etc.
        """
        with self._lock:
            return {
                "total_spans_recorded": self._total_spans_recorded,
                "buffer_size": len(self._spans),
                "metrics_count": len(self._metrics),
                "sequence_num": self._sequence_num,
                "streaming_file": str(self._streaming_file_path)
                if self._streaming_file_path
                else None,
            }

    def shutdown(self) -> None:
        """Shutdown the collector, closing any open files.

        Should be called when the job completes.
        """
        if self._streaming_file is not None:
            try:
                self._streaming_file.close()
            except Exception:
                pass
            self._streaming_file = None


class NoOpCollector:
    """A no-op collector that does nothing.

    Used when telemetry is disabled to avoid None checks everywhere.
    """

    def __init__(self, site: str = "", job_id: str = ""):
        self._site = site
        self._job_id = job_id

    @property
    def site(self) -> str:
        return self._site

    @property
    def job_id(self) -> str:
        return self._job_id

    def record_span(self, span: SpanData) -> None:
        pass

    def record_metric(self, metric: MetricData) -> None:
        pass

    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        pass

    def record_gauge(
        self,
        name: str,
        value: float,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        pass

    def record_histogram(
        self,
        name: str,
        value: float,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        pass

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> Generator[SpanBuilder, None, None]:
        yield SpanBuilder(
            name=name,
            collector=None,  # Won't record anything
            trace_id="0" * 32,  # Dummy trace ID
            site=self._site,
            job_id=self._job_id,
        )

    def flush(self) -> TelemetryBatch:
        return TelemetryBatch(site=self._site, job_id=self._job_id)

    def get_pending_count(self) -> tuple[int, int]:
        return (0, 0)

    def is_empty(self) -> bool:
        return True


def create_collector(site: str, job_id: str) -> SiteTelemetryCollector | NoOpCollector:
    """Factory function to create the appropriate collector.

    Returns a real collector if telemetry is enabled, otherwise a no-op.
    The trace_id is derived from job_id automatically.

    Args:
        site: The DFM site name
        job_id: The DFM job ID

    Returns:
        SiteTelemetryCollector if enabled, NoOpCollector otherwise
    """
    if telemetry_enabled():
        return SiteTelemetryCollector(site=site, job_id=job_id)
    return NoOpCollector(site=site, job_id=job_id)
