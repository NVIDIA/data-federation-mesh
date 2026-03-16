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

"""Tests for telemetry data models."""

import pytest

from nv_dfm_core.telemetry._models import (
    MetricData,
    MetricType,
    SpanData,
    SpanKind,
    SpanStatus,
    TelemetryBatch,
)

# Valid test IDs (32 chars for trace_id, 16 chars for span_id)
TEST_TRACE_ID = "a" * 32
TEST_SPAN_ID = "b" * 16
TEST_PARENT_SPAN_ID = "c" * 16


class TestSpanData:
    """Tests for SpanData model."""

    def test_create_span(self):
        """Test creating a basic span."""
        span = SpanData(
            trace_id=TEST_TRACE_ID,
            span_id=TEST_SPAN_ID,
            name="test.operation",
            kind=SpanKind.INTERNAL,
            start_time_ns=1000000000,
            end_time_ns=2000000000,
            status=SpanStatus.OK,
            site="test-site",
            job_id="job-123",
        )
        assert span.trace_id == TEST_TRACE_ID
        assert span.span_id == TEST_SPAN_ID
        assert span.name == "test.operation"
        assert span.duration_ns == 1000000000
        assert span.status == SpanStatus.OK

    def test_span_with_parent(self):
        """Test span with parent span ID."""
        span = SpanData(
            trace_id=TEST_TRACE_ID,
            span_id=TEST_SPAN_ID,
            parent_span_id=TEST_PARENT_SPAN_ID,
            name="child.operation",
            kind=SpanKind.INTERNAL,
            start_time_ns=1000000000,
            end_time_ns=2000000000,
            status=SpanStatus.OK,
            site="test-site",
            job_id="job-123",
        )
        assert span.parent_span_id == TEST_PARENT_SPAN_ID

    def test_span_with_attributes(self):
        """Test span with custom attributes."""
        span = SpanData(
            trace_id=TEST_TRACE_ID,
            span_id=TEST_SPAN_ID,
            name="test.operation",
            kind=SpanKind.CLIENT,
            start_time_ns=1000000000,
            end_time_ns=2000000000,
            status=SpanStatus.OK,
            attributes={"key": "value", "count": 42},
            site="test-site",
            job_id="job-123",
        )
        assert span.attributes["key"] == "value"
        assert span.attributes["count"] == 42

    def test_span_error_status(self):
        """Test span with error status."""
        span = SpanData(
            trace_id=TEST_TRACE_ID,
            span_id=TEST_SPAN_ID,
            name="test.operation",
            kind=SpanKind.INTERNAL,
            start_time_ns=1000000000,
            end_time_ns=2000000000,
            status=SpanStatus.ERROR,
            status_message="Something went wrong",
            site="test-site",
            job_id="job-123",
        )
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Something went wrong"

    def test_duration_calculation(self):
        """Test duration_ns property calculation."""
        span = SpanData(
            trace_id=TEST_TRACE_ID,
            span_id=TEST_SPAN_ID,
            name="test.operation",
            kind=SpanKind.INTERNAL,
            start_time_ns=1000000000,
            end_time_ns=1500000000,
            status=SpanStatus.OK,
            site="test-site",
            job_id="job-123",
        )
        assert span.duration_ns == 500000000  # 500ms in nanoseconds


class TestMetricData:
    """Tests for MetricData model."""

    def test_counter_metric(self):
        """Test creating a counter metric."""
        metric = MetricData(
            name="requests.total",
            type=MetricType.COUNTER,
            value=100,
            timestamp_ns=1000000000,
            site="test-site",
            job_id="job-123",
        )
        assert metric.name == "requests.total"
        assert metric.type == MetricType.COUNTER
        assert metric.value == 100

    def test_gauge_metric(self):
        """Test creating a gauge metric."""
        metric = MetricData(
            name="queue.size",
            type=MetricType.GAUGE,
            value=42.5,
            timestamp_ns=1000000000,
            site="test-site",
            job_id="job-123",
        )
        assert metric.type == MetricType.GAUGE
        assert metric.value == 42.5

    def test_metric_with_attributes(self):
        """Test metric with custom attributes."""
        metric = MetricData(
            name="requests.total",
            type=MetricType.COUNTER,
            value=100,
            timestamp_ns=1000000000,
            attributes={"endpoint": "/api/v1", "method": "GET"},
            site="test-site",
            job_id="job-123",
        )
        assert metric.attributes["endpoint"] == "/api/v1"
        assert metric.attributes["method"] == "GET"


class TestTelemetryBatch:
    """Tests for TelemetryBatch model."""

    def test_empty_batch(self):
        """Test creating an empty batch."""
        batch = TelemetryBatch(site="test-site", job_id="job-123")
        assert batch.is_empty is True
        assert batch.span_count == 0
        assert batch.metric_count == 0

    def test_batch_with_spans(self):
        """Test batch with spans."""
        spans = [
            SpanData(
                trace_id=TEST_TRACE_ID,
                span_id=f"{i:016x}",  # Valid 16-char hex
                name=f"operation{i}",
                kind=SpanKind.INTERNAL,
                start_time_ns=1000000000,
                end_time_ns=2000000000,
                status=SpanStatus.OK,
                site="test-site",
                job_id="job-123",
            )
            for i in range(3)
        ]
        batch = TelemetryBatch(site="test-site", job_id="job-123", spans=spans)
        assert batch.is_empty is False
        assert batch.span_count == 3

    def test_batch_with_metrics(self):
        """Test batch with metrics."""
        metrics = [
            MetricData(
                name=f"metric{i}",
                type=MetricType.COUNTER,
                value=i * 10,
                timestamp_ns=1000000000,
                site="test-site",
                job_id="job-123",
            )
            for i in range(2)
        ]
        batch = TelemetryBatch(site="test-site", job_id="job-123", metrics=metrics)
        assert batch.is_empty is False
        assert batch.metric_count == 2

    def test_batch_serialization(self):
        """Test batch can be serialized to dict and back."""
        span = SpanData(
            trace_id=TEST_TRACE_ID,
            span_id=TEST_SPAN_ID,
            name="test.operation",
            kind=SpanKind.INTERNAL,
            start_time_ns=1000000000,
            end_time_ns=2000000000,
            status=SpanStatus.OK,
            site="test-site",
            job_id="job-123",
        )
        batch = TelemetryBatch(site="test-site", job_id="job-123", spans=[span])

        # Serialize to dict
        batch_dict = batch.model_dump()
        assert batch_dict["site"] == "test-site"
        assert len(batch_dict["spans"]) == 1

        # Deserialize back
        restored = TelemetryBatch.model_validate(batch_dict)
        assert restored.site == "test-site"
        assert restored.span_count == 1
        assert restored.spans[0].name == "test.operation"

    def test_batch_has_sequence_number(self):
        """Test batch has a sequence number."""
        batch = TelemetryBatch(site="test-site", job_id="job-123")
        # Sequence number should be an integer
        assert isinstance(batch.sequence_num, int)
