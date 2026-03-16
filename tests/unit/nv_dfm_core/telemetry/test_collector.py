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

"""Tests for telemetry collector."""

import os
from unittest.mock import patch

import pytest

from nv_dfm_core.telemetry._collector import (
    NoOpCollector,
    SiteTelemetryCollector,
    create_collector,
)
from nv_dfm_core.telemetry._config import reset_config_cache
from nv_dfm_core.telemetry._models import SpanStatus


@pytest.fixture(autouse=True)
def reset_env():
    """Reset environment and caches before and after each test."""
    reset_config_cache()
    yield
    reset_config_cache()


class TestSiteTelemetryCollector:
    """Tests for SiteTelemetryCollector."""

    def test_create_collector(self):
        """Test collector creation."""
        # Disable streaming for this test
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            collector = SiteTelemetryCollector(site="my-site", job_id="job-123")
            assert collector.site == "my-site"
            assert collector.job_id == "job-123"

    def test_record_span_context_manager(self):
        """Test recording a span using context manager."""
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            collector = SiteTelemetryCollector(site="test-site", job_id="test-job")

            with collector.span("test.operation") as span:
                span.set_attribute("key", "value")
                span.set_ok()

            batch = collector.flush()
            assert batch.span_count == 1
            assert batch.spans[0].name == "test.operation"
            assert batch.spans[0].status == SpanStatus.OK
            assert batch.spans[0].attributes["key"] == "value"

    def test_nested_spans(self):
        """Test recording nested spans.

        Note: We no longer track parent-child relationships automatically.
        All spans share the same trace_id (derived from job_id) which is
        sufficient for Jaeger to group them together.
        """
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            collector = SiteTelemetryCollector(site="test-site", job_id="test-job")

            with collector.span("parent.operation") as parent:
                parent.set_ok()
                with collector.span("child.operation") as child:
                    child.set_ok()

            batch = collector.flush()
            assert batch.span_count == 2

            # Find parent and child spans
            parent_span = next(s for s in batch.spans if s.name == "parent.operation")
            child_span = next(s for s in batch.spans if s.name == "child.operation")

            # Both spans share the same trace_id (derived from job_id)
            assert parent_span.trace_id == child_span.trace_id
            # They have different span_ids
            assert parent_span.span_id != child_span.span_id

    def test_span_error(self):
        """Test recording a span with error."""
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            collector = SiteTelemetryCollector(site="test-site", job_id="test-job")

            with collector.span("failing.operation") as span:
                span.set_error("Something went wrong")

            batch = collector.flush()
            assert batch.span_count == 1
            assert batch.spans[0].status == SpanStatus.ERROR
            assert batch.spans[0].status_message == "Something went wrong"

    def test_flush_returns_batch_and_clears(self):
        """Test flush returns batch and clears buffer."""
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            collector = SiteTelemetryCollector(site="test-site", job_id="test-job")

            with collector.span("operation1") as span:
                span.set_ok()
            with collector.span("operation2") as span:
                span.set_ok()

            # First flush should return both spans
            batch1 = collector.flush()
            assert batch1.span_count == 2

            # Second flush should return empty batch
            batch2 = collector.flush()
            assert batch2.span_count == 0

    def test_same_trace_id_with_fixed_trace(self):
        """Test all spans share the same trace ID when fixed trace_id provided."""
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            # trace_id is now derived from job_id
            collector = SiteTelemetryCollector(site="test-site", job_id="test-job")

            with collector.span("op1") as span:
                span.set_ok()
            with collector.span("op2") as span:
                span.set_ok()

            batch = collector.flush()
            trace_ids = {s.trace_id for s in batch.spans}
            assert len(trace_ids) == 1  # All spans have same trace_id
            # trace_id is derived from job_id using job_id_to_trace_id()
            from nv_dfm_core.telemetry import job_id_to_trace_id

            expected_trace_id = job_id_to_trace_id("test-job")
            assert batch.spans[0].trace_id == expected_trace_id


class TestNoOpCollector:
    """Tests for NoOpCollector."""

    def test_noop_span_context_manager(self):
        """Test NoOpCollector span context manager does nothing."""
        collector = NoOpCollector()

        with collector.span("test.operation") as span:
            span.set_attribute("key", "value")
            span.set_ok()

        batch = collector.flush()
        assert batch.is_empty

    def test_noop_is_empty(self):
        """Test NoOpCollector is_empty returns True."""
        collector = NoOpCollector()
        assert collector.is_empty() is True


class TestCreateCollector:
    """Tests for create_collector factory function."""

    def test_creates_noop_when_disabled(self):
        """Test creates NoOpCollector when telemetry disabled."""
        with patch.dict(os.environ, {"DFM_TELEMETRY_ENABLED": "false"}, clear=True):
            reset_config_cache()
            collector = create_collector(site="test", job_id="job")
            assert isinstance(collector, NoOpCollector)

    def test_creates_real_collector_when_enabled(self):
        """Test creates real collector when telemetry enabled."""
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            collector = create_collector(site="test", job_id="job")
            assert isinstance(collector, SiteTelemetryCollector)

    def test_derives_trace_id_from_job_id(self):
        """Test collector derives trace_id from job_id."""
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            from nv_dfm_core.telemetry import job_id_to_trace_id

            job_id = "my-unique-job-id"
            expected_trace_id = job_id_to_trace_id(job_id)

            collector = create_collector(site="test", job_id=job_id)

            with collector.span("test.op") as span:
                span.set_ok()

            batch = collector.flush()
            assert batch.spans[0].trace_id == expected_trace_id


class TestPeriodicFlushRestore:
    """Tests for periodic flush callback failure recovery."""

    def test_spans_restored_on_callback_failure(self):
        """Test spans are restored to buffer when flush callback fails."""
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
            "DFM_TELEMETRY_FLUSH_THRESHOLD_SPANS": "2",  # Trigger after 2 spans
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            collector = SiteTelemetryCollector(site="test", job_id="job")

            # Set a callback that always fails
            callback_called = []

            def failing_callback(batch):
                callback_called.append(batch)
                raise Exception("Simulated failure")

            collector.set_flush_callback(failing_callback)

            # Record enough spans to trigger periodic flush
            with collector.span("span1") as s:
                s.set_ok()
            with collector.span("span2") as s:
                s.set_ok()
            with collector.span("span3") as s:
                s.set_ok()

            # Callback should have been called (flush triggered)
            assert len(callback_called) >= 1

            # Final flush should still have all spans (restored after callback failure)
            final_batch = collector.flush()
            # We should have at least 2 spans (the ones that were restored)
            # The exact number depends on timing, but nothing should be lost
            assert final_batch.span_count >= 2

    def test_spans_not_lost_on_successful_callback(self):
        """Test spans are properly sent when callback succeeds."""
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
            "DFM_TELEMETRY_FLUSH_THRESHOLD_SPANS": "2",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            collector = SiteTelemetryCollector(site="test", job_id="job")

            received_batches = []

            def success_callback(batch):
                received_batches.append(batch)

            collector.set_flush_callback(success_callback)

            # Record spans
            with collector.span("span1") as s:
                s.set_ok()
            with collector.span("span2") as s:
                s.set_ok()
            with collector.span("span3") as s:
                s.set_ok()

            # Final flush
            final_batch = collector.flush()

            # Total spans should be 3 (across all batches)
            total_spans = (
                sum(b.span_count for b in received_batches) + final_batch.span_count
            )
            assert total_spans == 3
