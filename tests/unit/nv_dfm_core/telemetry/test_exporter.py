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

"""Tests for telemetry exporters."""

import io
import os
from unittest.mock import MagicMock, patch

import pytest

from nv_dfm_core.telemetry._config import reset_config_cache
from nv_dfm_core.telemetry._exporter import (
    ConsoleExporter,
    NoOpExporter,
    TelemetryAggregator,
    create_exporter,
)
from nv_dfm_core.telemetry._models import (
    SpanData,
    SpanKind,
    SpanStatus,
    TelemetryBatch,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the config cache before and after each test."""
    reset_config_cache()
    yield
    reset_config_cache()


# Valid test IDs (32 chars for trace_id, 16 chars for span_id)
TEST_TRACE_ID = "a" * 32
TEST_SPAN_ID = "b" * 16


def create_test_span(name: str = "test.operation", **kwargs) -> SpanData:
    """Helper to create test spans."""
    defaults = {
        "trace_id": TEST_TRACE_ID,
        "span_id": TEST_SPAN_ID,
        "name": name,
        "kind": SpanKind.INTERNAL,
        "start_time_ns": 1000000000,
        "end_time_ns": 2000000000,
        "status": SpanStatus.OK,
        "site": "test-site",
        "job_id": "job-123",
    }
    defaults.update(kwargs)
    return SpanData(**defaults)


def create_test_batch(**kwargs) -> TelemetryBatch:
    """Helper to create test batches."""
    defaults = {
        "site": "test-site",
        "job_id": "job-123",
        "spans": [create_test_span()],
    }
    defaults.update(kwargs)
    return TelemetryBatch(**defaults)


class TestConsoleExporter:
    """Tests for ConsoleExporter."""

    def test_export_span(self):
        """Test exporting a span to console."""
        output = io.StringIO()
        exporter = ConsoleExporter(output=output)

        span = create_test_span(name="test.operation")
        exporter.export_span(span)

        result = output.getvalue()
        assert "test.operation" in result
        assert "OK" in result

    def test_export_batch(self):
        """Test exporting a batch to console."""
        output = io.StringIO()
        exporter = ConsoleExporter(output=output)

        batch = create_test_batch(
            spans=[
                create_test_span("op1", span_id="c" * 16),
                create_test_span("op2", span_id="d" * 16),
            ]
        )
        exporter.export_batch(batch)

        result = output.getvalue()
        assert "Telemetry Batch" in result
        assert "op1" in result
        assert "op2" in result

    def test_export_empty_batch_no_output(self):
        """Test empty batch produces no output."""
        output = io.StringIO()
        exporter = ConsoleExporter(output=output)

        batch = TelemetryBatch(site="test", job_id="job")
        exporter.export_batch(batch)

        assert output.getvalue() == ""


class TestNoOpExporter:
    """Tests for NoOpExporter."""

    def test_noop_export_does_nothing(self):
        """Test NoOpExporter does nothing."""
        exporter = NoOpExporter()
        span = create_test_span()
        batch = create_test_batch()

        # Should not raise
        exporter.export_span(span)
        exporter.export_batch(batch)


class TestTelemetryAggregator:
    """Tests for TelemetryAggregator."""

    def test_add_batch_exports_immediately(self):
        """Test adding a batch exports it immediately."""
        mock_exporter = MagicMock()
        aggregator = TelemetryAggregator(exporter=mock_exporter)

        batch = create_test_batch()
        aggregator.add_batch(batch)

        mock_exporter.export_batch.assert_called_once_with(batch)

    def test_multiple_batches(self):
        """Test handling multiple batches."""
        mock_exporter = MagicMock()
        aggregator = TelemetryAggregator(exporter=mock_exporter)

        batch1 = create_test_batch(site="site1")
        batch2 = create_test_batch(site="site2")

        aggregator.add_batch(batch1)
        aggregator.add_batch(batch2)

        assert mock_exporter.export_batch.call_count == 2


class TestCreateExporter:
    """Tests for create_exporter factory function."""

    def test_creates_console_exporter_by_default(self):
        """Test creates ConsoleExporter by default."""
        with patch.dict(os.environ, {"DFM_TELEMETRY_ENABLED": "true"}, clear=True):
            reset_config_cache()
            exporter = create_exporter()
            assert isinstance(exporter, ConsoleExporter)

    def test_creates_noop_exporter(self):
        """Test creates NoOpExporter when configured."""
        env = {
            "DFM_TELEMETRY_ENABLED": "true",
            "DFM_TELEMETRY_EXPORTER": "none",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            exporter = create_exporter()
            assert isinstance(exporter, NoOpExporter)
