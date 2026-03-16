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

"""Tests for telemetry context utilities."""

from nv_dfm_core.telemetry._context import (
    generate_span_id,
    generate_trace_id,
    get_current_time_ns,
    job_id_to_trace_id,
)


class TestGenerateIds:
    """Tests for ID generation functions."""

    def test_generate_trace_id_length(self):
        """Test trace ID has correct length."""
        trace_id = generate_trace_id()
        assert len(trace_id) == 32  # 16 bytes hex encoded

    def test_generate_trace_id_uniqueness(self):
        """Test trace IDs are unique."""
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_span_id_length(self):
        """Test span ID has correct length."""
        span_id = generate_span_id()
        assert len(span_id) == 16  # 8 bytes hex encoded

    def test_generate_span_id_uniqueness(self):
        """Test span IDs are unique."""
        ids = {generate_span_id() for _ in range(100)}
        assert len(ids) == 100

    def test_ids_are_valid_hex(self):
        """Test IDs are valid hex strings."""
        trace_id = generate_trace_id()
        span_id = generate_span_id()

        # Should not raise
        int(trace_id, 16)
        int(span_id, 16)


class TestJobIdToTraceId:
    """Tests for job_id_to_trace_id function."""

    def test_produces_valid_trace_id(self):
        """Test output is a valid 32-char hex trace ID."""
        trace_id = job_id_to_trace_id("my-job-123")
        assert len(trace_id) == 32
        int(trace_id, 16)  # Should not raise

    def test_deterministic(self):
        """Test same job_id always produces same trace_id."""
        job_id = "job-abc-456"
        trace_id_1 = job_id_to_trace_id(job_id)
        trace_id_2 = job_id_to_trace_id(job_id)
        assert trace_id_1 == trace_id_2

    def test_different_job_ids_produce_different_trace_ids(self):
        """Test different job_ids produce different trace_ids."""
        trace_id_1 = job_id_to_trace_id("job-1")
        trace_id_2 = job_id_to_trace_id("job-2")
        assert trace_id_1 != trace_id_2


class TestGetCurrentTimeNs:
    """Tests for get_current_time_ns function."""

    def test_returns_positive_integer(self):
        """Test returns a positive integer."""
        time_ns = get_current_time_ns()
        assert isinstance(time_ns, int)
        assert time_ns > 0

    def test_monotonic(self):
        """Test time is monotonically increasing."""
        times = [get_current_time_ns() for _ in range(100)]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]
