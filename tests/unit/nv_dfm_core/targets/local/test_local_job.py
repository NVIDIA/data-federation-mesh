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

# pyright: reportPrivateUsage=false
"""
Tests for LocalJob detach and reattach functionality.
"""

import queue
from unittest.mock import MagicMock, Mock

import pytest

from nv_dfm_core.api import PreparedPipeline
from nv_dfm_core.exec import Frame
from nv_dfm_core.session import DirectDispatcher, JobStatus
from nv_dfm_core.targets.local import FederationRunner, JobHandle, LocalJob


@pytest.fixture
def mock_handle():
    """Create a mock JobHandle."""
    handle = Mock(spec=JobHandle)
    handle.job_id = "test_job_123"
    handle.homesite = "homesite"
    handle.pipeline = MagicMock(spec=PreparedPipeline)
    handle.next_frame = Frame.start_frame(0)
    handle.federation_name = "test_fed"
    handle.pipeline_api_version = "1.0"
    handle.force_modgen = False
    handle.execution = MagicMock()  # Mock execution
    return handle


@pytest.fixture
def callback_runner():
    """Create a callback runner with a sync callback."""
    dispatcher = DirectDispatcher()
    return dispatcher.create_runner(default_callback=MagicMock())


@pytest.fixture
def callback_runner_with_places():
    """Create a callback runner with default and place callbacks."""
    dispatcher = DirectDispatcher()
    return dispatcher.create_runner(
        default_callback=MagicMock(),
        place_callbacks={"place1": MagicMock(), "place2": MagicMock()},
    )


class TestLocalJobDetachReattach:
    """Test LocalJob detach and reattach functionality."""

    def test_detach_clears_references(self, mock_handle, callback_runner):
        """Test that detach() properly clears callback runner and references."""
        federation = MagicMock(spec=FederationRunner)
        federation.attach_client.return_value = ("consumer1", queue.Queue())

        # Create job from handle
        job = LocalJob.create_from_handle(
            handle=mock_handle,
            federation=federation,
            logger=MagicMock(),
            callback_runner=callback_runner,
        )

        # Stop consumer thread started by create_from_handle
        job._cleanup_consumer_thread()  # pyright: ignore[reportPrivateUsage]

        # Verify initial state
        assert job._callback_runner is callback_runner
        assert job._federation is federation
        assert job._detached is False

        # Detach
        job.detach()

        # Verify cleanup
        assert job._callback_runner is None
        assert job._federation is None
        assert job._execution is None
        assert job._detached is True
        federation.detach_client.assert_called_once()

    def test_detach_is_idempotent(self, mock_handle):
        """Test that calling detach() multiple times is safe."""
        federation = MagicMock(spec=FederationRunner)
        federation.attach_client.return_value = None
        job = LocalJob.create_from_handle(
            handle=mock_handle,
            federation=federation,
            logger=MagicMock(),
            callback_runner=None,
        )

        # First detach
        job.detach()
        assert job._detached is True

        # Second detach should not raise
        job.detach()
        assert job._detached is True

    def test_detached_job_raises_on_get_status(self, mock_handle):
        """Test that detached jobs raise errors on get_status."""
        federation = MagicMock(spec=FederationRunner)
        federation.attach_client.return_value = None
        job = LocalJob.create_from_handle(
            handle=mock_handle,
            federation=federation,
            logger=MagicMock(),
            callback_runner=None,
        )

        job.detach()

        with pytest.raises(
            RuntimeError, match="Cannot perform operation on detached job"
        ):
            job.get_status()

    def test_detached_job_raises_on_wait(self, mock_handle):
        """Test that detached jobs raise errors on wait_until_finished."""
        federation = MagicMock(spec=FederationRunner)
        federation.attach_client.return_value = None
        job = LocalJob.create_from_handle(
            handle=mock_handle,
            federation=federation,
            logger=MagicMock(),
            callback_runner=None,
        )

        job.detach()

        with pytest.raises(
            RuntimeError, match="Cannot perform operation on detached job"
        ):
            job.wait_until_finished()

    def test_detached_job_raises_on_cancel(self, mock_handle):
        """Test that detached jobs raise errors on cancel."""
        federation = MagicMock(spec=FederationRunner)
        federation.attach_client.return_value = None
        job = LocalJob.create_from_handle(
            handle=mock_handle,
            federation=federation,
            logger=MagicMock(),
            callback_runner=None,
        )

        job.detach()

        with pytest.raises(
            RuntimeError, match="Cannot perform operation on detached job"
        ):
            job.cancel()

    def test_reattach_creates_new_job_from_handle(self, mock_handle):
        """Test that reattach creates a fresh job from the same handle."""
        federation = MagicMock(spec=FederationRunner)
        federation.attach_client.return_value = ("consumer1", queue.Queue())

        # Create first job
        dispatcher = DirectDispatcher()
        runner1 = dispatcher.create_runner(default_callback=MagicMock())
        job1 = LocalJob.create_from_handle(
            handle=mock_handle,
            federation=federation,
            logger=MagicMock(),
            callback_runner=runner1,
        )

        job1._cleanup_consumer_thread()  # pyright: ignore[reportPrivateUsage]

        # Detach first job
        job1.detach()
        assert job1._detached is True

        # Create second job (simulating reattach)
        runner2 = dispatcher.create_runner(default_callback=MagicMock())
        federation.attach_client.return_value = ("consumer2", queue.Queue())
        job2 = LocalJob.create_from_handle(
            handle=mock_handle,
            federation=federation,
            logger=MagicMock(),
            callback_runner=runner2,
        )

        job2._cleanup_consumer_thread()  # pyright: ignore[reportPrivateUsage]

        # Verify new job has new callback runner
        assert job2._callback_runner is runner2
        assert job2._detached is False
        assert job2.job_id == job1.job_id  # Same job ID
        assert job2 is not job1  # But different object

    def test_create_stub_returns_unknown_status(self):
        """Test that create_stub returns a job with UNKNOWN status."""
        stub_job = LocalJob.create_stub(
            job_id="unknown_job_999",
            homesite="homesite",
            logger=MagicMock(),
        )

        assert stub_job.job_id == "unknown_job_999"
        assert stub_job.get_status() == JobStatus.UNKNOWN
        assert stub_job._pipeline is None
        assert stub_job._next_frame is None
        assert stub_job._federation is None

    def test_create_from_handle_with_callbacks(
        self, mock_handle, callback_runner_with_places
    ):
        """Test creating job from handle with callback runner."""
        federation = MagicMock(spec=FederationRunner)
        federation.attach_client.return_value = ("consumer1", queue.Queue())

        job = LocalJob.create_from_handle(
            handle=mock_handle,
            federation=federation,
            logger=MagicMock(),
            callback_runner=callback_runner_with_places,
        )

        job._cleanup_consumer_thread()  # pyright: ignore[reportPrivateUsage]

        assert job.job_id == "test_job_123"
        assert job._callback_runner is callback_runner_with_places
        assert job._execution is mock_handle.execution

    def test_create_from_handle_without_callbacks(self, mock_handle):
        """Test creating informational job from handle without callback runner."""
        federation = MagicMock(spec=FederationRunner)
        federation.attach_client.return_value = None
        job = LocalJob.create_from_handle(
            handle=mock_handle,
            federation=federation,
            logger=MagicMock(),
            callback_runner=None,
        )

        assert job.job_id == "test_job_123"
        assert job._callback_runner is None
