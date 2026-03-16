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

# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false
"""
Tests for FlareJob detach and reattach functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from nv_dfm_core.session import DirectDispatcher
from nv_dfm_core.targets.flare._job import Job as FlareJob


@pytest.fixture
def mock_flare_session():
    """Create a mock Flare session."""
    return MagicMock()


@pytest.fixture
def callback_runner():
    """Create a callback runner with a sync callback."""
    dispatcher = DirectDispatcher()
    return dispatcher.create_runner(default_callback=MagicMock())


@pytest.fixture
def flare_job(mock_flare_session, callback_runner):
    """Create a FlareJob for testing."""
    with patch("nv_dfm_core.targets.flare._job.TokenPackage"):
        job = FlareJob(
            homesite="homesite",
            job_id="test_job_456",
            flare_session=mock_flare_session,
            runtime_module=MagicMock(),
            pipeline=None,
            next_frame=None,
            logger=MagicMock(),
            callback_runner=callback_runner,
        )
        # Stop the background thread immediately for testing
        job._stop_token_receiving_thread_event.set()
        if job._token_receiving_thread.is_alive():
            job._token_receiving_thread.join(timeout=1.0)
        yield job


class TestFlareJobDetach:
    """Test FlareJob detach functionality."""

    def test_detach_clears_callbacks(self, flare_job):
        """Test that detach() clears callback runner."""
        # Verify initial state
        assert flare_job._callback_runner is not None
        assert flare_job._detached is False

        # Detach
        flare_job.detach()

        # Verify cleanup
        assert flare_job._callback_runner is None
        assert flare_job._default_callback is None
        assert flare_job._place_callbacks == {}
        assert flare_job._detached is True

    def test_detach_is_idempotent(self, flare_job):
        """Test that calling detach() multiple times is safe."""
        flare_job.detach()
        assert flare_job._detached is True

        # Second call should not raise
        flare_job.detach()
        assert flare_job._detached is True

    def test_detached_job_raises_on_wait(self, flare_job):
        """Test that detached jobs raise errors on wait."""
        flare_job.detach()

        with pytest.raises(
            RuntimeError, match="Cannot perform operation on detached job"
        ):
            flare_job.wait_until_finished()

    def test_reattached_job_without_pipeline(self, mock_flare_session, callback_runner):
        """Test creating a FlareJob for reattachment (without pipeline)."""
        with patch("nv_dfm_core.targets.flare._job.TokenPackage"):
            job = FlareJob(
                homesite="homesite",
                job_id="existing_job_789",
                flare_session=mock_flare_session,
                runtime_module=MagicMock(),
                pipeline=None,  # Reattached jobs don't have pipeline
                next_frame=None,
                logger=MagicMock(),
                callback_runner=callback_runner,
            )

            # Stop background thread
            job._stop_token_receiving_thread_event.set()
            if job._token_receiving_thread.is_alive():
                job._token_receiving_thread.join(timeout=1.0)

            # Verify job was created successfully
            assert job.job_id == "existing_job_789"
            assert job._pipeline is None
            assert job._next_frame is None
            assert job._detached is False

    def test_detach_keeps_flare_session_for_status(self, flare_job):
        """Test that detach() keeps flare_session for status queries."""
        original_session = flare_job._flare_session

        flare_job.detach()

        # Session should still be available for status queries
        assert flare_job._flare_session is original_session
