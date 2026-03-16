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
import threading
import time
from unittest.mock import MagicMock

from nv_dfm_core.exec import Frame, TokenPackage
from nv_dfm_core.session import JobStatus
from nv_dfm_core.targets.local._federation_runner import (
    FederationRunner,
    JobRecord,
    LOCAL_COMPLETION_PLACE,
)


def _make_handle(job_id: str):
    handle = MagicMock()
    handle.job_id = job_id
    handle.homesite = "homesite"
    handle.pipeline = MagicMock()
    handle.pipeline.api_version = "1.0"
    handle.next_frame = Frame.start_frame(0)
    handle.federation_name = "test_fed"
    handle.pipeline_api_version = "1.0"
    handle.force_modgen = False
    handle.execution = MagicMock()
    return handle


def test_multi_site_completion_logic():
    # Setup federation with 2 sites
    sites = ["site1", "site2"]
    job_id = "job123"

    # We use __new__ to avoid starting the background thread and other side effects
    fed = FederationRunner.__new__(FederationRunner)
    fed._executions_lock = threading.Lock()
    fed._logger = MagicMock()
    fed._running_jobs = {}
    fed._jobs_by_id = {}

    handle = _make_handle(job_id)
    rec = JobRecord(
        job_id=job_id,
        handle=handle,
        created_at=time.time(),
        status=JobStatus.RUNNING,
        participating_sites=set(sites),
    )

    fed._jobs_by_id[job_id] = rec
    fed._running_jobs[job_id] = handle

    # Mock the yield queue for the execution
    execution = handle.execution
    yield_queue = MagicMock()
    execution.yield_queue = yield_queue

    # Simulate site1 completion token
    token1 = TokenPackage.wrap_data(
        source_site="site1",
        source_node=None,
        source_job=job_id,
        target_site="homesite",
        target_place=LOCAL_COMPLETION_PLACE,
        target_job=job_id,
        is_yield=True,
        frame=Frame.stop_frame(),
        data=None,
    )

    # Simulate site2 completion token
    token2 = TokenPackage.wrap_data(
        source_site="site2",
        source_node=None,
        source_job=job_id,
        target_site="homesite",
        target_place=LOCAL_COMPLETION_PLACE,
        target_job=job_id,
        is_yield=True,
        frame=Frame.stop_frame(),
        data=None,
    )

    # Mock yield_queue.get_nowait to return token1 then token2
    yield_queue.get_nowait.side_effect = [
        token1,
        Exception("Empty"),
    ]  # Using Exception to break the loop

    # Run one tick of drain
    # Since we use Exception to break the loop, we need to catch it or mock it properly
    # Actually _drain_yields_and_update_jobs catches Empty, not general Exception.
    from queue import Empty

    yield_queue.get_nowait.side_effect = [token1, Empty()]

    fed._drain_yields_and_update_jobs()

    # Check status - should still be RUNNING because site2 is missing
    assert rec.status == JobStatus.RUNNING
    assert "site1" in rec.completed_sites
    assert "site2" not in rec.completed_sites
    assert not rec.completion_event.is_set()

    # Now provide token2
    yield_queue.get_nowait.side_effect = [token2, Empty()]
    fed._drain_yields_and_update_jobs()

    # Now it should be FINISHED
    assert rec.status == JobStatus.FINISHED
    assert "site2" in rec.completed_sites
    assert rec.completion_event.is_set()
    assert rec.finished_at is not None


def test_cleanup_does_not_abort_finished_job():
    job_id = "job_finished"
    fed = FederationRunner.__new__(FederationRunner)
    fed._executions_lock = threading.Lock()
    fed._logger = MagicMock()

    handle = _make_handle(job_id)
    rec = JobRecord(
        job_id=job_id,
        handle=handle,
        created_at=time.time(),
        status=JobStatus.FINISHED,
        participating_sites={"site1"},
    )
    rec.finish_observed_at = time.time()

    fed._jobs_by_id = {job_id: rec}
    fed._running_jobs = {job_id: handle}

    # Mock release_execution to track calls
    fed.release_execution = MagicMock()

    fed._cleanup_all_jobs()

    # Verify release_execution was called with abort=False
    fed.release_execution.assert_called_once_with(handle.execution, abort=False)


def test_cleanup_aborts_running_job():
    job_id = "job_running"
    fed = FederationRunner.__new__(FederationRunner)
    fed._executions_lock = threading.Lock()
    fed._logger = MagicMock()

    handle = _make_handle(job_id)
    rec = JobRecord(
        job_id=job_id,
        handle=handle,
        created_at=time.time(),
        status=JobStatus.RUNNING,
        participating_sites={"site1"},
    )

    fed._jobs_by_id = {job_id: rec}
    fed._running_jobs = {job_id: handle}

    # Mock release_execution
    fed.release_execution = MagicMock()
    fed._mark_job_aborted = MagicMock()

    fed._cleanup_all_jobs()

    # Verify release_execution was called with abort=True
    fed.release_execution.assert_called_once_with(handle.execution, abort=True)
    fed._mark_job_aborted.assert_called_once_with(rec)
