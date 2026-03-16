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
from unittest.mock import MagicMock

from nv_dfm_core.exec import Frame, TokenPackage
from nv_dfm_core.session import JobStatus
from nv_dfm_core.targets.local._federation_runner import FederationRunner, JobRecord
from nv_dfm_core.targets.local._session_delegate import LocalSessionDelegate


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
    handle.execution = None
    return handle


def _make_federation_with_record(job_id: str, status: JobStatus) -> FederationRunner:
    fed = FederationRunner.__new__(FederationRunner)
    fed._executions_lock = threading.Lock()
    fed._logger = MagicMock()
    fed._running_jobs = {}
    handle = _make_handle(job_id)
    rec = JobRecord(job_id=job_id, handle=handle, created_at=0.0, status=status)
    fed._jobs_by_id = {job_id: rec}
    return fed


def test_registry_known_id_is_not_unknown():
    fed = _make_federation_with_record("job1", JobStatus.RUNNING)
    assert fed.is_known_job_id("job1") is True
    assert fed.get_job_status("job1") == JobStatus.RUNNING
    assert fed.get_job_status("does_not_exist") == JobStatus.UNKNOWN


def test_attach_client_flushes_buffer():
    fed = _make_federation_with_record("job1", JobStatus.RUNNING)
    with fed._executions_lock:
        rec = fed._jobs_by_id["job1"]
        rec.buffer.append(
            TokenPackage.wrap_data(
                source_site="s",
                source_node=None,
                source_job="job1",
                target_site="homesite",
                target_place="p",
                target_job="job1",
                is_yield=True,
                frame=Frame.start_frame(0),
                data={"x": 1},
            )
        )
        rec.buffer.append(
            TokenPackage.wrap_data(
                source_site="s",
                source_node=None,
                source_job="job1",
                target_site="homesite",
                target_place="p2",
                target_job="job1",
                is_yield=True,
                frame=Frame.start_frame(0),
                data={"y": 2},
            )
        )

    attached = fed.attach_client("job1")
    assert attached is not None
    _, q = attached

    t1 = q.get(timeout=0.5)
    t2 = q.get(timeout=0.5)
    assert t1.target_place == "p"
    assert t2.target_place == "p2"

    with fed._executions_lock:
        assert len(fed._jobs_by_id["job1"].buffer) == 0


def test_local_session_delegate_reattach_returns_stub_only_for_unknown_id():
    fed = _make_federation_with_record("job1", JobStatus.FINISHED)

    delegate = LocalSessionDelegate(
        session=MagicMock(),
        federation_name="test_fed",
        homesite="homesite",
        logger=MagicMock(),
    )
    delegate._federation = fed  # simulate connect()

    job = delegate.reattach(job_id="job1", callback_runner=None)
    assert job.was_found is True
    assert job.get_status() == JobStatus.FINISHED

    stub = delegate.reattach(job_id="unknown", callback_runner=None)
    assert stub.was_found is False
    assert stub.get_status() == JobStatus.UNKNOWN
