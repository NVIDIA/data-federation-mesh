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

# pyright: reportPrivateUsage=false, reportMissingTypeArgument=false, reportUnknownVariableType=false, reportUnknownParameterType=false
import threading
import time
from multiprocessing import Queue
from unittest.mock import MagicMock, patch

import pytest

from nv_dfm_core.exec import TokenPackage
from nv_dfm_core.exec._frame import Frame
from nv_dfm_core.gen.modgen.ir import (
    START_PLACE_NAME,
    AdapterCallStmt,
    BoundNetIR,
    InPlace,
    NetIR,
    ReadPlaceStmt,
    StmtRef,
    Transition,
)
from nv_dfm_core.gen.modgen.ir._activation_statements import ActivateWhenPlacesReady
from nv_dfm_core.gen.modgen.ir._ir_stmts import IRStmt, TokenSend
from nv_dfm_core.targets.local._job_runner import JobRunner, JobSubmission


@pytest.fixture
def sample_net_ir():
    """Create a sample NetIR for testing."""
    fire_body: list[IRStmt] = [
        ReadPlaceStmt(stmt_id="data", place="input"),
        AdapterCallStmt(
            stmt_id="result",
            has_users=True,
            adapter="process",
            literal_params={},
            stmt_params={"data": StmtRef(stmt_id="data")},
            is_async=True,
        ),
        TokenSend(
            job=None,
            node_id=None,
            site="output_site",
            place="output",
            is_yield=False,
            kind="data",
            data=StmtRef(stmt_id="result"),
        ),
    ]
    return NetIR(
        pipeline_name="test",
        site="test_site",
        transitions=[
            Transition(
                control_place=InPlace(
                    name=START_PLACE_NAME,
                    kind="control",
                    origin="external",
                    flavor="seq_control",
                    type="FlowInfo",
                ),
                data_places=[
                    InPlace(
                        name="input",
                        kind="data",
                        origin="external",
                        flavor="scoped",
                        type="str",
                    )
                ],
                try_activate_func=ActivateWhenPlacesReady(),
                fire_body=fire_body,
                signal_error_body=[],
                signal_stop_body=[],
            )
        ],
    )


@pytest.fixture
def sample_bound_net_ir(sample_net_ir: NetIR):
    """Create a sample BoundNetIR for testing."""
    bound_net_ir = BoundNetIR(
        site="test_site",
        ir=sample_net_ir,
        tagged_input_params=[(Frame.start_frame(0), {"input": ("json", "test_value")})],
    )
    return bound_net_ir


@pytest.fixture
def sample_job_submission(sample_bound_net_ir: BoundNetIR):
    """Create a sample JobSubmission for testing."""
    return JobSubmission(
        pipeline_api_version="1.0",
        federation_name="test_federation",
        job_id="test_job_123",
        homesite="home_site",
        netir=sample_bound_net_ir,
        force_modgen=True,
    )


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock()


@pytest.fixture
def sample_channels() -> dict[str, Queue]:
    """Create sample channels for testing."""
    return {
        "test_site": Queue(),
        "other_site": Queue(),
    }


@pytest.fixture
def sample_queues() -> dict[str, Queue]:
    """Create sample queues for testing."""
    return {
        "yield_queue": Queue(),
        "inter_job_queue": Queue(),
    }


@pytest.fixture
def job_runner(
    sample_channels: dict[str, Queue],
    sample_queues: dict[str, Queue],
    mock_logger: MagicMock,
):
    """Create a JobRunner instance for testing."""
    return JobRunner(
        site="test_site",
        channels=sample_channels,
        yield_queue=sample_queues["yield_queue"],
        inter_job_queue=sample_queues["inter_job_queue"],
        logger=mock_logger,
    )


class TestJobRunnerInitialization:
    """Test JobRunner initialization and setup."""

    def test_job_runner_initialization(
        self,
        sample_channels: dict[str, Queue],
        sample_queues: dict[str, Queue],
        mock_logger: MagicMock,
    ):
        """Test that JobRunner can be initialized correctly."""
        job_runner = JobRunner(
            site="test_site",
            channels=sample_channels,
            yield_queue=sample_queues["yield_queue"],
            inter_job_queue=sample_queues["inter_job_queue"],
            logger=mock_logger,
        )

        assert job_runner._site == "test_site"
        assert job_runner._channels == sample_channels
        assert job_runner._yield_queue == sample_queues["yield_queue"]
        assert job_runner._inter_job_queue == sample_queues["inter_job_queue"]
        assert job_runner._logger == mock_logger
        assert job_runner.daemon is True
        assert hasattr(job_runner.shutdown_event, "is_set")
        assert hasattr(job_runner.abort_netrunner_event, "is_set")
        assert job_runner._routing_thread is None
        assert isinstance(job_runner._router_thread_shutdown_event, threading.Event)

    def test_job_runner_initialization_site_not_in_channels(
        self, sample_queues: dict[str, Queue], mock_logger: MagicMock
    ):
        """Test that JobRunner raises an error if the site is not in channels."""
        channels: dict[str, Queue] = {"other_site": Queue()}

        with pytest.raises(
            AssertionError, match="Site test_site not found in channels"
        ):
            _ = JobRunner(
                site="test_site",
                channels=channels,
                yield_queue=sample_queues["yield_queue"],
                inter_job_queue=sample_queues["inter_job_queue"],
                logger=mock_logger,
            )


class TestJobRunnerRoutingThread:
    """Test JobRunner routing thread management."""

    def test_start_routing_thread(self, job_runner: JobRunner):
        """Test starting the routing thread."""
        mock_router = MagicMock()

        # Initially no routing thread
        assert job_runner._routing_thread is None

        job_runner._start_routing_thread(mock_router)

        # Thread should be created and started
        assert job_runner._routing_thread is not None
        assert job_runner._routing_thread.is_alive()
        assert job_runner._routing_thread.daemon is True

    def test_start_routing_thread_already_running(self, job_runner: JobRunner):
        """Test starting routing thread when one is already running."""
        mock_router = MagicMock()

        # Start first thread
        job_runner._start_routing_thread(mock_router)
        first_thread = job_runner._routing_thread

        # Start second thread
        job_runner._start_routing_thread(mock_router)
        second_thread = job_runner._routing_thread

        # Should be the same thread instance
        assert first_thread is second_thread

    def test_stop_routing_thread(self, job_runner: JobRunner):
        """Test stopping the routing thread."""
        mock_router = MagicMock()

        # Start thread
        job_runner._start_routing_thread(mock_router)
        assert job_runner._routing_thread is not None
        assert job_runner._routing_thread.is_alive()

        # Stop thread
        job_runner._stop_routing_thread()

        # Thread should be stopped
        assert job_runner._routing_thread is None

    def test_stop_routing_thread_no_thread(self, job_runner: JobRunner):
        """Test stopping routing thread when none exists."""
        # Should not raise an error
        job_runner._stop_routing_thread()
        assert job_runner._routing_thread is None

    def test_monitor_and_route_input_queue(self, job_runner: JobRunner):
        """Test the background monitoring thread functionality."""
        mock_router = MagicMock()
        # Create a real TokenPackage instead of a mock to avoid pickling issues
        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="test_site",
            target_place="target_place",
            target_job="target_job",
            is_yield=False,
            frame=Frame.start_frame(0),
            data={"test": "data"},
        )

        # Put a token in the input channel
        job_runner._channels["test_site"].put(token_package)

        # Start the monitoring thread
        job_runner._start_routing_thread(mock_router)

        # Give it time to process
        time.sleep(0.2)

        # Stop the thread
        job_runner._stop_routing_thread()

        # Verify the router was called
        mock_router.route_token_package_sync.assert_called_with(token_package)

    def test_monitor_and_route_input_queue_empty_queue(self, job_runner: JobRunner):
        """Test monitoring thread with empty queue."""
        mock_router = MagicMock()

        # Start the monitoring thread
        job_runner._start_routing_thread(mock_router)

        # Give it time to process
        time.sleep(0.2)

        # Stop the thread
        job_runner._stop_routing_thread()

        # Router should not be called since queue is empty
        mock_router.route_token_package_sync.assert_not_called()

    def test_monitor_and_route_input_queue_none_token(self, job_runner: JobRunner):
        """Test monitoring thread with None token."""
        mock_router = MagicMock()

        # Put None in the input channel
        job_runner._channels["test_site"].put(None)  # pyright: ignore[reportArgumentType]

        # Start the monitoring thread
        job_runner._start_routing_thread(mock_router)

        # Give it time to process
        time.sleep(0.2)

        # Stop the thread
        job_runner._stop_routing_thread()

        # Router should not be called for None token
        mock_router.route_token_package_sync.assert_not_called()

    def test_monitor_and_route_input_queue_exception(self, job_runner: JobRunner):
        """Test monitoring thread handles exceptions gracefully."""
        mock_router = MagicMock()
        mock_router.route_token_package_sync.side_effect = Exception("Test error")
        # Create a real TokenPackage instead of a mock to avoid pickling issues
        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="test_site",
            target_place="target_place",
            target_job="target_job",
            is_yield=False,
            frame=Frame.start_frame(0),
            data={"test": "data"},
        )

        # Put a token in the input channel
        job_runner._channels["test_site"].put(token_package)

        # Start the monitoring thread
        job_runner._start_routing_thread(mock_router)

        # Give it time to process
        time.sleep(0.2)

        # Stop the thread
        job_runner._stop_routing_thread()

        # Verify error was logged
        assert isinstance(job_runner._logger.error, MagicMock)
        job_runner._logger.error.assert_called()
        error_call = job_runner._logger.error.call_args[0][0]
        assert "Background monitor thread error" in error_call
        assert "Test error" in error_call


class TestJobRunnerJobSubmission:
    """Test JobRunner job submission handling."""

    @patch("nv_dfm_core.targets.local._job_runner.LocalRouter")
    @patch("nv_dfm_core.targets.local._job_runner.JobController")
    def test_run_with_job_submission(
        self,
        mock_job_controller_class: MagicMock,
        mock_local_router_class: MagicMock,
        job_runner: JobRunner,
        sample_job_submission: JobSubmission,
    ):
        """Test running JobRunner with a job submission."""
        mock_router = MagicMock()
        mock_local_router_class.return_value = mock_router

        mock_controller = MagicMock()
        mock_job_controller_class.return_value = mock_controller

        # Put job submission in command queue

        # Set shutdown event to stop after processing one job
        def stop_after_job():
            time.sleep(0.1)
            job_runner.command_queue.put(sample_job_submission)
            time.sleep(0.1)
            job_runner.shutdown_event.set()

        stop_thread = threading.Thread(target=stop_after_job)
        stop_thread.daemon = True
        stop_thread.start()

        # Run the job runner
        job_runner.run()

        # Verify LocalRouter was created with correct parameters
        mock_local_router_class.assert_called_once_with(
            channels=job_runner._channels,
            inter_job_queue=job_runner._inter_job_queue,
            yield_queue=job_runner._yield_queue,
        )

        # Verify JobController was created with correct parameters
        # (trace_id is now derived from job_id inside the telemetry collector)
        mock_job_controller_class.assert_called_once_with(
            router=mock_router,
            pipeline_api_version=sample_job_submission.pipeline_api_version,
            federation_name=sample_job_submission.federation_name,
            homesite=sample_job_submission.homesite,
            this_site=job_runner._site,
            job_id=sample_job_submission.job_id,
            netir=sample_job_submission.netir,
            logger=job_runner._logger,
            force_modgen=sample_job_submission.force_modgen,
        )

        # Verify controller methods were called
        mock_controller.start.assert_called_once()
        mock_controller.submit_initial_tokens.assert_called_once()
        mock_controller.wait_for_done.assert_called_once()

    def test_run_with_wrong_site_netir(
        self, job_runner: JobRunner, sample_bound_net_ir: BoundNetIR
    ):
        """Test that JobRunner raises an error for netir with wrong site."""
        # Create job submission with netir for different site
        wrong_site_netir = sample_bound_net_ir
        wrong_site_netir.site = "wrong_site"

        job_submission = JobSubmission(
            pipeline_api_version="1.0",
            federation_name="test_federation",
            job_id="test_job_123",
            homesite="home_site",
            netir=wrong_site_netir,
            force_modgen=True,
        )

        # Put job submission in command queue
        job_runner.command_queue.put(job_submission)

        # Set shutdown event to stop after processing
        def stop_after_job():
            time.sleep(0.1)
            job_runner.shutdown_event.set()

        stop_thread = threading.Thread(target=stop_after_job)
        stop_thread.daemon = True
        stop_thread.start()

        # Run the job runner
        job_runner.run()

        # Verify error was logged
        assert isinstance(job_runner._logger.error, MagicMock)
        job_runner._logger.error.assert_called()
        error_call = job_runner._logger.error.call_args[0][0]
        assert (
            "JobRunner for site test_site received a netir for wrong site wrong_site"
            in error_call
        )

    def test_run_with_empty_queue(self, job_runner: JobRunner):
        """Test running JobRunner with empty command queue."""

        # Set shutdown event to stop after a short time
        def stop_after_time():
            time.sleep(0.1)
            job_runner.shutdown_event.set()

        stop_thread = threading.Thread(target=stop_after_time)
        stop_thread.daemon = True
        stop_thread.start()

        # Run the job runner
        job_runner.run()

        # Should not raise any errors and should exit cleanly

    def test_run_with_none_job(self, job_runner: JobRunner):
        """Test running JobRunner with None job submission."""
        # Put None in command queue
        job_runner.command_queue.put(None)  # pyright: ignore[reportArgumentType]

        # Set shutdown event to stop after processing
        def stop_after_job():
            time.sleep(0.1)
            job_runner.shutdown_event.set()

        stop_thread = threading.Thread(target=stop_after_job)
        stop_thread.daemon = True
        stop_thread.start()

        # Run the job runner
        job_runner.run()

        # Should not raise any errors and should exit cleanly

    @patch("nv_dfm_core.targets.local._job_runner.LocalRouter")
    @patch("nv_dfm_core.targets.local._job_runner.JobController")
    def test_run_with_controller_exception(
        self,
        mock_job_controller_class: MagicMock,
        mock_local_router_class: MagicMock,
        job_runner: JobRunner,
        sample_job_submission: JobSubmission,
    ):
        """Test running JobRunner when controller raises an exception."""
        mock_router = MagicMock()
        mock_local_router_class.return_value = mock_router

        mock_controller = MagicMock()
        mock_controller.start.side_effect = Exception("Controller error")
        mock_job_controller_class.return_value = mock_controller

        # Put job submission in command queue
        job_runner.command_queue.put(sample_job_submission)

        # Set shutdown event to stop after processing one job
        def stop_after_job():
            time.sleep(0.1)
            job_runner.shutdown_event.set()

        stop_thread = threading.Thread(target=stop_after_job)
        stop_thread.daemon = True
        stop_thread.start()

        # Run the job runner
        job_runner.run()

        # Verify error was logged
        assert isinstance(job_runner._logger.error, MagicMock)
        job_runner._logger.error.assert_called()
        error_call = job_runner._logger.error.call_args[0][0]
        assert "An error occurred during job execution" in error_call
        assert "Controller error" in error_call

    @patch("nv_dfm_core.targets.local._job_runner.LocalRouter")
    @patch("nv_dfm_core.targets.local._job_runner.JobController")
    def test_run_cleanup_on_exception(
        self,
        mock_job_controller_class: MagicMock,
        mock_local_router_class: MagicMock,
        job_runner: JobRunner,
        sample_job_submission: JobSubmission,
    ):
        """Test that cleanup happens even when controller raises an exception."""
        mock_router = MagicMock()
        mock_local_router_class.return_value = mock_router

        mock_controller = MagicMock()
        mock_controller.start.side_effect = Exception("Controller error")
        mock_job_controller_class.return_value = mock_controller

        # Put job submission in command queue
        job_runner.command_queue.put(sample_job_submission)

        # Set shutdown event to stop after processing one job
        def stop_after_job():
            time.sleep(0.1)
            job_runner.shutdown_event.set()

        stop_thread = threading.Thread(target=stop_after_job)
        stop_thread.daemon = True
        stop_thread.start()

        # Run the job runner
        job_runner.run()

        # Verify abort event was set
        assert job_runner.abort_netrunner_event.is_set()


class TestJobRunnerIntegration:
    """Integration tests for JobRunner."""

    def test_job_runner_lifecycle(
        self,
        sample_channels: dict[str, Queue],
        sample_queues: dict[str, Queue],
        mock_logger: MagicMock,
        sample_job_submission: JobSubmission,
    ):
        """Test complete JobRunner lifecycle."""
        job_runner = JobRunner(
            site="test_site",
            channels=sample_channels,
            yield_queue=sample_queues["yield_queue"],
            inter_job_queue=sample_queues["inter_job_queue"],
            logger=mock_logger,
        )

        # Verify initial state
        assert job_runner._routing_thread is None
        assert not job_runner.shutdown_event.is_set()
        assert not job_runner.abort_netrunner_event.is_set()

        # Test that we can put a job submission in the queue
        job_runner.command_queue.put(sample_job_submission)

        # Verify the job submission is in the queue
        retrieved_job = job_runner.command_queue.get()
        assert retrieved_job == sample_job_submission
        assert retrieved_job.pipeline_api_version == "1.0"
        assert retrieved_job.federation_name == "test_federation"
        assert retrieved_job.job_id == "test_job_123"
        assert retrieved_job.homesite == "home_site"
        assert retrieved_job.netir.site == "test_site"

    def test_job_runner_multiple_sites(
        self, sample_queues: dict[str, Queue], mock_logger: MagicMock
    ):
        """Test JobRunner with multiple sites in channels."""
        channels = {
            "site1": Queue(),
            "site2": Queue(),
            "site3": Queue(),
        }

        job_runner = JobRunner(
            site="site2",
            channels=channels,
            yield_queue=sample_queues["yield_queue"],
            inter_job_queue=sample_queues["inter_job_queue"],
            logger=mock_logger,
        )

        assert job_runner._site == "site2"
        assert "site1" in job_runner._channels
        assert "site2" in job_runner._channels
        assert "site3" in job_runner._channels
        assert job_runner._channels["site2"] is not None


class TestJobSubmission:
    """Test JobSubmission dataclass."""

    def test_job_submission_creation(self, sample_bound_net_ir: BoundNetIR):
        """Test creating a JobSubmission instance."""
        job_submission = JobSubmission(
            pipeline_api_version="2.0",
            federation_name="my_federation",
            job_id="job_456",
            homesite="my_home",
            netir=sample_bound_net_ir,
            force_modgen=True,
        )

        assert job_submission.pipeline_api_version == "2.0"
        assert job_submission.federation_name == "my_federation"
        assert job_submission.job_id == "job_456"
        assert job_submission.homesite == "my_home"
        assert job_submission.netir == sample_bound_net_ir

    def test_job_submission_equality(self, sample_bound_net_ir: BoundNetIR):
        """Test JobSubmission equality."""
        job1 = JobSubmission(
            pipeline_api_version="1.0",
            federation_name="test",
            job_id="123",
            homesite="home",
            netir=sample_bound_net_ir,
            force_modgen=True,
        )

        job2 = JobSubmission(
            pipeline_api_version="1.0",
            federation_name="test",
            job_id="123",
            homesite="home",
            netir=sample_bound_net_ir,
            force_modgen=True,
        )

        assert job1 == job2
