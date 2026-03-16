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

import logging
import threading
import time
import types
from dataclasses import dataclass
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nv_dfm_core.api import ErrorToken
from nv_dfm_core.exec import NetRunner, QueuePlace
from nv_dfm_core.exec._activation_functions import (
    Activation,
    TransitionTryActivateFunc,
    activate_when_places_ready,
)
from nv_dfm_core.exec._dfm_context import DfmContext
from nv_dfm_core.exec._frame import FlowInfo, Frame
from nv_dfm_core.exec._net import Net
from nv_dfm_core.exec._places import ControlPlace
from nv_dfm_core.exec._site import Site


@dataclass
class ThisNet(Net):
    t1_control: ControlPlace = ControlPlace("t1_control")
    t2_control: ControlPlace = ControlPlace("t2_control")
    t1_data: QueuePlace = QueuePlace("t1_data")

    def __init__(self):
        # hm, not sure why this is needed but without this empty constructor, the
        # test apparently only creates a single instance of this Net class and then
        # the places aren't empty in the beginning of the test. This is nothing that should
        # happen at runtime, but it does here. Maybe some pytest optimization when creating
        # fixtures?
        super().__init__()

    def get_activation_functions(self) -> list[TransitionTryActivateFunc]:
        return [
            self.t1_try_activate,
            self.t2_try_activate,
        ]

    # Add transition functions
    def t1_try_activate(self) -> list[Activation]:
        res = activate_when_places_ready(
            self.t1_fire,
            self.t1_signal_error,
            self.t1_signal_stop,
            self.t1_control,
            self.t1_data,
        )
        return res

    async def t1_fire(
        self,
        site: Site,  # pyright: ignore[reportUnusedParameter]
        dfm_context: DfmContext,
        frame: Frame,
        data: dict[str, Any] | ErrorToken,
    ):
        assert isinstance(data, dict)
        await dfm_context.send_to_place(
            to_job=None,
            to_site="remote",
            to_place="output",
            data=data["t1_data"],
            is_yield=False,
            frame=frame,
            node_id=None,
        )

    async def t1_signal_stop(
        self,
        site: Site,  # pyright: ignore[reportUnusedParameter]
        dfm_context: DfmContext,
        frame: Frame,
        data: dict[str, Any] | ErrorToken,
    ):
        await dfm_context.send_to_place(
            to_job=None,
            to_site="remote",
            to_place="output",
            data=data,
            is_yield=False,
            frame=frame,
            node_id=None,
        )

    async def t1_signal_error(
        self,
        site: Site,  # pyright: ignore[reportUnusedParameter]
        dfm_context: DfmContext,
        frame: Frame,
        data: dict[str, Any] | ErrorToken,
    ):
        await dfm_context.send_to_place(
            to_job=None,
            to_site="remote",
            to_place="output",
            data=data,
            is_yield=False,
            frame=frame,
            node_id=None,
        )

    def t2_try_activate(self) -> list[Activation]:
        res = activate_when_places_ready(
            self.t2_fire,
            self.t1_signal_error,  # we just reuse the same error and stop functions
            self.t1_signal_stop,
            self.t2_control,
        )
        return res

    async def t2_fire(
        self,
        site: Site,  # pyright: ignore[reportUnusedParameter]
        dfm_context: DfmContext,  # pyright: ignore[reportUnusedParameter]
        frame: Frame,  # pyright: ignore[reportUnusedParameter]
        data: dict[str, Any] | ErrorToken,  # pyright: ignore[reportUnusedParameter]
    ):
        # NOTE: in compiled code, there will be try/excepts around the core body of a transition and "normal" exceptions
        # should not be raised up to the netrunner. This test is testing an exception that does escap into the netrunner.
        raise ValueError("Expected uncaught test error")


def create_mock_net_module() -> ModuleType:
    """Create a mock net module with places and transitions."""
    # Create a new module
    module = types.ModuleType("mock_net_module")

    # Create places

    setattr(module, "ThisNet", ThisNet)

    return module


@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def net_runner(mock_logger: logging.Logger):
    net_module = create_mock_net_module()
    # from nv_dfm_core.session._session import configure_session_logging
    logger = mock_logger  # configure_session_logging()

    context = MagicMock(spec=DfmContext)
    context.yield_error = AsyncMock()
    context.yield_control_token = AsyncMock()
    context.send_to_place = AsyncMock()

    return NetRunner(context, net_module, logger)


def test_places_are_created_correctly(net_runner: NetRunner):
    """Test that places are created and accessible."""
    assert "t1_control" in net_runner._places
    assert "t1_data" in net_runner._places
    assert isinstance(net_runner._places["t1_control"], ControlPlace)
    assert isinstance(net_runner._places["t1_data"], QueuePlace)


def test_receive_token_transaction(net_runner: NetRunner):
    """Test that receive_token_transaction properly manages the lock."""
    # Test that we can't access places outside transaction
    with pytest.raises(ValueError):
        with net_runner.receive_token_transaction() as transaction:
            transaction.receive_token("unknown_place", Frame.start_frame(0), "test1")

    # Test that we can access places inside transaction
    with net_runner.receive_token_transaction() as transaction:
        transaction.receive_token("t1_control", Frame.start_frame(0), FlowInfo())
        transaction.receive_token("t1_data", Frame.start_frame(0), "test1")

    # Verify the data was actually put in the places

    assert isinstance(net_runner._places["t1_control"], ControlPlace)
    assert isinstance(net_runner._places["t1_data"], QueuePlace)
    assert net_runner._places["t1_control"].peek_frames_copy() == {
        Frame.start_frame(0): FlowInfo()
    }
    assert net_runner._places["t1_data"]._queues[Frame.start_frame(0)] == "test1"


def test_start_and_shutdown(net_runner: NetRunner):
    """Test that start and shutdown work correctly."""
    # Start the net runner
    net_runner.start()
    assert net_runner._check_activations_thread is not None

    # Wait for thread to become alive (with timeout)
    for _ in range(10):  # Try for up to 1 second
        if net_runner._check_activations_thread.is_alive():
            break
        time.sleep(0.1)
    assert net_runner._check_activations_thread.is_alive()

    # Shutdown should stop the thread and set it to None
    net_runner.shutdown()
    assert net_runner._check_activations_thread is None


def test_wait_for_done(net_runner: NetRunner):
    """Test that wait_for_done works correctly."""
    net_runner.start()

    # Create a mock abort signal
    abort_signal = MagicMock()
    abort_signal.triggered = False

    # Start a thread that will trigger the abort signal after a short delay
    def trigger_abort():
        time.sleep(0.1)
        abort_signal.triggered = True

    threading.Thread(target=trigger_abort, daemon=True).start()

    # wait_for_done should return when abort signal is triggered
    net_runner.wait_for_done(abort_signal)
    assert abort_signal.triggered


@pytest.mark.asyncio
async def test_transition_exception_handling(net_runner: NetRunner):
    """Test that exceptions in transitions are properly handled."""

    # Start the net runner
    net_runner.start()

    # Put some data in the input place to trigger the transition
    with net_runner.receive_token_transaction() as transaction:
        transaction.receive_token("t2_control", Frame.start_frame(0), FlowInfo())

    net_runner.wait_for_done(None)

    assert net_runner.error_occurred()

    # Cleanup
    net_runner.shutdown()


@pytest.mark.asyncio
async def test_transition_success(net_runner: NetRunner):
    """Test that successful transitions work correctly."""
    # Start the net runner
    net_runner.start()

    # Put some data in the input place to trigger the transition
    with net_runner.receive_token_transaction() as transaction:
        transaction.receive_token("t1_control", Frame.start_frame(0), FlowInfo())
        transaction.receive_token("t1_data", Frame.start_frame(0), "test1")

    # Wait for the transition to complete
    time.sleep(0.1)

    # Verify that the data was sent to the remote place
    mock_dfm_context = net_runner.dfm_context
    assert isinstance(mock_dfm_context, MagicMock)
    mock_dfm_context.send_to_place.assert_called_once_with(
        to_job=None,
        to_site="remote",
        to_place="output",
        data="test1",
        is_yield=False,
        frame=Frame(token="@dfm-control-token", frame=[0]),
        node_id=None,
    )

    # Cleanup
    net_runner.shutdown()


def test_stop_frame_handling(net_runner: NetRunner):
    """Test that stop tokens are handled correctly."""
    # Start the net runner
    net_runner.start()

    len_before = len(net_runner._transitions)

    # Put a stop token in the input place
    with net_runner.receive_token_transaction() as transaction:
        transaction.receive_token("t1_control", Frame.stop_frame(), FlowInfo())

    # Give the transition some time to complete, but don't wait forever
    start_time = time.time()
    while time.time() - start_time < 2:
        if len(net_runner._transitions) == len_before - 1:
            break
        time.sleep(0.01)

    # Verify that the stop token was sent to the output place
    mock_dfm_context = net_runner.dfm_context
    assert isinstance(mock_dfm_context, MagicMock)
    mock_dfm_context.send_to_place.assert_called_once_with(
        to_job=None,
        to_site="remote",
        to_place="output",
        data={},
        is_yield=False,
        frame=Frame.stop_frame(),
        node_id=None,
    )

    # Verify that the transition was removed
    assert len(net_runner._transitions) == len_before - 1

    # Cleanup
    net_runner.shutdown()
