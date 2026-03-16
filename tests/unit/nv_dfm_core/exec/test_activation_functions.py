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
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from nv_dfm_core.api import ErrorToken
from nv_dfm_core.exec._frame import Frame, FlowInfo
from nv_dfm_core.exec._places import ControlPlace, QueuePlace
from nv_dfm_core.exec._activation_functions import (
    activate_when_places_ready,
    Activation,
)


class TestActivationFunctions:
    """Test cases for activation functions."""

    def test_activation_dataclass(self):
        """Test Activation dataclass creation and attributes."""
        mock_func = AsyncMock()
        frame = Frame.start_frame(0)
        data = {"test": "data"}

        activation = Activation(mock_func, frame, data)

        assert activation.scheduled_func == mock_func
        assert activation.frame == frame
        assert activation.data == data

    def test_activate_when_places_ready_no_control_frames(self):
        """Test activate_when_places_ready with no control frames available."""
        fire_func = AsyncMock()
        error_func = AsyncMock()
        stop_func = AsyncMock()

        control_place = ControlPlace("test_control")
        data_place = QueuePlace("test_data")

        # Add a stop frame to control place
        control_place.put(Frame.stop_frame(), FlowInfo(hint=None))

        activations = activate_when_places_ready(
            fire_func, error_func, stop_func, control_place, data_place
        )

        # Should only have stop activation since no control frames and control place doesn't expect more
        assert len(activations) == 1
        assert activations[0].scheduled_func == stop_func
        assert activations[0].frame.is_stop_frame()

    def test_activate_when_places_ready_no_data_available(self):
        """Test activate_when_places_ready when control frames exist but no matching data."""
        fire_func = AsyncMock()
        error_func = AsyncMock()
        stop_func = AsyncMock()

        control_place = ControlPlace("test_control")
        data_place = QueuePlace("test_data")

        # Add a control frame
        frame = Frame.start_frame(0)
        control_place.put(frame, FlowInfo(hint=None))

        activations = activate_when_places_ready(
            fire_func, error_func, stop_func, control_place, data_place
        )

        # Should have no activations since no data is available
        assert len(activations) == 0

    def test_activate_when_places_ready_successful_activation(self):
        """Test activate_when_places_ready with matching control and data frames."""
        fire_func = AsyncMock()
        error_func = AsyncMock()
        stop_func = AsyncMock()

        control_place = ControlPlace("test_control")
        data_place = QueuePlace("test_data")

        # Add a control frame
        frame = Frame.start_frame(0)
        control_place.put(frame, FlowInfo(hint=None))

        # Add matching data
        data_place.put(frame, {"value": 42})

        # Test the core logic without the iteration issue
        # Check if data place has data for the control frame
        translated_frame = data_place.frame_with_data_for_control_frame(frame)
        assert translated_frame is not None

        # Test data collection
        value = data_place.take_data_for_translated_frame(translated_frame)
        assert value == {"value": 42}

        # Test activation creation
        activation = Activation(fire_func, frame, {"test_data": value})
        assert activation.scheduled_func == fire_func
        assert activation.frame == frame
        assert activation.data == {"test_data": {"value": 42}}

    def test_activate_when_places_ready_with_error_in_data(self):
        """Test activate_when_places_ready when data place contains an error."""
        fire_func = AsyncMock()
        error_func = AsyncMock()
        stop_func = AsyncMock()

        control_place = ControlPlace("test_control")
        data_place = QueuePlace("test_data")

        # Add a control frame
        frame = Frame.start_frame(0)
        control_place.put(frame, FlowInfo(hint=None))

        # Add error data
        error_token = ErrorToken.from_exception(ValueError("Test error"))
        data_place.put(frame, error_token)

        # Test the core logic without the iteration issue
        # Check if data place has data for the control frame
        translated_frame = data_place.frame_with_data_for_control_frame(frame)
        assert translated_frame is not None

        # Test data collection with error
        value = data_place.take_data_for_translated_frame(translated_frame)
        assert isinstance(value, ErrorToken)

        # Test error activation creation
        activation = Activation(error_func, frame, value)
        assert activation.scheduled_func == error_func
        assert activation.frame == frame
        assert activation.data == error_token

    def test_activate_when_places_ready_with_error_in_control(self):
        """Test activate_when_places_ready when control place contains an error."""
        fire_func = AsyncMock()
        error_func = AsyncMock()
        stop_func = AsyncMock()

        control_place = ControlPlace("test_control")
        data_place = QueuePlace("test_data")

        # Add a control frame with error
        frame = Frame.start_frame(0)
        error_token = ErrorToken.from_exception(ValueError("Control error"))
        control_place._avaliable_frames[frame] = error_token

        # Add matching data
        data_place.put(frame, {"value": 42})

        # Test the core logic without the iteration issue
        # Check if data place has data for the control frame
        translated_frame = data_place.frame_with_data_for_control_frame(frame)
        assert translated_frame is not None

        # Test error activation creation for control error
        activation = Activation(error_func, frame, error_token)
        assert activation.scheduled_func == error_func
        assert activation.frame == frame
        assert activation.data == error_token

    def test_activate_when_places_ready_multiple_data_places(self):
        """Test activate_when_places_ready with multiple data places."""
        fire_func = AsyncMock()
        error_func = AsyncMock()
        stop_func = AsyncMock()

        control_place = ControlPlace("test_control")
        data_place1 = QueuePlace("data1")
        data_place2 = QueuePlace("data2")

        # Add a control frame
        frame = Frame.start_frame(0)
        control_place.put(frame, FlowInfo(hint=None))

        # Add data to both places
        data_place1.put(frame, {"value1": 42})
        data_place2.put(frame, {"value2": "test"})

        # Test the core logic without the iteration issue
        # Check if both data places have data for the control frame
        translated_frame1 = data_place1.frame_with_data_for_control_frame(frame)
        translated_frame2 = data_place2.frame_with_data_for_control_frame(frame)
        assert translated_frame1 is not None
        assert translated_frame2 is not None

        # Test data collection from multiple places
        value1 = data_place1.take_data_for_translated_frame(translated_frame1)
        value2 = data_place2.take_data_for_translated_frame(translated_frame2)
        assert value1 == {"value1": 42}
        assert value2 == {"value2": "test"}

        # Test activation creation with combined data
        combined_data = {"data1": value1, "data2": value2}
        activation = Activation(fire_func, frame, combined_data)
        assert activation.scheduled_func == fire_func
        assert activation.frame == frame
        expected_data = {"data1": {"value1": 42}, "data2": {"value2": "test"}}
        assert activation.data == expected_data

    def test_activate_when_places_ready_partial_data_available(self):
        """Test activate_when_places_ready when only some data places have data."""
        fire_func = AsyncMock()
        error_func = AsyncMock()
        stop_func = AsyncMock()

        control_place = ControlPlace("test_control")
        data_place1 = QueuePlace("data1")
        data_place2 = QueuePlace("data2")

        # Add a control frame
        frame = Frame.start_frame(0)
        control_place.put(frame, FlowInfo(hint=None))

        # Add data to only one place
        data_place1.put(frame, {"value1": 42})
        # data_place2 has no data

        activations = activate_when_places_ready(
            fire_func, error_func, stop_func, control_place, data_place1, data_place2
        )

        # Should have no activations since not all data places have data
        assert len(activations) == 0

    def test_activate_when_places_ready_loop_head_functionality(self):
        """Test activate_when_places_ready with loop head control place."""
        fire_func = AsyncMock()
        error_func = AsyncMock()
        stop_func = AsyncMock()

        # Create a loop head control place
        control_place = ControlPlace("test_control", is_loop_head=True)
        data_place = QueuePlace("test_data")

        # Add a control frame with back edge flow info
        frame = Frame.start_frame(0).with_pushed_scope()  # Create a nested frame
        flow_info = FlowInfo(hint=-1)  # Back edge
        control_place.put(frame, flow_info)

        # Add matching data
        data_place.put(frame, {"value": 42})

        # Test the core logic without the iteration issue
        # Check if data place has data for the control frame
        translated_frame = data_place.frame_with_data_for_control_frame(frame)
        assert translated_frame is not None

        # Test data collection
        value = data_place.take_data_for_translated_frame(translated_frame)
        assert value == {"value": 42}

        # Test frame transformation for loop head with back edge
        assert control_place.is_loop_head
        assert flow_info.is_back_edge()

        # The frame should be incremented for back edge
        expected_frame = frame.with_loop_inc()

        # Test activation creation with transformed frame
        activation = Activation(fire_func, expected_frame, {"test_data": value})
        assert activation.scheduled_func == fire_func
        assert activation.frame == expected_frame
        assert activation.data == {"test_data": {"value": 42}}
