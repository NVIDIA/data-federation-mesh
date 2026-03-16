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

from nv_dfm_core.exec._frame import Frame, FlowInfo
from nv_dfm_core.exec._places import ControlPlace, QueuePlace, CountingPlace


class TestControlPlace:
    """Test cases for ControlPlace class."""

    def test_initialization(self):
        """Test ControlPlace initialization."""
        place = ControlPlace("test_control")
        assert place.name == "test_control"
        assert not place._received_stop_frame
        assert place._avaliable_frames == {}
        assert place._remembered_frames == set()

    def test_initialization_with_loop_head(self):
        """Test ControlPlace initialization with loop head flag."""
        place = ControlPlace("test_control", is_loop_head=True)
        assert place.name == "test_control"

    def test_put_flow_info(self):
        """Test putting FlowInfo data into control place."""
        place = ControlPlace("test_control")
        frame = Frame.start_frame(0)
        flow_info = FlowInfo(hint=None)

        place.put(frame, flow_info)
        assert frame in place._avaliable_frames
        assert place._avaliable_frames[frame] == flow_info

    def test_put_stop_frame(self):
        """Test putting stop frame into control place."""
        place = ControlPlace("test_control")
        stop_frame = Frame.stop_frame()
        flow_info = FlowInfo(hint=None)

        place.put(stop_frame, flow_info)
        assert place._received_stop_frame
        assert stop_frame not in place._avaliable_frames

    def test_put_invalid_data_type(self):
        """Test that putting non-FlowInfo data raises assertion error."""
        place = ControlPlace("test_control")
        frame = Frame.start_frame(0)

        with pytest.raises(
            AssertionError,
            match="Control Place test_control: control frames must come with FlowInfo or ErrorToken data, got <class 'str'>",
        ):
            place.put(frame, "invalid_data")

    def test_remember_and_forget_frame(self):
        """Test remembering and forgetting frames."""
        place = ControlPlace("test_control")
        frame = Frame.start_frame(0)

        place.remember_frame(frame)
        assert frame in place._remembered_frames

        place.forget_frame(frame)
        assert frame not in place._remembered_frames

    def test_forget_nonexistent_frame(self):
        """Test that forgetting a non-remembered frame raises assertion error."""
        place = ControlPlace("test_control")
        frame = Frame.start_frame(0)

        with pytest.raises(AssertionError, match="Frame.*not in remembered frames"):
            place.forget_frame(frame)

    def test_expects_more_no_stop_frame(self):
        """Test expects_more when no stop frame received."""
        place = ControlPlace("test_control")
        assert place.expects_more()

    def test_expects_more_with_stop_frame_no_outstanding_work(self):
        """Test expects_more when stop frame received but no outstanding work."""
        place = ControlPlace("test_control")
        stop_frame = Frame.stop_frame()
        place.put(stop_frame, FlowInfo(hint=None))

        assert not place.expects_more()

    def test_expects_more_with_stop_frame_and_available_frames(self):
        """Test expects_more when stop frame received but available frames remain."""
        place = ControlPlace("test_control")
        stop_frame = Frame.stop_frame()
        place.put(stop_frame, FlowInfo(hint=None))

        frame = Frame.start_frame(0)
        place.put(frame, FlowInfo(hint=None))

        assert place.expects_more()

    def test_expects_more_with_stop_frame_and_remembered_frames(self):
        """Test expects_more when stop frame received but remembered frames remain."""
        place = ControlPlace("test_control")
        stop_frame = Frame.stop_frame()
        place.put(stop_frame, FlowInfo(hint=None))

        frame = Frame.start_frame(0)
        place.remember_frame(frame)

        assert place.expects_more()

    def test_peek_frames(self):
        """Test peeking at available frames."""
        place = ControlPlace("test_control")
        frame1 = Frame.start_frame(0)
        frame2 = Frame.start_frame(1)
        flow_info1 = FlowInfo(hint=None)
        flow_info2 = FlowInfo(hint=2)

        place.put(frame1, flow_info1)
        place.put(frame2, flow_info2)

        peeked = place.peek_frames_copy()
        assert peeked == {frame1: flow_info1, frame2: flow_info2}

    def test_take_for_processing(self):
        """Test taking a frame for processing."""
        place = ControlPlace("test_control")
        frame = Frame.start_frame(0)
        flow_info = FlowInfo(hint=None)

        place.put(frame, flow_info)
        assert frame in place._avaliable_frames

        place.take_for_processing(frame)
        assert frame not in place._avaliable_frames

    def test_take_for_processing_nonexistent_frame(self):
        """Test that taking a non-existent frame raises assertion error."""
        place = ControlPlace("test_control")
        frame = Frame.start_frame(0)

        with pytest.raises(AssertionError, match="Frame.*not in control place"):
            place.take_for_processing(frame)


class TestQueuePlace:
    """Test cases for QueuePlace class."""

    def test_initialization(self):
        """Test QueuePlace initialization."""
        place = QueuePlace("test_queue")
        assert place.name == "test_queue"
        assert place._queues == {}
        assert not place._is_sticky

    def test_initialization_sticky(self):
        """Test QueuePlace initialization with sticky flag."""
        place = QueuePlace("test_queue", is_sticky=True)
        assert place._is_sticky

    def test_put_data(self):
        """Test putting data into queue place."""
        place = QueuePlace("test_queue")
        frame = Frame.start_frame(0)
        data = {"key": "value"}

        place.put(frame, data)
        assert place._queues[frame] == data

    def test_put_duplicate_frame(self):
        """Test that putting data for existing frame raises assertion error."""
        place = QueuePlace("test_queue")
        frame = Frame.start_frame(0)
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}

        place.put(frame, data1)

        with pytest.raises(AssertionError, match="Frame.*already has data in queue"):
            place.put(frame, data2)

    def test_has_data_for_control_frame_exact_match(self):
        """Test has_data_for_control_frame with exact frame match."""
        place = QueuePlace("test_queue")
        frame = Frame.start_frame(0)
        data = {"key": "value"}

        place.put(frame, data)
        result = place.frame_with_data_for_control_frame(frame)
        assert result == frame

    def test_has_data_for_control_frame_ancestor_match(self):
        """Test has_data_for_control_frame with ancestor frame match."""
        place = QueuePlace("test_queue")
        parent_frame = Frame.start_frame(0)
        child_frame = parent_frame.with_pushed_scope()
        data = {"key": "value"}

        place.put(parent_frame, data)
        result = place.frame_with_data_for_control_frame(child_frame)
        assert result == parent_frame

    def test_has_data_for_control_frame_no_match(self):
        """Test has_data_for_control_frame with no matching frame."""
        place = QueuePlace("test_queue")
        frame = Frame.start_frame(0)

        result = place.frame_with_data_for_control_frame(frame)
        assert result is None

    def test_has_data_for_control_frame_best_match(self):
        """Test has_data_for_control_frame returns best (closest) match."""
        place = QueuePlace("test_queue")
        grandparent = Frame.start_frame(0)
        parent = grandparent.with_pushed_scope()
        child = parent.with_pushed_scope()

        place.put(grandparent, "grandparent_data")
        place.put(parent, "parent_data")

        result = place.frame_with_data_for_control_frame(child)
        assert result == parent  # Should return closest ancestor

    def test_take_data_for_translated_frame(self):
        """Test taking data for a translated frame."""
        place = QueuePlace("test_queue")
        frame = Frame.start_frame(0)
        data = {"key": "value"}

        place.put(frame, data)
        result = place.take_data_for_translated_frame(frame)
        assert result == data

    def test_take_data_for_translated_frame_nonexistent(self):
        """Test that taking data for non-existent frame raises assertion error."""
        place = QueuePlace("test_queue")
        frame = Frame.start_frame(0)

        with pytest.raises(AssertionError, match="Frame.*not in queue"):
            _ = place.take_data_for_translated_frame(frame)

    def test_take_data_for_translated_frame_non_sticky(self):
        """Test that non-sticky place removes data after taking."""
        place = QueuePlace("test_queue", is_sticky=False)
        frame = Frame.start_frame(0)
        data = {"key": "value"}

        place.put(frame, data)
        assert frame in place._queues

        _ = place.take_data_for_translated_frame(frame)
        assert frame not in place._queues

    def test_take_data_for_translated_frame_sticky(self):
        """Test that sticky place keeps data after taking."""
        place = QueuePlace("test_queue", is_sticky=True)
        frame = Frame.start_frame(0)
        data = {"key": "value"}

        place.put(frame, data)
        assert frame in place._queues

        _ = place.take_data_for_translated_frame(frame)
        assert frame in place._queues  # Data should remain


class TestCountingPlace:
    """Test cases for CountingPlace class."""

    def test_initialization(self):
        """Test CountingPlace initialization."""
        place = CountingPlace("test_counting")
        assert place.name == "test_counting"
        assert place._parent_counts == {}

    def test_put_data_increments_parent_count(self):
        """Test that putting data increments the parent frame count."""
        place = CountingPlace("test_counting")
        parent_frame = Frame.start_frame(0)
        child_frame = parent_frame.with_pushed_scope()

        place.put(child_frame, "some_data")
        assert place._parent_counts[parent_frame] == 1

    def test_put_data_multiple_increments(self):
        """Test that multiple puts increment the parent count correctly."""
        place = CountingPlace("test_counting")
        parent_frame = Frame.start_frame(0)
        child_frame1 = parent_frame.with_pushed_scope()
        child_frame2 = child_frame1.with_loop_inc()

        place.put(child_frame1, "data1")
        place.put(child_frame2, "data2")

        assert place._parent_counts[parent_frame] == 2

    def test_has_data_for_control_frame_exact_match(self):
        """Test has_data_for_control_frame with exact frame match."""
        place = CountingPlace("test_counting")
        frame = Frame.start_frame(0)
        child_frame = frame.with_pushed_scope()

        place.put(child_frame, "data")
        result = place.frame_with_data_for_control_frame(frame)
        assert result == frame

    def test_has_data_for_control_frame_no_match(self):
        """Test has_data_for_control_frame with no matching frame."""
        place = CountingPlace("test_counting")
        frame = Frame.start_frame(0)

        result = place.frame_with_data_for_control_frame(frame)
        assert result is None

    def test_has_data_for_control_frame_wrong_scope(self):
        """Test has_data_for_control_frame with frame from inside loop."""
        place = CountingPlace("test_counting")
        parent_frame = Frame.start_frame(0)
        child_frame = parent_frame.with_pushed_scope()

        place.put(child_frame, "data")
        # Try to get data using child frame instead of parent
        result = place.frame_with_data_for_control_frame(child_frame)
        assert result is None

    def test_take_data_for_translated_frame(self):
        """Test taking data for a translated frame returns correct count."""
        place = CountingPlace("test_counting")
        parent_frame = Frame.start_frame(0)
        child_frame1 = parent_frame.with_pushed_scope()
        child_frame2 = child_frame1.with_loop_inc()

        place.put(child_frame1, "data1")
        place.put(child_frame2, "data2")

        result = place.take_data_for_translated_frame(parent_frame)
        assert result == 2

    def test_take_data_for_translated_frame_nonexistent(self):
        """Test that taking data for non-existent frame raises assertion error."""
        place = CountingPlace("test_counting")
        frame = Frame.start_frame(0)

        with pytest.raises(AssertionError, match="Frame.*not in queue"):
            _ = place.take_data_for_translated_frame(frame)

    def test_multiple_parent_frames(self):
        """Test counting with multiple parent frames."""
        place = CountingPlace("test_counting")
        parent1 = Frame.start_frame(0)
        parent2 = Frame.start_frame(1)

        child1_1 = parent1.with_pushed_scope()
        child1_2 = child1_1.with_loop_inc()
        child2_1 = parent2.with_pushed_scope()

        place.put(child1_1, "data1")
        place.put(child1_2, "data2")
        place.put(child2_1, "data3")

        assert place._parent_counts[parent1] == 2
        assert place._parent_counts[parent2] == 1

        assert place.take_data_for_translated_frame(parent1) == 2
        assert place.take_data_for_translated_frame(parent2) == 1


class TestFrameIntegration:
    """Test cases for frame integration with places."""

    def test_frame_hierarchy_with_queue_place(self):
        """Test frame hierarchy matching in QueuePlace."""
        place = QueuePlace("test_queue")

        # Create frame hierarchy: (0) -> (0, 0) -> (0, 0, 0)
        root_frame = Frame.start_frame(0)
        child_frame = root_frame.with_pushed_scope()
        grandchild_frame = child_frame.with_pushed_scope()

        # Put data at different levels
        place.put(root_frame, "root_data")
        place.put(child_frame, "child_data")

        # Test matching from different levels
        assert place.frame_with_data_for_control_frame(grandchild_frame) == child_frame
        assert place.frame_with_data_for_control_frame(child_frame) == child_frame
        assert place.frame_with_data_for_control_frame(root_frame) == root_frame

    def test_frame_hierarchy_with_counting_place(self):
        """Test frame hierarchy with CountingPlace."""
        place = CountingPlace("test_counting")

        # Create frame hierarchy: (0) -> (0, 0) -> (0, 0, 0)
        root_frame = Frame.start_frame(0)
        child_frame = root_frame.with_pushed_scope()
        grandchild_frame = child_frame.with_pushed_scope()

        # Put data at different levels
        place.put(child_frame, "data1")  # increments root_frame count
        place.put(grandchild_frame, "data2")  # increments child_frame count

        # child_frame should have count 1 (from grandchild_frame)
        # root_frame should have count 1 (from child_frame)
        assert place._parent_counts[child_frame] == 1
        assert place._parent_counts[root_frame] == 1
        assert place.take_data_for_translated_frame(child_frame) == 1
        assert place.take_data_for_translated_frame(root_frame) == 1

    def test_stop_frame_handling(self):
        """Test stop frame handling in ControlPlace."""
        place = ControlPlace("test_control")
        stop_frame = Frame.stop_frame()
        flow_info = FlowInfo(hint=None)

        place.put(stop_frame, flow_info)
        assert place._received_stop_frame
        assert stop_frame not in place._avaliable_frames

    def test_flow_info_variants(self):
        """Test different FlowInfo variants with ControlPlace."""
        place = ControlPlace("test_control")
        frame = Frame.start_frame(0)

        # Test jump-like flow
        jump_flow = FlowInfo(hint=None)
        place.put(frame, jump_flow)
        assert place._avaliable_frames[frame] == jump_flow

        # Test back edge flow
        back_flow = FlowInfo(hint=-1)
        frame2 = Frame.start_frame(1)
        place.put(frame2, back_flow)
        assert place._avaliable_frames[frame2] == back_flow

        # Test forked edge flow
        fork_flow = FlowInfo(hint=3)
        frame3 = Frame.start_frame(2)
        place.put(frame3, fork_flow)
        assert place._avaliable_frames[frame3] == fork_flow
