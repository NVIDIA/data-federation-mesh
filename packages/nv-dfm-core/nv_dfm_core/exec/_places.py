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

from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import override

from nv_dfm_core.api._error_token import ErrorToken

from ._frame import FlowInfo, Frame


class Place(ABC):
    """Base class for control and data places."""

    def __init__(self, name: str):
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def put(self, frame: Frame, data: Any) -> None:
        pass

    @abstractmethod
    def num_tokens(self) -> int:
        pass

    @abstractmethod
    def assert_empty(self) -> None:
        pass


class ControlPlace(Place):
    """
    This place is used to collect control frames for a transition. The control frames
    are then matched up with the corresponding data frames.
    """

    def __init__(self, name: str, is_loop_head: bool = False):
        super().__init__(name)
        self._received_stop_frame: bool = False
        # the frames we have received
        self._avaliable_frames: dict[
            Frame, FlowInfo | ErrorToken
        ] = {}  # NOT stop frames
        # the try_activate and fire transition functions may want to remember frames
        # (e.g. while a loop is executing), to prevent the place from sending a stop frame too early
        self._remembered_frames: set[Frame] = set()
        self._is_loop_head: bool = is_loop_head

    @property
    def is_loop_head(self) -> bool:
        return self._is_loop_head

    @override
    def put(self, frame: Frame, data: Any) -> None:
        """Called when receiving a frame."""
        assert isinstance(data, (FlowInfo, ErrorToken)), (
            f"Control Place {self.name}: control frames must come with FlowInfo or ErrorToken data, got {type(data)}"
        )

        if frame.is_stop_frame():
            self._received_stop_frame = True
            return
        self._avaliable_frames[frame] = data

    def remember_frame(self, frame: Frame) -> None:
        self._remembered_frames.add(frame)

    def forget_frame(self, frame: Frame) -> None:
        """Called by the loop head region when it is done with a loop."""
        assert frame in self._remembered_frames, (
            f"Frame {frame} not in remembered frames"
        )
        self._remembered_frames.remove(frame)

    def expects_more(self) -> bool:
        if not self._received_stop_frame:
            return True
        # got a stop frame; do we have any outstanding work left?
        return len(self._avaliable_frames) > 0 or len(self._remembered_frames) > 0

    def peek_frames_copy(self) -> dict[Frame, FlowInfo | ErrorToken]:
        """Used to peek at the frames when checking if a transition can be activated.
        available_frames does not have Stop frames in it.
        Returns a copy so users can loop over it and still call take_for_processing."""
        return self._avaliable_frames.copy()

    def take_for_processing(self, frame: Frame) -> None:
        """And then we remove the frames that we were able to match."""
        assert frame in self._avaliable_frames, (
            f"Frame {frame} not in control place {self._name}"
        )
        del self._avaliable_frames[frame]

    @override
    def num_tokens(self) -> int:
        return len(self._avaliable_frames) + len(self._remembered_frames)

    @override
    def assert_empty(self) -> None:
        assert len(self._avaliable_frames) == 0, (
            f"Control place {self.name} is not empty"
        )
        assert len(self._remembered_frames) == 0, (
            f"Control place {self.name} is not empty"
        )

    @override
    def __repr__(self) -> str:
        return f"ControlPlace({self.name}, {len(self._avaliable_frames)} available frames, {len(self._remembered_frames)} remembered frames)"


class DataPlace(Place, ABC):
    def __init__(self, name: str):
        super().__init__(name)

    @abstractmethod
    def frame_with_data_for_control_frame(self, frame: Frame) -> Frame | None: ...

    """Checks if this place has data for the given frame. It returns a new frame under which the
    data is stored. This may be the same frame or translated to a parent, if the data is scoped."""

    @abstractmethod
    def take_data_for_translated_frame(
        self, matching_frame: Frame
    ) -> Any | ErrorToken: ...


class QueuePlace(DataPlace):
    """Matches frames by scope. Can be sticky, keeping the data in the place."""

    def __init__(self, name: str, is_sticky: bool = False):
        # all places are locked by the NetRunner lock, so we don't need to
        # take care of multithreading here.
        super().__init__(name)
        self._queues: dict[Frame, Any] = {}
        self._is_sticky: bool = is_sticky

    @override
    def frame_with_data_for_control_frame(self, frame: Frame) -> Frame | None:
        """Here, control frames may be from inside a loop. Look for a value in this frame scope"""
        best_match: Frame | None = None
        for earlier_frame in self._queues.keys():
            if earlier_frame.is_ancestor_of(frame):
                if best_match is None or best_match.is_ancestor_of(earlier_frame):
                    best_match = earlier_frame
        return best_match

    @override
    def put(self, frame: Frame, data: Any) -> None:
        assert frame not in self._queues, (
            f"Frame {frame} already has data in queue {self.name}. Tried to put value {repr(data)}, already had {repr(self._queues[frame])}"
        )
        self._queues[frame] = data

    @override
    def take_data_for_translated_frame(self, matching_frame: Frame) -> Any | ErrorToken:
        assert matching_frame in self._queues, (
            f"Frame {matching_frame} not in queue {self.name}"
        )
        value = self._queues[matching_frame]
        if not self._is_sticky:
            del self._queues[matching_frame]
        return value

    @override
    def num_tokens(self) -> int:
        return len(self._queues)

    @override
    def assert_empty(self) -> None:
        assert len(self._queues) == 0, (
            f"QueuePlace {self.name} is not empty: {self._queues}"
        )

    @override
    def __repr__(self) -> str:
        return f"QueuePlace({self.name}, {len(self._queues)} frames ({self._queues.keys()}), {self._is_sticky} sticky)"


class CountingPlace(DataPlace):
    """Counts the number of frames of the parent scope"""

    def __init__(self, name: str):
        # all places are locked by the NetRunner lock, so we don't need to
        # take care of multithreading here.
        super().__init__(name)
        self._parent_counts: dict[Frame, int] = {}

    @override
    def frame_with_data_for_control_frame(self, frame: Frame) -> Frame | None:
        """Frame needs to be the exact frame (i.e. the loop frame), not from inside the loop."""
        if frame not in self._parent_counts:
            return None
        return frame

    @override
    def put(self, frame: Frame, data: Any) -> None:
        """Increments the counter for the parent frame"""
        parent = frame.parent()
        if parent not in self._parent_counts:
            self._parent_counts[parent] = 0
        self._parent_counts[parent] += 1

    @override
    def take_data_for_translated_frame(self, matching_frame: Frame) -> int:
        assert matching_frame in self._parent_counts, (
            f"Frame {matching_frame} not in queue {self.name}"
        )
        return self._parent_counts[matching_frame]

    @override
    def num_tokens(self) -> int:
        return sum(self._parent_counts.values())

    @override
    def assert_empty(self) -> None:
        assert len(self._parent_counts) == 0, f"CountingPlace {self.name} is not empty"
