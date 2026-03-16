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

from typing import Literal

from pydantic import BaseModel
from typing_extensions import override


class FlowInfo(BaseModel, frozen=True):
    """
    Additional information flowing along a control edge to let the receiver know what's happening.
    The hint is None for a normal jump-like edge, -1 for a back-edge,
    and >= 0 indicating the number of values expected for a forked edge.
    """

    hint: None | int = None

    def is_jump_like(self) -> bool:
        return self.hint is None

    def is_back_edge(self) -> bool:
        return isinstance(self.hint, int) and self.hint == -1

    def is_expect_n_forked(self) -> bool:
        return isinstance(self.hint, int) and self.hint >= 0

    def expect_n_forks(self) -> int:
        assert isinstance(self.hint, int) and self.hint >= 0, (
            "expect_n_forks() can only be called on forked edges"
        )
        return self.hint


class Frame(BaseModel, frozen=True):
    """
    A frame represents the execution context. Each token (control or data) is associated
    with a frame and the frames are used to match corresponding tokens from multiple places
    during transition activation.

    The frame is a hierarchical list of "executions". The first parameter set
    to a pipeline is frame (0,). A loop inside this execution may result in
    frames like (0, 1) or (0, 2).
    """

    token: Literal["@dfm-control-token"] = "@dfm-control-token"
    # The frame is a hierarchical list of "executions".
    frame: list[int] | Literal["stop"]

    @staticmethod
    def stop_frame() -> "Frame":
        return Frame(frame="stop")

    @staticmethod
    def start_frame(num: int) -> "Frame":
        return Frame(frame=[num])

    def input_paramset_index(self) -> int:
        assert self.frame != "stop", "Cannot get the input index of a stop frame"
        assert len(self.frame) > 0, "Cannot get the input index of the root frame"
        return self.frame[0]

    def is_stop_frame(self) -> bool:
        return self.frame == "stop"

    def with_loop_inc(self) -> "Frame":
        assert self.frame != "stop", "Cannot increment the a stop frame"
        assert len(self.frame) > 0, "Cannot increment empty frame"
        # increment the last element of the frame stack
        return Frame(frame=self.frame[:-1] + [self.frame[-1] + 1])

    def with_pushed_scope(self) -> "Frame":
        assert self.frame != "stop", "Cannot push on a stop frame"
        return Frame(frame=self.frame + [0])

    def with_popped_scope(self) -> "Frame":
        assert self.frame != "stop", "Cannot pop a stop frame"
        assert len(self.frame) > 0, "Cannot pop the root frame"
        return Frame(frame=self.frame[:-1])

    def is_ancestor_of(self, other: "Frame") -> bool:
        assert self.frame != "stop" and other.frame != "stop", (
            "Cannot compare stop frames"
        )
        # this frame is a parent of other if the first n elements of other are the same
        return self.frame == other.frame[: len(self.frame)]

    def parent(self) -> "Frame":
        assert self.frame != "stop", "Cannot get the parent of a stop frame"
        assert len(self.frame) > 1, "Cannot get the parent of the root frame"
        return Frame(frame=self.frame[:-1])

    @override
    def __hash__(self) -> int:
        if isinstance(self.frame, list):
            return hash(tuple(self.frame))
        return hash(self.frame)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frame):
            return False
        return self.frame == other.frame
