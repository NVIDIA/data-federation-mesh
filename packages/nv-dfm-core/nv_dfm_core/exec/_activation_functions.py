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

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

from nv_dfm_core.api import ErrorToken

from ._dfm_context import DfmContext
from ._frame import FlowInfo, Frame
from ._places import ControlPlace, DataPlace
from ._site import Site

TransitionTryActivateFunc: TypeAlias = Callable[[], list["Activation"]]
ScheduledTransitionFunc: TypeAlias = Callable[
    [Site, DfmContext, Frame, dict[str, Any] | ErrorToken], Awaitable[None]
]


@dataclass
class Activation:
    scheduled_func: ScheduledTransitionFunc
    frame: Frame
    data: dict[str, Any] | ErrorToken


# The activation functions implement the logic of when transitions can be fired.
# The activation functions return lists of activations to the NetRunner. Activations are
# any function that can be called; this is usually the transition fire function, but could
# also be the signal_error or signal_stop function. That's for the activation function to decide.


def activate_when_places_ready(
    fire_func: ScheduledTransitionFunc,
    error_func: ScheduledTransitionFunc,
    stop_func: ScheduledTransitionFunc,
    control_place: "ControlPlace",
    *data_places: "DataPlace",
) -> list[Activation]:
    """
    This function is called from generated _try_activate functions for a transition.
    It checks which frames can be matched up with the data frames.
    This function is called indirectly by the NetRunner with the places lock held.

    IMPORTANT: the list of returned activations can contain a Stop frame activation IF AND ONLY IF
    the control place does not expect any more frames.
    """

    for data_place in data_places:
        assert isinstance(data_place, DataPlace), (
            f"Data place {data_place.name} is not a data place"
        )

    activations: list[Activation] = []
    for frame, flow_info_or_error in control_place.peek_frames_copy().items():
        # check if all data places have data for this frame
        translated_frames = [
            data_place.frame_with_data_for_control_frame(frame)
            for data_place in data_places
        ]
        if all(translated_frames):
            # yes, check it out
            control_place.take_for_processing(frame)
            # collect all the data
            data: dict[str, Any] = {}
            found_errors: list[ErrorToken] = []
            for i, data_place in enumerate(data_places):
                translated_frame = translated_frames[i]
                assert translated_frame
                value = data_place.take_data_for_translated_frame(translated_frame)
                data[data_place.name] = value
                if isinstance(value, ErrorToken):
                    found_errors.append(value)

            # now we have checked out all the data from all places

            # check if we need to propagate an error or can fire the transition
            # add the control error, if there was one
            if isinstance(flow_info_or_error, ErrorToken):
                found_errors.append(flow_info_or_error)

            if len(found_errors) > 1:
                activations.append(
                    Activation(
                        error_func, frame, ErrorToken.from_error_tokens(found_errors)
                    )
                )
                continue
            elif len(found_errors) == 1:
                activations.append(Activation(error_func, frame, found_errors[0]))
                continue

            flow_info = flow_info_or_error
            # we handled any error tokens above
            assert isinstance(flow_info, FlowInfo)

            assert not frame.is_stop_frame()

            # update the frame for the first/next loop iteration, if needed
            if control_place.is_loop_head:
                if flow_info.is_back_edge():
                    frame = frame.with_loop_inc()
                else:
                    frame = frame.with_pushed_scope()

            activations.append(Activation(fire_func, frame, data))

    # now check if the control place expects more frames, after the ones we checked out
    if not control_place.expects_more():
        # expects_more() is False, which is the promise that the control_place has no
        # outstanding frames left and any future call to peek would return an empty dict
        activations.append(Activation(stop_func, Frame.stop_frame(), {}))

    return activations
