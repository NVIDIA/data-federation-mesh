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

from typing import Any, Literal

from pydantic import BaseModel, JsonValue
from typing_extensions import override

from nv_dfm_core.exec import Frame, any_object_to_tagged_json_value
from nv_dfm_core.exec._frame import FlowInfo

from ._gen_context import GenContext
from ._in_place import START_PLACE_NAME
from ._transition import Transition


class NetIR(BaseModel):
    """Intermediate representation of a Petri net for a specific site.

    Contains all the transitions, places, and execution logic for one site
    in the federation.
    """

    pipeline_name: str | None
    site: str
    transitions: list[Transition]
    # a hash ID under which the netir can be cached
    fingerprint: str = ""

    @override
    def model_post_init(self, __context: Any):
        # Only compute the fingerprint if it's not already set
        if not getattr(self, "fingerprint", None) or getattr(self, "fingerprint") != "":
            import hashlib
            import json

            base_data = self.model_dump(exclude={"fingerprint"}, mode="json")
            json_str = json.dumps(base_data, sort_keys=True, separators=(",", ":"))
            object.__setattr__(
                self,
                "fingerprint",
                hashlib.sha256(json_str.encode("utf-8")).hexdigest(),
            )

    def find_places(
        self,
        kind: Literal["data", "control", None],
        origin: Literal["internal", "external", None],
    ) -> set[str]:
        """Return all required places (places with kind='param').
        kind == None returns places of any kind.
        origin == None returns places of any origin."""
        places: set[str] = set()
        for transition in self.transitions:
            if (kind is None or transition.control_place.kind == kind) and (
                origin is None or transition.control_place.origin == origin
            ):
                places.add(transition.control_place.name)
            for place in transition.data_places:
                if (kind is None or place.kind == kind) and (
                    origin is None or place.origin == origin
                ):
                    places.add(place.name)

        return places

    def emit_python(self, ctx: GenContext) -> None:
        """Emit the Python code for the Net IR."""

        def transition_name(i: int) -> str:
            return f"t{i + 1}"

        ctx.emit("@dataclass")
        ctx.emit("class ThisNet(Net):")
        ctx.enter_scope()
        if self.pipeline_name is None:
            ctx.emit(f'"""Net for unnamed pipeline at site {self.site}"""')
        else:
            ctx.emit(
                f'"""Net for pipeline "{self.pipeline_name}" at site {self.site}"""'
            )
        ctx.emit("")
        ctx.emit("# Places")
        for i, transition in enumerate(self.transitions):
            ctx.prep_for_transition(transition_name=transition_name(i))
            transition.emit_place_declarations(ctx)
        ctx.emit("")

        ctx.emit(
            "def get_activation_functions(self) -> list[TransitionTryActivateFunc]:"
        )
        ctx.enter_scope()
        ctx.emit("return [")
        ctx.enter_scope()
        for i, transition in enumerate(self.transitions):
            ctx.prep_for_transition(transition_name=transition_name(i))
            ctx.emit(f"self.{ctx.try_activate_function_name()},")
        ctx.exit_scope()
        ctx.emit("]")
        ctx.exit_scope()
        ctx.emit("")

        # Generate transition functions
        for i, transition in enumerate(self.transitions):
            ctx.prep_for_transition(transition_name=transition_name(i))
            transition.emit_python(ctx)
        # done
        ctx.exit_scope()

    def pick_input_params(
        self,
        input_params: list[tuple[Frame, dict[str, Any]]],
    ) -> list[tuple[Frame, dict[str, tuple[str, JsonValue]]]]:
        """Picks all input parameters from the 'global' parameter set that are consumed by the given
        netIR. Also translates the values into tagged JSON values so we can send them."""

        # all places of this NetIR that expect external input
        my_param_places = self.find_places(kind=None, origin="external")

        # for each parameter set, pick and translate the values into tagged JSON values so we can send them
        params_to_send: list[tuple[Frame, dict[str, tuple[str, JsonValue]]]] = []
        for frame, paramset in input_params:
            translated: dict[str, tuple[str, JsonValue]] = {}
            if frame.is_stop_frame() and START_PLACE_NAME in my_param_places:
                translated[START_PLACE_NAME] = any_object_to_tagged_json_value(
                    FlowInfo()
                )
            else:
                # pick all the params for this net
                for pname in my_param_places:
                    if pname not in paramset:
                        raise ValueError(
                            f"Frame {frame}, parameter set {paramset} is missing required parameter {pname}"
                        )
                    translated[pname] = any_object_to_tagged_json_value(
                        paramset.get(pname, None)
                    )
            # add this frame to the list
            params_to_send.append((frame, translated))

        return params_to_send
