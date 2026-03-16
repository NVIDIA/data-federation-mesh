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

from pydantic import BaseModel, field_validator

from ._gen_context import GenContext

START_PLACE_NAME = "__start_place__"


class InPlace(BaseModel):
    name: str
    kind: Literal["control", "data"]
    origin: Literal["internal", "external"]
    flavor: Literal[
        "seq_control", "loop_control", "scoped", "sticky", "framecount", "yield"
    ]
    type: str

    @field_validator("flavor")
    @classmethod
    def validate_control_flavor(cls, v: Any, info: Any) -> Any:
        if info.data and info.data.get("kind") == "control":
            if v not in ["seq_control", "loop_control"]:
                raise ValueError(
                    f"Places of kind 'control' must have flavor 'seq_control' or 'loop_control'. Place {info.data.get('name')} has flavor {v} ({info.data})"
                )
        return v

    def decl_name(self) -> str:
        return f"{self.name}"

    def emit_place_declaration(self, ctx: GenContext) -> None:
        if self.flavor == "seq_control":
            ctx.emit(
                f"{self.decl_name()}: Place = field(default_factory=lambda: ControlPlace(name={repr(self.name)}, is_loop_head=False))"
            )
        elif self.flavor == "loop_control":
            ctx.emit(
                f"{self.decl_name()}: Place = field(default_factory=lambda: ControlPlace(name={repr(self.name)}, is_loop_head=True))"
            )
        elif self.flavor == "scoped":
            ctx.emit(
                f"{self.decl_name()}: Place = field(default_factory=lambda: QueuePlace(name={repr(self.name)}, is_sticky=False))"
            )
        elif self.flavor == "sticky":
            ctx.emit(
                f"{self.decl_name()}: Place = field(default_factory=lambda: QueuePlace(name={repr(self.name)}, is_sticky=True))"
            )
        elif self.flavor == "framecount":
            ctx.emit(
                f"{self.decl_name()}: Place = field(default_factory=lambda: CountingPlace(name={repr(self.name)}))"
            )
        elif self.flavor == "yield":
            raise ValueError(f"Yield place {self.name} should not appear in the net IR")
