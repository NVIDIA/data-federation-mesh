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
from typing import Literal

from pydantic import ConfigDict
from typing_extensions import override

from nv_dfm_core.api.pydantic import PolymorphicBaseModel

from ._gen_context import GenContext

START_PLACE_NAME = "__start_place__"


class ActivationFunction(PolymorphicBaseModel, ABC):
    """Abstract base class for activation functions that determine when transitions fire."""

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True)  # pyright: ignore[reportIncompatibleVariableOverride]

    @abstractmethod
    def emit_python(
        self, ctx: GenContext, control_place: str, data_places: list[str]
    ) -> None: ...


class ActivateWhenPlacesReady(ActivationFunction):
    """Activation function that fires when all required places are ready."""

    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.ActivateWhenPlacesReady"] = (
        "nv_dfm_core.gen.modgen.ir.ActivateWhenPlacesReady"
    )

    @override
    def emit_python(
        self, ctx: GenContext, control_place: str, data_places: list[str]
    ) -> None:
        ctx.emit("return activate_when_places_ready(")
        ctx.emit_indented(f"self.{ctx.fire_function_name()},")
        ctx.emit_indented(f"self.{ctx.signal_error_function_name()},")
        ctx.emit_indented(f"self.{ctx.signal_stop_function_name()},")
        ctx.emit_indented(f"self.{control_place},")
        for p in sorted(data_places):
            ctx.emit_indented(f"self.{p},")
        ctx.emit(")")
