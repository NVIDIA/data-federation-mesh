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

from pydantic import BaseModel

from ._activation_statements import ActivationFunction
from ._gen_context import GenContext
from ._in_place import InPlace
from ._ir_stmts import IRStmt


class Transition(BaseModel):
    control_place: InPlace
    data_places: list[InPlace]

    #
    try_activate_func: ActivationFunction
    #
    fire_body: list[IRStmt]
    #
    signal_error_body: list[IRStmt]
    #
    signal_stop_body: list[IRStmt]

    def emit_place_declarations(self, ctx: GenContext) -> None:
        ctx.emit(f"# Transition {ctx.transition_name}")
        self.control_place.emit_place_declaration(ctx)
        for place in sorted(self.data_places, key=lambda p: p.name):
            place.emit_place_declaration(ctx)

    def emit_python(self, ctx: GenContext) -> None:
        ctx.emit(f"# Transition {ctx.transition_name}")
        self._emit_try_activate(ctx)
        self._emit_signal_error(ctx)
        self._emit_signal_stop(ctx)
        self._emit_fire(ctx)

    def _emit_try_activate(self, ctx: GenContext) -> None:
        # Generate _try_activate function
        ctx.emit(ctx.try_activate_function_def())
        ctx.enter_scope()
        self.try_activate_func.emit_python(
            ctx,
            control_place=self.control_place.decl_name(),
            data_places=[place.decl_name() for place in self.data_places],
        )
        ctx.exit_scope()
        ctx.emit("")

    def _emit_signal_error(self, ctx: GenContext) -> None:
        # Generate _signal_error function
        ctx.emit(ctx.signal_error_function_def())
        ctx.enter_scope()
        if self.signal_error_body:
            for stmt in self.signal_error_body:
                stmt.emit_python(ctx)
        else:
            ctx.emit("pass")
        ctx.exit_scope()
        ctx.emit("")

    def _emit_signal_stop(self, ctx: GenContext) -> None:
        # Generate _signal_stop function
        ctx.emit(ctx.signal_stop_function_def())
        ctx.enter_scope()
        if self.signal_stop_body:
            for stmt in self.signal_stop_body:
                stmt.emit_python(ctx)
        else:
            ctx.emit("pass")
        ctx.exit_scope()
        ctx.emit("")

    def _emit_fire(self, ctx: GenContext) -> None:
        # Generate _fire function
        ctx.emit(ctx.fire_function_def())
        ctx.enter_scope()
        if self.fire_body:
            for stmt in self.fire_body:
                stmt.emit_python(ctx)
        else:
            ctx.emit("pass")
        ctx.exit_scope()
        ctx.emit("")
