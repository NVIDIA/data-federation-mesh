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

# pyright: reportImportCycles=false


INDENT_SIZE = 4


class GenContext:
    """Context object for generating Python code from NetIR, tracking state and indentation."""

    def __init__(self, lines: list[str] | None = None):
        self._transition_name: str | None = None
        self._lines: list[str] = lines if lines is not None else []
        self._var_counter: int = 0
        self._indent: int = 0

    def prep_for_transition(self, transition_name: str) -> None:
        self._transition_name = transition_name
        self._var_counter = 0

    @property
    def transition_name(self) -> str:
        assert self._transition_name is not None, "transition_name is not set"
        return self._transition_name

    def site_param_name(self) -> str:
        return "_site_"

    def dfm_context_param_name(self) -> str:
        return "_dfm_context_"

    def frame_param_name(self) -> str:
        return "_frame_"

    def data_param_name(self) -> str:
        return "_data_"

    def error_param_name(self) -> str:
        return "_error_"

    def try_activate_function_name(self) -> str:
        return f"{self.transition_name}_try_activate"

    def try_activate_function_def(self) -> str:
        return f"def {self.try_activate_function_name()}(self) -> list[Activation]:"

    def fire_function_name(self) -> str:
        return f"{self.transition_name}_fire"

    def fire_function_def(self) -> str:
        return f"async def {self.fire_function_name()}(self, {self.site_param_name()}: ThisSite, {self.dfm_context_param_name()}: DfmContext, {self.frame_param_name()}: Frame, {self.data_param_name()}: dict[str, Any]) -> None:"

    def signal_error_function_name(self) -> str:
        return f"{self.transition_name}_signal_error"

    def signal_error_function_def(self) -> str:
        return f"async def {self.signal_error_function_name()}(self, {self.site_param_name()}: ThisSite, {self.dfm_context_param_name()}: DfmContext, {self.frame_param_name()}: Frame, {self.error_param_name()}: ErrorToken) -> None:"

    def signal_error_function_call(self) -> str:
        return f"await self.{self.signal_error_function_name()}({self.site_param_name()}={self.site_param_name()}, {self.dfm_context_param_name()}={self.dfm_context_param_name()}, {self.frame_param_name()}={self.frame_param_name()}, {self.error_param_name()}={self.error_param_name()})"

    def signal_stop_function_name(self) -> str:
        return f"{self.transition_name}_signal_stop"

    def signal_stop_function_def(self) -> str:
        return f"async def {self.signal_stop_function_name()}(self, {self.site_param_name()}: ThisSite, {self.dfm_context_param_name()}: DfmContext, {self.frame_param_name()}: Frame, {self.data_param_name()}: dict[str, Any]) -> None:"

    @property
    def lines(self) -> list[str]:
        return self._lines

    def fresh_var(self, prefix: str = "_tmp") -> str:
        self._var_counter += 1
        return f"{prefix}_{self._var_counter}"

    def enter_scope(self):
        self._indent += INDENT_SIZE

    def exit_scope(self):
        self._indent -= INDENT_SIZE

    def emit(self, line: str):
        if line.strip() == "":
            self._lines.append("")
            return
        self._lines.append(" " * self._indent + line)

    def emit_indented(self, line: str):
        """convenience method to avoid entering and exiting for individual lines"""
        if line.strip() == "":
            self._lines.append("")
            return
        self._lines.append(" " * (self._indent + INDENT_SIZE) + line)
