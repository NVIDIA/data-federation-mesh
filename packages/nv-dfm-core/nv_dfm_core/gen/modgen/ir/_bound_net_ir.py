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

from typing import Any

from pydantic import BaseModel, JsonValue

# from ....api import PythonObject
from nv_dfm_core.exec import Frame, tagged_json_value_to_object

from ._net_ir import NetIR


class BoundNetIR(BaseModel):
    """A NetIR bound with specific input parameters ready for execution."""

    site: str
    ir: NetIR
    tagged_input_params: list[tuple[Frame, dict[str, tuple[str, JsonValue]]]]

    @classmethod
    def bind_netir(
        cls,
        net_ir: NetIR,
        input_params: list[tuple[Frame, dict[str, Any]]],
    ) -> "BoundNetIR":
        """Bind a NetIR with input parameters by picking relevant parameters and creating a BoundNetIR instance."""
        self = cls(
            site=net_ir.site,
            ir=net_ir,
            tagged_input_params=net_ir.pick_input_params(input_params),
        )
        return self

    def deserialized_input_params(self) -> list[tuple[Frame, dict[str, Any]]]:
        """Convert tagged JSON input parameters back to their original Python objects."""
        # for each parameter set, translate the tagged JSON values back into the original values
        return [
            (frame, {k: tagged_json_value_to_object(v) for k, v in paramset.items()})
            for frame, paramset in self.tagged_input_params
        ]
