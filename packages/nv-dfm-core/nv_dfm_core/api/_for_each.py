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

"""
Returns a value (usually the output of an Operation) to the client.
"""

from typing import Any, Literal

from pydantic import ConfigDict
from typing_extensions import override

from ._block import Block
from ._node_id import NodeRef
from ._statement import Statement


class ForEach(Statement, Block):
    """Takes an iterable and iterates over it, executing the body for each element.

    Usage:
    seq = ...
    with ForEach(seq=seq) as e:
        # body
        Yield(value=e)
    """

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True)

    dfm_class_name: Literal["nv_dfm_core.api.ForEach"] = "nv_dfm_core.api.ForEach"
    seq: NodeRef

    @override
    def accept(self, visitor: Any) -> None:
        visitor.visit_for_each(self)
