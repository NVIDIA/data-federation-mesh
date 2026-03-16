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

from ._node_id import NodeRef
from ._statement import Statement

# the yield place for status tokens, in particular ErrorToken and StopToken
STATUS_PLACE_NAME = "_dfm_status_"
# the yield place for discovery results
DISCOVERY_PLACE_NAME = "_dfm_discovery_"


class Yield(Statement):
    """Sends a value back to a place located in the app.
    By default, all places in a pipeline must be unique (this includes PaceParam places),
    to avoid accidental aliasing. If multiple Yields should reference the
    same place, multiuse must be set to True on ALL Yields for this place.
    The place name must still be different from all PlaceParam places, however."""

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True)

    dfm_class_name: Literal["nv_dfm_core.api.Yield"] = "nv_dfm_core.api.Yield"
    place: str = "yield"
    value: NodeRef | Any
    multiuse: bool = False

    @override
    def accept(self, visitor: Any) -> None:
        return visitor.visit_yield(self)
