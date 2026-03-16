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


class TryFromCache(Statement, Block):
    """Takes a key and tries to get the value from the cache.
    If the value is not in the cache, the body is executed.

    Usage:
    data = ...
    key = KeyFromExpr(expr=RenderImage(data=data))
    with TryFromCache(key=key) as img:
        # body
        content = RenderImage(data=dat)
        img.cache(content)

    Yield(value=img)
    """

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True)

    dfm_class_name: Literal["nv_dfm_core.api.TryFromCache"] = (
        "nv_dfm_core.api.TryFromCache"
    )
    key: NodeRef

    def cache(self, value: NodeRef) -> "WriteToCache":
        return WriteToCache(key=self.key, value=value)

    @override
    def accept(self, visitor: Any) -> None:
        visitor.visit_try_from_cache(self)


class WriteToCache(Statement):
    """Writes a value to the cache. This should not be used directly, but only
    via the TryFromCache.cache method.
    """

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True)

    dfm_class_name: Literal["nv_dfm_core.api._WriteToCache"] = (
        "nv_dfm_core.api._WriteToCache"
    )
    key: NodeRef
    value: NodeRef

    @override
    def accept(self, visitor: Any) -> None:
        visitor.visit_write_to_cache(self)
