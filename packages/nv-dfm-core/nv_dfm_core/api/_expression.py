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
from typing import Any

from pydantic import ConfigDict
from typing_extensions import override

from ._node_id import NodeRef
from ._statement import Statement


class Expression(Statement, ABC):
    """An expression is a Statement that produces a value that can be
    referenced by other Statements."""

    model_config: ConfigDict = ConfigDict(
        frozen=True,
    )

    def fieldref(self, selector: int | str, issubs: bool | None = None) -> NodeRef:
        return self.dfm_node_id.to_ref(sel=selector, issubs=issubs)

    @abstractmethod
    @override
    def accept(self, visitor: Any) -> None:
        pass
