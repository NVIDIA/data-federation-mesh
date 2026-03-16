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

# pyright: reportPrivateUsage=false
from typing_extensions import override

from .._graph import GraphState, GraphTransformPass
from .._graph_elements import (
    ExitNode,
    NodeState,
    Region,
)


class PrunePass(GraphTransformPass):
    """
    Remove all nodes that are not selected
    """

    @override
    def apply(self) -> None:
        for node in list(self._graph._nodes_by_global_id.values()):
            if node.state != NodeState.SELECTED:
                assert not isinstance(node, (Region, ExitNode)), (
                    f"Node {node} is a region or exit node. Those should be SELECTED and never be pruned"
                )
                node.remove()

    @override
    def target_state(self) -> GraphState:
        return GraphState.PRUNED
