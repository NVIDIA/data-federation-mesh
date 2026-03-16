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


class RemoveEmptySitesPass(GraphTransformPass):
    """
    Remove all sites that have no operation nodes after pruning.
    """

    @override
    def apply(self) -> None:
        # find which sites have no operation nodes
        sizes = self._graph.num_operation_nodes_by_site()
        to_delete = [site for site, size in sizes.items() if size == 0]

        # and delete all those
        if len(to_delete) > 0:
            nodes = self._graph.get_all_nodes_copy()
            for node in nodes:
                # don't delete the yield nodes
                if node.region() == self._graph.yield_region():
                    continue
                if node == self._graph.yield_region():
                    continue
                # Note: we are deleting nodes in an unordered way. It's possible that our
                # nodes list is outdated already, that's why we check with the graph if it still exists
                if node.site() in to_delete and self._graph.has_node(node.global_id()):
                    node.remove(force=True)

    @override
    def target_state(self) -> GraphState:
        return GraphState.EMPTY_SITES_REMOVED
