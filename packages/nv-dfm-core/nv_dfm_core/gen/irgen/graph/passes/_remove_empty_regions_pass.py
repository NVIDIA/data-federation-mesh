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
    FlowEdge,
    Jump,
    Region,
)


class RemoveEmptyRegionsPass(GraphTransformPass):
    """
    After the earlier transformations, some regions may be empty and exit with a direct jump.
    Those regions can be removed by redirecting the incoming flow edges to the jump target.
    """

    @override
    def apply(self) -> None:
        for region in self._graph.get_regions_copy():
            if region.size() == 0 and isinstance(region.exit_node(), Jump):
                region_in_slot = region.in_slot("__flow__")
                jump_out_slot = region.exit_node().out_slot("__flow__")
                if (
                    region_in_slot.kind() == jump_out_slot.kind()
                    and region_in_slot.type() == jump_out_slot.type()
                ):
                    # we can forward all incoming edges to the jump target
                    jump_edges = jump_out_slot.outgoing_edges()
                    assert len(jump_edges) == 1, (
                        f"Region {region.global_id()} has {len(jump_edges)} jump targets, expected 1"
                    )
                    jump_target_slot = jump_edges[0].dst()
                    jump_target_region = jump_target_slot.node()
                    assert isinstance(jump_target_region, Region), (
                        f"Jump target {jump_target_region.global_id()} is not a region"
                    )

                    # migrate all places to the jump target
                    for place in region.places():
                        place.migrate(jump_target_region)

                    # forward all incoming edges to the jump target
                    for edge in region_in_slot.incoming_edges():
                        _ne = FlowEdge(
                            src=edge.src(),
                            dst=jump_target_slot,
                            send_cost_info=self._graph._fed_info.find_send_cost(
                                fromsite=edge.src().node().site(),
                                tosite=jump_target_region.site(),
                                logger=self._graph.logger,
                            ),
                            is_weak=edge.is_weak(),
                        )
                        # delete the old edge
                        edge.remove()

                    # delete the region
                    region.remove(force=True)

    @override
    def target_state(self) -> GraphState:
        return GraphState.EMPTY_REGIONS_REMOVED
