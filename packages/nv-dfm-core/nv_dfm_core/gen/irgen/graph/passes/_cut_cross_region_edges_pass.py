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
    FlowEdge,
    Place,
    Send,
)


class CutCrossRegionEdgesPass(GraphTransformPass):
    """
    This pass cuts cross-region edges by creating a send node and a place node.
    The send node is created in the source region and the place node is created in the target region.
    The send node sends the data to the place node and the place node receives the data.
    """

    @override
    def apply(self) -> None:
        for edge in list(self._graph._edges):
            if (
                # cross regions
                edge.src().node().region() != edge.dst().node().region()
                # also break up edges from own exit node to itself
                or (
                    isinstance(edge.src().node(), ExitNode)
                    and edge.src().node().region() == edge.dst().node().region()
                )
            ):
                if edge.is_weak():
                    continue
                source_region = edge.src().node().region()
                source_slot = edge.src()

                target_region = edge.dst().node().region()
                target_node = edge.dst().node()
                target_slot = edge.dst()

                # create a dedicated send node
                send_node = Send(
                    graph=self._graph,
                    region=source_region,
                    site_local_logid=self._graph._get_fresh_per_nodetype_logid(Send),
                    data_kind=source_slot.kind(),
                    type=source_slot.type(),
                )

                # reuse a place node, if it already exists
                # check if there's a place for the target slot already
                placename = f"{target_node.site_local_logid()}_{target_slot.name()}"
                if placename in target_region._places:
                    place_node = target_region._places[placename]
                else:
                    place_node = Place(
                        graph=self._graph,
                        region=target_region,
                        site_local_logid=placename,
                        kind=target_slot.kind(),
                        origin="internal",
                        flavor="control"
                        if target_slot.kind() == "control"
                        else "scoped",
                        type=target_slot.type(),
                    )
                    # connect the place to the target slot
                    _ = FlowEdge(
                        src=place_node.out_slot("__data__"),
                        dst=target_slot,
                        send_cost_info=self._graph._fed_info.find_send_cost(
                            place_node.site(),
                            target_slot.node().site(),
                            logger=self._graph.logger,
                        ),
                        is_weak=False,
                    )

                edge.remove()

                _ = FlowEdge(
                    src=source_slot,
                    dst=send_node.in_slot("value"),
                    send_cost_info=self._graph._fed_info.find_send_cost(
                        source_slot.node().site(),
                        send_node.site(),
                        logger=self._graph.logger,
                    ),
                    is_weak=False,
                )
                _ = FlowEdge(
                    src=send_node.out_slot("__data__"),
                    dst=place_node.in_slot("__data__"),
                    send_cost_info=self._graph._fed_info.find_send_cost(
                        send_node.site(), place_node.site(), logger=self._graph.logger
                    ),
                    is_weak=True,
                )

    @override
    def target_state(self) -> GraphState:
        return GraphState.CROSS_CUT
