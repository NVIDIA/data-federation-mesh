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
    Jump,
    Node,
    Region,
)


class ResolveDeadlocksPass(GraphTransformPass):
    """
    It can happen that within a region A, data is sent to a different site via an edge 1
    and this data indirectly causes an incoming edge 2 back into A.
    When edge 2 is cut, it will result in an incoming place P to A, but receiving data in this
    place is casually connected to A having executed. This results in a deadlock, because A
    can only execute once it has data in all its places, which it cannot for A.
    This pass identifies this situation and resolves it by splitting region A into "before
    and after edge 2" parts.

    NOTE: I *believe* that this deadlock situation can only happen through edges that cross
    sites but are within the same logical region. Edges to and from other regions don't
    cause this problem. But this should be formally proven one day to make sure that's correct.
    """

    def _all_incoming_cross_site_edges_from_sibling_regions(
        self, region: Region
    ) -> list[FlowEdge]:
        cross_site_edges: list[FlowEdge] = []
        for node in region.nodes_copy():
            for in_slot in node.in_slots().values():
                for edge in in_slot.incoming_edges():
                    from_node = edge.src().node()
                    # edge crosses sites but within the same logical region
                    if (
                        from_node.region().site_local_logid()
                        == region.site_local_logid()
                        and from_node.site() != region.site()
                    ):
                        cross_site_edges.append(edge)
        return cross_site_edges

    def _all_outgoing_cross_site_edges_to_sibling_regions(
        self, region: Region
    ) -> list[FlowEdge]:
        cross_site_edges: list[FlowEdge] = []
        for node in region.nodes_copy():
            for out_slot in node.out_slots().values():
                for edge in out_slot.outgoing_edges():
                    to_node = edge.dst().node()
                    if (
                        to_node.region().site_local_logid() == region.site_local_logid()
                        and to_node.site() != region.site()
                    ):
                        cross_site_edges.append(edge)
        return cross_site_edges

    def _check_reachability(
        self, outgoing_edge: FlowEdge, targets: list[Node]
    ) -> set[Node]:
        if len(targets) == 0:
            return set()
        my_site = outgoing_edge.src().node().site()
        my_logical_region = outgoing_edge.src().node().region().site_local_logid()
        assert outgoing_edge.dst().node().site() != my_site
        assert all(target.site() == my_site for target in targets)

        # dfs
        visited: set[Node] = set()
        stack: list[Node] = [outgoing_edge.dst().node()]  # guaranteed to be remote
        tgt_set: set[Node] = set(targets)
        reachable: set[Node] = set()

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in tgt_set:
                reachable.add(node)
            # we don't follow ExitNodes further, this region is done and outgoing edges
            # fall into a different activation
            if isinstance(node, ExitNode):
                continue
            # Explore outgoing edges
            for out_slot in node.out_slots().values():
                for outgoing_edge in out_slot.outgoing_edges():
                    next_node = outgoing_edge.dst().node()
                    # we know that our first node was outside our region, therefore we can follow
                    # all edges as long as we stay in the same logical region
                    if next_node.region().site_local_logid() == my_logical_region:
                        # note that we may add ExitNodes here. ExitNodes end this region and
                        # we don't follow their outgoing edges further (see above).
                        # An ExitNode may directly looping back to its own region, but that's okay
                        stack.append(next_node)
        return reachable

    def _find_reachable_inner_nodes_within_region(
        self, this_region: Region, start_nodes: set[Node]
    ) -> set[Node]:
        reachable_nodes: set[Node] = set()
        visited: set[Node] = set()
        stack: list[Node] = list(start_nodes)
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            # we don't want to add any Region or ExitNode(s) to the split
            if isinstance(node, ExitNode) or isinstance(node, Region):
                continue

            # if this node is in our region, add it to the reachable nodes
            if node.region() == this_region:
                reachable_nodes.add(node)

            # follow outgoing edges, including cross sites, but stay within the same logical region
            for out_slot in node.out_slots().values():
                for outgoing_edge in out_slot.outgoing_edges():
                    next_node = outgoing_edge.dst().node()
                    if (
                        next_node.region().site_local_logid()
                        == this_region.site_local_logid()
                    ):
                        stack.append(next_node)

        return reachable_nodes

    def _split_region(self, this_region: Region, targets: set[Node]) -> Region:
        # First, find all nodes in this region reachable from the targets
        reachable_nodes = self._find_reachable_inner_nodes_within_region(
            this_region, targets
        )

        region_size_before = this_region.size()
        # create a new region node and migrate the old exit node to it
        new_region = Region(
            graph=self._graph,
            site=this_region.site(),
            site_local_logid=self._graph._get_fresh_per_nodetype_logid(Region),
            is_loop_head=False,
        )
        this_region.exit_node().migrate(new_region)

        # migrate all the reachable nodes to the new region
        for node in reachable_nodes:
            node.migrate(new_region)

        # in the old region, insert a jump to the new region
        jump_node = Jump(self._graph, this_region)
        _ = FlowEdge(
            src=jump_node.out_slot("__flow__"),
            dst=new_region.in_slot("__flow__"),
            send_cost_info=self._graph._fed_info.find_send_cost(
                jump_node.site(), new_region.site(), logger=self._graph.logger
            ),
            is_weak=False,
        )

        assert this_region.size() < region_size_before, (
            f"Region {this_region.global_id()} size must decrease, otherwise infinite loop. Was {region_size_before}, now it's {this_region.size()}"
        )
        return new_region

    def _handle_region(self, region: Region, worklist: list[Region]) -> None:
        # step 1: find all edges that enter or exit this region to/from the same logical
        # region but on a different site
        my_outgoing_edges: list[FlowEdge] = (
            self._all_outgoing_cross_site_edges_to_sibling_regions(region)
        )
        my_incoming_edges: list[FlowEdge] = (
            self._all_incoming_cross_site_edges_from_sibling_regions(region)
        )
        my_receiving_nodes: list[Node] = [
            edge.dst().node() for edge in my_incoming_edges
        ]

        # if no incoming and/or outgoing edges, continue
        if not my_outgoing_edges or not my_incoming_edges:
            return

        # step 2: check if we can reach any of the receiving nodes from any outgoing cross-site
        # edge to sibling regions
        # NOTE: this essentially handles one conflicting edge at a time in arbitrary order
        for outgoing_edge in my_outgoing_edges:
            reachable_receiving_nodes = self._check_reachability(
                outgoing_edge=outgoing_edge, targets=my_receiving_nodes
            )
            if len(reachable_receiving_nodes) > 0:
                # This is an offending edge, we can reach at least one of the receiving nodes, so we need to split the region
                new_region = self._split_region(region, reachable_receiving_nodes)
                # and add the old and new regions to the worklist again, there may be more cycles
                worklist.append(region)
                worklist.append(new_region)

    @override
    def apply(self) -> None:
        worklist = self._graph.get_regions_copy()
        while len(worklist) > 0:
            region = worklist.pop(0)
            self._handle_region(region, worklist)

    @override
    def target_state(self) -> GraphState:
        return GraphState.DEADLOCKS_RESOLVED
