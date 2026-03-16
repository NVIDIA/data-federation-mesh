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

from .._cost import Cost
from .._graph import GraphState, GraphTransformPass
from .._graph_elements import (
    ExitNode,
    Node,
    NodeState,
    Place,
    Region,
)


class SolvePass(GraphTransformPass):
    """
    Propagate costs through the graph and select the nodes that should remain.
    """

    def _find_leaf_candidates(self) -> list[Node]:
        candidates: list[Node] = []
        for node in self._inner_nodes:
            if node.state == NodeState.CANDIDATE and node.is_leaf():
                candidates.append(node)
        return candidates

    def _reset_all_candidates(self) -> None:
        for node in self._inner_nodes:
            if node.state == NodeState.CANDIDATE:
                node.reset_cost()

    def _propagate_cost(self) -> None:
        """Propagate costs through the graph using a work queue.

        This method:
        1. Creates a work queue of graph elements that need and can compute their cost
        2. Processes elements from the queue until it's empty
        3. For each processed element, adds its successors to the queue if they need and can compute their cost
        4. Verifies that all elements have computed costs at the end

        Raises:
            AssertionError: If any element remains without a computed cost after propagation.
        """
        # Create initial work queue
        stack: list[Node] = []
        for node in self._inner_nodes:
            if node.needs_compute_cost() and node.can_compute_cost():
                stack.append(node)

        # Process elements until queue is empty
        while stack:
            node = stack.pop()
            if isinstance(node, (Region, ExitNode)):
                continue
            node.compute_cost_and_cheapest_incoming_edges()
            assert not node.needs_compute_cost(), (
                f"Node {node.global_id()} still needs compute cost after computing its cost"
            )

            # propagate the cost along the edges
            for out_slot in node.out_slots().values():
                for edge in out_slot.outgoing_edges():
                    if (
                        edge.dst().node().needs_compute_cost()
                        and edge.dst().node().can_compute_cost()
                    ):
                        # check if the destination node is ready to compute its cost
                        stack.append(edge.dst().node())
        assert all(not node.needs_compute_cost() for node in self._inner_nodes), (
            f"Not all inner nodes have computed costs: {[node.global_id() for node in self._inner_nodes if node.needs_compute_cost()]}"
        )

    def _pick_best_leaf_candidate(self, leaf_candidates: list[Node]) -> Node:
        """Find the best leaf candidate node.

        This method:
        2. Finds the most expensive leaf
        3. Within the logical group of this most expensive leaf, finds the cheapest alternative
        4. Returns the cheapest alternative, or None if no candidate leaves exist

        Returns:
            The cheapest alternative of the most expensive candidate leaf node, or None if no candidate leaves exist.
        """
        assert len(leaf_candidates) > 0, "No leaf candidates to pick from"

        # Find the most expensive leaf
        def get_cost(node: Node) -> Cost:
            cost = node.cost()
            assert cost is not None, f"Node {node.global_id()} has no computed cost"
            return cost

        most_expensive_leaf = max(leaf_candidates, key=get_cost)

        # Find the cheapest alternative
        siblings = self._graph.get_nodes_for_logid(
            most_expensive_leaf.site_local_logid(), Node
        )
        return min(siblings, key=get_cost)

    def _select_backwards(self, start_node: Node) -> None:
        stack: list[Node] = [start_node]
        while stack:
            selected = stack.pop()
            if (
                selected.state == NodeState.SELECTED
                or selected.state == NodeState.DISCARDED
            ):
                continue
            assert selected.state == NodeState.CANDIDATE, (
                f"Node {selected.global_id()} is not a candidate"
            )
            selected.state = NodeState.SELECTED

            siblings = self._graph.get_nodes_for_logid(
                selected.site_local_logid(), Node
            )
            for member in siblings:
                if member != selected:
                    member.state = (
                        NodeState.DISCARDED
                    )  # we don't propagate discardedness backwards

            # recurse upwards
            for edge in selected.cheapest_incoming_edges():
                stack.append(edge.src().node())

    @override
    def apply(self) -> None:
        """
        picking the cheapest inner node for every operation is done iteratively, one
        leaf node at a time. When we select a leaf node to stay in the graph, all
        its siblings from the same logical node are set to discarded, and then we recurse upwards
        in the graph to select all predecessors of the selected node and set them to selected,
        too (and discard their siblings). Then we find the next best leaf.

        The "best" leaf is heuristically defined as the cheapest variant of the most expensive
        node that does not have any outgoing edges. We do this because it's most important to get
        the most expensive logical operation as fast as possible and we use "the most expensive node"
        as a proxy for finding the "most expensive logical operation".

        We select only one node at a time because two leaf nodes may differ in which predecessors they would choose.
        This is why we need to pick the best leaf candidate iteratively.
        """

        # first, set all region and exit nodes to Selected and
        # find nodes that are ready to compute their cost
        all_nodes = self._graph.get_all_nodes_copy()
        assert all(node.state == NodeState.CANDIDATE for node in all_nodes), (
            "All nodes must be candidates before solving"
        )

        # we assume that all Place nodes explicitly introduced into the graph by the user
        # are fixed. This includes the START_PLACE_NAME node
        edge_nodes: list[Region | ExitNode | Place] = [
            node
            for node in all_nodes
            if isinstance(node, Region)
            or isinstance(node, ExitNode)
            or isinstance(node, Place)
        ]
        self._inner_nodes: list[Node] = [
            node
            for node in all_nodes
            if not isinstance(node, Region) and not isinstance(node, ExitNode)
        ]

        # first select all edge nodes, they will stay in the graph and don't have any cost
        for node in edge_nodes:
            node.state = NodeState.SELECTED
            node._cost = Cost(0, 0)

        while leaf_candiates := self._find_leaf_candidates():
            self._reset_all_candidates()
            self._propagate_cost()
            best_leaf = self._pick_best_leaf_candidate(leaf_candiates)
            assert best_leaf is not None, "No best leaf candidate found"
            self._select_backwards(best_leaf)

        assert all(node.state != NodeState.CANDIDATE for node in self._inner_nodes), (
            f"Some inner nodes are still CANDIDATES after solving: {[node.global_id() for node in self._inner_nodes if node.state == NodeState.CANDIDATE]}"
        )

    @override
    def target_state(self) -> GraphState:
        return GraphState.SOLVED
