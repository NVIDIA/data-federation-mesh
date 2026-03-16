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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Literal

from pydantic import JsonValue
from typing_extensions import override

from nv_dfm_core.api import Operation as OperationStmt
from nv_dfm_core.api import Yield as YieldStmt
from nv_dfm_core.api._bool_expressions import BooleanExpression
from nv_dfm_core.api._node_id import NodeId, NodeRef

from .._fed_info import ComputeCostInfo, OperationInfo, SendCostInfo
from ._cost import Cost


def global_id(site: str, site_local_logid: str) -> str:
    return f"{site}#{site_local_logid}"


class InSlot:
    """An input slot on a graph node that can receive data or control flow."""

    def __init__(
        self,
        node: "Node",
        name: str,
        kind: Literal["data", "control"],
        type: str,
        expect_weak: bool = False,
    ):
        self._node: "Node" = node
        self._name: str = name
        self._kind: Literal["data", "control"] = kind
        self._type: str = type
        self._incoming_edges: list["FlowEdge"] = []
        self._expect_weak: bool = expect_weak

    def node(self) -> "Node":
        return self._node

    def name(self) -> str:
        return self._name

    def kind(self) -> Literal["data", "control"]:
        return self._kind

    def type(self) -> str:
        return self._type

    def incoming_edges(self) -> list["FlowEdge"]:
        return self._incoming_edges

    def can_find_cheapest_edge(self) -> bool:
        for edge in self.incoming_edges():
            if edge.src().node().state == NodeState.SELECTED:
                return True
        for edge in self.incoming_edges():
            if not edge.has_cost():
                return False
        return True

    def find_cheapest_edge(self) -> "FlowEdge|None":
        # first check for a selected predecessor
        for edge in self.incoming_edges():
            if edge.src().node().state == NodeState.SELECTED:
                return edge

        # none found, find the cheapest
        cheapest = None
        for edge in self.incoming_edges():
            # edge is not a selected edge yet
            if cheapest is None or edge.cost() < cheapest.cost():
                cheapest = edge
        return cheapest

    def _register_incoming_edge(self, edge: "FlowEdge"):
        if self._expect_weak and not edge.is_weak():
            raise ValueError(
                f"Weak edge cannot be added to non-weak slot {self.name()}"
            )
        len_before = len(self._incoming_edges)
        self._incoming_edges.append(edge)
        len_after = len(self._incoming_edges)
        assert len_before + 1 == len_after, (
            f"Edge {edge.src().node().global_id()}:{edge.src().name()} -> {edge.dst().node().global_id()}:{edge.dst().name()} not added"
        )

    def _remove_incoming_edge(self, edge: "FlowEdge"):
        len_before = len(self._incoming_edges)
        self._incoming_edges.remove(edge)
        len_after = len(self._incoming_edges)
        assert len_before - 1 == len_after, (
            f"Edge {edge.src().node().global_id()}:{edge.src().name()} -> {edge.dst().node().global_id()}:{edge.dst().name()} not removed"
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutSlot):
            return False
        return self.node() == other.node() and self.name() == other.name()

    @override
    def __hash__(self) -> int:
        return hash((self.node(), self.name()))

    @override
    def __str__(self) -> str:
        return f"{self.node().global_id()}:{self.name()}"


class OutSlot:
    """An output slot on a graph node that can send data or control flow."""

    def __init__(
        self,
        node: "Node",
        name: str,
        kind: Literal["data", "control"],
        type: str,
        expect_weak: bool = False,
    ):
        self._node: "Node" = node
        self._name: str = name
        self._kind: Literal["data", "control"] = kind
        self._type: str = type
        self._outgoing_edges: list["FlowEdge"] = []
        self._expect_weak: bool = expect_weak

    def node(self) -> "Node":
        return self._node

    def name(self) -> str:
        return self._name

    def kind(self) -> Literal["data", "control"]:
        return self._kind

    def type(self) -> str:
        return self._type

    def is_subtype_of(self, other_type: str) -> bool:
        return self._type == other_type or other_type == "Any"

    def outgoing_edges(self) -> list["FlowEdge"]:
        return self._outgoing_edges

    def _register_outgoing_edge(self, edge: "FlowEdge"):
        if self._expect_weak and not edge.is_weak():
            raise ValueError(
                f"Weak edge cannot be added to non-weak slot {self.name()}"
            )
        len_before = len(self._outgoing_edges)
        self._outgoing_edges.append(edge)
        len_after = len(self._outgoing_edges)
        assert len_before + 1 == len_after, (
            f"Edge {edge.src().node().global_id()}:{edge.src().name()} -> {edge.dst().node().global_id()}:{edge.dst().name()} not added"
        )

    def _remove_outgoing_edge(self, edge: "FlowEdge"):
        len_before = len(self._outgoing_edges)
        self._outgoing_edges.remove(edge)
        len_after = len(self._outgoing_edges)
        assert len_before - 1 == len_after, (
            f"Edge {edge.src().node().global_id()}:{edge.src().name()} -> {edge.dst().node().global_id()}:{edge.dst().name()} not removed"
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutSlot):
            return False
        return self.node() == other.node() and self.name() == other.name()

    @override
    def __hash__(self) -> int:
        return hash((self.node(), self.name()))

    @override
    def __str__(self) -> str:
        return f"{self.node().global_id()}:{self.name()}"


class FlowEdge:
    """An edge in the execution graph connecting an output slot to an input slot."""

    def __init__(
        self,
        src: OutSlot,
        dst: InSlot,
        send_cost_info: SendCostInfo | None,
        is_weak: bool,
    ):
        assert src.kind() == dst.kind(), (
            f"Flow edge type mismatch: {src.kind()} != {dst.kind()} in {src} -> {dst}"
        )
        assert src.node().graph() == dst.node().graph(), (
            f"Flow edge src {src} and dst {dst} must be in the same graph"
        )
        assert src.is_subtype_of(dst.type()), (
            f"Flow edge {src}->{dst} src type {src.type()} is not a subtype of dst type {dst.type()}"
        )
        self._src: OutSlot = src
        self._dst: InSlot = dst
        self._send_cost_info: SendCostInfo | None = send_cost_info
        self._is_weak: bool = is_weak
        src._register_outgoing_edge(self)
        dst._register_incoming_edge(self)
        src.node().graph()._register_edge(self)

    def remove(self):
        """Remove this edge from both source and destination slots and from the graph."""
        self._src._remove_outgoing_edge(self)
        self._dst._remove_incoming_edge(self)
        self._src.node().graph()._deregister_edge(self)

    def has_cost(self) -> bool:
        return self.src().node().has_cost()

    def cost(self) -> Cost:
        """Calculate the total cost of this edge including source node cost and transfer time."""
        incoming_cost = self.src().node().cost()

        incoming_time = incoming_cost.time
        incoming_size = incoming_cost.size

        transfer_time: float = (
            self._send_cost_info.compute_time(incoming_size)
            if self._send_cost_info is not None
            else 0
        )
        total_time = incoming_time + transfer_time
        total_size = incoming_size
        return Cost(total_time, total_size)

    def is_weak(self) -> bool:
        return self._is_weak

    def src(self) -> OutSlot:
        return self._src

    def dst(self) -> InSlot:
        return self._dst

    @override
    def __str__(self) -> str:
        if self.is_weak():
            return f"{self.src()} =(weak,{self.src().kind()})=> {self.dst()}"
        else:
            return f"{self.src()} =({self.src().kind()})=> {self.dst()}"

    @override
    def __repr__(self) -> str:
        return f"FlowEdge({self.src()}, {self.dst()}, {self.is_weak()})"

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlowEdge):
            return False
        return self.src() == other.src() and self.dst() == other.dst()

    @override
    def __hash__(self) -> int:
        return hash((self.src(), self.dst()))

    def to_graphviz(self, lines: list[str]) -> None:
        """Generate Graphviz DOT representation of this edge with appropriate styling."""
        src_id = self.src().node().global_id()
        dst_id = self.dst().node().global_id()
        style = "dashed" if self.is_weak() else "solid"
        if self.src().kind() == "data":
            if self in self.dst().node().cheapest_incoming_edges():
                color = "black"
            else:
                color = "gray"
        else:
            color = "blue"
        weight = ", weight=100" if isinstance(self.src().node(), Place) else ""

        lines.append(
            f'    "{src_id}":"out_{self.src().name()}"	 -> "{dst_id}":"in_{self.dst().name()}" [style="{style}", color="{color}"{weight}];'
        )


class NodeState(Enum):
    """State of a node during graph optimization."""

    CANDIDATE = "CANDIDATE"
    SELECTED = "SELECTED"
    DISCARDED = "DISCARDED"
    LEADER = "LEADER"  # in later phases, a selected node that is chosen to be the leader of the logical node


class Node(ABC):
    """Abstract base class for all nodes in the execution graph."""

    def __init_subclass__(cls, **kwargs: Any):
        """We wrap the __init__ method because we want to automatically register this
        Node in the graph with the __init__ call, but the graph registration assumes that
        the Node has been fully initialized, to get its ID etc. We wrap the __init__ and
        call the __post_init__ method after the __init__ call."""

        def init_decorator(previous_init: Callable[..., None]) -> Callable[..., None]:
            def new_init(self: "Node", *args: Any, **kwargs: Any) -> None:
                previous_init(self, *args, **kwargs)
                if type(self) is cls:
                    self.__post_init__()

            return new_init

        cls.__init__ = init_decorator(cls.__init__)

    def __post_init__(self):
        self._region._register_node(self)
        self._graph._register_node(self)

    def __init__(
        self,
        graph: Any,
        region: "Region",
        site_local_logid: str,
        compute_cost_info: ComputeCostInfo | None,
    ):
        assert not site_local_logid[0].isupper(), (
            f"Site local logid {site_local_logid} should start with a lowercase letter (or underscore)"
        )
        assert site_local_logid.isidentifier(), (
            f"Site local logid {site_local_logid} is not a valid Python identifier"
        )
        self._site_local_logid: str = site_local_logid
        self._graph: Any = graph
        self._region: "Region" = region
        self.state: NodeState = NodeState.CANDIDATE
        self._compute_cost_info: ComputeCostInfo | None = compute_cost_info
        self._cost: Cost | None = None
        self._cheapest_edges: list["FlowEdge"] = []

    def reset_cost(self) -> None:
        assert self.state == NodeState.CANDIDATE, (
            "Only candidate nodes can reset cost computation"
        )
        self._cost = None
        self._cheapest_edges = []

    def has_cost(self) -> bool:
        return self._cost is not None

    def cost(self) -> Cost:
        assert self._cost is not None, f"Node {self.global_id()} has no cost"
        return self._cost

    def cheapest_incoming_edges(self) -> list["FlowEdge"]:
        return self._cheapest_edges

    def can_compute_cost(self) -> bool:
        for in_slot in self.in_slots().values():
            if not in_slot.can_find_cheapest_edge():
                return False
        return True

    def needs_compute_cost(self) -> bool:
        return self._cost is None

    def compute_cost_and_cheapest_incoming_edges(self) -> None:
        # this is defined on node, but essentially with inner nodes in mind
        # (that is, no Region or ExitNode nodes). But outer nodes are automatically
        # selected anyways and compute_cost will never be called on them.

        # each slot computes its cost and the edge it uses for this cost
        edge_costs: list[Cost] = []
        for in_slot in self.in_slots().values():
            edge = in_slot.find_cheapest_edge()
            if edge is not None:
                edge_costs.append(edge.cost())
                self._cheapest_edges.append(edge)

        # We sum up the slot costs, because we need all of them
        incoming_cost = Cost.sum(edge_costs)

        # now derive from the incoming cost the cost of this node
        if self._compute_cost_info:
            additional_compute_time = self._compute_cost_info.compute_time(
                incoming_cost.size
            )
            output_size = self._compute_cost_info.compute_size(incoming_cost.size)
        else:
            additional_compute_time = 0
            # assume it stays the same
            output_size = incoming_cost.size

        total_time = incoming_cost.time + additional_compute_time

        self._cost = Cost(total_time, output_size)

    @abstractmethod
    def site(self) -> str:
        pass

    @abstractmethod
    def in_slots(self) -> "dict[str, InSlot]":
        pass

    @abstractmethod
    def out_slots(self) -> "dict[str, OutSlot]":
        pass

    @abstractmethod
    def accept(self, visitor: "GraphVisitor") -> Any:
        pass

    def remove(self, force: bool = False):
        for slot in self.in_slots().values():
            # Note: IMPORTANT: make sure to clone this list (with list()), because we'll remove from it
            for edge in list(slot._incoming_edges):
                edge.remove()
        for slot in self.out_slots().values():
            # Note: IMPORTANT: make sure to clone this list, because we'll remove from it
            for edge in list(slot._outgoing_edges):
                edge.remove()
        self.region()._deregister_node(self)
        self._graph._deregister_node(self, force=force)

    def site_local_logid(self) -> str:
        """logid is the logical ID of this node. Logical ID, because this logical node (e.g.
        a region enter) may be replicated across multiple sites."""
        return self._site_local_logid

    def global_id(self) -> str:
        """Global ID is the ID of this node in the global graph. It is unique across all sites."""
        return global_id(self.site(), self.site_local_logid())

    def graph(self) -> Any:
        return self._graph

    def region(self) -> "Region":
        return self._region

    def in_slot(self, name: str) -> InSlot:
        assert name in self.in_slots(), f"Node {self} has no in slot {name}"
        return self.in_slots()[name]

    def out_slot(self, name: str) -> OutSlot:
        assert name in self.out_slots(), f"Node {self} has no out slot {name}"
        return self.out_slots()[name]

    def migrate(self, new_region: "Region"):
        glob_id_before = self.global_id()
        self._region._deregister_node(self)
        self._region = new_region
        self._region._register_node(self)
        assert glob_id_before == self.global_id(), (
            f"Node {self} migrated to {self.global_id()} and global ID changed unexpectedly"
        )

    def is_leaf(self) -> bool:
        # a node is a leaf if it either doesn't have outgoing edges
        # or all its outgoing edges lead to already selected nodes
        # (generally exit or region nodes)
        for slot in self.out_slots().values():
            for edge in slot.outgoing_edges():
                if edge.dst().node().state == NodeState.CANDIDATE:
                    return False
        return True

    @override
    def __str__(self) -> str:
        return f"{type(self).__name__}@{self.site()}:{self.site_local_logid()}"

    def graphviz_node_name(self) -> str:
        return type(self).__name__

    def to_graphviz(self, lines: list[str]) -> None:
        # Use the node's global_id as the node name, and label as its class name and logid
        node_id = self.global_id().replace(":", "__")

        cost = f"\\n{self.cost()}" if self.has_cost() else ""
        base_label = f"{self.graphviz_node_name()}@{self.site()}\\n{self.site_local_logid()}{cost}"
        in_slots = " | ".join([f"<in_{slot}>{slot}" for slot in self.in_slots().keys()])
        out_slots = " | ".join(
            [f"<out_{slot}>{slot}" for slot in self.out_slots().keys()]
        )
        label = f'"<_root_>{base_label} | {{ {{ {in_slots} }} | {{ {out_slots} }} }}"'
        shape = "record"

        if self.state == NodeState.CANDIDATE:
            attribs = 'style="filled", fillcolor="gray", '
        elif self.state == NodeState.DISCARDED:
            attribs = 'style="filled", fillcolor="red", '
        else:
            attribs = ""

        if isinstance(self, Place):
            shape = "Mrecord"
        elif isinstance(self, Send):
            shape = "oval"
            label = f'"{self.site()}\\n{self.site_local_logid()}"'
        elif isinstance(self, Region) or isinstance(self, ExitNode):
            if self.state == NodeState.LEADER:
                attribs = 'style="filled", fillcolor="lightgreen", '
            else:
                attribs = 'style="filled", fillcolor="lightblue", '

        lines.append(
            f'    "{node_id}" [{attribs}shape={shape}, label={label} {attribs}];'
        )


class Region(Node):
    """A region node representing a scope or block of execution within the graph."""

    def __init__(
        self,
        graph: Any,
        site: str,
        site_local_logid: str,
        is_loop_head: bool,
    ):
        super().__init__(
            graph=graph,
            region=self,
            site_local_logid=site_local_logid,
            # we don't compute cost for Regions
            compute_cost_info=None,
        )
        self._site: str = site
        self._flow_in_slot: InSlot = InSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )
        self._nodes: dict[str, "Node"] = {}
        self._places: dict[str, "Place"] = {}
        self._exit_node: "ExitNode|None" = None
        self._is_loop_head: bool = is_loop_head

    def _register_node(self, node: "Node"):
        if isinstance(node, Place):
            assert node.site_local_logid() not in self._places, (
                f"Place {node.site_local_logid()} already in region {self.site_local_logid()}"
            )
            self._places[node.site_local_logid()] = node
        elif isinstance(node, ExitNode):
            assert self._exit_node is None, "Region can only have one exit node"
            self._exit_node = node
        elif isinstance(node, Region):
            # region constructor will call register with itself, which is okay
            assert node == self, "Region constructor should only register itself"
        else:
            assert node.site_local_logid() not in self._nodes, (
                f"Node {node.site_local_logid()} already in region {self.site_local_logid()}"
            )
            self._nodes[node.site_local_logid()] = node

    def _deregister_node(self, node: "Node"):
        if node == self:
            # happens when a region node is removed, because it is its own region
            return
        if isinstance(node, Place):
            assert node.site_local_logid() in self._places, (
                f"Place {node.site_local_logid()} not in region {self.site_local_logid()}"
            )
            assert self._places[node.site_local_logid()] == node, (
                f"Place {node.site_local_logid()} not in region {self.site_local_logid()}"
            )
            if node.site_local_logid() in self._places:
                del self._places[node.site_local_logid()]
        elif isinstance(node, ExitNode):
            assert node == self._exit_node, (
                f"Exit node {node.site_local_logid()} not in region {self.site_local_logid()}"
            )
            self._exit_node = None
        else:
            assert node.site_local_logid() in self._nodes, (
                f"Node {node.site_local_logid()} not in region {self.site_local_logid()}"
            )
            assert self._nodes[node.site_local_logid()] == node, (
                f"Node {node.site_local_logid()} not in region {self.site_local_logid()}"
            )
            del self._nodes[node.site_local_logid()]

    def size(self) -> int:
        return len(self._nodes)

    def nodes_copy(self) -> list["Node"]:
        return list(self._nodes.values())

    def places(self) -> list["Place"]:
        return list(self._places.values())

    def exit_node(self) -> "ExitNode":
        assert self._exit_node is not None, (
            f"Region {self.global_id()} has no exit node"
        )
        return self._exit_node

    def is_loop_head(self) -> bool:
        return self._is_loop_head

    def flow_in_place_after_cut(self) -> "Place":
        assert len(self._flow_in_slot.incoming_edges()) == 1, (
            f"Region {self.global_id()} has {len(self._flow_in_slot.incoming_edges())} incoming edges"
        )
        src = self._flow_in_slot.incoming_edges()[0].src().node()
        assert isinstance(src, Place), (
            f"Region {self.global_id()} flow in place has source {src}, expected a Place"
        )
        return src

    @override
    def migrate(self, new_region: "Region"):
        raise ValueError("Region cannot be migrated")

    @override
    def remove(self, force: bool = False):
        assert force, "Region cannot be removed without force"
        for node in list(self._nodes.values()):
            node.remove(force=force)
        for place in list(self._places.values()):
            place.remove(force=force)
        if self._exit_node is not None:
            self._exit_node.remove(force=force)
        super().remove(force=force)

    @override
    def region(self) -> "Region":
        return self

    @override
    def site(self) -> str:
        return self._site

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {"__flow__": self._flow_in_slot}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {}

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_region(self)

    def visit_in_topological_order(self, visitor: "GraphVisitor") -> None:
        """Visit all nodes in the region in topological order.

        The order is:
        1. Region node itself
        2. All places
        3. All nodes in topological order (excluding places and exit nodes)
        4. The single exit node
        """
        # 1. Visit the region node itself
        visitor.visit_region(self)

        # 2. Visit all places
        places = list(self._places.values())
        places.sort(key=lambda p: p.global_id())
        for place in places:
            place.accept(visitor)

        # 3. Visit all nodes in topological order (excluding places and exit nodes)
        self._visit_nodes_topologically(visitor)

        # 4. Visit the exit node
        assert self._exit_node is not None
        self._exit_node.accept(visitor)

    def _visit_nodes_topologically(self, visitor: "GraphVisitor") -> None:
        """Visit nodes in topological order using Kahn's algorithm.
        This sort should be stable to simplify testing:
        ties are broken by sorting nodes by their site_local_logid"""
        # Build adjacency list and in-degree count
        adjacency: dict[Node, list[Node]] = {}
        in_degree: dict[Node, int] = {}

        # Initialize all nodes
        for node in self._nodes.values():
            adjacency[node] = []
            in_degree[node] = 0

        # Build adjacency list and count in-degrees
        for node in self._nodes.values():
            for out_slot in node.out_slots().values():
                for edge in out_slot.outgoing_edges():
                    dst_node = edge.dst().node()
                    # Only consider edges to other nodes in this region (not places or exit nodes)
                    if dst_node.region() == self:
                        adjacency[node].append(dst_node)
                        in_degree[dst_node] += 1

        # Find all nodes with in-degree 0
        queue: list[Node] = [
            node for node in self._nodes.values() if in_degree[node] == 0
        ]
        # Sort by site_local_logid for stable ordering
        queue.sort(key=lambda node: node.site_local_logid())
        visited: list[Node] = []

        # Process nodes in topological order
        while queue:
            current = queue.pop(0)
            visited.append(current)

            # Visit the current node
            current.accept(visitor)

            # Reduce in-degree of all neighbors
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

            # Re-sort queue to maintain stable ordering for newly added nodes
            queue.sort(key=lambda node: node.site_local_logid())

        # Check for cycles (if not all nodes were visited)
        if len(visited) != len(self._nodes):
            remaining = [node for node in self._nodes.values() if node not in visited]
            raise ValueError(
                f"Graph contains cycles. Unvisited nodes: {[node.global_id() for node in remaining]}"
            )

    @override
    def to_graphviz(self, lines: list[str], show_region_edges: bool = True) -> None:
        super().to_graphviz(lines)
        if show_region_edges:
            attribs = 'style="dashed", color="orange"'
            for node in self._nodes.values():
                lines.append(
                    f'    "{self.global_id()}":"_root_"	 -> "{node.global_id()}":"_root_" [{attribs}];'
                )
            # for place in self._places.values():
            #     lines.append(f'    "{self.global_id()}"	 -> "{place.global_id()}" [{attribs}];')
            if self._exit_node is not None:
                lines.append(
                    f'    "{self.global_id()}":"_root_"	 -> "{self._exit_node.global_id()}":"_root_" [{attribs}];'
                )
                for node in self._nodes.values():
                    lines.append(
                        f'    "{node.global_id()}":"_root_"	 -> "{self._exit_node.global_id()}":"_root_" [{attribs}];'
                    )


class ExitNode(Node, ABC):
    """Abstract base class for nodes that represent exits from regions."""

    def __init__(self, graph: Any, region: "Region"):
        super().__init__(
            graph=region.graph(),
            region=region,
            site_local_logid=f"{region.site_local_logid()}_{type(self).__name__}",
            # we don't compute cost for ExitNodes
            compute_cost_info=None,
        )

    @override
    def region(self) -> "Region":
        return self._region

    @override
    def migrate(self, new_region: "Region"):
        self._graph._deregister_node(self, force=True)
        self._region._deregister_node(self)
        self._region: Region = new_region
        self._site_local_logid: str = (
            f"{self._region.site_local_logid()}_{type(self).__name__}"
        )
        self._region._register_node(self)
        self._graph._register_node(self)

    @override
    def site(self) -> str:
        return self._region.site()


class Place(Node):
    """
    flavors:
        queue: the place is a normal queue
        prefix: the place matches frames by prefix. E.g. frame (32,) matches place data (32, 1)
        sticky: the place matches by prefix but doesn't dequeue the values.
        framecount: the place counts incoming frames, discarding values
        yield: the place is a yield place, which is used to send data back to the client
    """

    def __init__(
        self,
        graph: Any,
        region: "Region",
        site_local_logid: str,
        kind: Literal["control", "data"],
        origin: Literal["internal", "external"],
        # control is a control place (already encoded in kind) that keeps exact frames
        # scoped is a place that keeps values in a scoped fashion. E.g. if a value was received for frame [1,3,2], then that value is valid also for frame [1,3,2,99]
        # sticky is a scoped place that doesn't delete values
        # framecount counts incoming frames of the same level. E.g. if it receives [1,2,3] and [1,2,3,4], it will count 2 frames for scope [1,2,3]
        flavor: Literal["control", "scoped", "sticky", "framecount", "yield"],
        type: str,
    ):
        super().__init__(
            graph=graph,
            region=region,
            site_local_logid=site_local_logid,
            # places forward the data they receive, we assume no computational overhead
            compute_cost_info=None,
        )
        if origin == "internal":
            self._in_slot: InSlot | None = InSlot(
                self, name="__data__", kind=kind, type=type
            )
        else:
            self._in_slot = None
        self._out_slot: OutSlot = OutSlot(self, name="__data__", kind=kind, type=type)
        self.kind: Literal["control", "data"] = kind
        self.origin: Literal["internal", "external"] = origin
        self.flavor: Literal["control", "scoped", "sticky", "framecount", "yield"] = (
            flavor
        )
        if self.kind == "control":
            assert self.flavor in ("control", "framecount"), (
                f"Places of kind 'control' must have flavor 'control' or 'framecount'. Place '{self.global_id()}' has flavor {self.flavor}"
            )
        else:
            assert self.flavor in ("scoped", "sticky", "yield"), (
                f"Places of kind 'data' must have flavor 'scoped', 'sticky', or 'framecount'. Place '{self.global_id()}' has flavor {self.flavor}"
            )
        self.type: str = type

    @override
    def region(self) -> "Region":
        return self._region

    @override
    def site(self) -> str:
        return self._region.site()

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        if self._in_slot is None:
            return {}
        else:
            return {"__data__": self._in_slot}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {"__data__": self._out_slot}

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_place(self)


class Operation(Node):
    """A node representing an operation to be executed."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
        operation: OperationStmt,
        operation_info: OperationInfo,
    ):
        super().__init__(
            graph=graph,
            region=region,
            site_local_logid=operation.dfm_node_id.as_identifier(),
            compute_cost_info=operation_info.compute_cost,
        )
        self.operation: OperationStmt = operation
        self.operation_info: OperationInfo = operation_info

        self._flow_in_slot: InSlot = InSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )

        self._param_in_slots: dict[str, InSlot] = {}
        for (
            param_name,
            _value,
        ) in operation.get_noderef_and_placeparam_pydantic_fields():
            # NOTE: using "Any" here until we have a way to figure out the actual type. Need to implement type checking.
            self._param_in_slots[param_name] = InSlot(
                self, param_name, "data", type="Any"
            )

        self._flow_out_slot: OutSlot = OutSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )
        self._data_out_slot: OutSlot = OutSlot(
            self,
            name="__data__",
            kind="data",
            # NOTE: see above, we need to implement type checking
            type="Any",
        )

    def param_source_after_cut(self, name: str) -> Node:
        """Get the source node for a parameter after cross-region edge cutting."""
        slot = self._param_in_slots[name]
        assert len(slot.incoming_edges()) == 1, (
            f"Operation {self.operation.__api_name__} has multiple or no incoming edges for param field {name}"
        )
        source = slot.incoming_edges()[0].src().node()
        return source

    @override
    def region(self) -> "Region":
        return self._region

    @override
    def site(self) -> str:
        return self._region.site()

    def has_users(self) -> bool:
        return len(self._data_out_slot.outgoing_edges()) > 0

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {"__flow__": self._flow_in_slot, **self._param_in_slots}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {"__flow__": self._flow_out_slot, "__data__": self._data_out_slot}

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_operation(self)

    @override
    def graphviz_node_name(self) -> str:
        return f"{self.operation.__api_name__}()"


class BoolValue(Node):
    """A node that evaluates a boolean expression."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
        site_local_logid: str,
        bool_expr: BooleanExpression,
        params: set[NodeId],
    ):
        super().__init__(
            graph=graph,
            region=region,
            site_local_logid=site_local_logid,
            compute_cost_info=ComputeCostInfo(
                fixed_size=1,  # only a bool
                output_factor=0.0,  # none of the expression data is sent along
            ),
        )
        self.bool_expr: BooleanExpression = bool_expr
        self._params: set[NodeId] = params

        self._flow_in_slot: InSlot = InSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )

        self._param_in_slots: dict[str, InSlot] = {}
        for param in params:
            assert isinstance(param, NodeId), (
                f"Param {param} is not a NodeId. Was {type(param)}: {param}"
            )
            # NOTE: need to implement type checking
            self._param_in_slots[param.as_identifier()] = InSlot(
                self, param.as_identifier(), "data", type="Any"
            )

        self._flow_out_slot: OutSlot = OutSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )

        self._data_out_slot: OutSlot = OutSlot(
            self, name="__condition__", kind="data", type="Any"
        )

    @override
    def region(self) -> "Region":
        return self._region

    @override
    def site(self) -> str:
        return self._region.site()

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {"__flow__": self._flow_in_slot, **self._param_in_slots}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {"__flow__": self._flow_out_slot, "__condition__": self._data_out_slot}

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_bool_value(self)


class Next(Node):
    """The Next() node is more conceptual in nature. The SeqForEach and ParForEach ExitNodes
    actually control and manage the call to next(). However, we want next() to be represented
    in the graph as its own data node, because we want the graph passes to work normally
    (e.g finding best site, pruning, cross-site edges etc)."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
        site_local_logid: str,
        in_type: str,
        out_type: str,
        is_async: bool,
    ):
        super().__init__(
            graph=graph,
            region=region,
            site_local_logid=site_local_logid,
            compute_cost_info=ComputeCostInfo(
                output_factor=1.0,  # all of the data is sent along
            ),
        )
        self.is_async: bool = is_async

        self._flow_in_slot: InSlot = InSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )

        self._iterator_in_slot: InSlot = InSlot(
            self, name="__iterator__", kind="data", type=in_type
        )

        self._flow_out_slot: OutSlot = OutSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )

        self._data_out_slot: OutSlot = OutSlot(
            self, name="__data__", kind="data", type=out_type
        )

        self._has_next_out_slot: OutSlot = OutSlot(  # this slot is used to connect the Next node to the SeqForEach or the ParForEach
            self, name="__has_next__", kind="data", type="bool"
        )

    def data_targets_after_cut(self) -> list[Place]:
        targets: list[Place] = []
        for edge in self.out_slot("__data__").outgoing_edges():
            tgt = edge.dst().node()
            assert isinstance(tgt, Send)
            targets.append(tgt.send_target())
        return targets

    def iterator_source_after_cut(self) -> Node:
        assert len(self.in_slot("__iterator__").incoming_edges()) == 1, (
            f"Next {self.site_local_logid()} has multiple incoming edges"
        )
        src = self.in_slot("__iterator__").incoming_edges()[0].src().node()
        return src

    @override
    def region(self) -> "Region":
        return self._region

    @override
    def site(self) -> str:
        return self._region.site()

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {
            "__flow__": self._flow_in_slot,
            "__iterator__": self._iterator_in_slot,
        }

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {
            "__flow__": self._flow_out_slot,
            "__data__": self._data_out_slot,
            "__has_next__": self._has_next_out_slot,
        }

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_next(self)


class Yield(Node):
    """A node that yields data back to the client."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
        yld: YieldStmt,
        # the yield cost is the sending of the data back to the client
        send_cost_info: SendCostInfo,
    ):
        super().__init__(
            graph=graph,
            region=region,
            site_local_logid=yld.dfm_node_id.as_identifier(),
            compute_cost_info=ComputeCostInfo(
                compute_throughput=send_cost_info.bandwidth,  # the "compute" is the sending of the data back to the client
                output_factor=0.0,  # yield doesn't return any data
            ),
        )
        self.yld: YieldStmt = yld

        self._flow_in_slot: InSlot = InSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )

        self._value_in_slot: InSlot | None = None
        if isinstance(yld.value, NodeRef):
            self._value_in_slot = InSlot(self, name="value", kind="data", type="Any")

        self._flow_out_slot: OutSlot = OutSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )
        self._data_out_slot: OutSlot = OutSlot(
            self, name="__data__", kind="data", type="Any"
        )

    @override
    def region(self) -> "Region":
        return self._region

    @override
    def site(self) -> str:
        return self._region.site()

    def yield_target_after_cut(self) -> Place:
        assert len(self.out_slot("__data__").outgoing_edges()) == 1, (
            f"Yield {self.site_local_logid()} has multiple outgoing edges"
        )
        target = self.out_slot("__data__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Place)
        return target

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        if self._value_in_slot is None:
            return {
                "__flow__": self._flow_in_slot,
            }
        else:
            return {
                "__flow__": self._flow_in_slot,
                "value": self._value_in_slot,
            }

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {
            "__flow__": self._flow_out_slot,
            "__data__": self._data_out_slot,
        }

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_yield(self)


class Send(Node):
    """A node that sends data or control flow to another node, potentially across sites."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
        site_local_logid: str,
        data_kind: Literal["data", "control"],
        type: str,
        # tagged json value if this send node should send a literal
        literal_data: tuple[str, JsonValue] | None = None,
    ):
        self.literal_data: tuple[str, JsonValue] | None = literal_data
        super().__init__(
            graph=graph,
            region=region,
            site_local_logid=site_local_logid,
            # Send sends the same data it received, we assume no computational overhead
            compute_cost_info=None,
        )
        self._flow_in_slot: InSlot = InSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )
        # using a normal "value" instead of __data__ because Send may become a user-exposed Statement at some point
        self._value_in_slot: InSlot = InSlot(
            self, name="value", kind=data_kind, type=type
        )
        self._flow_out_slot: OutSlot = OutSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )
        self._data_out_slot: OutSlot = OutSlot(
            self, name="__data__", kind=data_kind, type=type
        )

    def is_exit_node_send(self) -> bool:
        assert len(self._value_in_slot.incoming_edges()) == 1, (
            f"Send {self.site_local_logid()} has multiple incoming edges"
        )
        return isinstance(
            self._value_in_slot.incoming_edges()[0].src().node(), ExitNode
        )

    def send_target(self) -> Place:
        assert len(self.out_slot("__data__").outgoing_edges()) == 1, (
            f"Send {self.site_local_logid()} has multiple outgoing edges"
        )
        target = self.out_slot("__data__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Place)
        return target

    @override
    def region(self) -> "Region":
        return self._region

    @override
    def site(self) -> str:
        return self._region.site()

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {
            "__flow__": self._flow_in_slot,
            "value": self._value_in_slot,
        }

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {
            "__flow__": self._flow_out_slot,
            "__data__": self._data_out_slot,
        }

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_send(self)


class Jump(ExitNode):
    """An exit node that jumps to another region."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
    ):
        super().__init__(graph, region)
        self._flow_out_slot: OutSlot = OutSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )

    def jump_target_after_cut(self) -> Place:
        assert len(self.out_slot("__flow__").outgoing_edges()) == 1, (
            f"Jump {self.site_local_logid()} has multiple outgoing edges"
        )
        target = self.out_slot("__flow__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {"__flow__": self._flow_out_slot}

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_jump(self)


class CannotReach(ExitNode):
    """An exit node for regions that cannot exit. That's usually because
    the region is not meant to be executed (e.g. the yield region)."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
    ):
        super().__init__(graph, region)

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {}

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_cannot_reach(self)


class Loop(ExitNode):
    """An exit node representing a loop back to a loop head region."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
    ):
        super().__init__(graph, region)
        self._flow_out_slot: OutSlot = OutSlot(
            self, name="__flow__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )

    def loop_target_after_cut(self) -> Place:
        assert len(self.out_slot("__flow__").outgoing_edges()) == 1, (
            f"Loop {self.site_local_logid()} has multiple outgoing edges"
        )
        target = self.out_slot("__flow__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    def loop_head_after_cut(self) -> Region:
        place = self.loop_target_after_cut()
        assert (
            place.region()
            == place.out_slot("__flow__").outgoing_edges()[0].dst().node()
        ), (
            f"Loop {self.site_local_logid()} target place does not input to the same region"
        )
        return place.region()

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {"__flow__": self._flow_out_slot}

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_loop(self)


class EndFork(ExitNode):
    """An exit node that ends a forked execution branch."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
    ):
        super().__init__(graph, region)
        self._flow_out_slot: OutSlot = OutSlot(
            self,
            name="__fork_frame__",
            kind="control",
            type="nv_dfm_core.exec.FlowInfo",
        )

    def end_fork_target_after_cut(self) -> Place:
        assert len(self.out_slot("__fork_frame__").outgoing_edges()) == 1, (
            f"EndFork {self.site_local_logid()} has multiple outgoing edges"
        )
        target = self.out_slot("__fork_frame__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {"__fork_frame__": self._flow_out_slot}

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_end_fork(self)


class Stop(ExitNode):
    """An exit node that stops execution and optionally yields to a place."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
        yield_place: str,
    ):
        super().__init__(graph, region)
        self.yield_place: str = yield_place
        self._sync_in_slots: dict[str, InSlot] = {}
        self._sync_out_slot: OutSlot = OutSlot(
            self, name="__sync__", kind="data", type="Any"
        )

    def sync_target_after_cut(self) -> Place:
        assert len(self.out_slot("__sync__").outgoing_edges()) == 1, (
            f"Stop {self.global_id()} has multiple or no outgoing edges: {self.out_slot('__sync__').outgoing_edges()}"
        )
        target = self.out_slot("__sync__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    def create_sync_input_slot(self, name: str):
        self._sync_in_slots[name] = InSlot(self, name=name, kind="data", type="Any")

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {**self._sync_in_slots}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {"__sync__": self._sync_out_slot}

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_stop(self)


class Branch(ExitNode):
    """An exit node representing a conditional branch based on a boolean condition."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
    ):
        super().__init__(graph, region)
        self._data_in_slot: InSlot = InSlot(
            self, name="__condition__", kind="data", type="Any"
        )
        self._branch_taken_flow_out_slot: OutSlot = OutSlot(
            self,
            name="__branch_taken__",
            kind="control",
            type="nv_dfm_core.exec.FlowInfo",
        )
        self._branch_not_taken_flow_out_slot: OutSlot = OutSlot(
            self,
            name="__branch_not_taken__",
            kind="control",
            type="nv_dfm_core.exec.FlowInfo",
        )
        self._sync_out_slot: OutSlot = OutSlot(
            self, name="__sync__", kind="data", type="Any"
        )

    def condition_source(self) -> Node:
        assert len(self.in_slot("__condition__").incoming_edges()) == 1, (
            f"Branch {self.site_local_logid()} __condition__ has multiple incoming edges"
        )
        return self.in_slot("__condition__").incoming_edges()[0].src().node()

    def branch_taken_target_after_cut(self) -> Place:
        assert len(self.out_slot("__branch_taken__").outgoing_edges()) == 1, (
            f"Branch {self.site_local_logid()} __branch_taken__ has multiple outgoing edges"
        )
        target = self.out_slot("__branch_taken__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    def branch_not_taken_target_after_cut(self) -> Place:
        assert len(self.out_slot("__branch_not_taken__").outgoing_edges()) == 1, (
            f"Branch {self.site_local_logid()} __branch_not_taken__ has multiple outgoing edges"
        )
        target = self.out_slot("__branch_not_taken__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    def after_target_after_cut(self) -> Place:
        assert len(self.out_slot("__after__").outgoing_edges()) == 1, (
            f"Branch {self.site_local_logid()} __after__ has multiple outgoing edges"
        )
        target = self.out_slot("__after__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    def sync_targets_after_cut(self) -> list[Place]:
        targets: list[Place] = []
        for edge in self.out_slot("__sync__").outgoing_edges():
            tgt = edge.dst().node()
            assert isinstance(tgt, Send)
            targets.append(tgt.send_target())
        return targets

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {"__condition__": self._data_in_slot}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {
            "__branch_taken__": self._branch_taken_flow_out_slot,
            "__branch_not_taken__": self._branch_not_taken_flow_out_slot,
            "__sync__": self._sync_out_slot,
        }

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_branch(self)


class SeqForEach(ExitNode):
    """A synchronous iterator node."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
        foreach_node_id: NodeId,
    ):
        super().__init__(graph, region)
        # the ID of the original loop node. ExitNodes generate their own names and don't
        # use the original node ID, but in case it's needed, it's here
        self.foreach_node_id: NodeId = foreach_node_id
        self._has_next_in_slot: InSlot = InSlot(  # this slot is used to connect the Next node to the SeqForEach or the ParForEach
            self,
            name="__has_next__",
            kind="data",
            type="bool",
        )
        self._has_more_flow_out_slot: OutSlot = OutSlot(
            self,
            name="__next_iteration__",
            kind="control",
            type="nv_dfm_core.exec.FlowInfo",
        )
        self._stop_iteration_flow_out_slot: OutSlot = OutSlot(
            self,
            name="__stop_iteration__",
            kind="control",
            type="nv_dfm_core.exec.FlowInfo",
        )
        self._sync_out_slot: OutSlot = OutSlot(
            self, name="__sync__", kind="data", type="bool"
        )

    def has_next_source(self) -> Next:
        # the leader should have a direct connection to the next node
        assert len(self.in_slot("__has_next__").incoming_edges()) == 1, (
            f"SeqIterate {self.site_local_logid()} has multiple incoming edges to __has_next__"
        )
        node = self.in_slot("__has_next__").incoming_edges()[0].src().node()
        assert isinstance(node, Next)
        return node

    def has_next_place(self) -> Place:
        # followers have an input place for the has_next
        assert len(self.in_slot("__has_next__").incoming_edges()) == 1, (
            f"SeqIterate {self.site_local_logid()} has multiple incoming edges to __has_next__"
        )
        node = self.in_slot("__has_next__").incoming_edges()[0].src().node()
        assert isinstance(node, Place)
        return node

    def next_iteration_target_after_cut(self) -> Place:
        assert len(self.out_slot("__next_iteration__").outgoing_edges()) == 1, (
            f"SeqIterate {self.site_local_logid()} has multiple outgoing edges"
        )
        target = self.out_slot("__next_iteration__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    def stop_iteration_target_after_cut(self) -> Place:
        assert len(self.out_slot("__stop_iteration__").outgoing_edges()) == 1, (
            f"SeqIterate {self.site_local_logid()} has multiple outgoing edges"
        )
        target = self.out_slot("__stop_iteration__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    def sync_targets_after_cut(self) -> list[Place]:
        targets: list[Place] = []
        for edge in self.out_slot("__sync__").outgoing_edges():
            tgt = edge.dst().node()
            assert isinstance(tgt, Send)
            targets.append(tgt.send_target())
        return targets

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {"__has_next__": self._has_next_in_slot}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {
            "__next_iteration__": self._has_more_flow_out_slot,
            "__stop_iteration__": self._stop_iteration_flow_out_slot,
            "__sync__": self._sync_out_slot,
        }

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_seq_iterate(self)


class ParForEach(ExitNode):
    """A parallel iterator node. The ParForEach node iterates through the sequence
    and sends a new Frame through its __fork__ slot and a data item through its
    __data__ slot for every element. And when the list is done, it sends the original Frame
    with the number of elements as the frame info to the __stop_iteration__ slot where it then can be synced."""

    def __init__(
        self,
        graph: Any,
        region: "Region",
        foreach_node_id: NodeId,
    ):
        super().__init__(graph, region)
        self.foreach_node_id: NodeId = foreach_node_id
        self._has_next_in_slot: InSlot = InSlot(
            self,
            name="__has_next__",
            kind="data",
            type="bool",
        )
        self._fork_out_slot: OutSlot = OutSlot(
            self, name="__fork__", kind="control", type="nv_dfm_core.exec.FlowInfo"
        )
        self._stop_iteration_flow_out_slot: OutSlot = OutSlot(
            self,
            name="__stop_iteration__",
            kind="control",
            type="nv_dfm_core.exec.FlowInfo",
        )
        self._sync_out_slot: OutSlot = OutSlot(
            self, name="__sync__", kind="data", type="bool"
        )

    def has_next_source(self) -> Next:
        assert len(self.in_slot("__has_next__").incoming_edges()) == 1, (
            f"ParForEach {self.site_local_logid()} has multiple incoming edges to __has_next__"
        )
        node = self.in_slot("__has_next__").incoming_edges()[0].src().node()
        assert isinstance(node, Next)
        return node

    def has_next_place(self) -> Place:
        # followers have an input place for the has_next
        assert len(self.in_slot("__has_next__").incoming_edges()) == 1, (
            f"SeqIterate {self.site_local_logid()} has multiple incoming edges to __has_next__"
        )
        node = self.in_slot("__has_next__").incoming_edges()[0].src().node()
        assert isinstance(node, Place)
        return node

    def fork_target_after_cut(self) -> Place:
        assert len(self.out_slot("__fork__").outgoing_edges()) == 1, (
            f"ParForEach {self.site_local_logid()} has multiple outgoing edges"
        )
        target = self.out_slot("__fork__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    def stop_iteration_target_after_cut(self) -> Place:
        assert len(self.out_slot("__stop_iteration__").outgoing_edges()) == 1, (
            f"ParForEach {self.site_local_logid()} has multiple outgoing edges"
        )
        target = self.out_slot("__stop_iteration__").outgoing_edges()[0].dst().node()
        assert isinstance(target, Send)
        return target.send_target()

    def sync_targets_after_cut(self) -> list[Place]:
        targets: list[Place] = []
        for edge in self.out_slot("__sync__").outgoing_edges():
            tgt = edge.dst().node()
            assert isinstance(tgt, Send)
            targets.append(tgt.send_target())
        return targets

    @override
    def in_slots(self) -> "dict[str, InSlot]":
        return {"__has_next__": self._has_next_in_slot}

    @override
    def out_slots(self) -> "dict[str, OutSlot]":
        return {
            "__fork__": self._fork_out_slot,
            "__stop_iteration__": self._stop_iteration_flow_out_slot,
            "__sync__": self._sync_out_slot,
        }

    @override
    def accept(self, visitor: "GraphVisitor") -> Any:
        return visitor.visit_par_iterate(self)


class GraphVisitor(ABC):
    """Abstract visitor class for visiting graph nodes."""

    @abstractmethod
    def visit_region(self, region: "Region") -> Any:
        pass

    @abstractmethod
    def visit_place(self, place: "Place") -> Any:
        pass

    @abstractmethod
    def visit_operation(self, operation: "Operation") -> Any:
        pass

    @abstractmethod
    def visit_bool_value(self, bool_value: "BoolValue") -> Any:
        pass

    @abstractmethod
    def visit_next(self, next: "Next") -> Any:
        pass

    @abstractmethod
    def visit_yield(self, yield_node: "Yield") -> Any:
        pass

    @abstractmethod
    def visit_send(self, send: "Send") -> Any:
        pass

    @abstractmethod
    def visit_jump(self, jump: "Jump") -> Any:
        pass

    @abstractmethod
    def visit_cannot_reach(self, cannot_reach: "CannotReach") -> Any:
        pass

    @abstractmethod
    def visit_loop(self, loop: "Loop") -> Any:
        pass

    @abstractmethod
    def visit_end_fork(self, end_fork: "EndFork") -> Any:
        pass

    @abstractmethod
    def visit_stop(self, stop: "Stop") -> Any:
        pass

    @abstractmethod
    def visit_branch(self, branch: "Branch") -> Any:
        pass

    @abstractmethod
    def visit_seq_iterate(self, seq_iterate: "SeqForEach") -> Any:
        pass

    @abstractmethod
    def visit_par_iterate(self, par_iterate: "ParForEach") -> Any:
        pass
