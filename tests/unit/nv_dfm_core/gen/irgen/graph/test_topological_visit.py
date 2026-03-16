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

from typing import Any

from typing_extensions import override
from unittest.mock import MagicMock

from nv_dfm_core.api._node_id import NodeId
from nv_dfm_core.gen.irgen._fed_info import ComputeCostInfo, OperationInfo, SendCostInfo
from nv_dfm_core.gen.irgen.graph import (
    BoolValue,
    Branch,
    CannotReach,
    EndFork,
    FlowEdge,
    GraphVisitor,
    Jump,
    Loop,
    Operation,
    ParForEach,
    Place,
    Region,
    Send,
    SeqForEach,
    Stop,
    Yield,
)
from nv_dfm_core.gen.irgen.graph._graph_elements import Next


class TopologicalTestVisitor(GraphVisitor):
    """A test visitor that records the order of visits."""

    def __init__(self):
        self.visit_order: list[str] = []

    @override
    def visit_region(self, region: "Region") -> Any:
        self.visit_order.append(f"Region:{region.site_local_logid()}")
        return None

    @override
    def visit_place(self, place: "Place") -> Any:
        self.visit_order.append(f"Place:{place.site_local_logid()}")
        return None

    @override
    def visit_operation(self, operation: "Operation") -> Any:
        self.visit_order.append(f"Operation:{operation.site_local_logid()}")
        return None

    @override
    def visit_bool_value(self, bool_value: "BoolValue") -> Any:
        self.visit_order.append(f"BoolValue:{bool_value.site_local_logid()}")
        return None

    @override
    def visit_yield(self, yield_node: "Yield") -> Any:
        self.visit_order.append(f"Yield:{yield_node.site_local_logid()}")
        return None

    @override
    def visit_next(self, next: "Next") -> Any:
        self.visit_order.append(f"Next:{next.site_local_logid()}")
        return None

    @override
    def visit_send(self, send: "Send") -> Any:
        self.visit_order.append(f"Send:{send.site_local_logid()}")
        return None

    @override
    def visit_jump(self, jump: "Jump") -> Any:
        self.visit_order.append(f"Jump:{jump.site_local_logid()}")
        return None

    @override
    def visit_cannot_reach(self, cannot_reach: "CannotReach") -> Any:
        self.visit_order.append(f"CannotReach:{cannot_reach.site_local_logid()}")
        return None

    @override
    def visit_loop(self, loop: "Loop") -> Any:
        self.visit_order.append(f"Loop:{loop.site_local_logid()}")
        return None

    @override
    def visit_end_fork(self, end_fork: "EndFork") -> Any:
        self.visit_order.append(f"EndFork:{end_fork.site_local_logid()}")
        return None

    @override
    def visit_stop(self, stop: "Stop") -> Any:
        self.visit_order.append(f"Stop:{stop.site_local_logid()}")
        return None

    @override
    def visit_branch(self, branch: "Branch") -> Any:
        self.visit_order.append(f"Branch:{branch.site_local_logid()}")
        return None

    @override
    def visit_seq_iterate(self, seq_iterate: "SeqForEach") -> Any:
        self.visit_order.append(f"SeqIterate:{seq_iterate.site_local_logid()}")
        return None

    @override
    def visit_par_iterate(self, par_iterate: "ParForEach") -> Any:
        self.visit_order.append(f"ParIterate:{par_iterate.site_local_logid()}")
        return None


def test_topological_visit():
    """Test the topological visit functionality."""

    # Create a mock graph
    class MockGraph:
        def _register_node(self, _node: Any):
            pass

        def _deregister_node(self, _node: Any, _force: bool = False):
            pass

        def _register_edge(self, _edge: Any):
            pass

    mock_graph = MockGraph()

    # Create a region
    region = Region(mock_graph, "test_site", "test_region", is_loop_head=False)

    # create a different region
    region_outside = Region(
        mock_graph, "test_site", "test_region_outside", is_loop_head=False
    )

    # Create some places
    _place1 = Place(mock_graph, region, "place1", "data", "internal", "sticky", "Any")
    _place2 = Place(mock_graph, region, "place2", "data", "external", "sticky", "Any")

    # Create some operations
    op_a = MagicMock(
        site="SiteA",
        provider=None,
        __api_name__="A",
        dfm_node_id=NodeId(ident="nodeA"),
    )
    op_b = MagicMock(
        site="SiteA",
        provider=None,
        __api_name__="B",
        dfm_node_id=NodeId(ident="nodeB"),
        get_noderef_and_placeparam_pydantic_fields=lambda: [("val", None)],
    )
    # create in reverse order to make sure the ordering works
    send1 = Send(mock_graph, region, "send1", "data", "Any")
    op2 = Operation(
        mock_graph,
        region,
        op_b,
        OperationInfo(operation="B", compute_cost=ComputeCostInfo()),
    )
    op1 = Operation(
        mock_graph,
        region,
        op_a,
        OperationInfo(operation="A", compute_cost=ComputeCostInfo()),
    )

    _ = FlowEdge(
        src=op1.out_slots()["__data__"],
        dst=op2.in_slots()["val"],
        send_cost_info=SendCostInfo(fixed_time=0.1, bandwidth=1000),
        is_weak=False,
    )
    _ = FlowEdge(
        src=op2.out_slots()["__data__"],
        dst=send1.in_slots()["value"],
        send_cost_info=SendCostInfo(fixed_time=0.1, bandwidth=1000),
        is_weak=False,
    )

    # add a node outside the region, we don't want to visit that
    op3 = Operation(
        mock_graph,
        region_outside,
        op_b,
        OperationInfo(operation="B", compute_cost=ComputeCostInfo()),
    )
    _ = FlowEdge(
        src=op1.out_slots()["__data__"],
        dst=op3.in_slots()["val"],
        send_cost_info=SendCostInfo(fixed_time=0.1, bandwidth=1000),
        is_weak=False,
    )

    # Create an exit node
    from nv_dfm_core.gen.irgen.graph._graph_elements import Jump

    _exit_node = Jump(mock_graph, region)

    # Create visitor and test
    visitor = TopologicalTestVisitor()

    # Test the topological visit
    region.visit_in_topological_order(visitor)

    # Print the visit order
    DEBUG = False
    if DEBUG:
        print("Visit order:")
        for i, visit in enumerate(visitor.visit_order):
            print(f"  {i + 1}. {visit}")

    # Verify the order follows the expected pattern:
    # 1. Region should be first
    # 2. Places should come next
    # 3. Operations should come after places
    # 4. Exit node should be last

    assert visitor.visit_order[0].startswith("Region:"), (
        "Region should be visited first"
    )
    assert any(visit.startswith("Place:") for visit in visitor.visit_order[1:3]), (
        "Places should be visited after region"
    )
    assert visitor.visit_order[3].startswith("Operation:nodeA"), (
        f"Operation A should be visited after places. Was: {visitor.visit_order[3]}"
    )
    assert visitor.visit_order[4].startswith("Operation:nodeB"), (
        f"Operation B should be visited after places. Was: {visitor.visit_order[4]}"
    )
    assert visitor.visit_order[5].startswith("Send:send1"), (
        f"Send should be visited after places. Was: {visitor.visit_order[5]}"
    )
    assert visitor.visit_order[-1].startswith("Jump:"), (
        "Exit node should be visited last"
    )
