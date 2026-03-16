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
from unittest.mock import MagicMock

import pytest

from nv_dfm_core.api import BestOf, NodeId
from nv_dfm_core.gen.irgen._fed_info import (
    ComputeCostInfo,
    FedInfo,
    OperationInfo,
    SendCostInfo,
    SiteInfo,
)
from nv_dfm_core.gen.irgen.graph import (
    Graph,
)
from nv_dfm_core.gen.irgen.graph._graph_elements import (
    NodeState,
    Operation,
    Region,
)
from nv_dfm_core.gen.irgen.graph.passes._solve_pass import SolvePass


@pytest.fixture
def simple_fed_info() -> FedInfo:
    """Create a simple FedInfo with two sites for testing."""
    return FedInfo(
        sites={
            "SiteA": SiteInfo(
                interface={
                    "A": OperationInfo(
                        operation="A",
                        compute_cost=ComputeCostInfo(fixed_time=1.0, fixed_size=10),
                        is_async=True,
                    ),
                    "B": OperationInfo(
                        operation="B",
                        compute_cost=ComputeCostInfo(fixed_time=5.0, fixed_size=20),
                        is_async=True,
                    ),
                },
                providers={},
                send_cost={
                    "SiteB": SendCostInfo(fixed_time=0.1, bandwidth=1000),
                },
            ),
            "SiteB": SiteInfo(
                interface={
                    "A": OperationInfo(
                        operation="A",
                        compute_cost=ComputeCostInfo(fixed_time=10.0, fixed_size=10),
                        is_async=True,
                    ),
                    "B": OperationInfo(
                        operation="B",
                        compute_cost=ComputeCostInfo(fixed_time=2.0, fixed_size=20),
                        is_async=True,
                    ),
                },
                providers={},
                send_cost={
                    "SiteA": SendCostInfo(fixed_time=0.1, bandwidth=1000),
                },
            ),
        }
    )


@pytest.fixture
def op_a() -> Any:
    op_a = MagicMock(
        site=BestOf(),
        provider=None,
        __api_name__="A",
        dfm_node_id=NodeId(ident="nodeA"),
    )
    return op_a


@pytest.fixture
def op_b() -> Any:
    op_b = MagicMock(
        site=BestOf(),
        provider=None,
        __api_name__="B",
        dfm_node_id=NodeId(ident="nodeB"),
        get_noderef_and_placeparam_pydantic_fields=lambda: [("val", None)],
    )
    return op_b


@pytest.fixture
def simple_graph(simple_fed_info: FedInfo) -> Graph:
    """Create a simple graph for testing."""
    return Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=simple_fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )


def test_solve_pass_basic_selection(simple_graph: Graph, op_a: Any, op_b: Any) -> None:
    """Test that SolvePass correctly selects the cheapest nodes."""
    # Create a simple graph with two operations A and B
    region = simple_graph.create_log_start_region()

    # Create operation A on both sites
    # costs 1 on site A and 10 on site B
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    # Create operation B on both sites
    # costs 5 on site A and 2 on site B
    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Connect A to B
    simple_graph.create_cross_site_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
    )

    # Apply SolvePass
    solve_pass = SolvePass(simple_graph, simple_graph.logger)
    solve_pass.apply()

    # Check that the cheapest nodes are selected
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        node_b = simple_graph.get_node(site, op_b_logid, Operation)

        # SiteA should be selected for A (cheaper: 1.0 vs 10.0)
        if site == "SiteA":
            assert node_a.state == NodeState.SELECTED
            assert node_b.state == NodeState.DISCARDED
        else:  # SiteB
            assert node_a.state == NodeState.DISCARDED
            assert node_b.state == NodeState.SELECTED


def test_solve_pass_leaf_selection(simple_graph: Graph, op_a: Any, op_b: Any) -> None:
    """Test that SolvePass correctly identifies and selects leaf nodes."""
    # Create a graph with a chain of operations
    region = simple_graph.create_log_start_region()

    # Create operations A -> B -> C
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Connect A to B
    simple_graph.create_cross_site_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
    )

    # Apply SolvePass
    solve_pass = SolvePass(simple_graph, simple_graph.logger)
    solve_pass.apply()

    # Check that all nodes have been processed
    all_nodes = simple_graph.get_all_nodes_copy()
    for node in all_nodes:
        if hasattr(node, "state"):
            assert node.state != NodeState.CANDIDATE, (
                f"Node {node.global_id()} is still a candidate"
            )


def test_solve_pass_cost_propagation(simple_graph: Graph, op_a: Any, op_b: Any) -> None:
    """Test that costs are properly propagated through the graph."""
    # Create a simple chain A -> B
    region = simple_graph.create_log_start_region()

    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Connect A to B
    simple_graph.create_cross_site_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
    )

    # Apply SolvePass
    solve_pass = SolvePass(simple_graph, simple_graph.logger)
    solve_pass.apply()

    # Check that costs have been computed
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        node_b = simple_graph.get_node(site, op_b_logid, Operation)

        if node_a.state == NodeState.SELECTED:
            assert node_a.has_cost()
            assert node_b.has_cost()

            # B's cost should be higher than A's cost (due to propagation)
            assert node_b.cost().time > node_a.cost().time


def test_solve_pass_place_nodes_fixed(simple_graph: Graph) -> None:
    """Test that Place nodes are automatically selected."""
    # Create a graph with a place node
    region = simple_graph.create_log_start_region()

    # The start region already has a place node
    # Apply SolvePass
    solve_pass = SolvePass(simple_graph, simple_graph.logger)
    solve_pass.apply()

    # Check that place nodes are selected
    for site in ["SiteA", "SiteB"]:
        region_node = simple_graph.get_node(site, region, Region)
        places = region_node.places()
        for place in places:
            assert place.state == NodeState.SELECTED


def test_solve_pass_empty_graph(simple_graph: Graph) -> None:
    """Test that SolvePass handles empty graphs gracefully."""
    # Create a graph with just regions and exit nodes
    region = simple_graph.create_log_start_region()
    _ = simple_graph.create_log_stop_node(region_logid=region)

    # Apply SolvePass
    solve_pass = SolvePass(simple_graph, simple_graph.logger)
    solve_pass.apply()

    # Should complete without errors
    assert True
