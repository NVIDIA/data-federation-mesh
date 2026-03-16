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
    ExitNode,
    Graph,
    Place,
    Region,
)
from nv_dfm_core.gen.irgen.graph._graph_elements import NodeState, Operation, Stop
from nv_dfm_core.gen.irgen.graph.passes._prune_pass import PrunePass


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


def test_prune_pass_removes_discarded_nodes(
    simple_graph: Graph, op_a: Any, op_b: Any
) -> None:
    """Test that PrunePass removes all discarded nodes."""
    # Create a graph with operations
    region = simple_graph.create_log_start_region()

    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Manually set some nodes to discarded
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        node_b = simple_graph.get_node(site, op_b_logid, Operation)

        node_a.state = NodeState.DISCARDED
        node_b.state = NodeState.SELECTED

    # Count nodes before pruning
    nodes_before = len(simple_graph.get_all_nodes_copy())

    # simulate the SOLVE pass that will have selected all Region, ExitNode, and Place nodes
    for node in simple_graph.get_all_nodes_copy():
        if isinstance(node, (Region, ExitNode, Place)):
            node.state = NodeState.SELECTED
    prune_pass = PrunePass(simple_graph, simple_graph.logger)
    prune_pass.apply()

    # Apply PrunePass
    # Count nodes after pruning
    nodes_after = len(simple_graph.get_all_nodes_copy())

    # Should have removed 2 discarded nodes (one from each site)
    assert nodes_after == nodes_before - 2

    # Check that discarded nodes are gone
    for site in ["SiteA", "SiteB"]:
        # This should raise an exception since the node should be removed
        with pytest.raises(KeyError):
            _ = simple_graph.get_node(site, "A", Operation)


def test_prune_pass_keeps_selected_nodes(
    simple_graph: Graph, op_a: Any, op_b: Any
) -> None:
    """Test that PrunePass keeps selected nodes."""
    # Create a graph with operations
    region = simple_graph.create_log_start_region()

    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Manually set some nodes to selected
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        node_b = simple_graph.get_node(site, op_b_logid, Operation)

        node_a.state = NodeState.SELECTED
        node_b.state = NodeState.DISCARDED

    # Set all Region, ExitNode, and Place nodes to SELECTED (required by PrunePass)
    for node in simple_graph.get_all_nodes_copy():
        if isinstance(node, (Region, ExitNode, Place)):
            node.state = NodeState.SELECTED

    # Apply PrunePass
    prune_pass = PrunePass(simple_graph, simple_graph.logger)
    prune_pass.apply()

    # Check that selected nodes are still there
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        assert node_a.state == NodeState.SELECTED

    # Check that discarded nodes are gone
    for site in ["SiteA", "SiteB"]:
        with pytest.raises(KeyError):
            _ = simple_graph.get_node(site, op_b_logid, Operation)


def test_prune_pass_keeps_region_and_exit_nodes(simple_graph: Graph) -> None:
    """Test that PrunePass keeps region and exit nodes."""
    # Create a graph with regions and exit nodes
    region = simple_graph.create_log_start_region()
    stop = simple_graph.create_log_stop_node(region_logid=region)

    # Set all Region, ExitNode, and Place nodes to SELECTED (required by PrunePass)
    for node in simple_graph.get_all_nodes_copy():
        if isinstance(node, (Region, ExitNode, Place)):
            node.state = NodeState.SELECTED

    # Apply PrunePass
    prune_pass = PrunePass(simple_graph, simple_graph.logger)
    prune_pass.apply()

    # Check that regions and exit nodes are still there
    regions = simple_graph.get_regions_copy()
    assert len(regions) > 0

    for site in ["SiteA", "SiteB"]:
        stop_node = simple_graph.get_node(site, stop, Stop)
        assert stop_node is not None


def test_prune_pass_empty_graph(simple_graph: Graph) -> None:
    """Test that PrunePass handles empty graphs gracefully."""
    # Create a graph with just regions and exit nodes
    region = simple_graph.create_log_start_region()
    _ = simple_graph.create_log_stop_node(region_logid=region)

    # Set all Region, ExitNode, and Place nodes to SELECTED (required by PrunePass)
    for node in simple_graph.get_all_nodes_copy():
        if isinstance(node, (Region, ExitNode, Place)):
            node.state = NodeState.SELECTED

    # Apply PrunePass
    prune_pass = PrunePass(simple_graph, simple_graph.logger)
    prune_pass.apply()

    # Should complete without errors
    assert True
