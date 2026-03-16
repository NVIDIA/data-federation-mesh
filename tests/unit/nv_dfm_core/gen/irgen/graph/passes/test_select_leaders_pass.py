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
    Branch,
    Next,
    NodeState,
    Operation,
    ParForEach,
    SeqForEach,
    Stop,
)
from nv_dfm_core.gen.irgen.graph.passes._select_leaders_pass import SelectLeadersPass


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


def test_branch_with_condition(simple_graph: Graph, op_a: Any) -> None:
    """Test that SelectLeadersPass selects the branch with incoming condition as leader."""
    # Create a graph with branches
    region = simple_graph.create_log_start_region()

    # Create operations to provide condition
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    # Create branches
    branch = simple_graph.create_log_branch_node(region_logid=region)

    # Connect condition to SiteA branch only
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=branch,
        dst_in_slot="__condition__",
    )

    # Simulate Solve and Prune passes: remove all but one of the operation nodes
    op_a_variant2 = simple_graph.get_node("SiteB", op_a_logid, Operation)
    op_a_variant2.remove()

    # Set all nodes to selected
    for node in simple_graph.get_all_nodes_copy():
        node.state = NodeState.SELECTED

    # Apply SelectLeadersPass
    select_leaders_pass = SelectLeadersPass(simple_graph, simple_graph.logger)
    select_leaders_pass.apply()

    # Check that SiteA branch is selected as leader (has condition input)
    for site in ["SiteA", "SiteB"]:
        branch_node = simple_graph.get_node(site, branch, Branch)
        if site == "SiteA":
            assert branch_node.state == NodeState.LEADER
        else:
            assert branch_node.state == NodeState.SELECTED  # Not leader


def test_seq_iterate_with_condition(simple_graph: Graph, op_a: Any) -> None:
    """Test that SelectLeadersPass selects the branch with incoming condition as leader."""
    # Create a graph with branches
    region = simple_graph.create_log_start_region()

    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    next_node = simple_graph.create_log_next_node_where_needed(
        region_logid=region,
        foreach_node_id=NodeId(ident="seq_for_each_node0"),
        iterator_logid=op_a_logid,
        in_type="Any",
        out_type="Any",
        is_async=False,
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=next_node,
        dst_in_slot="__iterator__",
    )

    # Create branches
    seq_iterate = simple_graph.create_log_seq_foreach_node(
        region_logid=region,
        foreach_node_id=NodeId(ident="seq_for_each_node0"),
    )

    # Connect condition to SiteA branch only
    simple_graph.create_site_internal_flow_edges(
        src_logid=next_node,
        src_out_slot="__has_next__",
        dst_logid=seq_iterate,
        dst_in_slot="__has_next__",
    )

    # Simulate Solve and Prune passes: remove all but one of the operation nodes
    next_node_variant2 = simple_graph.get_node("SiteB", next_node, Next)
    next_node_variant2.remove()

    # Set all nodes to selected
    for node in simple_graph.get_all_nodes_copy():
        node.state = NodeState.SELECTED

    # Apply SelectLeadersPass
    select_leaders_pass = SelectLeadersPass(simple_graph, simple_graph.logger)
    select_leaders_pass.apply()

    # Check that SiteA branch is selected as leader (has condition input)
    for site in ["SiteA", "SiteB"]:
        seq_iterate_node = simple_graph.get_node(site, seq_iterate, SeqForEach)
        if site == "SiteA":
            assert seq_iterate_node.state == NodeState.LEADER
        else:
            assert seq_iterate_node.state == NodeState.SELECTED  # Not leader


def test_par_iterate_with_condition(simple_graph: Graph, op_a: Any) -> None:
    """Test that SelectLeadersPass selects the branch with incoming condition as leader."""
    # Create a graph with branches
    region = simple_graph.create_log_start_region()

    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )
    # Create operations to provide condition
    next_node = simple_graph.create_log_next_node_where_needed(
        region_logid=region,
        foreach_node_id=NodeId(ident="seq_for_each_node0"),
        iterator_logid=op_a_logid,
        in_type="Any",
        out_type="Any",
        is_async=False,
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=next_node,
        dst_in_slot="__iterator__",
    )

    # Create branches
    par_iterate = simple_graph.create_log_par_foreach_node(
        region_logid=region,
        foreach_node_id=NodeId(ident="par_for_each_node0"),
    )

    # Connect condition to SiteA branch only
    simple_graph.create_site_internal_flow_edges(
        src_logid=next_node,
        src_out_slot="__has_next__",
        dst_logid=par_iterate,
        dst_in_slot="__has_next__",
    )

    # Simulate Solve and Prune passes: remove all but one of the operation nodes
    next_node_variant2 = simple_graph.get_node("SiteB", next_node, Next)
    next_node_variant2.remove()

    # Set all nodes to selected
    for node in simple_graph.get_all_nodes_copy():
        node.state = NodeState.SELECTED

    # Apply SelectLeadersPass
    select_leaders_pass = SelectLeadersPass(simple_graph, simple_graph.logger)
    select_leaders_pass.apply()

    # Check that SiteA branch is selected as leader (has condition input)
    for site in ["SiteA", "SiteB"]:
        par_iterate_node = simple_graph.get_node(site, par_iterate, ParForEach)
        if site == "SiteA":
            assert par_iterate_node.state == NodeState.LEADER
        else:
            assert par_iterate_node.state == NodeState.SELECTED  # Not leader


def test_branch_no_condition(simple_graph: Graph) -> None:
    """Test that SelectLeadersPass handles branches without condition inputs."""
    # Create a graph with branches but no condition input
    region = simple_graph.create_log_start_region()

    # Create branches
    branch = simple_graph.create_log_branch_node(region_logid=region)

    # Set all nodes to selected
    for site in ["SiteA", "SiteB"]:
        branch_node = simple_graph.get_node(site, branch, Branch)
        branch_node.state = NodeState.SELECTED

    # Apply SelectLeadersPass - should raise assertion error since no leader found
    select_leaders_pass = SelectLeadersPass(simple_graph, simple_graph.logger)
    with pytest.raises(AssertionError, match="No leader found"):
        select_leaders_pass.apply()


def test_stop_selection(simple_graph: Graph, op_a: Any, op_b: Any) -> None:
    """Test that SelectLeadersPass selects the stop node with most nodes in region."""
    # Create a graph with stops
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations in region1
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_b
    )

    # Create operations in region2 (empty)

    # Create stops
    stop1 = simple_graph.create_log_stop_node(region_logid=region1)
    stop2 = simple_graph.create_log_stop_node(region_logid=region2)

    # Set all nodes to selected
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        node_b = simple_graph.get_node(site, op_b_logid, Operation)
        stop1_node = simple_graph.get_node(site, stop1, Stop)
        stop2_node = simple_graph.get_node(site, stop2, Stop)

        node_a.state = NodeState.SELECTED
        node_b.state = NodeState.SELECTED
        stop1_node.state = NodeState.SELECTED
        stop2_node.state = NodeState.SELECTED

    # Apply SelectLeadersPass
    select_leaders_pass = SelectLeadersPass(simple_graph, simple_graph.logger)
    select_leaders_pass.apply()

    # Check that each stop group gets one leader
    # SiteA has operations in region1, so SiteA stop1 should be leader
    # SiteA also has more nodes overall, so SiteA stop2 should be leader too
    for site in ["SiteA", "SiteB"]:
        stop1_node = simple_graph.get_node(site, stop1, Stop)
        stop2_node = simple_graph.get_node(site, stop2, Stop)

        if site == "SiteA":
            assert stop1_node.state == NodeState.LEADER
            assert stop2_node.state == NodeState.LEADER
        else:  # SiteB
            assert stop1_node.state == NodeState.SELECTED
            assert stop2_node.state == NodeState.SELECTED


def test_stop_empty_regions(simple_graph: Graph, op_a: Any) -> None:
    """Test that SelectLeadersPass selects stop from site with most nodes when regions are empty."""
    # Create a graph with empty regions
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations only on SiteA
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    # Create stops
    stop1 = simple_graph.create_log_stop_node(region_logid=region1)
    stop2 = simple_graph.create_log_stop_node(region_logid=region2)

    # Set SiteA nodes to selected, SiteB nodes to discarded
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        stop1_node = simple_graph.get_node(site, stop1, Stop)
        stop2_node = simple_graph.get_node(site, stop2, Stop)

        if site == "SiteA":
            node_a.state = NodeState.SELECTED
            stop1_node.state = NodeState.SELECTED
            stop2_node.state = NodeState.SELECTED
        else:
            node_a.state = NodeState.DISCARDED
            stop1_node.state = NodeState.SELECTED
            stop2_node.state = NodeState.SELECTED

    # Apply SelectLeadersPass
    select_leaders_pass = SelectLeadersPass(simple_graph, simple_graph.logger)
    select_leaders_pass.apply()

    # Check that each stop group gets one leader
    # SiteA has more operation nodes, so SiteA stops should be leaders
    for site in ["SiteA", "SiteB"]:
        stop1_node = simple_graph.get_node(site, stop1, Stop)
        stop2_node = simple_graph.get_node(site, stop2, Stop)

        if site == "SiteA":
            assert stop1_node.state == NodeState.LEADER
            assert stop2_node.state == NodeState.LEADER
        else:
            assert stop1_node.state == NodeState.SELECTED
            assert stop2_node.state == NodeState.SELECTED


def test_multiple_branch_groups(simple_graph: Graph, op_a: Any, op_b: Any) -> None:
    """Test that SelectLeadersPass handles multiple branch groups correctly."""
    # Create a graph with multiple branch groups
    region = simple_graph.create_log_start_region()

    # Create operations to provide conditions
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    # Create branches in different regions
    branch1 = simple_graph.create_log_branch_node(region_logid=region)
    region2 = simple_graph.create_log_region_node(is_loop_head=False)
    branch2 = simple_graph.create_log_branch_node(region_logid=region2)

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region2, operation=op_b
    )

    # Connect condition to SiteA branches only
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=branch1,
        dst_in_slot="__condition__",
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_b_logid,
        src_out_slot="__data__",
        dst_logid=branch2,
        dst_in_slot="__condition__",
    )

    # Simulate Solve and Prune passes: remove all but one of the operation nodes
    op_a_variant2 = simple_graph.get_node("SiteB", op_a_logid, Operation)
    op_a_variant2.remove()
    op_b_variant1 = simple_graph.get_node("SiteA", op_b_logid, Operation)
    op_b_variant1.remove()

    # Set all nodes to selected
    for node in simple_graph.get_all_nodes_copy():
        node.state = NodeState.SELECTED

    # Apply SelectLeadersPass
    select_leaders_pass = SelectLeadersPass(simple_graph, simple_graph.logger)
    select_leaders_pass.apply()

    # Check that SiteA branches are selected as leaders
    for site in ["SiteA", "SiteB"]:
        branch1_node = simple_graph.get_node(site, branch1, Branch)
        branch2_node = simple_graph.get_node(site, branch2, Branch)

        if site == "SiteA":
            assert branch1_node.state == NodeState.LEADER
            assert branch2_node.state == NodeState.SELECTED
        else:
            assert branch1_node.state == NodeState.SELECTED
            assert branch2_node.state == NodeState.LEADER
