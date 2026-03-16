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
    Jump,
    Next,
    NodeState,
    Operation,
    ParForEach,
    SeqForEach,
    Stop,
)
from nv_dfm_core.gen.irgen.graph.passes._sync_exit_nodes_pass import SyncExitNodesPass


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


def test_sync_exit_nodes_pass_branch_synchronization(
    simple_graph: Graph, op_a: Any
) -> None:
    """Test that SyncExitNodesPass creates sync edges for branch groups."""
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

    # Set all nodes to selected and mark SiteA as leader
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        branch_node = simple_graph.get_node(site, branch, Branch)

        node_a.state = NodeState.SELECTED
        if site == "SiteA":
            branch_node.state = NodeState.LEADER
        else:
            branch_node.state = NodeState.SELECTED

    # Apply SyncExitNodesPass
    sync_pass = SyncExitNodesPass(simple_graph, simple_graph.logger)
    sync_pass.apply()

    # Check that new regions were created for followers
    regions = simple_graph.get_regions_copy()
    assert len(regions) > 2  # Should have created new regions for followers

    # Check that sync edges were created
    edges = simple_graph.get_all_edges()
    sync_edges = [
        edge
        for edge in edges
        if edge.src().name() == "__sync__" and edge.dst().name() == "__condition__"
    ]
    assert len(sync_edges) > 0


def test_sync_exit_nodes_pass_stop_synchronization(
    simple_graph: Graph, op_a: Any
) -> None:
    """Test that SyncExitNodesPass creates sync edges for stop groups."""
    # Create a graph with stops
    region = simple_graph.create_log_start_region()

    # Create operations
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    # Create stops
    stop = simple_graph.create_log_stop_node(region_logid=region)

    # Set all nodes to selected and mark SiteA as leader
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        stop_node = simple_graph.get_node(site, stop, Stop)

        node_a.state = NodeState.SELECTED
        if site == "SiteA":
            stop_node.state = NodeState.LEADER
        else:
            stop_node.state = NodeState.SELECTED

    # Apply SyncExitNodesPass
    sync_pass = SyncExitNodesPass(simple_graph, simple_graph.logger)
    sync_pass.apply()

    # Check that new regions were created for followers
    regions = simple_graph.get_regions_copy()
    assert len(regions) > 2  # Should have created new regions for followers

    # Check that sync edges were created
    edges = simple_graph.get_all_edges()
    sync_edges = [
        edge
        for edge in edges
        if edge.src().name() == "__sync__" and "stop_from_" in edge.dst().name()
    ]
    assert len(sync_edges) > 0


def test_sync_exit_nodes_pass_jump_no_sync(simple_graph: Graph) -> None:
    """Test that SyncExitNodesPass doesn't create sync for Jump nodes."""
    # Create a graph with jumps
    region = simple_graph.create_log_start_region()

    # Create jumps
    jump = simple_graph.create_log_jump_node(region_logid=region)

    # Set all nodes to selected
    for site in ["SiteA", "SiteB"]:
        jump_node = simple_graph.get_node(site, jump, Jump)
        jump_node.state = NodeState.SELECTED

    # Count edges before sync
    edges_before = len(simple_graph.get_all_edges())

    # Apply SyncExitNodesPass
    sync_pass = SyncExitNodesPass(simple_graph, simple_graph.logger)
    sync_pass.apply()

    # Count edges after sync
    edges_after = len(simple_graph.get_all_edges())

    # Should not have created any new edges for jumps
    assert edges_after == edges_before


def test_sync_exit_nodes_pass_multiple_branch_groups(
    simple_graph: Graph, op_a: Any, op_b: Any
) -> None:
    """Test that SyncExitNodesPass handles multiple branch groups correctly."""
    # Create a graph with multiple branch groups
    region = simple_graph.create_log_start_region()

    # Create operations to provide conditions
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Create branches in different regions
    branch1 = simple_graph.create_log_branch_node(region_logid=region)
    region2 = simple_graph.create_log_region_node(is_loop_head=False)
    branch2 = simple_graph.create_log_branch_node(region_logid=region2)

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

    # Set all nodes to selected and mark SiteA as leader
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        node_b = simple_graph.get_node(site, op_b_logid, Operation)
        branch1_node = simple_graph.get_node(site, branch1, Branch)
        branch2_node = simple_graph.get_node(site, branch2, Branch)

        node_a.state = NodeState.SELECTED
        node_b.state = NodeState.SELECTED
        if site == "SiteA":
            branch1_node.state = NodeState.LEADER
            branch2_node.state = NodeState.LEADER
        else:
            branch1_node.state = NodeState.SELECTED
            branch2_node.state = NodeState.SELECTED

    # Apply SyncExitNodesPass
    sync_pass = SyncExitNodesPass(simple_graph, simple_graph.logger)
    sync_pass.apply()

    # Check that sync edges were created for both branch groups
    edges = simple_graph.get_all_edges()
    sync_edges = [
        edge
        for edge in edges
        if edge.src().name() == "__sync__" and edge.dst().name() == "__condition__"
    ]
    assert len(sync_edges) >= 2  # Should have sync edges for both branch groups


def test_sync_exit_nodes_pass_multiple_seq_iterate_groups(
    simple_graph: Graph, op_a: Any, op_b: Any
) -> None:
    """Test that SyncExitNodesPass handles multiple branch groups correctly."""
    # Create a graph with multiple branch groups
    region = simple_graph.create_log_start_region()

    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    # Create operations to provide conditions
    next_node_a = simple_graph.create_log_next_node_where_needed(
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
        dst_logid=next_node_a,
        dst_in_slot="__iterator__",
    )
    seq_iterate1 = simple_graph.create_log_seq_foreach_node(
        region_logid=region, foreach_node_id=NodeId(ident="seq_for_each_node0")
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=next_node_a,
        src_out_slot="__has_next__",
        dst_logid=seq_iterate1,
        dst_in_slot="__has_next__",
    )

    # create a second region
    region2 = simple_graph.create_log_region_node(is_loop_head=False)

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region2, operation=op_b
    )

    next_node_b = simple_graph.create_log_next_node_where_needed(
        region_logid=region2,
        foreach_node_id=NodeId(ident="seq_for_each_node1"),
        iterator_logid=op_b_logid,
        in_type="Any",
        out_type="Any",
        is_async=False,
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_b_logid,
        src_out_slot="__data__",
        dst_logid=next_node_b,
        dst_in_slot="__iterator__",
    )
    seq_iterate2 = simple_graph.create_log_seq_foreach_node(
        region_logid=region2, foreach_node_id=NodeId(ident="seq_for_each_node1")
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=next_node_b,
        src_out_slot="__has_next__",
        dst_logid=seq_iterate2,
        dst_in_slot="__has_next__",
    )

    # Set all nodes to selected and mark SiteA as leader
    for site in ["SiteA", "SiteB"]:
        next_a = simple_graph.get_node(site, next_node_a, Next)
        next_b = simple_graph.get_node(site, next_node_b, Next)
        seq_iterate1_node = simple_graph.get_node(site, seq_iterate1, SeqForEach)
        seq_iterate2_node = simple_graph.get_node(site, seq_iterate2, SeqForEach)
        assert isinstance(seq_iterate1_node, SeqForEach)
        assert isinstance(seq_iterate2_node, SeqForEach)

        if site == "SiteA":
            next_a.state = NodeState.SELECTED
            next_b.state = NodeState.SELECTED
            seq_iterate1_node.state = NodeState.LEADER
            seq_iterate2_node.state = NodeState.LEADER
        else:
            next_a.remove()
            next_b.remove()
            seq_iterate1_node.state = NodeState.SELECTED
            seq_iterate2_node.state = NodeState.SELECTED

    # Apply SyncExitNodesPass
    sync_pass = SyncExitNodesPass(simple_graph, simple_graph.logger)
    sync_pass.apply()

    # Check that sync edges were created for both branch groups
    edges = simple_graph.get_all_edges()
    sync_edges = [
        edge
        for edge in edges
        if edge.src().name() == "__sync__" and edge.dst().name() == "__has_next__"
    ]
    assert len(sync_edges) >= 2  # Should have sync edges for both branch groups


def test_sync_exit_nodes_pass_multiple_par_iterate_groups(
    simple_graph: Graph, op_a: Any, op_b: Any
) -> None:
    """Test that SyncExitNodesPass handles multiple branch groups correctly."""
    # Create a graph with multiple branch groups
    region = simple_graph.create_log_start_region()
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )
    next_node_a = simple_graph.create_log_next_node_where_needed(
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
        dst_logid=next_node_a,
        dst_in_slot="__iterator__",
    )
    par_iterate1 = simple_graph.create_log_par_foreach_node(
        region_logid=region, foreach_node_id=NodeId(ident="par_for_each_node0")
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=next_node_a,
        src_out_slot="__has_next__",
        dst_logid=par_iterate1,
        dst_in_slot="__has_next__",
    )

    region2 = simple_graph.create_log_region_node(is_loop_head=False)
    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region2, operation=op_b
    )
    next_node_b = simple_graph.create_log_next_node_where_needed(
        region_logid=region2,
        foreach_node_id=NodeId(ident="seq_for_each_node1"),
        iterator_logid=op_b_logid,
        in_type="Any",
        out_type="Any",
        is_async=False,
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_b_logid,
        src_out_slot="__data__",
        dst_logid=next_node_b,
        dst_in_slot="__iterator__",
    )
    par_iterate2 = simple_graph.create_log_par_foreach_node(
        region_logid=region2, foreach_node_id=NodeId(ident="par_for_each_node1")
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=next_node_b,
        src_out_slot="__has_next__",
        dst_logid=par_iterate2,
        dst_in_slot="__has_next__",
    )

    # Set all nodes to selected and mark SiteA as leader
    for site in ["SiteA", "SiteB"]:
        next_a = simple_graph.get_node(site, next_node_a, Next)
        next_b = simple_graph.get_node(site, next_node_b, Next)
        par_iterate1_node = simple_graph.get_node(site, par_iterate1, ParForEach)
        par_iterate2_node = simple_graph.get_node(site, par_iterate2, ParForEach)
        assert isinstance(par_iterate1_node, ParForEach)
        assert isinstance(par_iterate2_node, ParForEach)

        if site == "SiteA":
            next_a.state = NodeState.SELECTED
            next_b.state = NodeState.SELECTED
            par_iterate1_node.state = NodeState.LEADER
            par_iterate2_node.state = NodeState.LEADER
        else:
            next_a.remove()
            next_b.remove()
            par_iterate1_node.state = NodeState.SELECTED
            par_iterate2_node.state = NodeState.SELECTED

    # Apply SyncExitNodesPass
    sync_pass = SyncExitNodesPass(simple_graph, simple_graph.logger)
    sync_pass.apply()

    # Check that sync edges were created for both branch groups
    edges = simple_graph.get_all_edges()
    sync_edges = [
        edge
        for edge in edges
        if edge.src().name() == "__sync__" and edge.dst().name() == "__has_next__"
    ]
    assert len(sync_edges) >= 2  # Should have sync edges for both branch groups


def test_sync_exit_nodes_pass_empty_groups(simple_graph: Graph, op_a: Any) -> None:
    """Test that SyncExitNodesPass handles empty node groups gracefully."""
    # Create a graph with no exit nodes
    region = simple_graph.create_log_start_region()

    # Create operations
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    # Set all nodes to selected
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        node_a.state = NodeState.SELECTED

    # Apply SyncExitNodesPass - should complete without errors
    sync_pass = SyncExitNodesPass(simple_graph, simple_graph.logger)
    sync_pass.apply()

    # Should complete without errors
    assert True


def test_sync_exit_nodes_pass_mixed_exit_types(simple_graph: Graph, op_a: Any) -> None:
    """Test that SyncExitNodesPass handles mixed exit node types correctly."""
    # Create a graph with different types of exit nodes
    region = simple_graph.create_log_start_region()

    # Create operations
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    # Create different exit nodes in different regions
    branch = simple_graph.create_log_branch_node(region_logid=region)
    region2 = simple_graph.create_log_region_node(is_loop_head=False)
    jump = simple_graph.create_log_jump_node(region_logid=region2)
    region3 = simple_graph.create_log_region_node(is_loop_head=False)
    stop = simple_graph.create_log_stop_node(region_logid=region3)

    # Connect condition to branch
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=branch,
        dst_in_slot="__condition__",
    )

    # Set all nodes to selected and mark SiteA as leader
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        branch_node = simple_graph.get_node(site, branch, Branch)
        jump_node = simple_graph.get_node(site, jump, Jump)
        stop_node = simple_graph.get_node(site, stop, Stop)

        node_a.state = NodeState.SELECTED
        branch_node.state = NodeState.SELECTED
        jump_node.state = NodeState.SELECTED
        stop_node.state = NodeState.SELECTED

        if site == "SiteA":
            branch_node.state = NodeState.LEADER
            stop_node.state = NodeState.LEADER

    # Apply SyncExitNodesPass
    sync_pass = SyncExitNodesPass(simple_graph, simple_graph.logger)
    sync_pass.apply()

    # Check that sync edges were created for branch and stop, but not jump
    edges = simple_graph.get_all_edges()
    branch_sync_edges = [
        edge
        for edge in edges
        if edge.src().name() == "__sync__" and edge.dst().name() == "__condition__"
    ]
    stop_sync_edges = [
        edge
        for edge in edges
        if edge.src().name() == "__sync__" and "stop_from_" in edge.dst().name()
    ]

    assert len(branch_sync_edges) > 0
    assert len(stop_sync_edges) > 0


def test_sync_exit_nodes_pass_creates_new_regions(
    simple_graph: Graph, op_a: Any
) -> None:
    """Test that SyncExitNodesPass creates new regions for followers."""
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

    # Set all nodes to selected and mark SiteA as leader
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        branch_node = simple_graph.get_node(site, branch, Branch)

        node_a.state = NodeState.SELECTED
        if site == "SiteA":
            branch_node.state = NodeState.LEADER
        else:
            branch_node.state = NodeState.SELECTED

    # Count regions before sync
    regions_before = len(simple_graph.get_regions_copy())

    # Apply SyncExitNodesPass
    sync_pass = SyncExitNodesPass(simple_graph, simple_graph.logger)
    sync_pass.apply()

    # Count regions after sync
    regions_after = len(simple_graph.get_regions_copy())

    # Should have created new regions for followers
    assert regions_after > regions_before
