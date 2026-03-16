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

from unittest.mock import MagicMock

import pytest

from nv_dfm_core.api import Operation
from nv_dfm_core.api._best_of import BestOf
from nv_dfm_core.api._node_id import NodeId
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
from nv_dfm_core.gen.irgen.graph._graph_elements import Place, Send
from nv_dfm_core.gen.irgen.graph.passes._cut_cross_region_edges_pass import (
    CutCrossRegionEdgesPass,
)


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
                    "C": OperationInfo(
                        operation="C",
                        compute_cost=ComputeCostInfo(fixed_time=3.0, fixed_size=15),
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
                    "C": OperationInfo(
                        operation="C",
                        compute_cost=ComputeCostInfo(fixed_time=4.0, fixed_size=15),
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
def simple_graph(simple_fed_info: FedInfo) -> Graph:
    """Create a simple graph for testing."""
    return Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=simple_fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )


@pytest.fixture
def op_a() -> Operation:
    op_a = MagicMock(
        site=BestOf(),
        provider=None,
        __api_name__="A",
        dfm_node_id=NodeId(ident="nodeA"),
    )
    return op_a


@pytest.fixture
def op_b() -> Operation:
    op_b = MagicMock(
        site=BestOf(),
        provider=None,
        __api_name__="B",
        dfm_node_id=NodeId(ident="nodeB"),
        get_noderef_and_placeparam_pydantic_fields=lambda: [("val", None)],
    )
    return op_b


def test_cut_cross_region_edges_pass_cross_region_edges(
    simple_graph: Graph, op_a: Operation, op_b: Operation
) -> None:
    """Test that CutCrossRegionEdgesPass cuts edges between different regions."""
    # Create a graph with cross-region edges
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations in different regions
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )
    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region2, operation=op_b
    )

    # Connect operations across regions
    simple_graph.create_cross_site_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
    )

    # Count edges before cutting
    edges_before = len(simple_graph.get_all_edges())

    # Apply CutCrossRegionEdgesPass
    cut_pass = CutCrossRegionEdgesPass(simple_graph, simple_graph.logger)
    cut_pass.apply()

    # Count edges after cutting
    edges_after = len(simple_graph.get_all_edges())

    # Should have created new edges (send/place pairs)
    assert edges_after > edges_before

    # Check that Send and Place nodes were created
    send_nodes = [
        node for node in simple_graph.get_all_nodes_copy() if isinstance(node, Send)
    ]
    place_nodes = [
        node for node in simple_graph.get_all_nodes_copy() if isinstance(node, Place)
    ]

    assert len(send_nodes) > 0
    assert len(place_nodes) > 0


def test_cut_cross_region_edges_pass_same_region_edges(
    simple_graph: Graph, op_a: Operation, op_b: Operation
) -> None:
    """Test that CutCrossRegionEdgesPass doesn't cut edges within same region."""
    # Create a graph with same-region edges
    region = simple_graph.create_log_start_region()

    # Create operations in same region
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Connect operations within same region
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
    )

    # Count edges before cutting
    edges_before = len(simple_graph.get_all_edges())

    # Apply CutCrossRegionEdgesPass
    cut_pass = CutCrossRegionEdgesPass(simple_graph, simple_graph.logger)
    cut_pass.apply()

    # Count edges after cutting
    edges_after = len(simple_graph.get_all_edges())

    # Should not have created new edges
    assert edges_after == edges_before


def test_cut_cross_region_edges_pass_weak_edges(
    simple_graph: Graph, op_a: Operation, op_b: Operation
) -> None:
    """Test that CutCrossRegionEdgesPass doesn't cut weak edges."""
    # Create a graph with weak cross-region edges
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations in different regions
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region2, operation=op_b
    )

    # Connect operations with weak cross-region edges
    simple_graph.create_cross_site_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
        is_weak=True,
    )

    # Count edges before cutting
    edges_before = len(simple_graph.get_all_edges())

    # Apply CutCrossRegionEdgesPass
    cut_pass = CutCrossRegionEdgesPass(simple_graph, simple_graph.logger)
    cut_pass.apply()

    # Count edges after cutting
    edges_after = len(simple_graph.get_all_edges())

    # Should not have created new edges for weak edges
    assert edges_after == edges_before


def test_cut_cross_region_edges_pass_handles_data_and_control_edges(
    simple_graph: Graph, op_a: Operation, op_b: Operation
) -> None:
    """Test that CutCrossRegionEdgesPass handles both data and control edges."""
    # Create a graph with both data and control cross-region edges
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations in different regions
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region2, operation=op_b
    )

    # Connect operations with data edge
    simple_graph.create_cross_site_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
    )

    # Connect operations with control edge
    simple_graph.create_cross_site_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__flow__",
        dst_logid=op_b_logid,
        dst_in_slot="__flow__",
    )

    # Count edges before cutting
    edges_before = len(simple_graph.get_all_edges())

    # Apply CutCrossRegionEdgesPass
    cut_pass = CutCrossRegionEdgesPass(simple_graph, simple_graph.logger)
    cut_pass.apply()

    # Count edges after cutting
    edges_after = len(simple_graph.get_all_edges())

    # Should have created new edges for both data and control edges
    assert edges_after > edges_before

    # Check that Send and Place nodes were created for both types
    send_nodes = [
        node for node in simple_graph.get_all_nodes_copy() if isinstance(node, Send)
    ]
    place_nodes = [
        node for node in simple_graph.get_all_nodes_copy() if isinstance(node, Place)
    ]

    assert len(send_nodes) >= 2  # At least one for data, one for control
    assert len(place_nodes) >= 2  # At least one for data, one for control


def test_cut_cross_region_edges_pass_handles_no_cross_region_edges(
    simple_graph: Graph, op_a: Operation, op_b: Operation
) -> None:
    """Test that CutCrossRegionEdgesPass handles graphs with no cross-region edges."""
    # Create a graph with only same-region edges
    region = simple_graph.create_log_start_region()

    # Create operations in same region
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Connect operations within same region
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
    )

    # Count edges before cutting
    edges_before = len(simple_graph.get_all_edges())

    # Apply CutCrossRegionEdgesPass
    cut_pass = CutCrossRegionEdgesPass(simple_graph, simple_graph.logger)
    cut_pass.apply()

    # Count edges after cutting
    edges_after = len(simple_graph.get_all_edges())

    # Should not have created new edges
    assert edges_after == edges_before
