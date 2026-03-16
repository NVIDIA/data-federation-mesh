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
from nv_dfm_core.gen.irgen.graph._graph_elements import NodeState, Operation
from nv_dfm_core.gen.irgen.graph.passes._resolve_deadlock_pass import (
    ResolveDeadlocksPass,
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


def test_resolve_deadlocks_pass_no_deadlocks(
    simple_graph: Graph, op_a: Any, op_b: Any
) -> None:
    """Test that ResolveDeadlocksPass handles graphs with no deadlocks."""
    # Create a simple graph with no cross-site edges within same logical region
    region = simple_graph.create_log_start_region()

    # Create operations
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Connect operations within same site
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
    )

    # Count regions before resolution
    regions_before = len(simple_graph.get_regions_copy())

    # Apply ResolveDeadlocksPass
    resolve_pass = ResolveDeadlocksPass(simple_graph, simple_graph.logger)
    resolve_pass.apply()

    # Count regions after resolution
    regions_after = len(simple_graph.get_regions_copy())

    # Should not have created any new regions
    assert regions_after == regions_before


def test_resolve_deadlocks_pass_handles_empty_regions(
    simple_graph: Graph, op_a: Any
) -> None:
    """Test that ResolveDeadlocksPass handles empty regions gracefully."""
    # Create a graph with empty regions
    region = simple_graph.create_log_start_region()

    # Create operations only on one site
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    # Set SiteA nodes to selected, SiteB nodes to discarded
    for site in ["SiteA", "SiteB"]:
        node_a = simple_graph.get_node(site, op_a_logid, Operation)
        if site == "SiteA":
            node_a.state = NodeState.SELECTED
        else:
            node_a.state = NodeState.DISCARDED

    # Apply ResolveDeadlocksPass
    resolve_pass = ResolveDeadlocksPass(simple_graph, simple_graph.logger)
    resolve_pass.apply()

    # Should complete without errors
    assert True


def test_resolve_deadlocks_pass_handles_no_cross_site_edges(
    simple_graph: Graph, op_a: Any, op_b: Any
) -> None:
    """Test that ResolveDeadlocksPass handles graphs with no cross-site edges."""
    # Create a graph with only site-internal edges
    region = simple_graph.create_log_start_region()

    # Create operations
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Connect operations within same site only
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
    )

    # Count regions before resolution
    regions_before = len(simple_graph.get_regions_copy())

    # Apply ResolveDeadlocksPass
    resolve_pass = ResolveDeadlocksPass(simple_graph, simple_graph.logger)
    resolve_pass.apply()

    # Count regions after resolution
    regions_after = len(simple_graph.get_regions_copy())

    # Should not have created any new regions
    assert regions_after == regions_before


def test_resolve_deadlocks_pass_handles_weak_edges(
    simple_graph: Graph, op_a: Any, op_b: Any
) -> None:
    """Test that ResolveDeadlocksPass handles weak edges correctly."""
    # Create a graph with weak cross-site edges
    region = simple_graph.create_log_start_region()

    # Create operations
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_a
    )

    op_b_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region, operation=op_b
    )

    # Connect operations with weak cross-site edges
    simple_graph.create_cross_site_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__data__",
        dst_logid=op_b_logid,
        dst_in_slot="val",
        is_weak=True,
    )

    # Count regions before resolution
    regions_before = len(simple_graph.get_regions_copy())

    # Apply ResolveDeadlocksPass
    resolve_pass = ResolveDeadlocksPass(simple_graph, simple_graph.logger)
    resolve_pass.apply()

    # Count regions after resolution
    regions_after = len(simple_graph.get_regions_copy())

    # Should not have created new regions for weak edges
    assert regions_after == regions_before
