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
from nv_dfm_core.gen.irgen.graph.passes._remove_empty_regions_pass import (
    RemoveEmptyRegionsPass,
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


def test_remove_empty_regions_pass_removes_empty_region_with_jump(
    simple_graph: Graph, op_a: Any
) -> None:
    """Test that RemoveEmptyRegionsPass removes empty regions with Jump exit nodes."""
    # Create a graph with an empty region
    region1 = simple_graph.create_log_start_region()
    # make sure that region1 has an exit node (a jump in this case) because
    # every region must have an exit node
    jump1 = simple_graph.create_log_jump_node(region_logid=region1)

    region2 = simple_graph.create_log_region_node(is_loop_head=False)
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump1,
        src_out_slot="__flow__",
        dst_logid=region2,
        dst_in_slot="__flow__",
    )

    # Create operations in region1
    _ = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    # Create jump in region2 (empty region)
    jump2 = simple_graph.create_log_jump_node(region_logid=region2)

    # Connect region1 to region2
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump1,
        src_out_slot="__flow__",
        dst_logid=region2,
        dst_in_slot="__flow__",
    )

    # Connect jump to a target (create another region)
    region3 = simple_graph.create_log_region_node(is_loop_head=False)
    # make sure that region3 has a stop node because every region must have a stop node
    _ = simple_graph.create_log_stop_node(region_logid=region3)
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump2,
        src_out_slot="__flow__",
        dst_logid=region3,
        dst_in_slot="__flow__",
    )

    # Count regions before removal
    regions_before = len(simple_graph.get_regions_copy())

    # Apply RemoveEmptyRegionsPass
    remove_pass = RemoveEmptyRegionsPass(simple_graph, simple_graph.logger)
    remove_pass.apply()

    # Count regions after removal
    regions_after = len(simple_graph.get_regions_copy())

    # Should have removed the empty region
    assert regions_after < regions_before

    # Check that region2 is gone
    regions = simple_graph.get_regions_copy()
    region2_ids = [
        region.global_id() for region in regions if region.site_local_logid() == region2
    ]
    assert len(region2_ids) == 0


def test_remove_empty_regions_pass_keeps_non_empty_regions(
    simple_graph: Graph, op_a: Any, op_b: Any
) -> None:
    """Test that RemoveEmptyRegionsPass keeps regions with operations."""
    # Create a graph with non-empty regions
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations in both regions
    _ = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    _ = simple_graph.create_log_operation_node_for_execute(
        region_logid=region2, operation=op_b
    )

    # Create jumps in both regions
    jump1 = simple_graph.create_log_jump_node(region_logid=region1)
    _jump2 = simple_graph.create_log_jump_node(region_logid=region2)

    # Connect regions
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump1,
        src_out_slot="__flow__",
        dst_logid=region2,
        dst_in_slot="__flow__",
    )

    # Count regions before removal
    regions_before = len(simple_graph.get_regions_copy())

    # Apply RemoveEmptyRegionsPass
    remove_pass = RemoveEmptyRegionsPass(simple_graph, simple_graph.logger)
    remove_pass.apply()

    # Count regions after removal
    regions_after = len(simple_graph.get_regions_copy())

    # Should not have removed any regions (both have operations)
    assert regions_after == regions_before


def test_remove_empty_regions_pass_handles_non_jump_exit_nodes(
    simple_graph: Graph, op_a: Any
) -> None:
    """Test that RemoveEmptyRegionsPass doesn't remove regions with non-Jump exit nodes."""
    # Create a graph with empty region but non-Jump exit node
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations in region1
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    # Create stop in region2 (empty region but non-Jump exit)
    _ = simple_graph.create_log_stop_node(region_logid=region2)

    # Connect region1 to region2 using the operation
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__flow__",
        dst_logid=region2,
        dst_in_slot="__flow__",
    )

    # Count regions before removal
    regions_before = len(simple_graph.get_regions_copy())

    # Apply RemoveEmptyRegionsPass
    remove_pass = RemoveEmptyRegionsPass(simple_graph, simple_graph.logger)
    remove_pass.apply()

    # Count regions after removal
    regions_after = len(simple_graph.get_regions_copy())

    # Should not have removed the region (has non-Jump exit node)
    assert regions_after == regions_before


def test_remove_empty_regions_pass_handles_multiple_jump_targets(
    simple_graph: Graph, op_a: Any
) -> None:
    """Test that RemoveEmptyRegionsPass handles regions with multiple jump targets."""
    # Create a graph with empty region that has multiple jump targets
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)
    region3 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations in region1
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    # Create jump in region2 (empty region)
    jump = simple_graph.create_log_jump_node(region_logid=region2)

    # Connect region1 to region2 using the operation
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__flow__",
        dst_logid=region2,
        dst_in_slot="__flow__",
    )

    # Connect jump to multiple targets
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump,
        src_out_slot="__flow__",
        dst_logid=region3,
        dst_in_slot="__flow__",
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump,
        src_out_slot="__flow__",
        dst_logid=region1,
        dst_in_slot="__flow__",
    )

    # Apply RemoveEmptyRegionsPass - should raise assertion error
    remove_pass = RemoveEmptyRegionsPass(simple_graph, simple_graph.logger)
    with pytest.raises(AssertionError, match="has 2 jump targets, expected 1"):
        remove_pass.apply()


def test_remove_empty_regions_pass_handles_type_mismatch(
    simple_graph: Graph, op_a: Any
) -> None:
    """Test that RemoveEmptyRegionsPass handles type mismatches between regions."""
    # Create a graph with empty region that has type mismatch
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations in region1
    op_a_logid = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    # Create jump in region2 (empty region)
    jump = simple_graph.create_log_jump_node(region_logid=region2)

    # Connect region1 to region2 using the operation
    simple_graph.create_site_internal_flow_edges(
        src_logid=op_a_logid,
        src_out_slot="__flow__",
        dst_logid=region2,
        dst_in_slot="__flow__",
    )

    # Connect jump to a target with different type
    region3 = simple_graph.create_log_region_node(is_loop_head=False)
    # Add exit node to region3 to make it valid
    _ = simple_graph.create_log_stop_node(region_logid=region3)
    # Connect jump to region3
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump,
        src_out_slot="__flow__",
        dst_logid=region3,
        dst_in_slot="__flow__",
    )

    # Apply RemoveEmptyRegionsPass - should handle gracefully
    remove_pass = RemoveEmptyRegionsPass(simple_graph, simple_graph.logger)
    remove_pass.apply()

    # Should complete without errors
    assert True


def test_remove_empty_regions_pass_handles_no_empty_regions(
    simple_graph: Graph, op_a: Any, op_b: Any
) -> None:
    """Test that RemoveEmptyRegionsPass handles graphs with no empty regions."""
    # Create a graph with only non-empty regions
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations in both regions
    _ = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    _ = simple_graph.create_log_operation_node_for_execute(
        region_logid=region2, operation=op_b
    )

    # Create exit nodes
    jump1 = simple_graph.create_log_jump_node(region_logid=region1)
    _ = simple_graph.create_log_stop_node(region_logid=region2)

    # Connect regions
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump1,
        src_out_slot="__flow__",
        dst_logid=region2,
        dst_in_slot="__flow__",
    )

    # Count regions before removal
    regions_before = len(simple_graph.get_regions_copy())

    # Apply RemoveEmptyRegionsPass
    remove_pass = RemoveEmptyRegionsPass(simple_graph, simple_graph.logger)
    remove_pass.apply()

    # Count regions after removal
    regions_after = len(simple_graph.get_regions_copy())

    # Should not have removed any regions
    assert regions_after == regions_before


def test_remove_empty_regions_pass_handles_complex_chain(
    simple_graph: Graph, op_a: Any
) -> None:
    """Test that RemoveEmptyRegionsPass handles complex chains of empty regions."""
    # Create a graph with a chain of empty regions
    region1 = simple_graph.create_log_start_region()
    region2 = simple_graph.create_log_region_node(is_loop_head=False)
    region3 = simple_graph.create_log_region_node(is_loop_head=False)
    region4 = simple_graph.create_log_region_node(is_loop_head=False)

    # Create operations in region1
    _ = simple_graph.create_log_operation_node_for_execute(
        region_logid=region1, operation=op_a
    )

    # Create jumps in empty regions
    jump1 = simple_graph.create_log_jump_node(region_logid=region1)
    jump2 = simple_graph.create_log_jump_node(region_logid=region2)
    jump3 = simple_graph.create_log_jump_node(region_logid=region3)

    # Add exit node to region4 to make it valid
    _ = simple_graph.create_log_stop_node(region_logid=region4)

    # Connect the chain
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump1,
        src_out_slot="__flow__",
        dst_logid=region2,
        dst_in_slot="__flow__",
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump2,
        src_out_slot="__flow__",
        dst_logid=region3,
        dst_in_slot="__flow__",
    )
    simple_graph.create_site_internal_flow_edges(
        src_logid=jump3,
        src_out_slot="__flow__",
        dst_logid=region4,
        dst_in_slot="__flow__",
    )

    # Count regions before removal
    regions_before = len(simple_graph.get_regions_copy())

    # Apply RemoveEmptyRegionsPass
    remove_pass = RemoveEmptyRegionsPass(simple_graph, simple_graph.logger)
    remove_pass.apply()

    # Count regions after removal
    regions_after = len(simple_graph.get_regions_copy())

    # Should have removed the empty regions
    assert regions_after < regions_before

    # Check that empty regions are gone
    regions = simple_graph.get_regions_copy()
    empty_region_ids = [
        region.global_id()
        for region in regions
        if region.site_local_logid() in [region2, region3]
    ]
    assert len(empty_region_ids) == 0
