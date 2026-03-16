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
from nv_dfm_core.api._bool_expressions import Equal
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


@pytest.fixture
def fed_info() -> FedInfo:
    return FedInfo(
        sites={
            "SiteA": SiteInfo(
                interface={
                    "A": OperationInfo(
                        operation="A",
                        compute_cost=ComputeCostInfo(fixed_time=1),
                        is_async=True,
                    ),
                    "B": OperationInfo(
                        operation="B",
                        compute_cost=ComputeCostInfo(fixed_time=1000),
                        is_async=True,
                    ),
                    "C": OperationInfo(
                        operation="C",
                        compute_cost=ComputeCostInfo(fixed_time=1),
                        is_async=True,
                    ),
                    "D": OperationInfo(
                        operation="D",
                        compute_cost=ComputeCostInfo(fixed_time=10),
                        is_async=True,
                    ),
                    "E": OperationInfo(
                        operation="E",
                        compute_cost=ComputeCostInfo(fixed_time=1),
                        is_async=True,
                    ),
                },
                providers={},
                send_cost={
                    "SiteB": SendCostInfo(),
                    "SiteC": SendCostInfo(),
                },
            ),
            "SiteB": SiteInfo(
                interface={
                    "A": OperationInfo(
                        operation="A",
                        compute_cost=ComputeCostInfo(fixed_time=10),
                        is_async=True,
                    ),
                    "B": OperationInfo(
                        operation="B",
                        compute_cost=ComputeCostInfo(fixed_time=1),
                        is_async=True,
                    ),
                    "C": OperationInfo(
                        operation="C",
                        compute_cost=ComputeCostInfo(fixed_time=10),
                        is_async=True,
                    ),
                    "D": OperationInfo(
                        operation="D",
                        compute_cost=ComputeCostInfo(fixed_time=1),
                        is_async=True,
                    ),
                    "E": OperationInfo(
                        operation="E",
                        compute_cost=ComputeCostInfo(fixed_time=10),
                        is_async=True,
                    ),
                },
                providers={},
                send_cost={
                    "SiteA": SendCostInfo(),
                    "SiteC": SendCostInfo(),
                },
            ),
            "SiteC": SiteInfo(
                interface={
                    "A": OperationInfo(
                        operation="A",
                        compute_cost=ComputeCostInfo(fixed_time=10),
                        is_async=True,
                    ),
                    "B": OperationInfo(
                        operation="B",
                        compute_cost=ComputeCostInfo(fixed_time=10),
                        is_async=True,
                    ),
                    "C": OperationInfo(
                        operation="C",
                        compute_cost=ComputeCostInfo(fixed_time=10),
                        is_async=True,
                    ),
                    "D": OperationInfo(
                        operation="D",
                        compute_cost=ComputeCostInfo(fixed_time=10),
                        is_async=True,
                    ),
                    "E": OperationInfo(
                        operation="E",
                        compute_cost=ComputeCostInfo(fixed_time=10),
                        is_async=True,
                    ),
                },
                providers={},
                send_cost={
                    "SiteA": SendCostInfo(),
                    "SiteB": SendCostInfo(),
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
def op_c() -> Any:
    op_c = MagicMock(
        site=BestOf(),
        provider=None,
        __api_name__="C",
        dfm_node_id=NodeId(ident="nodeC"),
        get_noderef_and_placeparam_pydantic_fields=lambda: [("age", None)],
    )
    return op_c


@pytest.fixture
def op_d() -> Any:
    op_d = MagicMock(
        site=BestOf(),
        provider=None,
        __api_name__="D",
        dfm_node_id=NodeId(ident="nodeD"),
        get_noderef_and_placeparam_pydantic_fields=lambda: [("age", None)],
    )
    return op_d


@pytest.fixture
def op_e() -> Any:
    op_e = MagicMock(
        site=BestOf(),
        provider=None,
        __api_name__="E",
        dfm_node_id=NodeId(ident="nodeE"),
        get_noderef_and_placeparam_pydantic_fields=lambda: [("mya", None)],
    )
    return op_e


def test_for_each_loop_with_if(
    fed_info: FedInfo, op_a: Any, op_b: Any, op_c: Any, op_d: Any, op_e: Any
):
    """The program being tested is something like this:

    as = A()
    parallel for each a in as:
        if a > 42:
            b = B(a)
            c = C(b)
            d = D(c)
            Yield(d)
        e = E(a)
        <end fork>
    <sync par for>
    Yield('Hello, world!')
    """

    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB", "SiteC"],
        logger=None,
    )

    # pipeline nodeid to graph node group logical id
    symtab: dict[NodeId, str] = {}

    def start_region() -> str:
        # start -> as = A() -> jump
        region = graph.create_log_start_region()
        as_equals_A = graph.create_log_operation_node_for_execute(
            region_logid=region, operation=op_a
        )
        symtab[op_a.dfm_node_id] = as_equals_A
        return graph.create_log_jump_node(region_logid=region)

    def loop_head_region(pred_exit: str) -> str:
        for_loop_node = NodeId(ident="par_for_each_node0")

        # enter -> parallel_iterate(as)
        region = graph.create_log_region_node(is_loop_head=True)
        # connect to predecessor
        graph.create_site_internal_flow_edges(
            src_logid=pred_exit,
            src_out_slot="__flow__",
            dst_logid=region,
            dst_in_slot="__flow__",
        )
        # next(as)
        as_equals_A = symtab[op_a.dfm_node_id]
        next_node = graph.create_log_next_node_where_needed(
            region_logid=region,
            foreach_node_id=for_loop_node,
            iterator_logid=as_equals_A,
            in_type="Any",
            out_type="str",
            is_async=True,
        )
        graph.create_site_internal_flow_edges(
            src_logid=as_equals_A,
            src_out_slot="__data__",
            dst_logid=next_node,
            dst_in_slot="__iterator__",
        )
        symtab[for_loop_node] = next_node

        # par iterate
        par_iterate = graph.create_log_par_foreach_node(
            region_logid=region, foreach_node_id=for_loop_node
        )
        graph.create_site_internal_flow_edges(
            src_logid=next_node,
            src_out_slot="__has_next__",
            dst_logid=par_iterate,
            dst_in_slot="__has_next__",
        )

        return par_iterate

    def if_head_region(next_node: str, pred_exit: str) -> str:
        # enter -> evaluate if condition -> branch

        region = graph.create_log_region_node(is_loop_head=False)
        # connect to predecessor
        graph.create_site_internal_flow_edges(
            src_logid=pred_exit,
            src_out_slot="__fork__",
            dst_logid=region,
            dst_in_slot="__flow__",
        )
        # evaluate node
        eval_if_expression = graph.create_log_bool_value_node(
            region_logid=region,
            if_stmt=MagicMock(
                dfm_node_id=NodeId(ident="if_node0"),
                cond=Equal(operator="==", left=op_a, right=1),
            ),
            params=set([op_a.dfm_node_id]),
        )
        # connect inputs of evaluate node
        graph.create_cross_site_flow_edges(
            src_logid=next_node,
            src_out_slot="__data__",
            dst_logid=eval_if_expression,
            dst_in_slot=op_a.dfm_node_id.as_identifier(),
        )
        # branch
        branch_exit = graph.create_log_branch_node(region_logid=region)
        graph.create_site_internal_flow_edges(
            src_logid=eval_if_expression,
            src_out_slot="__condition__",
            dst_logid=branch_exit,
            dst_in_slot="__condition__",
        )
        return branch_exit

    def if_body_region(pred_exit: str) -> str:
        # enter -> b = B(a) -> c = C(b) -> d = D(c) -> Yield(c)
        region = graph.create_log_region_node(is_loop_head=False)
        graph.create_site_internal_flow_edges(
            src_logid=pred_exit,
            src_out_slot="__branch_taken__",
            dst_logid=region,
            dst_in_slot="__flow__",
        )

        # b = B(a)
        b_equals_B = graph.create_log_operation_node_for_execute(
            region_logid=region, operation=op_b
        )
        for_loop_node = NodeId(ident="par_for_each_node0")
        par_iterate = symtab[for_loop_node]
        graph.create_cross_site_flow_edges(
            src_logid=par_iterate,
            src_out_slot="__data__",
            dst_logid=b_equals_B,
            dst_in_slot="val",
        )

        # c = C(b)
        c_equals_C = graph.create_log_operation_node_for_execute(
            region_logid=region, operation=op_c
        )
        graph.create_cross_site_flow_edges(
            src_logid=b_equals_B,
            src_out_slot="__data__",
            dst_logid=c_equals_C,
            dst_in_slot="age",
        )

        # d = D(c)
        d_equals_D = graph.create_log_operation_node_for_execute(
            region_logid=region, operation=op_d
        )
        graph.create_cross_site_flow_edges(
            src_logid=c_equals_C,
            src_out_slot="__data__",
            dst_logid=d_equals_D,
            dst_in_slot="age",
        )

        # Yield(c)
        yield_c = graph.create_log_yield_node_where_needed(
            region_logid=region,
            yld_stmt=MagicMock(
                dfm_node_id=NodeId(ident="yield_node0"), value=op_d.dfm_node_id.to_ref()
            ),
        )
        graph.create_site_internal_flow_edges(
            src_logid=d_equals_D,
            src_out_slot="__data__",
            dst_logid=yield_c,
            dst_in_slot="value",
        )

        yp = graph.create_yield_place_node_if_needed(place=f"{yield_c}_yield_place")
        graph.create_cross_site_flow_edges(yield_c, "__data__", yp, "__data__")

        # exit
        jump = graph.create_log_jump_node(region_logid=region)
        return jump

    def in_loop_body_after_if(branch: str, branch_body: str) -> str:
        # enter -> sum place -> e = E(a) -> end fork
        region = graph.create_log_region_node(is_loop_head=False)
        graph.create_site_internal_flow_edges(
            src_logid=branch,
            src_out_slot="__branch_not_taken__",
            dst_logid=region,
            dst_in_slot="__flow__",
        )
        graph.create_site_internal_flow_edges(
            src_logid=branch_body,
            src_out_slot="__flow__",
            dst_logid=region,
            dst_in_slot="__flow__",
        )

        # e = E(a)
        for_loop_node = NodeId(ident="par_for_each_node0")
        par_iterate = symtab[for_loop_node]
        e_equals_E = graph.create_log_operation_node_for_execute(
            region_logid=region, operation=op_e
        )
        graph.create_cross_site_flow_edges(
            src_logid=par_iterate,
            src_out_slot="__data__",
            dst_logid=e_equals_E,
            dst_in_slot="mya",
        )

        # end of fork
        end_fork = graph.create_log_end_fork_node(region_logid=region)
        return end_fork

    def join_region(par_iterate: str, end_fork: str) -> str:
        # enter + sum place -> yield(42) -> stop
        region = graph.create_log_region_node(is_loop_head=False)
        graph.create_site_internal_flow_edges(
            src_logid=par_iterate,
            src_out_slot="__stop_iteration__",
            dst_logid=region,
            dst_in_slot="__flow__",
        )
        # connect forks to sum place
        sum_place = graph.create_log_place_node(
            region_logid=region,
            place=f"{par_iterate}_framecount",
            kind="control",
            origin="internal",
            flavor="framecount",
            type="Any",
        )
        graph.create_site_internal_flow_edges(
            src_logid=end_fork,
            src_out_slot="__fork_frame__",
            dst_logid=sum_place,
            dst_in_slot="__data__",
        )
        # Yield('Hello, world!')
        yield_hello_world = graph.create_log_yield_node_where_needed(
            region_logid=region,
            yld_stmt=MagicMock(
                dfm_node_id=NodeId(ident="yield_node1"), value="Hello, world!"
            ),
        )
        yp = graph.create_yield_place_node_if_needed(
            place=f"{yield_hello_world}_yield_place"
        )
        graph.create_cross_site_flow_edges(
            yield_hello_world, "__data__", yp, "__data__"
        )
        # exit
        stop = graph.create_log_stop_node(region_logid=region)
        return stop

    r0 = start_region()
    r1 = loop_head_region(r0)
    next_node = symtab[NodeId(ident="par_for_each_node0")]
    r2 = if_head_region(next_node, r1)
    r3 = if_body_region(r2)
    r4 = in_loop_body_after_if(r2, r3)
    _r5 = join_region(r1, r4)

    DEBUG = False

    if DEBUG:
        print("Graphviz:")
        print(graph.to_graphviz_by_site())

    for gpass in graph.get_transformation_passes():
        gpass.apply()
        if DEBUG:
            print("#" * 120)
            print(f"{gpass.target_state().value}:")
            print("#" * 120)
            print(graph.to_graphviz_by_site())
            print("#" * 120)

    if DEBUG:
        print("#" * 120)
        print("By region:")
        print("#" * 120)
        print(graph.to_graphviz_by_region())
