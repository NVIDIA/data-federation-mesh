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

from logging import Logger

from typing_extensions import override

from nv_dfm_core.api import (
    ApiVisitor,
    If,
    Operation,
    PlaceParam,
    Yield,
)
from nv_dfm_core.api._node_id import NodeId, NodeRef
from nv_dfm_core.api._pipeline import Pipeline
from nv_dfm_core.api._statement import Statement
from nv_dfm_core.gen.irgen.visitors._collect_bool_expr_node_ids import (
    CollectBoolExprNodeIds,
)

from .._fed_info import FedInfo
from ..graph import Graph


class PipelineToGraphVisitor(ApiVisitor):
    def __init__(
        self,
        fed_info: FedInfo,
        pipeline: Pipeline,
        homesite: str,
        candidate_sites: list[str],
        logger: Logger,
        debug: bool = False,
    ):
        self.graph: Graph = Graph(
            pipeline=pipeline,
            homesite=homesite,
            fed_info=fed_info,
            candidate_sites=candidate_sites,
            logger=logger,
        )
        self.logger: Logger = logger
        self.debug: bool = debug
        self.current_region_logid: str = "UNKNOWN"

    @override
    def visit_pipeline(self, pipeline: Pipeline):
        self.current_region_logid = self.graph.create_log_start_region()
        for stmt in pipeline.dfm_body:
            stmt.accept(self)
        _stop_node = self.graph.create_log_stop_node(self.current_region_logid)

    @override
    def visit_yield(self, yield_stmt: Yield) -> None:
        # create the yield place, if needed
        yield_place_logid = self.graph.create_yield_place_node_if_needed(
            yield_stmt.place
        )

        # create the yield "send" node
        yld = self.graph.create_log_yield_node_where_needed(
            self.current_region_logid, yld_stmt=yield_stmt
        )
        self._create_after_flow_edge_if_needed(yield_stmt, yld)

        # create flow edges for the yield input
        if isinstance(yield_stmt.value, NodeRef):
            input = yield_stmt.value.id.as_identifier()
            # the create_log_yield_node_where_needed ensures that we created a yield node
            # on a site if and only if the input node is present on that site.
            # yields can run on any site, we don't send them across sites.
            self.graph.create_site_internal_flow_edges(
                src_logid=input,
                src_out_slot="__data__",
                dst_logid=yld,
                dst_in_slot="value",
            )

        # and add a weak edge
        self.graph.create_cross_site_flow_edges(
            src_logid=yld,
            src_out_slot="__data__",
            dst_logid=yield_place_logid,
            dst_in_slot="__data__",
            is_weak=True,
        )

    @override
    def visit_if(self, if_stmt: If):
        # extract all free params from the condition
        expr = if_stmt.cond
        collector = CollectBoolExprNodeIds()
        collector.visit_boolean_expression(expr)
        expr_params = collector.node_ids

        # create the evaluation node. The expression node will have input slots with the names of the incoming node_ids
        expr_eval = self.graph.create_log_bool_value_node(
            self.current_region_logid, if_stmt=if_stmt, params=expr_params
        )
        self._create_after_flow_edge_if_needed(if_stmt, expr_eval)

        # create flow edges for the params
        for node_id in expr_params:
            self.graph.create_cross_site_flow_edges(
                src_logid=node_id.as_identifier(),
                src_out_slot="__data__",
                dst_logid=expr_eval,
                dst_in_slot=node_id.as_identifier(),
            )

        # close the current region with a branch node
        if_branch = self.graph.create_log_branch_node(self.current_region_logid)
        self.graph.create_site_internal_flow_edges(
            src_logid=expr_eval,
            src_out_slot="__condition__",
            dst_logid=if_branch,
            dst_in_slot="__condition__",
        )

        # create the true and false regions
        if_true_region = self.graph.create_log_region_node(is_loop_head=False)
        self.graph.create_site_internal_flow_edges(
            src_logid=if_branch,
            src_out_slot="__branch_taken__",
            dst_logid=if_true_region,
            dst_in_slot="__flow__",
        )

        if_false_region = self.graph.create_log_region_node(is_loop_head=False)
        self.graph.create_site_internal_flow_edges(
            src_logid=if_branch,
            src_out_slot="__branch_not_taken__",
            dst_logid=if_false_region,
            dst_in_slot="__flow__",
        )

        # set the true region as current and process the if body
        self.current_region_logid = if_true_region
        for stmt in if_stmt.dfm_body:
            stmt.accept(self)
        # from current block, jump to after block (the if_false_region)
        jump = self.graph.create_log_jump_node(self.current_region_logid)
        self.graph.create_site_internal_flow_edges(
            src_logid=jump,
            src_out_slot="__flow__",
            dst_logid=if_false_region,
            dst_in_slot="__flow__",
        )

        # done, continue and do the rest in the after region
        self.current_region_logid = if_false_region

    @override
    def visit_operation(self, operation: Operation) -> None:
        # create the operation node
        op_node = self.graph.create_log_operation_node_for_execute(
            self.current_region_logid, operation
        )

        self._create_after_flow_edge_if_needed(operation, op_node)

        # create flow edges for all the params
        for fieldname, value in operation.get_noderef_and_placeparam_pydantic_fields():
            if isinstance(value, PlaceParam):
                log_src_id = self.graph.create_place_param_node_if_needed(value)
                # we connect places only locally
                self.graph.create_site_internal_flow_edges(
                    src_logid=log_src_id,
                    src_out_slot="__data__",
                    dst_logid=op_node,
                    dst_in_slot=fieldname,
                )
            else:
                log_src_id = value.id.as_identifier()
                self.graph.create_cross_site_flow_edges(
                    src_logid=log_src_id,
                    src_out_slot="__data__",
                    dst_logid=op_node,
                    dst_in_slot=fieldname,
                )

    def _create_after_flow_edge_if_needed(
        self, statement: Statement, stmt_node_logid: str
    ) -> None:
        if statement.dfm_after:
            assert isinstance(statement.dfm_after, NodeId)
            self.graph.create_cross_site_flow_edges(
                src_logid=statement.dfm_after.as_identifier(),
                src_out_slot="__flow__",
                dst_logid=stmt_node_logid,
                dst_in_slot="__flow__",
            )
