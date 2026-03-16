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

from typing_extensions import override

from nv_dfm_core.api import BooleanExpressionVisitor, NodeId, NodeRef
from nv_dfm_core.api._bool_expressions import (
    And,
    Atom,
    BooleanExpression,
    Equal,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Not,
    NotEqual,
    Or,
)


class CollectBoolExprNodeIds(BooleanExpressionVisitor):
    """Visitor that collects all NodeIds from a boolean expression tree."""

    def __init__(self):
        self.node_ids: set[NodeId] = set()

    def visit_boolean_expression(self, expr: BooleanExpression):
        """Central method to handle any boolean expression.
        If it's an atom, handle it through visit_atom.
        Otherwise, let the expression handle itself via accept."""
        if isinstance(expr, (NodeRef, list, str, bool, int, float, type(None))):
            self.visit_atom(expr)
        else:
            expr.accept(self)

    @override
    def visit_and(self, expr: And):
        # Visit all expressions in the AND
        for e in expr.exp:
            self.visit_boolean_expression(e)

    @override
    def visit_or(self, expr: Or):
        # Visit all expressions in the OR
        for e in expr.exp:
            self.visit_boolean_expression(e)

    @override
    def visit_not(self, expr: Not):
        # Visit the negated expression
        self.visit_boolean_expression(expr.exp)

    @override
    def visit_equal(self, expr: Equal) -> None:
        self.visit_boolean_expression(expr.left)
        self.visit_boolean_expression(expr.right)

    @override
    def visit_not_equal(self, expr: NotEqual) -> None:
        self.visit_boolean_expression(expr.left)
        self.visit_boolean_expression(expr.right)

    @override
    def visit_greater_than(self, expr: GreaterThan) -> None:
        self.visit_boolean_expression(expr.left)
        self.visit_boolean_expression(expr.right)

    @override
    def visit_greater_than_or_equal(self, expr: GreaterThanOrEqual) -> None:
        self.visit_boolean_expression(expr.left)
        self.visit_boolean_expression(expr.right)

    @override
    def visit_less_than(self, expr: LessThan) -> None:
        self.visit_boolean_expression(expr.left)
        self.visit_boolean_expression(expr.right)

    @override
    def visit_less_than_or_equal(self, expr: LessThanOrEqual) -> None:
        self.visit_boolean_expression(expr.left)
        self.visit_boolean_expression(expr.right)

    @override
    def visit_atom(self, expr: Atom):
        # If the atom is a NodeRef, add its NodeId
        if isinstance(expr, NodeRef):
            self.node_ids.add(expr.id)
