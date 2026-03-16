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

from io import StringIO

from typing_extensions import override

from nv_dfm_core.api import (
    And,
    Atom,
    BooleanExpression,
    BooleanExpressionVisitor,
    Equal,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    NodeRef,
    Not,
    NotEqual,
    Or,
)
from nv_dfm_core.gen.modgen.ir._stmt_ref import StmtRef


class BoolExprToString(BooleanExpressionVisitor):
    """Visitor that converts a boolean expression to a string."""

    def __init__(self):
        self.out: StringIO = StringIO()
        self.ssa_uses: list[StmtRef] = []

    def get_string(self) -> str:
        return self.out.getvalue()

    def visit_boolean_expression(self, expr: BooleanExpression) -> None:
        """Central method to handle any boolean expression.
        If it's an atom, handle it through visit_atom.
        Otherwise, let the expression handle itself via accept."""
        if isinstance(expr, (NodeRef, list, str, bool, int, float, type(None))):
            self.visit_atom(expr)
        else:
            expr.accept(self)

    @override
    def visit_and(self, expr: And) -> None:
        # Visit all expressions in the AND
        _ = self.out.write("(")
        first = True
        for e in expr.exp:
            if not first:
                _ = self.out.write(" and ")
            first = False
            self.visit_boolean_expression(e)
        _ = self.out.write(")")

    @override
    def visit_or(self, expr: Or) -> None:
        # Visit all expressions in the OR
        _ = self.out.write("(")
        first = True
        for e in expr.exp:
            if not first:
                _ = self.out.write(" or ")
            first = False
            self.visit_boolean_expression(e)
        _ = self.out.write(")")

    @override
    def visit_not(self, expr: Not) -> None:
        # Visit the negated expression
        _ = self.out.write("not ")
        self.visit_boolean_expression(expr.exp)

    @override
    def visit_equal(self, expr: Equal) -> None:
        self.visit_boolean_expression(expr.left)
        _ = self.out.write(" == ")
        self.visit_boolean_expression(expr.right)

    @override
    def visit_not_equal(self, expr: NotEqual) -> None:
        self.visit_boolean_expression(expr.left)
        _ = self.out.write(" != ")
        self.visit_boolean_expression(expr.right)

    @override
    def visit_greater_than(self, expr: GreaterThan) -> None:
        self.visit_boolean_expression(expr.left)
        _ = self.out.write(" > ")
        self.visit_boolean_expression(expr.right)

    @override
    def visit_greater_than_or_equal(self, expr: GreaterThanOrEqual) -> None:
        self.visit_boolean_expression(expr.left)
        _ = self.out.write(" >= ")
        self.visit_boolean_expression(expr.right)

    @override
    def visit_less_than(self, expr: LessThan) -> None:
        self.visit_boolean_expression(expr.left)
        _ = self.out.write(" < ")
        self.visit_boolean_expression(expr.right)

    @override
    def visit_less_than_or_equal(self, expr: LessThanOrEqual) -> None:
        self.visit_boolean_expression(expr.left)
        _ = self.out.write(" <= ")
        self.visit_boolean_expression(expr.right)

    @override
    def visit_atom(self, expr: Atom) -> None:
        # If the atom is a NodeRef, add its NodeId
        if isinstance(expr, NodeRef):
            ref = StmtRef(
                stmt_id=expr.id.as_identifier(), sel=expr.sel, issubs=expr.issubs
            )
            _ = self.out.write(ref.to_python())
            self.ssa_uses.append(ref)
        else:
            _ = self.out.write(repr(expr))
