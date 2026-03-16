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

import pytest
from pydantic import BaseModel, ValidationError

from nv_dfm_core.api._bool_expressions import (
    And,
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
from nv_dfm_core.api._node_id import NodeId, NodeRef


class BoolUser(BaseModel):
    expr: BooleanExpression


def verify_serialization(expr: BooleanExpression):
    user = BoolUser(expr=expr)
    assert BoolUser.model_validate_json(user.model_dump_json()).expr == expr


def test_basic_boolean_operations():
    # Test AND operation
    and_expr = And(
        exp=[
            Equal(operator="==", left=1, right=1),
            Equal(operator="==", left=2, right=2),
        ]
    )
    assert and_expr.operator == "and"
    assert len(and_expr.exp) == 2
    verify_serialization(and_expr)

    # Test OR operation
    or_expr = Or(
        exp=[
            Equal(operator="==", left=1, right=2),
            Equal(operator="==", left=2, right=2),
        ]
    )
    assert or_expr.operator == "or"
    assert len(or_expr.exp) == 2
    verify_serialization(or_expr)

    # Test NOT operation
    not_expr = Not(exp=Equal(operator="==", left=1, right=2))
    assert not_expr.operator == "not"
    verify_serialization(not_expr)


def test_comparison_operations():
    # Test equality
    eq = Equal(operator="==", left=1, right=1)
    assert eq.operator == "=="
    assert eq.left == 1
    assert eq.right == 1
    verify_serialization(eq)
    # Test not equal
    neq = NotEqual(operator="!=", left=1, right=2)
    assert neq.operator == "!="
    assert neq.left == 1
    assert neq.right == 2
    verify_serialization(neq)

    # Test greater than
    gt = GreaterThan(operator=">", left=2, right=1)
    assert gt.operator == ">"
    assert gt.left == 2
    assert gt.right == 1
    verify_serialization(gt)

    # Test greater than or equal
    gte = GreaterThanOrEqual(operator=">=", left=2, right=2)
    assert gte.operator == ">="
    assert gte.left == 2
    assert gte.right == 2
    verify_serialization(gte)

    # Test less than
    lt = LessThan(operator="<", left=1, right=2)
    assert lt.operator == "<"
    assert lt.left == 1
    assert lt.right == 2
    verify_serialization(lt)
    # Test less than or equal
    lte = LessThanOrEqual(operator="<=", left=2, right=2)
    assert lte.operator == "<="
    assert lte.left == 2
    assert lte.right == 2
    verify_serialization(lte)


def test_nested_expressions():
    # Create a complex expression: (1 == 1) AND ((2 > 1) OR (3 < 4))
    complex_expr = And(
        exp=[
            Equal(operator="==", left=1, right=1),
            Or(
                exp=[
                    GreaterThan(operator=">", left=2, right=1),
                    LessThan(operator="<", left=3, right=4),
                ]
            ),
        ]
    )

    assert complex_expr.operator == "and"
    assert len(complex_expr.exp) == 2
    assert isinstance(complex_expr.exp[0], Equal)
    assert isinstance(complex_expr.exp[1], Or)
    assert len(complex_expr.exp[1].exp) == 2
    verify_serialization(complex_expr)


def test_node_references():
    # Test with NodeRef
    node_id = NodeId(tag="@dfm-node", ident="test_node")
    node_ref = NodeRef(id=node_id)
    expr = Equal(operator="==", left=node_ref, right=1)

    assert expr.left == node_ref
    assert expr.right == 1
    verify_serialization(expr)


def test_validation_errors():
    # the following should be okay
    _ = Equal(
        operator="==", left=None, right=1
    )  # None is not a valid ExpressionOperand
    _ = Not(exp=None)  # None should not be valid

    # Test missing required fields
    with pytest.raises(ValidationError):
        _ = And(exp=[])  # Empty list not okay
    with pytest.raises(ValidationError):
        _ = And(
            exp=[Equal(operator="==", left=1, right=1)]
        )  # And with one element not okay

    with pytest.raises(ValidationError):
        _ = Or(exp=[])  # Empty list not okay
    with pytest.raises(ValidationError):
        _ = Or(
            exp=[Equal(operator="==", left=1, right=1)]
        )  # Or with one element not okay
