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

from abc import ABC
from typing import Any, Literal

from pydantic import BaseModel, Field, JsonValue, field_validator

from ._node_param import NodeParam

json_without_dicts = list[JsonValue] | str | bool | int | float | None
# we cannot allow dicts as an atom because then serialization is ambiguous
# and the other Bool models will be de-serialized as simple dicts
Atom = NodeParam | list[JsonValue] | json_without_dicts


class And(BaseModel):
    """Represents a logical AND operation."""

    operator: Literal["and"] = Field(default="and")
    exp: list["BooleanExpression"] = Field(min_length=2)

    @field_validator("exp", mode="before")  # noqa: F821
    @classmethod
    def rewrite_expression_object_to_noderef(cls, v: Any) -> Any:
        from ._expression import Expression  # pylint: disable=import-outside-toplevel

        return [e.dfm_node_id.to_ref() if isinstance(e, Expression) else e for e in v]

    def accept(self, visitor: Any) -> None:
        visitor.visit_and(self)


class Or(BaseModel):
    """Represents a logical OR operation."""

    operator: Literal["or"] = Field(default="or")
    exp: list["BooleanExpression"] = Field(min_length=2)

    @field_validator("exp", mode="before")  # noqa: F821
    @classmethod
    def rewrite_expression_object_to_noderef(cls, v: Any) -> Any:
        from ._expression import Expression  # pylint: disable=import-outside-toplevel

        return [e.dfm_node_id.to_ref() if isinstance(e, Expression) else e for e in v]

    def accept(self, visitor: Any) -> None:
        visitor.visit_or(self)


class Not(BaseModel):
    """Represents a logical NOT operation."""

    operator: Literal["not"] = Field(default="not")
    exp: "BooleanExpression"

    @field_validator("exp", mode="before")  # noqa: F821
    @classmethod
    def rewrite_expression_object_to_noderef(cls, v: Any) -> Any:
        from ._expression import Expression  # pylint: disable=import-outside-toplevel

        return v.dfm_node_id.to_ref() if isinstance(v, Expression) else v

    def accept(self, visitor: Any) -> None:
        visitor.visit_not(self)


class BinaryOp(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """Base class for binary boolean operations (e.g., comparisons)."""

    left: "BooleanExpression"
    right: "BooleanExpression"

    @field_validator("left", mode="before")  # noqa: F821
    @classmethod
    def rewrite_left_object_to_noderef(cls, v: Any) -> Any:
        from ._expression import Expression  # pylint: disable=import-outside-toplevel

        return v.dfm_node_id.to_ref() if isinstance(v, Expression) else v

    @field_validator("right", mode="before")  # noqa: F821
    @classmethod
    def rewrite_right_object_to_noderef(cls, v: Any) -> Any:
        from ._expression import Expression  # pylint: disable=import-outside-toplevel

        return v.dfm_node_id.to_ref() if isinstance(v, Expression) else v


class Equal(BinaryOp):
    """Represents an equality comparison (==)."""

    operator: Literal["=="] = "=="

    def accept(self, visitor: Any) -> None:
        visitor.visit_equal(self)


class NotEqual(BinaryOp):
    """Represents a not equal comparison (!=)."""

    operator: Literal["!="] = "!="

    def accept(self, visitor: Any) -> None:
        visitor.visit_not_equal(self)


class GreaterThan(BinaryOp):
    """Represents a greater than comparison (>)."""

    operator: Literal[">"] = ">"

    def accept(self, visitor: Any) -> None:
        visitor.visit_greater_than(self)


class GreaterThanOrEqual(BinaryOp):
    """Represents a greater than or equal comparison (>=)."""

    operator: Literal[">="] = ">="

    def accept(self, visitor: Any) -> None:
        visitor.visit_greater_than_or_equal(self)


class LessThan(BinaryOp):
    """Represents a less than comparison (<)."""

    operator: Literal["<"] = "<"

    def accept(self, visitor: Any) -> None:
        visitor.visit_less_than(self)


class LessThanOrEqual(BinaryOp):
    """Represents a less than or equal comparison (<=)."""

    operator: Literal["<="] = "<="

    def accept(self, visitor: Any) -> None:
        visitor.visit_less_than_or_equal(self)


# A comparison expression can be any of the specific comparison types
# We use a discriminator on the 'operator' field.
ComparisonExpression = (
    Equal | NotEqual | GreaterThan | GreaterThanOrEqual | LessThan | LessThanOrEqual
)

# The main BooleanExpression type:
# This is where the magic of the discriminator truly shines for nested unions.
# Pydantic will first look for an 'operator' field.
# If 'operator' is present, it will try to match one of the operator-based models (And, Or, Not, ComparisonExpression).
# If 'operator' is not present, it will fall back to trying to match Atom.
BooleanExpression = Atom | And | Or | Not | ComparisonExpression
