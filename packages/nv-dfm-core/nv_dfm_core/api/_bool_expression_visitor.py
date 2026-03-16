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

from abc import ABC, abstractmethod

from ._bool_expressions import (
    And,
    Atom,
    Equal,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Not,
    NotEqual,
    Or,
)


class BooleanExpressionVisitor(ABC):
    """Visitor interface for boolean expressions."""

    @abstractmethod
    def visit_and(self, expr: And) -> None:
        """Visit an And expression."""
        pass

    @abstractmethod
    def visit_or(self, expr: Or) -> None:
        """Visit an Or expression."""
        pass

    @abstractmethod
    def visit_not(self, expr: Not) -> None:
        """Visit a Not expression."""
        pass

    @abstractmethod
    def visit_equal(self, expr: Equal) -> None:
        """Visit an Equal expression."""
        pass

    @abstractmethod
    def visit_not_equal(self, expr: NotEqual) -> None:
        """Visit a NotEqual expression."""
        pass

    @abstractmethod
    def visit_greater_than(self, expr: GreaterThan) -> None:
        """Visit a GreaterThan expression."""
        pass

    @abstractmethod
    def visit_greater_than_or_equal(self, expr: GreaterThanOrEqual) -> None:
        """Visit a GreaterThanOrEqual expression."""
        pass

    @abstractmethod
    def visit_less_than(self, expr: LessThan) -> None:
        """Visit a LessThan expression."""
        pass

    @abstractmethod
    def visit_less_than_or_equal(self, expr: LessThanOrEqual) -> None:
        """Visit a LessThanOrEqual expression."""
        pass

    @abstractmethod
    def visit_atom(self, expr: Atom) -> None:
        """Visit an Atom expression."""
        pass
