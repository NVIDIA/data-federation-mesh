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
from collections.abc import Iterable, Iterator
from typing import Literal, TypeAlias

from pydantic import BaseModel, JsonValue
from typing_extensions import override


class ErrorFieldAdvice(BaseModel):
    """Represents an error in field advice with an error message."""

    msg: str


class PartialError(Exception):
    """Raised when a path ends with a partial edge"""


class PartialFieldAdvice(BaseModel):
    """Represents a partial field advice result."""

    partial: Literal["partial"] = "partial"


EdgeT: TypeAlias = "BranchFieldAdvice | SingleFieldAdvice | ErrorFieldAdvice | PartialFieldAdvice | None"


def edge_has_target(edge: EdgeT) -> bool:
    return isinstance(edge, FieldAdvice)


def edge_get_target(edge: EdgeT) -> "FieldAdvice":
    if not isinstance(edge, FieldAdvice):
        raise ValueError("Edge does not have a target")
    return edge


def edge_has_error(edge: EdgeT) -> bool:
    return isinstance(edge, ErrorFieldAdvice)


def edge_get_error(edge: EdgeT) -> ErrorFieldAdvice:
    if not isinstance(edge, ErrorFieldAdvice):
        raise ValueError("Edge is not an error")
    return edge


def edge_is_partial(edge: EdgeT) -> bool:
    return isinstance(edge, PartialFieldAdvice)


def edge_is_good_path(edge: EdgeT) -> bool:
    if isinstance(edge, ErrorFieldAdvice):
        return False
    if isinstance(edge, FieldAdvice):
        return edge.has_good_options()
    # partial and None both (may) lead to a good outcome
    return True


def edge_collect_into(edge: EdgeT, field: str, error_map: dict[str, set[str]]):
    def add_error(msg: str):
        if field not in error_map:
            error_map[field] = set()
        error_map[field].add(msg)

    if edge_has_error(edge):
        add_error(edge_get_error(edge).msg)
    elif edge_has_target(edge):
        edge_get_target(edge).collect_into(error_map=error_map)


class FieldAdvice(BaseModel, ABC, Iterable[JsonValue], frozen=True):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """Abstract base class for field advice that provides guidance on valid field values."""

    @abstractmethod
    def has_good_options(self) -> bool:
        pass

    @abstractmethod
    def collect_into(self, error_map: dict[str, set[str]]):
        pass

    @abstractmethod
    def collect_error_messages(self) -> dict[str, set[str]]:
        pass

    @abstractmethod
    def select(self, value: JsonValue) -> "FieldAdvice | None":
        pass

    @abstractmethod
    @override
    def __iter__(self) -> Iterator[JsonValue]:  # pyright: ignore[reportIncompatibleMethodOverride]
        pass


class BranchFieldAdvice(FieldAdvice, frozen=True):
    """Field advice with multiple possible values (branches) for a field."""

    field: str
    branches: list[tuple[JsonValue, EdgeT]]

    @override
    def has_good_options(self) -> bool:
        return any(edge_is_good_path(edge) for _, edge in self.branches)

    @override
    def collect_into(self, error_map: dict[str, set[str]]):
        for _, edge in self.branches:
            edge_collect_into(edge, field=self.field, error_map=error_map)

    @override
    def collect_error_messages(self) -> dict[str, set[str]]:
        error_map: dict[str, set[str]] = {}
        self.collect_into(error_map=error_map)
        return error_map

    @override
    def select(self, value: JsonValue) -> "FieldAdvice | None":
        it = iter(self.branches)
        val, edge = next(it)
        while val != value:
            val, edge = next(it)
        if edge_has_error(edge):
            # if you checked that there's a good option and used that value, this shouldn't happen
            raise ValueError(f"Option resulted in error {edge_get_error(edge)}")
        if edge_is_partial(edge):
            raise PartialError()
        if edge_has_target(edge):
            return edge_get_target(edge)
        return None

    @override
    def __iter__(self) -> Iterator[JsonValue]:
        class ValueIterator(Iterator[JsonValue]):
            def __init__(self, advice: BranchFieldAdvice):
                self._branches: list[tuple[JsonValue, EdgeT]] = advice.branches
                self._it: Iterator[tuple[JsonValue, EdgeT]] = iter(advice.branches)

            @override
            def __next__(self) -> JsonValue:
                nxt_val, nxt_edge = next(self._it)
                while not edge_is_good_path(nxt_edge):
                    nxt_val, nxt_edge = next(self._it)
                return nxt_val

        return ValueIterator(self)


class SingleFieldAdvice(FieldAdvice, frozen=True):
    """Field advice with a single recommended value for a field."""

    field: str
    value: JsonValue
    edge: EdgeT = None

    @override
    def has_good_options(self) -> bool:
        return edge_is_good_path(self.edge)

    @override
    def collect_into(self, error_map: dict[str, set[str]]):
        edge_collect_into(self.edge, field=self.field, error_map=error_map)

    @override
    def collect_error_messages(self) -> dict[str, set[str]]:
        error_map: dict[str, set[str]] = {}
        self.collect_into(error_map=error_map)
        return error_map

    @override
    def select(self, value: JsonValue) -> "FieldAdvice | None":
        if edge_has_error(self.edge):
            raise ValueError(f"Option resulted in error {edge_get_error(self.edge)}")
        if edge_is_partial(self.edge):
            raise PartialError()
        if edge_has_target(self.edge):
            return edge_get_target(self.edge)
        return None

    @override
    def __iter__(self) -> Iterator[JsonValue]:
        if not edge_is_good_path(self.edge):
            raise StopIteration()
        if isinstance(self.value, Iterable) and not isinstance(self.value, str):
            return iter(self.value)
        return iter([self.value])
