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

from typing_extensions import override


class Cost:
    """Represents the cost of executing a graph node in terms of time and data size."""

    def __init__(self, time_: float, size_: int):
        self._time: float = time_
        self._size: int = size_

    @classmethod
    def zero(cls) -> "Cost":
        return Cost(0, 0)

    @property
    def time(self) -> float:
        return self._time

    @property
    def size(self) -> int:
        return self._size

    @property
    def value(self) -> tuple[float, int]:
        return self._time, self._size

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self._time + other._time, self._size + other._size)

    def __lt__(self, other: "Cost") -> bool:
        return self._time < other._time or (
            self._time == other._time and self._size < other._size
        )

    def __gt__(self, other: "Cost") -> bool:
        return self._time > other._time or (
            self._time == other._time and self._size > other._size
        )

    def __le__(self, other: "Cost") -> bool:
        return self._time <= other._time or (
            self._time == other._time and self._size <= other._size
        )

    def __ge__(self, other: "Cost") -> bool:
        return self._time >= other._time or (
            self._time == other._time and self._size >= other._size
        )

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Cost):
            return False
        return self._time == other._time and self._size == other._size

    @override
    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, Cost):
            return True
        return self._time != other._time or self._size != other._size

    @override
    def __repr__(self) -> str:
        return f"Cost(t={self._time}, s={self._size})"

    @override
    def __str__(self) -> str:
        return f"Cost(t={self._time}, s={self._size})"

    @classmethod
    def sum(cls, costs: list["Cost"]) -> "Cost":
        """Calculate the sum of multiple costs by accumulating time and size."""
        acc = cls.zero()
        for cost in costs:
            acc += cost
        return acc
