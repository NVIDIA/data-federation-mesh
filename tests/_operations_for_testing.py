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

from __future__ import annotations

from typing import Literal, Optional, Union

from nv_dfm_core.api import Advise, NodeParam, Operation, PlaceParam


class NullaryOp(Operation):
    class Config:
        arbitrary_types_allowed = True

    dfm_class_name: Optional[Literal["tests._operations_for_testing.NullaryOp"]] = (
        "tests._operations_for_testing.NullaryOp"
    )
    __api_name__: Optional[Literal["testing.NullaryOp"]] = "testing.NullaryOp"


class UnaryOp(Operation):
    class Config:
        arbitrary_types_allowed = True

    dfm_class_name: Optional[Literal["tests._operations_for_testing.UnaryOp"]] = (
        "tests._operations_for_testing.UnaryOp"
    )
    __api_name__: Optional[Literal["testing.UnaryOp"]] = "testing.UnaryOp"
    param1: Union[str, NodeParam, PlaceParam, Advise]


class BinaryOp(Operation):
    class Config:
        arbitrary_types_allowed = True

    dfm_class_name: Optional[Literal["tests._operations_for_testing.BinaryOp"]] = (
        "tests._operations_for_testing.BinaryOp"
    )
    __api_name__: Optional[Literal["testing.BinaryOp"]] = "testing.BinaryOp"
    param1: Union[str, NodeParam, PlaceParam, Advise]
    param2: Union[str, NodeParam, PlaceParam, Advise]
