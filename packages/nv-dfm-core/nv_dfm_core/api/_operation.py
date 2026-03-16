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
from typing import Any

from pydantic import ConfigDict
from typing_extensions import override

from ._expression import Expression
from ._located import Located


class Operation(Expression, Located, ABC):
    """An Operation is semantically similar to a remote method call on the
    containing provider object. E.g. a call to an operation named `LoadModel` defined
    on a provider `my_provider` is semantically similar to a call my_provider.LoadModel().

    An Operation without a provider is essentially a function
    call (that is, there is exactly one function with this name with no
    provider). In the code we don't distinguish between a function call
    and an operation call with no provider, however, to allow the site
    admins to be flexible; e.g. if the site admin wants to host two different
    implementations for the same function F() they can put the different
    F()'s into different providers."""

    model_config: ConfigDict = ConfigDict(frozen=True)  # pyright: ignore[reportIncompatibleVariableOverride]
    # the following fields will be overridden with constants
    # in generated operation classes, but they
    # logically belong here, and the compiler expects them here so the compiler
    # can figure out what are user-supplied fields and what are fields that are
    # DFM internal
    dfm_class_name: str
    __api_name__: str

    @override
    def accept(self, visitor: Any) -> None:
        return visitor.visit_operation(self)
