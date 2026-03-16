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
from typing import Any

from pydantic import BaseModel, ConfigDict

from ._pipeline_build_helper import PipelineBuildHelper
from ._statement import Statement


class Block(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A Block is a linear list of Statements (and subclasses of Statements).
    A Statement must only reference other Expressions (and subclasses of Expression)
    that come earlier in the block, or possibly earlier in the block containing
    the statement that contains this block. I.e. you can only reference values
    that have been defined before."""

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True)  # pyright: ignore[reportIncompatibleVariableOverride]

    dfm_body: list[Statement] = []

    def add_to_body(self, stmt: Statement):
        """Adds statement to the body. Called by Statement post init when pydantic
        model has been created"""
        self.dfm_body.append(stmt)

    @abstractmethod
    def accept(self, visitor: Any) -> None:
        pass

    def __enter__(self):
        PipelineBuildHelper.push_block(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the block context and pop it from the build helper stack."""
        PipelineBuildHelper.pop_block(self)
        if exc_type is not None:
            return False
        return True
