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

# pyright: reportUnusedParameter=false, reportImportCycles=false

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._for_each import ForEach
    from ._if import If
    from ._operation import Operation
    from ._pipeline import Pipeline
    from ._try_from_cache import TryFromCache, WriteToCache
    from ._yield import Yield
else:
    Pipeline = Any
    Yield = Any
    If = Any
    ForEach = Any
    TryFromCache = Any
    WriteToCache = Any
    Operation = Any


class ApiVisitor:
    """Base visitor class for traversing DFM API models.
    All visit methods have empty implementations by default.
    Subclasses should override the methods they are interested in."""

    def visit_pipeline(self, pipeline: Pipeline) -> None:
        """Visit a Pipeline instance."""
        pass

    def visit_yield(self, yield_stmt: Yield) -> None:
        """Visit a Yield statement."""
        pass

    def visit_if(self, if_stmt: If) -> None:
        """Visit an If statement."""
        pass

    def visit_for_each(self, for_each_stmt: ForEach) -> None:
        """Visit a ForEach statement."""
        pass

    def visit_try_from_cache(self, try_from_cache_stmt: TryFromCache) -> None:
        """Visit a ForEach statement."""
        pass

    def visit_write_to_cache(self, write_to_cache_stmt: WriteToCache) -> None:
        """Visit a ForEach statement."""
        pass

    def visit_operation(self, operation: Operation) -> None:
        """Visit an Operation instance."""
        pass
