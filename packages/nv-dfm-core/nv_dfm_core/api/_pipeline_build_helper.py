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

import threading
from _thread import _local  # pyright: ignore[reportPrivateUsage]
from dataclasses import dataclass, field
from typing import Any

from ._node_id import NodeId, make_auto_id


class PipelineBuildHelper:
    """The PipelineBuildHelper is a helper for the user to simplify writing DFM code
    that is embedded inside python. Therefore, it is only providing syntactic sugar
    to the user's code but functionality wise is not important.

    We mainly want to avoid a user having to invent node ids for every node and
    to avoid the user having to manually append each statement to the surrounding
    block. We also want to enable users to use Python `with` contexts as code
    segments.

    The PipelineBuildHelper provides static methods that the other dfm.api classes call to
    find the current surrounding block and pipeline and to add themselves to this block.
    """

    # In case we have multiple threads trying to build pipelines
    _thread_local: _local = threading.local()

    @dataclass
    class BuilderScope:
        """Scope information for building a pipeline, including the pipeline being built
        and tracking of node IDs and blocks."""

        pipeline: Any  # avoiding recursive imports, we don't need type support here
        next_id_counter: int = 0
        block_stack: list[Any] = field(default_factory=list)

        def get_id_and_inc(self):
            i = self.next_id_counter
            self.next_id_counter += 1
            return i

    @classmethod
    def build_helper_active(cls) -> bool:
        return hasattr(cls._thread_local, "scope")

    @classmethod
    def enter_pipeline(cls, pipe: Any) -> None:
        """Called by Pipeline's __enter__"""
        if hasattr(cls._thread_local, "scope"):
            raise RuntimeError("Can only enter one Pipeline context at a time.")
        # create new scope
        cls._thread_local.scope = PipelineBuildHelper.BuilderScope(pipe)
        # push the pipeline as the root block (Pipeline is a block)
        cls.push_block(pipe)

    @classmethod
    def exit_pipeline(cls, pipe: Any) -> None:
        """Called by Pipeline's __exit__"""
        if not hasattr(cls._thread_local, "scope"):
            raise RuntimeError(
                "Called exit_pipeline() without preceding enter_pipeline()"
                + " or not running on client."
            )
        if cls._thread_local.scope.pipeline != pipe:
            raise RuntimeError(
                "Illegal exit_pipeline: exiting a different pipeline that was entered."
            )
        if len(cls._thread_local.scope.block_stack) != 1:
            raise RuntimeError(
                "Illegal exit_pipeline. Stack expected to only contain the pipeline's"
                + " root block but had {len(cls._thread_local.scope.block_stack)} blocks on it."
            )
        # pop root block
        pipe = cls._thread_local.scope.pipeline
        cls.pop_block(pipe)
        # delete scope
        delattr(cls._thread_local, "scope")

    @classmethod
    def get_fresh_node_id(cls) -> NodeId:
        """Generate a fresh unique node ID for the current pipeline being built."""
        if not cls.build_helper_active():
            # get_fresh_node_id is somehow called during deserialization too, when
            # the build helper is not active. So we don't raise but return something
            # that hopefully raises an exception if used as a NodeId
            raise ValueError(
                "PipelineBuildHelper.get_fresh_node_id() was called when build helper was"
                + " not active. This is not a valid node ID and should not have been used as one."
            )
        ident = cls._thread_local.scope.get_id_and_inc()
        return make_auto_id(ident)

    @classmethod
    def push_block(cls, block: Any) -> None:
        """Push a new block onto the current pipeline's block stack."""
        if not cls.build_helper_active():
            raise RuntimeError(
                "No pipeline entered or not running on client."
                + " Please embed inside `with Pipeline() as p:` context."
            )
        cls._thread_local.scope.block_stack.append(block)

    @classmethod
    def pop_block(cls, block: Any) -> None:
        """Pop a block from the current pipeline's block stack after validation."""
        if not cls.build_helper_active():
            raise RuntimeError(
                "No pipeline entered or not running on client."
                + " Please embed inside `with Pipeline() as p:` context."
            )
        if cls._thread_local.scope.block_stack[-1] != block:
            raise RuntimeError(
                "Illegal pop from block stack: popping block that was not on top."
            )
        cls._thread_local.scope.block_stack.pop()

    @classmethod
    def get_current_block(cls) -> Any:
        """Get the currently active block in the pipeline being built."""
        if not cls.build_helper_active():
            raise RuntimeError(
                "No pipeline entered or not running on client."
                + " Please embed inside `with Pipeline() as p:` context."
            )
        if len(cls._thread_local.scope.block_stack) == 0:
            raise RuntimeError("No surrounding Process or block context found.")
        return cls._thread_local.scope.block_stack[-1]
