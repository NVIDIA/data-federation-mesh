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

from nv_dfm_core.api import PipelineBuildHelper, make_auto_id


def test_builder_context_direct_calls():
    assert PipelineBuildHelper._thread_local  # pylint: disable=protected-access
    assert not hasattr(PipelineBuildHelper._thread_local, "scope")  # pylint: disable=protected-access

    class Block:
        pass

    class Pipeline(Block):
        pass

    pipe = Pipeline()

    # initially, pipe is stored, counter is 0, root block is on stack
    assert not PipelineBuildHelper.build_helper_active()
    PipelineBuildHelper.enter_pipeline(pipe)
    assert PipelineBuildHelper.build_helper_active()
    assert PipelineBuildHelper._thread_local.scope  # pylint: disable=protected-access
    assert PipelineBuildHelper._thread_local.scope.pipeline == pipe  # pylint: disable=protected-access
    assert PipelineBuildHelper._thread_local.scope.next_id_counter == 0  # pylint: disable=protected-access
    assert PipelineBuildHelper._thread_local.scope.block_stack == [pipe]  # pylint: disable=protected-access
    assert PipelineBuildHelper.get_current_block() == pipe

    # creating some node ids
    i0 = PipelineBuildHelper.get_fresh_node_id()
    assert i0 == make_auto_id(0)
    i1 = PipelineBuildHelper.get_fresh_node_id()
    assert i1 == make_auto_id(1)

    # pushing a nested block
    b1 = Block()
    PipelineBuildHelper.push_block(b1)
    assert PipelineBuildHelper._thread_local.scope.block_stack == [pipe, b1]  # pylint: disable=protected-access
    assert PipelineBuildHelper.get_current_block() == b1
    # creating a node id inside the nested block
    i2 = PipelineBuildHelper.get_fresh_node_id()
    assert i2 == make_auto_id(2)

    # exiting the block, creating some more ids
    PipelineBuildHelper.pop_block(b1)
    assert PipelineBuildHelper._thread_local.scope.block_stack == [pipe]  # pylint: disable=protected-access
    assert PipelineBuildHelper.get_current_block() == pipe
    i3 = PipelineBuildHelper.get_fresh_node_id()
    assert i3 == make_auto_id(3)

    # done, everything is cleaned up
    PipelineBuildHelper.exit_pipeline(pipe)
    assert not hasattr(PipelineBuildHelper._thread_local, "scope")  # pylint: disable=protected-access
