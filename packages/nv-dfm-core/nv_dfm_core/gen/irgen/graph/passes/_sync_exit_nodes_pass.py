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

# pyright: reportPrivateUsage=false
from typing import Any

from typing_extensions import override

from .._graph import GraphState, GraphTransformPass
from .._graph_elements import (
    Branch,
    FlowEdge,
    Jump,
    NodeState,
    ParForEach,
    Region,
    SeqForEach,
    Stop,
)


class SyncExitNodesPass(GraphTransformPass):
    """Transform the graph to actually implement the leader/follower pattern for the exit node
    groups. For example, for a Jump there's nothing to synchronize, each site jumps individually.
    However, for a branch, the branch leader sends a message to the followers about which path
    to take. To translate this into petri net transitions, we move the follower branch into
    its own transition, awaiting the flow from the previous region plus the sync message from
    the leader. This "moving into its own transition" is done in this pass."""

    def _find_leader(self, nodes: list[Any]) -> tuple[Any, list[Any]]:
        leader: Any | None = None
        followers: list[Any] = []
        for node in nodes:
            if node.state == NodeState.LEADER:
                assert leader is None, (
                    f"Multiple leaders found for {node.site_local_logid()}"
                )
                leader = node
            else:
                followers.append(node)
        assert leader is not None, f"No leader found for {nodes[0].site_local_logid()}"
        return leader, followers

    def _handle_branch_group(self, nodes: list[Branch]) -> None:
        # separate leader and followers
        assert len(nodes) > 0, "No exit nodes to synchronize"
        leader, followers = self._find_leader(nodes)

        for follower in followers:
            # for each follower branch (which shouldn't have any incoming edges anymore), change the graph:
            old_region = follower.region()
            # Create a new region with the follower branch as the exit node
            new_region = Region(
                graph=self._graph,
                site=follower.site(),
                site_local_logid=self._graph._get_fresh_per_nodetype_logid(Region),
                is_loop_head=False,
            )
            follower.migrate(new_region)
            # Replace the branch exit node in the old region with a Jump to the new region
            jump = Jump(self._graph, old_region)
            _ = FlowEdge(
                src=jump.out_slot("__flow__"),
                dst=new_region.in_slot("__flow__"),
                send_cost_info=self._graph._fed_info.find_send_cost(
                    jump.site(), new_region.site(), logger=self._graph.logger
                ),
                is_weak=False,
            )
            # create an edge from the leader to the follower branch
            _ = FlowEdge(
                src=leader.out_slot("__sync__"),
                dst=follower.in_slot("__condition__"),
                send_cost_info=self._graph._fed_info.find_send_cost(
                    leader.site(), follower.site(), logger=self._graph.logger
                ),
                is_weak=False,
            )
            # This introduced new edges which will be cut with send/receive nodes later

    def _handle_seq_iterate_group(self, nodes: list[SeqForEach]) -> None:
        # separate leader and followers
        assert len(nodes) > 0, "No exit nodes to synchronize"
        leader, followers = self._find_leader(nodes)

        for follower in followers:
            # for each follower branch (which shouldn't have any incoming edges anymore), change the graph:
            old_region = follower.region()
            # Create a new region with the follower branch as the exit node
            new_region = Region(
                graph=self._graph,
                site=follower.site(),
                site_local_logid=self._graph._get_fresh_per_nodetype_logid(Region),
                is_loop_head=False,
            )
            follower.migrate(new_region)
            # Replace the branch exit node in the old region with a Jump to the new region
            jump = Jump(self._graph, old_region)
            _ = FlowEdge(
                src=jump.out_slot("__flow__"),
                dst=new_region.in_slot("__flow__"),
                send_cost_info=self._graph._fed_info.find_send_cost(
                    jump.site(), new_region.site(), logger=self._graph.logger
                ),
                is_weak=False,
            )
            # create an edge from the leader to the follower branch
            _ = FlowEdge(
                src=leader.out_slot("__sync__"),
                dst=follower.in_slot("__has_next__"),
                send_cost_info=self._graph._fed_info.find_send_cost(
                    leader.site(), follower.site(), logger=self._graph.logger
                ),
                is_weak=False,
            )
            # This introduced new edges which will be cut with send/receive nodes later

    def _handle_par_foreach_group(self, nodes: list[ParForEach]) -> None:
        # separate leader and followers
        assert len(nodes) > 0, "No exit nodes to synchronize"
        leader, followers = self._find_leader(nodes)

        for follower in followers:
            # for each follower branch (which shouldn't have any incoming edges anymore), change the graph:
            old_region = follower.region()
            # Create a new region with the follower branch as the exit node
            new_region = Region(
                graph=self._graph,
                site=follower.site(),
                site_local_logid=self._graph._get_fresh_per_nodetype_logid(Region),
                is_loop_head=False,
            )
            follower.migrate(new_region)
            # Replace the branch exit node in the old region with a Jump to the new region
            jump = Jump(self._graph, old_region)
            _ = FlowEdge(
                src=jump.out_slot("__flow__"),
                dst=new_region.in_slot("__flow__"),
                send_cost_info=self._graph._fed_info.find_send_cost(
                    jump.site(), new_region.site(), logger=self._graph.logger
                ),
                is_weak=False,
            )
            # create an edge from the leader to the follower branch
            _ = FlowEdge(
                src=leader.out_slot("__sync__"),
                dst=follower.in_slot("__has_next__"),
                send_cost_info=self._graph._fed_info.find_send_cost(
                    leader.site(), follower.site(), logger=self._graph.logger
                ),
                is_weak=False,
            )
            # This introduced new edges which will be cut with send/receive nodes later

    def _handle_stop_group(self, nodes: list[Stop]) -> None:
        # Very similar to Branch synchronization, but the other way around:
        # a stop is a Gather sync, a branch is a Scatter sync
        assert len(nodes) > 0, "No stops to synchronize"
        leader = None
        followers: list[Stop] = []
        for node in nodes:
            assert isinstance(node, Stop), f"Node {node.global_id()} is not a Stop"
            if node.state == NodeState.LEADER:
                assert leader is None, (
                    f"Multiple leaders found for {node.site_local_logid()}"
                )
                leader = node
            else:
                followers.append(node)

        if leader is None:
            raise RuntimeError(f"No leader found for {nodes[0].site_local_logid()}")

        old_region = leader.region()
        # create a new region with the leader as the exit node
        new_region = Region(
            self._graph,
            leader.site(),
            self._graph._get_fresh_per_nodetype_logid(Region),
            is_loop_head=False,
        )
        leader.migrate(new_region)
        # replace the old exit node with a jump
        jump = Jump(self._graph, old_region)
        _ = FlowEdge(
            src=jump.out_slot("__flow__"),
            dst=new_region.in_slot("__flow__"),
            send_cost_info=self._graph._fed_info.find_send_cost(
                jump.site(), new_region.site(), logger=self._graph.logger
            ),
            is_weak=False,
        )
        # create an edge from each follower to leader
        for stop in followers:
            in_slot_name = f"__stop_from_{stop.site()}__"
            leader.create_sync_input_slot(in_slot_name)
            _ = FlowEdge(
                src=stop.out_slot("__sync__"),
                dst=leader.in_slot(in_slot_name),
                send_cost_info=self._graph._fed_info.find_send_cost(
                    stop.site(), leader.site(), logger=self._graph.logger
                ),
                is_weak=False,
            )

    @override
    def apply(self) -> None:
        # clone, because we are modifying the graph in place
        nodes_by_logid = dict(self._graph.get_nodes_by_logids())
        for nodes in nodes_by_logid.values():
            if len(nodes) == 0:
                continue
            elif isinstance(nodes[0], Jump):
                # jumps don't synchronize. nothing to do
                continue
            elif isinstance(nodes[0], Stop):
                self._handle_stop_group(nodes)  # pyright: ignore[reportArgumentType]
            elif isinstance(nodes[0], Branch):
                self._handle_branch_group(nodes)  # pyright: ignore[reportArgumentType]
            elif isinstance(nodes[0], SeqForEach):
                self._handle_seq_iterate_group(nodes)  # pyright: ignore[reportArgumentType]
            elif isinstance(nodes[0], ParForEach):
                self._handle_par_foreach_group(nodes)  # pyright: ignore[reportArgumentType]
            else:
                # not an exit node, nothing to do
                continue

    @override
    def target_state(self) -> GraphState:
        return GraphState.EXITS_ARE_SYNCED
