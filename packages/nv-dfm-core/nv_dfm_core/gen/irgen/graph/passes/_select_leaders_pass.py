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
from typing_extensions import override

from .._graph import GraphState, GraphTransformPass
from .._graph_elements import (
    Branch,
    NodeState,
    ParForEach,
    SeqForEach,
    Stop,
)


class SelectLeadersPass(GraphTransformPass):
    """Each group of corresponding exit nodes selects a leader. The leader controls the flow
    across all sites for this logical exit node. For example, the leader in a branch group
    is the one checking the condition and then syncing with the followers."""

    def _handle_branch_group(self, group: list[Branch]) -> None:
        branch_leader: Branch | None = None
        for branch in group:
            # find the one branch node that still has an incoming edge. The others should have been pruned.
            assert isinstance(branch, Branch), (
                f"Node {branch.global_id()} is not a Branch"
            )
            cond_slot = branch.in_slot("__condition__")
            if len(cond_slot.incoming_edges()) > 0:
                assert len(cond_slot.incoming_edges()) == 1, (
                    f"Branch {branch.global_id()} has {len(cond_slot.incoming_edges())} incoming edges, expected 1"
                )
                assert branch_leader is None, (
                    f"Multiple leaders found for {group[0].site_local_logid()}"
                )
                branch_leader = branch
        assert branch_leader is not None, (
            f"No leader found for {group[0].site_local_logid()}"
        )
        # promote from SELECTED to LEADER
        if branch_leader is not None:
            branch_leader.state = NodeState.LEADER

    def _handle_seq_iterate_group(self, group: list[SeqForEach]) -> None:
        seq_iterate_leader: SeqForEach | None = None
        for seq_iterate in group:
            # find the one seq_iterate node that still has an incoming edge. The others should have been pruned.
            assert isinstance(seq_iterate, SeqForEach), (
                f"Node {seq_iterate.global_id()} is not a SeqIterate"
            )
            cond_slot = seq_iterate.in_slot("__has_next__")
            if len(cond_slot.incoming_edges()) > 0:
                assert len(cond_slot.incoming_edges()) == 1, (
                    f"SeqIterate {seq_iterate.global_id()} has {len(cond_slot.incoming_edges())} incoming edges, expected 1"
                )
                assert seq_iterate_leader is None, (
                    f"Multiple leaders found for {group[0].site_local_logid()}"
                )
                seq_iterate_leader = seq_iterate
        assert seq_iterate_leader is not None, (
            f"No leader found for {group[0].site_local_logid()}"
        )
        # promote from SELECTED to LEADER
        if seq_iterate_leader is not None:
            seq_iterate_leader.state = NodeState.LEADER

    def _handle_par_iterate_group(self, group: list[ParForEach]) -> None:
        par_iterate_leader: ParForEach | None = None
        for par_iterate in group:
            # find the one par_iterate node that still has an incoming edge. The others should have been pruned.
            assert isinstance(par_iterate, ParForEach), (
                f"Node {par_iterate.global_id()} is not a ParForEach"
            )
            cond_slot = par_iterate.in_slot("__has_next__")
            if len(cond_slot.incoming_edges()) > 0:
                assert len(cond_slot.incoming_edges()) == 1, (
                    f"ParForEach {par_iterate.global_id()} has {len(cond_slot.incoming_edges())} incoming edges, expected 1"
                )
                assert par_iterate_leader is None, (
                    f"Multiple leaders found for {group[0].site_local_logid()}"
                )
                par_iterate_leader = par_iterate
        assert par_iterate_leader is not None, (
            f"No leader found for {group[0].site_local_logid()}"
        )
        # promote from SELECTED to LEADER
        if par_iterate_leader is not None:
            par_iterate_leader.state = NodeState.LEADER

    def _handle_stop_group(self, group: list[Stop]) -> None:
        # Doesn't really matter who is leader, but pick one that looks sensible.
        stop_leader: Stop | None = None
        for stop in group:
            assert isinstance(stop, Stop), f"Node {stop.global_id()} is not a Stop"
            # first, find the stop node with the most nodes in its region
            if not stop_leader:
                stop_leader = stop
            # favor the larger regions
            elif stop.region().size() > stop_leader.region().size():
                stop_leader = stop
            # but if there's a tie use the site name to get a stable ordering
            elif (
                stop.region().size() == stop_leader.region().size()
                and stop.region().site() < stop_leader.region().site()
            ):
                stop_leader = stop
        if stop_leader is None:
            raise RuntimeError(f"No leader found for {group[0].site_local_logid()}")
        if stop_leader.region().size() == 0:
            # all stop node regions were empty,
            # Therefore pick the one from the site with the most nodes total
            num_nodes_by_site = self._graph.num_operation_nodes_by_site()
            largest = 0
            for stop in group:
                assert isinstance(stop, Stop), f"Node {stop.global_id()} is not a Stop"
                assert stop.site() in num_nodes_by_site, (
                    f"Site {stop.site()} not in num_nodes_by_site"
                )
                if num_nodes_by_site[stop.site()] > largest:
                    largest = num_nodes_by_site[stop.site()]
                    stop_leader = stop
        assert stop_leader is not None, (
            f"No leader found for {group[0].site_local_logid()}"
        )
        # promote from SELECTED to LEADER
        if stop_leader is not None:
            stop_leader.state = NodeState.LEADER

    @override
    def apply(self) -> None:
        """After pruning, branches don't have any inputs anymore. Those will be followers
        For stop nodes we just pick any that looks sensible.
        """
        for group in self._graph.get_nodes_by_logids().values():
            if len(group) == 0:
                continue
            if isinstance(group[0], Branch):
                self._handle_branch_group(group)  # pyright: ignore[reportArgumentType]
            elif isinstance(group[0], SeqForEach):
                self._handle_seq_iterate_group(group)  # pyright: ignore[reportArgumentType]
            elif isinstance(group[0], ParForEach):
                self._handle_par_iterate_group(group)  # pyright: ignore[reportArgumentType]
            elif isinstance(group[0], Stop):
                self._handle_stop_group(group)  # pyright: ignore[reportArgumentType]

    @override
    def target_state(self) -> GraphState:
        return GraphState.LEADERS_SELECTED
