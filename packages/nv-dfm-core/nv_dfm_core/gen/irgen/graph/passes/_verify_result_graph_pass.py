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

from nv_dfm_core.gen.modgen.ir._in_place import START_PLACE_NAME

from .._graph import GraphState, GraphTransformPass
from .._graph_elements import (
    Branch,
    EndFork,
    ExitNode,
    Jump,
    Loop,
    Next,
    NodeState,
    Operation,
    ParForEach,
    Place,
    Region,
    Send,
    SeqForEach,
    Stop,
    Yield,
)


class VerifyResultGraphPass(GraphTransformPass):
    """
    Remove all nodes that are not selected.
    """

    @override
    def apply(self) -> None:
        used_sites: set[str] = set()
        start_places: dict[str, int] = {}
        stop_nodes: dict[str, int] = {}

        for node in self._graph.get_all_nodes_copy():
            # we don't count the yield region or the yield places as using the site
            if (
                node.region() != self._graph.yield_region()
                and node != self._graph.yield_region()
            ):
                used_sites.add(node.site())
            # Note: SELECTED is only meaningful until the Prune pass. All nodes created afterwards don't really care
            # about the state field anymore, so we don't need to check it here.
            # assert node.state == NodeState.SELECTED or node.state == NodeState.LEADER, f"Node {node.global_id()} is not SELECTED or LEADER but {node.state}"

            # test for all exit nodes together: their out edges should all be to sends
            if isinstance(node, ExitNode):
                for out_slot in node.out_slots().values():
                    for out_edge in out_slot.outgoing_edges():
                        assert isinstance(out_edge.dst().node(), Send), (
                            f"Branch {node.global_id()} {out_slot.name()} is connected to a {out_edge.dst().node()} but should be connected to a Send"
                        )

            if isinstance(node, Place) and node.site_local_logid() == START_PLACE_NAME:
                if "__flow__" in node.site_local_logid():
                    assert node.flavor == "control", (
                        f"Flow place {node.global_id()} has flavor {node.flavor}, expected control"
                    )

                if node.site() not in start_places:
                    start_places[node.site()] = 0
                start_places[node.site()] += 1
                if node.origin == "internal":
                    assert len(node.in_slot("__data__").incoming_edges()) >= 1, (
                        f"External Place {node.global_id()} __data__ is not connected"
                    )
                    for in_edge in node.in_slot("__data__").incoming_edges():
                        assert isinstance(in_edge.src().node(), Send), (
                            f"Operation {node.global_id()} __data__ is connected to a {in_edge.src().node()} but should be connected to a Send"
                        )
                # Note: a place doesn't need to be connected to anything. Often they are just used for synchronization of the
                # transition's activation
            elif isinstance(node, Send):
                if node.literal_data is not None:
                    assert len(node.in_slot("value").incoming_edges()) == 0, (
                        f"Send {node.site_local_logid()} has a literal value, shouldn't have any incoming edges"
                    )
                else:
                    assert len(node.in_slot("value").incoming_edges()) == 1, (
                        f"Send {node.site_local_logid()} has multiple incoming edges"
                    )
            elif isinstance(node, Stop):
                if node.site() not in stop_nodes:
                    stop_nodes[node.site()] = 0
                stop_nodes[node.site()] += 1
            elif isinstance(node, Region):
                if node == self._graph.yield_region():
                    # make sure it's not connected
                    assert len(node.in_slot("__flow__").incoming_edges()) == 0, (
                        f"Yield Region {node.global_id()} should not be connected"
                    )
                else:
                    # make sure it's connected
                    assert len(node.in_slot("__flow__").incoming_edges()) >= 1, (
                        f"Region {node.global_id()} is not connected"
                    )
                assert node.exit_node() is not None, (
                    f"Region {node.global_id()} has no exit node"
                )
            elif isinstance(node, Branch):
                # make sure it's connected
                assert len(node.out_slot("__branch_taken__").outgoing_edges()) >= 1, (
                    f"Branch {node.global_id()} __branch_taken__ is not connected"
                )
                assert (
                    len(node.out_slot("__branch_not_taken__").outgoing_edges()) >= 1
                ), f"Branch {node.global_id()} __branch_not_taken__ is not connected"
                # check that condition input is connected
                if node.state == NodeState.LEADER:
                    assert len(node.in_slot("__condition__").incoming_edges()) == 1, (
                        f"Leader Branch {node.global_id()} is not connected"
                    )
                    assert len(node.out_slot("__sync__").outgoing_edges()) >= 1, (
                        f"Leader Branch {node.global_id()} __sync__ is not connected"
                    )
                else:
                    assert len(node.in_slot("__condition__").incoming_edges()) == 1, (
                        f"Follower Branch {node.global_id()} is connected but shouldn't be"
                    )
                    assert len(node.out_slot("__sync__").outgoing_edges()) == 0, (
                        f"Follower Branch {node.global_id()} __sync__ is connected but shouldn't be"
                    )
            elif isinstance(node, SeqForEach):
                # make sure it's connected
                assert len(node.out_slot("__next_iteration__").outgoing_edges()) >= 1, (
                    f"SeqForEach {node.global_id()} __next_iteration__ is not connected"
                )
                assert len(node.out_slot("__stop_iteration__").outgoing_edges()) >= 1, (
                    f"SeqForEach {node.global_id()} __stop_iteration__ is not connected"
                )
                # check that condition input is connected
                if node.state == NodeState.LEADER:
                    assert len(node.in_slot("__has_next__").incoming_edges()) == 1, (
                        f"Leader SeqForEach {node.global_id()} __has_next__ is not connected"
                    )
                    assert len(node.out_slot("__sync__").outgoing_edges()) >= 1, (
                        f"Leader SeqForEach {node.global_id()} __sync__ is not connected"
                    )
                else:
                    assert len(node.in_slot("__has_next__").incoming_edges()) == 1, (
                        f"Follower SeqForEach {node.global_id()} __has_next__ is connected but shouldn't be"
                    )
                    assert len(node.out_slot("__sync__").outgoing_edges()) == 0, (
                        f"Follower SeqForEach {node.global_id()} __sync__ is connected but shouldn't be"
                    )

            elif isinstance(node, ParForEach):
                # make sure it's connected
                assert len(node.out_slot("__fork__").outgoing_edges()) >= 1, (
                    f"ParForEach {node.global_id()} __fork__ is not connected"
                )
                assert len(node.out_slot("__stop_iteration__").outgoing_edges()) >= 1, (
                    f"ParForEach {node.global_id()} __stop_iteration__ is not connected"
                )
                # check that condition input is connected
                if node.state == NodeState.LEADER:
                    assert len(node.in_slot("__has_next__").incoming_edges()) == 1, (
                        f"Leader ParForEach {node.global_id()} __has_next__ is not connected"
                    )
                    assert len(node.out_slot("__sync__").outgoing_edges()) >= 1, (
                        f"Leader ParForEach {node.global_id()} __sync__ is not connected"
                    )
                else:
                    assert len(node.in_slot("__has_next__").incoming_edges()) == 1, (
                        f"Follower ParForEach {node.global_id()} __has_next__ is connected but shouldn't be"
                    )
                    assert len(node.out_slot("__sync__").outgoing_edges()) == 0, (
                        f"Follower ParForEach {node.global_id()} __sync__ is connected but shouldn't be"
                    )
            elif isinstance(node, Jump):
                # make sure it's connected
                assert len(node.out_slot("__flow__").outgoing_edges()) == 1, (
                    f"Jump {node.global_id()} is not connected"
                )
            elif isinstance(node, Loop):
                # make sure it's connected
                assert len(node.out_slot("__flow__").outgoing_edges()) == 1, (
                    f"Jump {node.global_id()} is not connected"
                )
            elif isinstance(node, EndFork):
                # make sure it's connected
                assert len(node.out_slot("__fork_frame__").outgoing_edges()) == 1, (
                    f"EndFork {node.global_id()} is not connected"
                )
            elif isinstance(node, Operation):
                for slot in node.in_slots().values():
                    # the control slots aren't necessarily connected
                    if slot.kind() == "data":
                        assert len(slot.incoming_edges()) >= 1, (
                            f"Operation {node.global_id()} {slot.name()} is not connected"
                        )
            elif isinstance(node, Yield):
                if "value" in node.in_slots():
                    assert len(node.in_slot("value").incoming_edges()) == 1, (
                        f"Yield {node.global_id()} input value is not connected"
                    )

                assert len(node.out_slot("__data__").outgoing_edges()) == 1, (
                    f"Yield {node.global_id()} output __data__ is not connected"
                )
            elif isinstance(node, Next):
                assert len(node.in_slot("__iterator__").incoming_edges()) == 1, (
                    f"Next {node.global_id()} __iterator__ is not connected"
                )
                assert len(node.out_slot("__has_next__").outgoing_edges()) == 1, (
                    f"Next {node.global_id()} output __has_next__ is not connected"
                )

        for site in used_sites:
            assert site in start_places, f"Site {site} has no start place"
            assert site in stop_nodes, f"Site {site} has no stop node"

            assert start_places[site] == 1, (
                f"Site {site} has {start_places[site]} start places, expected 1"
            )
            assert stop_nodes[site] == 1, (
                f"Site {site} has {stop_nodes[site]} stop nodes, expected 1"
            )

    @override
    def target_state(self) -> GraphState:
        return GraphState.VERIFIED
