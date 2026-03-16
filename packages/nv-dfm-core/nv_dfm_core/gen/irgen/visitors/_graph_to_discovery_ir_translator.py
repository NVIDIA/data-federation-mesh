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

from logging import Logger

from pydantic import JsonValue
from typing_extensions import override

from nv_dfm_core.api._advise import Advise
from nv_dfm_core.api._node_id import NodeRef
from nv_dfm_core.api._place_param import PlaceParam
from nv_dfm_core.api._stop_token import StopToken
from nv_dfm_core.api._yield import DISCOVERY_PLACE_NAME, STATUS_PLACE_NAME
from nv_dfm_core.exec import any_object_to_tagged_json_value
from nv_dfm_core.exec._frame import FlowInfo
from nv_dfm_core.gen.irgen.graph import (
    BoolValue,
    Branch,
    CannotReach,
    EndFork,
    Graph,
    GraphVisitor,
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
from nv_dfm_core.gen.modgen.ir import (
    START_PLACE_NAME,
    ActivateWhenPlacesReady,
    ActivationFunction,
    AdapterDiscoveryStmt,
    BodyEndMarker,
    BodyStartMarker,
    Comment,
    InPlace,
    IRStmt,
    NetIR,
    SendDiscoveryStmt,
    SendEndMarker,
    SendStartMarker,
    TokenSend,
    Transition,
    YieldPlace,
)


class GraphToDiscoveryIRTranslator:
    def __init__(self, graph: Graph, logger: Logger):
        self.graph: Graph = graph
        self._logger: Logger = logger
        self.net_irs: dict[str, NetIR] = {}
        self.yield_places: list[YieldPlace] = [
            YieldPlace(
                kind="control",
                place=DISCOVERY_PLACE_NAME,
                type="Any",
            )
        ]

    def get_netirs(self) -> dict[str, NetIR]:
        return self.net_irs

    def get_yield_places(self) -> list[YieldPlace]:
        return self.yield_places

    def _process_site(self, site: str) -> None:
        # For discovery there's only one large transition that simply calls all the
        # discovery functions.
        visitor = ToDiscoveryIRVisitor()
        # We sort the regions to make sure we get a stable output for testing
        for region in sorted(
            self.graph.non_yield_regions_for_site(site),
            key=lambda r: r.site_local_logid(),
        ):
            region.visit_in_topological_order(visitor)

        if len(visitor.discovery_statements) == 0:
            self._logger.info(
                f"Site {site} has no discovery statements. Not generating a netIR for it."
            )
            return

        assert visitor.stop_node is not None, f"Stop node not found for site {site}"

        # returns one or two transitions for this site's dicsovery
        transitions = visitor.assemble_transitions()

        self.net_irs[site] = NetIR(
            pipeline_name=self.graph.pipeline().name,
            site=site,
            transitions=transitions,
        )

    def translate_graph(self) -> None:
        # discovery does not deal with yield places, we keep them empty

        # process each region, except for the yield
        for site in self.graph.candidate_sites():
            self._process_site(site)


class ToDiscoveryIRVisitor(GraphVisitor):
    def __init__(self):
        # This visitor runs across all transitions/regions of a site and merges them into a single transition
        # we simply collect all the adapter discovery statements
        self.discovery_statements: list[AdapterDiscoveryStmt] = []
        # However we also piggybak on the stop node, but there's only one stop leader on one site
        # for most sites, this is empty.
        self.stop_node: Stop | None = None

    def _assemble_leader_stop_transition(self) -> Transition:
        assert self.stop_node is not None, "Stop node not found"
        assert self.stop_node.state == NodeState.LEADER, "Stop node is not the leader"

        stop_region = self.stop_node.region()

        flow_in_place = stop_region.flow_in_place_after_cut()

        # The control flow place from the successor
        control_in_place: InPlace = InPlace(
            name=flow_in_place.site_local_logid(),
            kind="control",
            origin="internal",
            flavor="seq_control",
            type="nv_dfm_core.exec.FlowInfo",
        )
        # create one data place for each follower's signal
        data_in_places: list[InPlace] = []
        for place in stop_region.places():
            if place == flow_in_place:
                continue
            data_in_places.append(
                InPlace(
                    name=place.site_local_logid(),
                    kind="data",
                    origin=place.origin,
                    flavor="scoped",
                    type="Any",
                )
            )

        try_activate_func: ActivationFunction = ActivateWhenPlacesReady()

        #
        signal_error_body: list[IRStmt] = []
        # when all is done, send the final stop token
        signal_stop_body: list[IRStmt] = [
            Comment(comment="Discovery transition"),
            BodyStartMarker(),
            SendStartMarker(),
            Comment(comment="Yield stop token"),
            TokenSend(
                job=None,
                site=self.stop_node.graph().homesite(),
                place=STATUS_PLACE_NAME,
                is_yield=True,
                kind="control",
                data=StopToken(),
                node_id=None,
            ),
            SendEndMarker(),
            BodyEndMarker(),
        ]

        # send the stop token for "this discovery frame" (where at the moment we only have one,
        # but we may have more in the future)
        fire_body: list[IRStmt] = [
            Comment(comment="Discovery transition"),
            BodyStartMarker(),
            SendStartMarker(),
            Comment(comment="Yield stop token"),
            TokenSend(
                job=None,
                site=self.stop_node.graph().homesite(),
                place=STATUS_PLACE_NAME,
                is_yield=True,
                kind="control",
                data=StopToken(),
                node_id=None,
            ),
            SendEndMarker(),
            BodyEndMarker(),
        ]
        stop_transition = Transition(
            control_place=control_in_place,
            data_places=data_in_places,
            try_activate_func=try_activate_func,
            fire_body=fire_body,
            signal_error_body=signal_error_body,
            signal_stop_body=signal_stop_body,
        )
        return stop_transition

    def assemble_transitions(self) -> list[Transition]:
        """Generally returns a single transition with all the discovery calls. But
        for the site with the stop node leader we return the stop transition as well."""
        assert self.stop_node is not None, "Stop node not found"

        # assemble the main transition
        # the main transition only has a start place
        control_in_place: InPlace = InPlace(
            name=START_PLACE_NAME,
            kind="control",
            origin="external",
            flavor="seq_control",
            type="nv_dfm_core.exec.FlowInfo",
        )
        # no data in places
        data_in_places: list[InPlace] = []

        try_activate_func: ActivationFunction = ActivateWhenPlacesReady()

        #
        signal_error_body: list[IRStmt] = []
        signal_stop_body: list[IRStmt] = []

        # distinguish between leader and follower of the stop node group
        # The followers send their signal to a special region in the leader
        # We need to insert those signal sends for the followers and insert
        # a jump signal from the leader's discovery region to the stop region
        epilog_body: list[IRStmt]
        if self.stop_node.state == NodeState.LEADER:
            stop_region = self.stop_node.region()
            control_place = stop_region.flow_in_place_after_cut()
            # add a jump to the stop region
            epilog_body = [
                SendDiscoveryStmt(),
                Comment(
                    comment="This is the stop group leader, jumping to stop region"
                ),
                TokenSend(
                    job=None,
                    site=control_place.site(),
                    place=control_place.site_local_logid(),
                    is_yield=False,
                    kind=control_place.kind,
                    data=FlowInfo(hint=None),
                    node_id=None,
                ),
            ]

            # and add this jump also to the signal stop body
            signal_stop_body.append(
                TokenSend(
                    job=None,
                    site=control_place.site(),
                    place=control_place.site_local_logid(),
                    is_yield=False,
                    kind=control_place.kind,
                    data=FlowInfo(hint=None),
                    node_id=None,
                )
            )
        else:
            # each follower sends its signal to a dedicated place in the leader to sync them all
            target = self.stop_node.sync_target_after_cut()
            epilog_body = [
                SendDiscoveryStmt(),
                Comment(comment="Signal stop to stop group leader"),
                TokenSend(
                    job=None,
                    site=target.site(),
                    place=target.site_local_logid(),
                    is_yield=False,
                    kind=target.kind,
                    data=FlowInfo(hint=None),
                    node_id=None,
                ),
            ]

        fire_body = (
            [Comment(comment="Discovery transition"), BodyStartMarker()]
            + self.discovery_statements
            + [SendStartMarker()]
            + epilog_body
            + [SendEndMarker(), BodyEndMarker()]
        )
        discovery_transition = Transition(
            control_place=control_in_place,
            data_places=data_in_places,
            try_activate_func=try_activate_func,
            fire_body=fire_body,
            signal_error_body=signal_error_body,
            signal_stop_body=signal_stop_body,
        )

        # and now, in case we are the leader, create the stop transition
        if self.stop_node.state == NodeState.LEADER:
            stop_transition = self._assemble_leader_stop_transition()
            return [discovery_transition, stop_transition]
        else:
            return [discovery_transition]

    @override
    def visit_region(self, region: "Region") -> None:
        # this visitor is visiting all the regions for the site, we are only
        # generating a single region for discovery.
        pass

    @override
    def visit_place(self, place: "Place") -> None:
        # we don't care about places during discovery
        pass

    @override
    def visit_operation(self, operation: "Operation") -> None:
        # Collect the literal parameters and the parameters that come from other nodes
        # NOTE: we could get the same info in two ways: we can look for incoming edges in the graph,
        # from preceeding nodes into the field slots of the operation, or we can look for the fields
        # in the original Operation pipeline object. We do the latter here since that also gives us
        # the literal values passed by the user.
        litparams: dict[str, tuple[str, JsonValue]] = {}
        for fieldname in operation.operation.model_fields_set:
            value = getattr(operation.operation, fieldname)
            if isinstance(value, NodeRef):
                # we don't care about ssa params during discovery
                pass
            elif isinstance(value, PlaceParam):
                # We interpret PlaceParams as Advise() nodes during discovery
                litparams[fieldname] = any_object_to_tagged_json_value(Advise())
            elif fieldname not in ("site", "provider", "node_id"):
                litparams[fieldname] = any_object_to_tagged_json_value(value)

        assert not isinstance(operation.operation.provider, Advise), (
            f"Provider {operation.operation.provider} is an Advise"
        )
        stmt = AdapterDiscoveryStmt(
            provider=operation.operation.provider,
            nodeid=operation.operation.dfm_node_id.ident,
            adapter=operation.operation.__api_name__,
            literal_params=litparams,
            is_async=operation.operation_info.is_async,
        )
        self.discovery_statements.append(stmt)

    @override
    def visit_bool_value(self, bool_value: "BoolValue") -> None:
        pass

    @override
    def visit_next(self, next: "Next") -> None:
        pass

    @override
    def visit_yield(self, yield_node: "Yield") -> None:
        pass

    @override
    def visit_send(self, send: "Send") -> None:
        pass

    @override
    def visit_jump(self, jump: "Jump") -> None:
        pass

    @override
    def visit_cannot_reach(self, cannot_reach: "CannotReach") -> None:
        assert False, "This should not be visited during IR generation"

    @override
    def visit_loop(self, loop: "Loop") -> None:
        pass

    @override
    def visit_end_fork(self, end_fork: "EndFork") -> None:
        pass

    @override
    def visit_stop(self, stop: "Stop") -> None:
        # for discovery, we are merging all transitions for this site into one. But we want to
        # piggyback on the stop node mechanism. This is handled above.
        self.stop_node = stop

    @override
    def visit_branch(self, branch: "Branch") -> None:
        pass

    @override
    def visit_seq_iterate(self, seq_iterate: "SeqForEach") -> None:
        pass

    @override
    def visit_par_iterate(self, par_iterate: "ParForEach") -> None:
        pass
