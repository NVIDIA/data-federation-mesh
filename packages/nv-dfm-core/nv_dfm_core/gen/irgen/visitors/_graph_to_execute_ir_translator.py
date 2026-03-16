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
from nv_dfm_core.api._yield import STATUS_PLACE_NAME
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
    Operation,
    ParForEach,
    Place,
    Region,
    Send,
    SeqForEach,
    Stop,
    Yield,
)
from nv_dfm_core.gen.irgen.graph._graph_elements import Next, NodeState
from nv_dfm_core.gen.modgen.ir import (
    ActivateWhenPlacesReady,
    ActivationFunction,
    AdapterCallStmt,
    BoolValueStmt,
    InPlace,
    IRStmt,
    NetIR,
    ReadPlaceStmt,
    StmtRef,
    TokenSend,
    Transition,
    YieldPlace,
)
from nv_dfm_core.gen.modgen.ir._ir_stmts import (
    BodyEndMarker,
    BodyStartMarker,
    BranchStmt,
    Comment,
    DrainIteratorStmt,
    ForkFromIteratorStmt,
    SendEndMarker,
    SendStartMarker,
)


class GraphToExecuteIRTranslator:
    def __init__(self, graph: Graph, logger: Logger):
        self.graph: Graph = graph
        self._logger: Logger = logger
        self.net_irs: dict[str, NetIR] = {}
        self.yield_places: list[YieldPlace] = []

    def get_netirs(self) -> dict[str, NetIR]:
        return self.net_irs

    def get_yield_places(self) -> list[YieldPlace]:
        return self.yield_places

    def _process_region(self, region: Region) -> Transition:
        # each region becomes a single transition
        assert region != self.graph.yield_region()

        visitor = ToExecuteIRVisitor()
        region.visit_in_topological_order(visitor)
        assert visitor.transition_docstring is not None, "Transition docstring not set"
        assert visitor.control_in_place is not None, "Control in place not set"

        transition = visitor.assemble_transition()
        return transition

    def _process_site(self, site: str) -> None:
        # Each site becomes a single NetIR object
        transitions: list[Transition] = []
        for region in self.graph.non_yield_regions_for_site(site):
            transitions.append(self._process_region(region))

        if len(transitions) == 0:
            self._logger.info(f"Site {site} has no transitions")
            return

        self.net_irs[site] = NetIR(
            pipeline_name=self.graph.pipeline().name,
            site=site,
            transitions=transitions,
        )

    def _process_yield_places(self, yield_region: Region) -> None:
        for place in yield_region.places():
            yield_place = YieldPlace(
                kind=place.kind,
                place=place.site_local_logid(),
                type=place.type,
            )
            self.yield_places.append(yield_place)

    def translate_graph(self) -> None:
        # collect yield places
        self._process_yield_places(self.graph.yield_region())

        # process each region, except for the yield
        for site in self.graph.candidate_sites():
            self._process_site(site)


class ToExecuteIRVisitor(GraphVisitor):
    def __init__(self):
        self.transition_docstring: str | None = None
        self.control_in_place_name: str | None = None
        self.control_in_place: InPlace | None = None
        self.data_in_places: list[InPlace] = []

        self.try_activate_func: ActivationFunction = ActivateWhenPlacesReady()
        self.fire_main_body: list[IRStmt] = []
        self.fire_epilog_body: list[IRStmt] = []

        self.control_stop_out_places: list[Place] = []
        self.error_propagation_out_places: list[Place] = []

    def assemble_transition(self) -> Transition:
        assert self.transition_docstring is not None, "Transition docstring not set"
        assert self.try_activate_func is not None, "Try activate function not set"
        assert self.control_in_place is not None, "Control in place not set"

        fire_body = (
            [Comment(comment=self.transition_docstring), BodyStartMarker()]
            + self.fire_main_body
            + [SendStartMarker()]
            + self.fire_epilog_body
            + [SendEndMarker()]
            + [BodyEndMarker()]
        )

        # assemble the signal stop body
        # the received frame is a stop frame, we just send it on to every
        # outgoing control place
        signal_stop_body: list[IRStmt] = []
        for place in self.control_stop_out_places:
            signal_stop_body.append(
                TokenSend(
                    job=None,
                    site=place.site(),
                    place=place.site_local_logid(),
                    is_yield=place.flavor == "yield",
                    kind=place.kind,
                    data=FlowInfo(hint=None)
                    if place.flavor != "yield"
                    else StopToken(),
                    node_id=None,
                )
            )

        # assemble the signal error body
        signal_error_body: list[IRStmt] = []
        for place in self.error_propagation_out_places:
            signal_error_body.append(
                TokenSend(
                    job=None,
                    site=place.site(),
                    place=place.site_local_logid(),
                    is_yield=place.flavor == "yield",
                    kind=place.kind,
                    data=StmtRef(stmt_id="_error_"),
                    node_id=None,
                )
            )

        return Transition(
            control_place=self.control_in_place,
            data_places=self.data_in_places,
            try_activate_func=self.try_activate_func,
            fire_body=fire_body,
            signal_error_body=signal_error_body,
            signal_stop_body=signal_stop_body,
        )

    def _node_ref_to_stmt_ref(self, node_ref: NodeRef) -> StmtRef:
        """NodeRefs come from the original Pipeline nodes. For pipeline nodes that are
        expressions (and therefore define values and therefore are the only nodes that can
        be used as references in other nodes params), we use the dfm_node_id as the site_local_logid
        and therefore both are equal"""
        return StmtRef(
            stmt_id=node_ref.id.as_identifier(),
            sel=node_ref.sel,
            issubs=node_ref.issubs,
        )

    @override
    def visit_region(self, region: "Region") -> None:
        # this is the entry point for this visitor
        place = region.in_slot("__flow__").incoming_edges()[0].src().node()
        assert isinstance(place, Place), "Control in place is not a place"
        self.control_in_place_name = place.site_local_logid()

    @override
    def visit_place(self, place: "Place") -> None:
        assert self.control_in_place_name is not None, "Control in place name not set"

        if place.flavor == "control":
            flavor = "loop_control" if place.region().is_loop_head() else "seq_control"
        else:
            flavor = place.flavor
        if place.site_local_logid() == self.control_in_place_name:
            in_place = InPlace(
                name=place.site_local_logid(),
                kind="control",
                origin=place.origin,
                flavor=flavor,
                type=place.type,
            )
            self.control_in_place = in_place
        else:
            # all the other places, can be data or control (at least in theory)
            in_place = InPlace(
                name=place.site_local_logid(),
                kind=place.kind,
                origin=place.origin,
                flavor=flavor,
                type=place.type,
            )
            self.data_in_places.append(in_place)

        # emit a "my_place = data['my_place']" statement in the body
        if place.flavor != "control":
            self.fire_main_body.append(
                ReadPlaceStmt(
                    stmt_id=place.site_local_logid(),
                    place=place.site_local_logid(),
                )
            )

    @override
    def visit_operation(self, operation: "Operation") -> None:
        # Collect the literal parameters and the parameters that come from other nodes
        # NOTE: we could get the same info in two ways: we can look for incoming edges in the graph,
        # from preceding nodes into the field slots of the operation, or we can look for the fields
        # in the original Operation pipeline object. We do the latter here since that also gives us
        # the literal values passed by the user.
        litparams: dict[str, tuple[str, JsonValue]] = {}
        ssaparams: dict[str, StmtRef] = {}
        # We used to only emit .model_fields_set, but that makes difficulties with default params
        # since the Operations default may differ from the adapter default. Therefore, we output
        # all the fields, including the ones that still have their default values.
        for fieldname in operation.operation.__class__.model_fields:
            value = getattr(operation.operation, fieldname)
            if isinstance(value, NodeRef):
                # we need to rewrite the name of the original param from the pipeline
                # because the value may not be directly from this source anymore but from a
                # place that has been inserted.
                source_node = operation.param_source_after_cut(fieldname)
                new_name = source_node.site_local_logid()
                # but we keep the original selector
                ssaparams[fieldname] = StmtRef(
                    stmt_id=new_name, sel=value.sel, issubs=value.issubs
                )
                assert len(operation.in_slot(fieldname).incoming_edges()) == 1, (
                    f"Operation {operation.global_id()} has {len(operation.in_slot(fieldname).incoming_edges())} incoming edges for nodref field {fieldname}. Expected 1."
                )
            elif isinstance(value, PlaceParam):
                # Similar to NodeRef, we need to get the actual place name after edge cutting
                # because the place may have been renamed during cross-site edge cutting
                source_node = operation.param_source_after_cut(fieldname)
                ssa_ref = StmtRef(stmt_id=source_node.site_local_logid())
                ssaparams[fieldname] = ssa_ref
                assert len(operation.in_slot(fieldname).incoming_edges()) == 1, (
                    f"Operation {operation.global_id()} has {len(operation.in_slot(fieldname).incoming_edges())} incoming edges for a place param field {fieldname}. Expected 1. {operation.in_slot(fieldname).incoming_edges()}"
                )
            elif fieldname not in (
                "site",
                "provider",
                "dfm_class_name",
                "dfm_after",
                "dfm_node_id",
            ):
                litparams[fieldname] = any_object_to_tagged_json_value(value)

        assert not isinstance(operation.operation.provider, Advise), (
            f"Provider {operation.operation.provider} is an Advise"
        )
        stmt = AdapterCallStmt(
            provider=operation.operation.provider,
            stmt_id=operation.site_local_logid(),
            has_users=operation.has_users(),
            adapter=operation.operation.__api_name__,
            literal_params=litparams,
            stmt_params=ssaparams,
            is_async=operation.operation_info.is_async,
        )
        self.fire_main_body.append(stmt)

    @override
    def visit_bool_value(self, bool_value: "BoolValue") -> None:
        stmt = BoolValueStmt(
            stmt_id=bool_value.site_local_logid(),
            cond=bool_value.bool_expr,
        )
        self.fire_main_body.append(stmt)

    @override
    def visit_next(self, next: "Next") -> None:
        # the next node is handled by the SeqForEach and ParForEach nodes and doesn't do anything by itself
        pass

    @override
    def visit_yield(self, yield_node: "Yield") -> None:
        val_or_noderef = yield_node.yld.value
        if isinstance(val_or_noderef, NodeRef):
            value_or_stmtref = self._node_ref_to_stmt_ref(val_or_noderef)
            value_node_id = val_or_noderef.id.ident
        else:
            value_or_stmtref = any_object_to_tagged_json_value(val_or_noderef)
            value_node_id = None
        target = yield_node.yield_target_after_cut()
        assert yield_node.yld.place == target.site_local_logid(), (
            "Yield place and target place must be the same"
        )

        token_send = TokenSend(
            job=None,
            site=target.site(),
            place=target.site_local_logid(),
            is_yield=True,
            kind=target.kind,
            data=value_or_stmtref,
            node_id=value_node_id,
        )
        # all sends go into the epilog
        self.fire_epilog_body.append(token_send)
        # propagate errors to the yield place
        self.error_propagation_out_places.append(target)

    @override
    def visit_send(self, send: "Send") -> None:
        if send.is_exit_node_send():
            # Don't traverse sends that belong to the exit node
            # The exit node will handle those
            return

        if send.literal_data is not None:
            assert len(send.in_slot("value").incoming_edges()) == 0
            value = send.literal_data
        else:
            assert len(send.in_slot("value").incoming_edges()) == 1
            source = send.in_slot("value").incoming_edges()[0].src().node()
            value = StmtRef(stmt_id=source.site_local_logid())

        target = send.send_target()

        token_send = TokenSend(
            job=None,
            site=target.site(),
            place=target.site_local_logid(),
            is_yield=False,
            kind=target.kind,
            data=value,
            node_id=None,
        )
        self.error_propagation_out_places.append(target)
        self.fire_epilog_body.append(token_send)

    @override
    def visit_jump(self, jump: "Jump") -> None:
        # this is an exit node
        self.transition_docstring = "jump block"

        target = jump.jump_target_after_cut()

        token_send = TokenSend(
            job=None,
            site=target.site(),
            place=target.site_local_logid(),
            is_yield=False,
            kind=target.kind,
            data=FlowInfo(hint=None),
            node_id=None,
        )
        self.fire_epilog_body.append(token_send)

        self.control_stop_out_places.append(target)
        self.error_propagation_out_places.append(target)

    @override
    def visit_cannot_reach(self, cannot_reach: "CannotReach") -> None:
        assert False, "This should not be visited during IR generation"

    @override
    def visit_loop(self, loop: "Loop") -> None:
        # this is an exit node
        self.transition_docstring = "loopback block"

        target = loop.loop_target_after_cut()

        token_send = TokenSend(
            job=None,
            site=target.site(),
            place=target.site_local_logid(),
            is_yield=False,
            kind=target.kind,
            data=FlowInfo(hint=-1),
            node_id=None,
        )
        self.fire_epilog_body.append(token_send)
        # don't propagate errors back to the loop head, this iteration is done
        # this is a back edge, we don't send the stop token there

    @override
    def visit_end_fork(self, end_fork: "EndFork") -> None:
        """The end of a fork sends the frame to a counting place and then ends."""
        self.transition_docstring = "end of fork block"

        target = end_fork.end_fork_target_after_cut()

        token_send = TokenSend(
            job=None,
            site=target.site(),
            place=target.site_local_logid(),
            is_yield=False,
            kind=target.kind,
            data=FlowInfo(hint=None),
            node_id=None,
        )
        self.fire_epilog_body.append(token_send)

        # this is the end of a fork, we don't send the stop token onwards
        # we also don't propagate errors to the counting place

    @override
    def visit_stop(self, stop: "Stop") -> None:
        # depends on whether we are the leader or follower
        # followers send their signal to the leader
        # the leader region is automatically synced by the places, as soon as it executes
        # it can simply stop
        if stop.state == NodeState.LEADER:
            self.transition_docstring = "stop group leader block"
            self.fire_epilog_body.append(Comment(comment="Yield stop token"))
            self.fire_epilog_body.append(
                TokenSend(
                    job=None,
                    site=stop.graph().homesite(),
                    place=STATUS_PLACE_NAME,
                    is_yield=True,
                    kind="control",
                    data=StopToken(),
                    node_id=None,
                )
            )
            stop_region = stop.graph().yield_region()
            assert isinstance(stop_region, Region), "Yield region is not a region"

            # if an error makes it to the stop node, send it to the control place
            status_place_name = stop.graph().create_yield_place_node_if_needed(
                STATUS_PLACE_NAME
            )
            control_place = stop.graph().get_nodes_for_logid(
                status_place_name, T=Place
            )[0]
            assert isinstance(control_place, Place), (
                "Status control place is not a place"
            )
            self.error_propagation_out_places.append(control_place)

            # also send the stop token at the very end
            self.control_stop_out_places.append(control_place)
        else:
            self.transition_docstring = "stop group follower block"
            target = stop.sync_target_after_cut()
            self.fire_epilog_body.append(
                TokenSend(
                    job=None,
                    site=target.site(),
                    place=target.site_local_logid(),
                    is_yield=False,
                    kind=target.kind,
                    data=FlowInfo(hint=None),
                    node_id=None,
                )
            )
            self.error_propagation_out_places.append(target)
        # this is the end of the pipeline, no reason to send the stop token any further

    @override
    def visit_branch(self, branch: "Branch") -> None:
        # the branch leader sends the condition to all the followers
        # and then branches
        # the followers don't need to do anything special, they simply
        # branch; the condition will have arrived through a place
        condition_stmt_id = branch.condition_source().site_local_logid()

        if branch.state == NodeState.LEADER:
            self.transition_docstring = "branch leader block"
            for target in branch.sync_targets_after_cut():
                self.fire_epilog_body.append(
                    TokenSend(
                        job=None,
                        site=target.site(),
                        place=target.site_local_logid(),
                        is_yield=False,
                        kind=target.kind,
                        data=StmtRef(stmt_id=condition_stmt_id),
                        node_id=None,
                    )
                )
        else:
            self.transition_docstring = "branch follower block"

        # from here, the leader and follower do the same thing
        taken_target = branch.branch_taken_target_after_cut()
        not_taken_target = branch.branch_not_taken_target_after_cut()

        self.fire_epilog_body.append(
            BranchStmt(
                condition=StmtRef(stmt_id=condition_stmt_id),
                taken_site=taken_target.site(),
                taken_place=taken_target.site_local_logid(),
                not_taken_site=not_taken_target.site(),
                not_taken_place=not_taken_target.site_local_logid(),
            )
        )

        self.control_stop_out_places.append(taken_target)
        self.control_stop_out_places.append(not_taken_target)
        # we only propagate errors along the exit branch
        self.error_propagation_out_places.append(not_taken_target)

    @override
    def visit_seq_iterate(self, seq_iterate: "SeqForEach") -> None:
        """
        Leader:
        # The frame should be the current iteration already, which is done by the activation function

            === In the body: DrainIteratorStmt ===
            try:
                data = next(next_node_input) # or await anext(next_node_input)
                has_next = True
            except StopIteration: # or StopAsyncIteration
                has_next = False
            === DrainIteratorStmt ===
            # epilog:
            <for each data receiver> send(frame, data)
            <for eacy sync> send(frame, has_next)

            branch (frame, has_next):
                if taken: (frame, __next_iteration__)
                else: (frame.pop(), __stop_iteration__)
        Follower:
            # data is handled, only need to deal with control
            branch (frame, __has_next__):
                if taken: (frame, __next_iteration__)
                else: (frame.pop(), __stop_iteration__)
        """
        if seq_iterate.state == NodeState.LEADER:
            self.transition_docstring = "seq iterate leader block"
            # takes the __has_next__ input node
            next_node = seq_iterate.has_next_source()
            iterator_source = next_node.iterator_source_after_cut()

            condition_stmt_id = seq_iterate.site_local_logid()
            # run the next() in the main body, in case there's an exception
            self.fire_main_body.append(
                DrainIteratorStmt(
                    data_stmt_id=next_node.site_local_logid(),
                    condition_stmt_id=condition_stmt_id,
                    iterator=StmtRef(stmt_id=iterator_source.site_local_logid()),
                    is_async=next_node.is_async,
                )
            )
            # send the data
            data_targets = next_node.data_targets_after_cut()
            for target in data_targets:
                self.fire_main_body.append(
                    TokenSend(
                        job=None,
                        site=target.site(),
                        place=target.site_local_logid(),
                        is_yield=False,
                        kind=target.kind,
                        data=StmtRef(stmt_id=next_node.site_local_logid()),
                        node_id=None,
                    )
                )
            # send the condition and the frame to the followers
            sync_targets = seq_iterate.sync_targets_after_cut()
            for target in sync_targets:
                self.fire_epilog_body.append(
                    TokenSend(
                        job=None,
                        site=target.site(),
                        place=target.site_local_logid(),
                        is_yield=False,
                        kind=target.kind,
                        data=StmtRef(stmt_id=condition_stmt_id),
                        node_id=None,
                    )
                )
        else:
            self.transition_docstring = "seq iterate follower block"
            condition_stmt_id = seq_iterate.has_next_place().site_local_logid()

        # The branch is the same for both, leader and follower
        taken_target = seq_iterate.next_iteration_target_after_cut()
        stop_iteration_target = seq_iterate.stop_iteration_target_after_cut()
        self.fire_epilog_body.append(
            BranchStmt(
                condition=StmtRef(stmt_id=condition_stmt_id),
                taken_site=taken_target.site(),
                taken_place=taken_target.site_local_logid(),
                not_taken_site=stop_iteration_target.site(),
                not_taken_place=stop_iteration_target.site_local_logid(),
                not_taken_branch_frame="pop",
            )
        )

        self.control_stop_out_places.append(taken_target)
        self.control_stop_out_places.append(stop_iteration_target)
        # only propagate errors to the loop exit
        self.error_propagation_out_places.append(stop_iteration_target)

    @override
    def visit_par_iterate(self, par_iterate: "ParForEach") -> None:
        """
        Leader:
            # For a par iterate, the incoming frame is NOT the loop, different to seq iterate.
            # This is also done by the activation function.
            === ForkFromIteratorStmt ===
            loop_frame = frame.push()
            loop_count = 0
            has_more = True
            while has_next:
                try: # isolate application errors that may happen in next()
                    data = next(it)
                except StopIteration:
                    has_more = False
                except Exception as e:
                    data = ErrorToken(error=e)
                <for each data receiver> send(loop_frame, data)
                <for each sync follower> send(loop_frame, FlowInfo())
                to fork target, send(loop_frame, FlowInfo())
            <for each sync follower> send(frame, expect(loop_count)) # NOT the loop frame
            send(frame, expect(loop_count)) to stop iteration
            === ForkFromIteratorStmt ===
        Follower:
            # data is handled, only need to deal with control
            branch (frame, flow_info, is_branch_on_flow_info=True):
                if taken: send(frame, FlowInfo()) to __next_iteration__
                else: send(frame, flow_info) to __stop_iteration__
        """
        if par_iterate.state == NodeState.LEADER:
            self.transition_docstring = "par iterate leader block"
            # takes the __has_next__ input node
            next_node = par_iterate.has_next_source()
            iterator_source = next_node.iterator_source_after_cut()
            data_targets = next_node.data_targets_after_cut()
            data_receivers = [
                (target.site(), target.site_local_logid()) for target in data_targets
            ]

            sync_targets = par_iterate.sync_targets_after_cut()
            sync_receivers = [
                (target.site(), target.site_local_logid()) for target in sync_targets
            ]

            fork_target = par_iterate.fork_target_after_cut()
            stop_iteration_target = par_iterate.stop_iteration_target_after_cut()

            self.fire_epilog_body.append(
                ForkFromIteratorStmt(
                    data_stmt_id=next_node.site_local_logid(),
                    iterator=StmtRef(stmt_id=iterator_source.site_local_logid()),
                    is_async=next_node.is_async,
                    data_receivers=data_receivers,
                    sync_receivers=sync_receivers,
                    fork_site=fork_target.site(),
                    fork_place=fork_target.site_local_logid(),
                    stop_iteration_site=stop_iteration_target.site(),
                    stop_iteration_place=stop_iteration_target.site_local_logid(),
                )
            )
        else:
            self.transition_docstring = "par iterate follower block"
            condition_stmt_id = par_iterate.has_next_place().site_local_logid()

            fork_target = par_iterate.fork_target_after_cut()
            stop_iteration_target = par_iterate.stop_iteration_target_after_cut()
            self.fire_epilog_body.append(
                BranchStmt(
                    condition=StmtRef(stmt_id=condition_stmt_id),
                    taken_site=fork_target.site(),
                    taken_place=fork_target.site_local_logid(),
                    not_taken_site=stop_iteration_target.site(),
                    not_taken_place=stop_iteration_target.site_local_logid(),
                )
            )

        self.control_stop_out_places.append(fork_target)
        self.control_stop_out_places.append(stop_iteration_target)
        # only propagate errors to the loop exit
        self.error_propagation_out_places.append(stop_iteration_target)
