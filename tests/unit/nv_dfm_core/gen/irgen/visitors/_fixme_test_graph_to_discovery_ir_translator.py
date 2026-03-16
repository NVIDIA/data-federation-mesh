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

from nv_dfm_core.api import Pipeline, PlaceParam, Yield
from nv_dfm_core.gen.irgen._fed_info import ComputeCostInfo, SendCostInfo
from nv_dfm_core.gen.irgen.ir import (
    OperationModel,
    PlaceSendModel,
    SlotEdge,
    SlotModel,
    FlowEdge,
    YieldModel,
)
from nv_dfm_core.gen.irgen.placement import Graph, GraphState, NodeState
from nv_dfm_core.gen.irgen.visitors import GraphToDiscoveryIRTranslator, ValidateVisitor
from nv_dfm_core.gen.modgen.ir import (
    AdapterDiscoveryStmt,
    InPlace,
    NetIR,
    ReadPlaceStmt,
    SendDiscoveryStmt,
    Transition,
)
from examplefed.fed.api.users import GreetMe


def test_empty_graph_creates_minimal_net_ir(caplog):
    """Test that an empty graph creates a minimal valid NetIR with just the start place."""
    # Create an empty graph
    graph = Graph(validation_visitor_class=ValidateVisitor)

    # Create translator with a logger
    logger = Logger("test")
    translator = GraphToDiscoveryIRTranslator(logger=logger)

    # Translate the empty graph
    net_irs = translator.create_net_irs_from_graph(graph=graph)

    # Assert that we got an empty dictionary (no locations in empty graph)
    assert net_irs == {}


def test_graph_with_greetme_and_yield_creates_discovery_net_ir(caplog):
    """Test that a graph with a GreetMe and Yield statement creates a valid discovery NetIR."""
    # Create a graph with a GreetMe and Yield statement
    graph = Graph(validation_visitor_class=ValidateVisitor)
    graph.state = GraphState.CUT_CROSS_SITES

    # Create translator with a logger
    logger = Logger("test")
    translator = GraphToDiscoveryIRTranslator(logger=logger)

    # Create a pipeline to get the statements
    with Pipeline(mode="discovery"):
        greet = GreetMe(site="concierge", name="test_user")
        yield_stmt = Yield(value=greet)

    # Create nodes at location "site1"
    greet_node = graph.create_node(
        ordering=0,
        location=("site1",),
        node_model=OperationModel(
            operation=greet, compute_cost_info=ComputeCostInfo(), is_async=True
        ),
    )
    greet_node.state = NodeState.SELECTED

    yield_node = graph.create_node(
        ordering=1,
        location=("site1",),
        node_model=YieldModel(
            yield_stmt=yield_stmt, compute_cost_info=ComputeCostInfo()
        ),
    )
    yield_node.state = NodeState.SELECTED

    # Create SetFieldModel node for the "value" field of yield_stmt
    slot_node = graph.create_node(
        ordering=0.75,  # Between greet and yield nodes
        location=("site1",),
        node_model=SlotModel(slotname="value"),
    )
    slot_node.state = NodeState.SELECTED

    # Create edges:
    # 1. SetFieldEdge from set_field_node to yield_node
    e1 = graph.create_edge(source=slot_node, target=yield_node, edge_model=SlotEdge())
    yield_node.used_edges.append(e1)

    # 2. SsaUseEdge from greet_node to set_field_node
    e2 = graph.create_edge(
        source=greet_node,
        target=slot_node,
        edge_model=FlowEdge(send_cost_info=None),  # No send cost since same location
    )
    slot_node.used_edges.append(e2)
    graph.validate()

    # Translate the graph
    net_irs = translator.create_net_irs_from_graph(graph=graph)

    # Assert we got a NetIR for site1
    assert "site1" in net_irs
    net_ir = net_irs["site1"]

    # Assert it's a valid NetIR
    assert isinstance(net_ir, NetIR)

    # Assert it has exactly one place (the start place)
    assert len(net_ir.places) == 1
    start_place = net_ir.places[0]
    assert isinstance(start_place, InPlace)
    assert start_place.name == START_PLACE_NAME
    assert start_place.kind == "special"
    assert not start_place.optional

    # Assert it has exactly one transition
    assert len(net_ir.trans) == 1
    transition = net_ir.trans[0]

    # Assert the transition only activates on the start place
    assert transition.act.data_places == [START_PLACE_NAME]
    assert transition.act.opt == []

    # Assert the transition body contains an AdapterDiscoveryStmt and SendDiscoveryStmt
    assert len(transition.body) == 2
    assert isinstance(transition.body[0], AdapterDiscoveryStmt)
    assert isinstance(transition.body[1], SendDiscoveryStmt)

    # Assert no out places
    assert transition.out == [OutPlace(job=None, to_site=None, to_place="discovery")]


def test_graph_with_islands_and_params(caplog):
    """Test that a graph with a GreetMe and Yield statement creates a valid execution NetIR."""
    # Create a graph with a GreetMe and Yield statement
    graph = Graph(validation_visitor_class=ValidateVisitor)
    graph.state = GraphState.CUT_CROSS_SITES

    # Create translator with a logger
    logger = Logger("test")
    translator = GraphToDiscoveryIRTranslator(logger=logger)

    # Create a pipeline to get the statements
    with Pipeline(mode="execute"):
        greet1 = GreetMe(site="server", name=PlaceParam(place="myname"))
        greet2 = GreetMe(site="reception", name=greet1)
        yield_stmt = Yield(value=greet2)

    assert greet1.dfm_node_id != greet2.dfm_node_id
    # Create the main nodes: greet1, greet2, yield
    greet1_node = graph.create_node(
        ordering=1,
        location=("server",),
        node_model=OperationModel(
            operation=greet1, compute_cost_info=ComputeCostInfo(), is_async=True
        ),
    )
    greet1_node.state = NodeState.SELECTED

    greet2_node = graph.create_node(
        ordering=2,
        location=("reception",),
        node_model=OperationModel(
            operation=greet2, compute_cost_info=ComputeCostInfo(), is_async=True
        ),
    )
    greet2_node.state = NodeState.SELECTED

    yield_node = graph.create_node(
        ordering=3,
        location=("reception",),
        node_model=YieldModel(
            yield_stmt=yield_stmt, compute_cost_info=ComputeCostInfo()
        ),
    )
    yield_node.state = NodeState.SELECTED

    # param and setfield nodes
    assert isinstance(greet1.name, PlaceParam)
    greet1_param_node = graph.create_node(
        ordering=0.5,  # Between greet and yield nodes
        location=("server",),
        node_model=ParamPlaceModel(param_place=greet1.name, send_cost=SendCostInfo()),
    )
    greet1_param_node.state = NodeState.SELECTED

    greet1_name_slot_node = graph.create_node(
        ordering=0.75,  # before the greet1 node
        location=("server",),
        node_model=SlotModel(slotname="name"),
    )
    greet1_name_slot_node.state = NodeState.SELECTED

    greet2_name_slot_node = graph.create_node(
        ordering=1.75,  # before the greet2 node
        location=("reception",),
        node_model=SlotModel(slotname="name"),
    )
    greet2_name_slot_node.state = NodeState.SELECTED

    yield_set_field_node = graph.create_node(
        ordering=2.75,  # before the yield node
        location=("reception",),
        node_model=SlotModel(slotname="value"),
    )
    yield_set_field_node.state = NodeState.SELECTED

    # cut nodes: edge went from greet1_node to greet2_name_slot_node
    greet_placename = node_id_to_ssa_def_name(greet2.dfm_node_id) + "_name"
    greet1_send_node = graph.create_node(
        ordering=1.5,
        location=("server",),
        node_model=PlaceSendModel(
            source_node=greet1_node,
            target_node=None,
            placename=greet_placename,
        ),
    )
    greet1_send_node.state = NodeState.SELECTED

    receive_slot_node = graph.create_node(
        ordering=1.75,
        location=("reception",),
        node_model=SlotModel(slotname="value"),
    )
    receive_slot_node.state = NodeState.SELECTED

    greet2_receive_node = graph.create_node(
        ordering=1.85,
        location=("reception",),
        node_model=PlaceReceiveModel(
            source_node=greet1_node,
            target_node=None,
            ssa_def=greet_placename,
            placename=greet_placename,
        ),
    )
    greet2_receive_node.state = NodeState.SELECTED

    # edges
    e = graph.create_edge(
        source=greet1_param_node,
        target=greet1_name_slot_node,
        edge_model=FlowEdge(send_cost_info=None),
    )
    e.target.edges_used_to_compute_cost.append(e)

    e = graph.create_edge(
        source=greet1_name_slot_node, target=greet1_node, edge_model=SlotEdge()
    )
    e.target.edges_used_to_compute_cost.append(e)

    e = graph.create_edge(
        source=greet1_node,
        target=greet1_send_node,
        edge_model=FlowEdge(send_cost_info=None),
    )
    e.target.edges_used_to_compute_cost.append(e)

    e = graph.create_edge(
        source=greet2_receive_node,
        target=greet2_name_slot_node,
        edge_model=FlowEdge(send_cost_info=None),
    )
    e.target.edges_used_to_compute_cost.append(e)

    e = graph.create_edge(
        source=greet2_name_slot_node, target=greet2_node, edge_model=SlotEdge()
    )
    e.target.edges_used_to_compute_cost.append(e)

    e = graph.create_edge(
        source=greet2_node,
        target=yield_set_field_node,
        edge_model=FlowEdge(send_cost_info=None),
    )
    e.target.edges_used_to_compute_cost.append(e)

    e = graph.create_edge(
        source=yield_set_field_node, target=yield_node, edge_model=SlotEdge()
    )
    e.target.edges_used_to_compute_cost.append(e)

    e = graph.create_edge(
        source=greet1_send_node,
        target=receive_slot_node,
        edge_model=WeakEdge(send_cost_info=None),
    )
    e.target.edges_used_to_compute_cost.append(e)

    e = graph.create_edge(
        source=receive_slot_node,
        target=greet2_receive_node,
        edge_model=SlotEdge(),
    )
    e.target.edges_used_to_compute_cost.append(e)

    # print(graph.to_graphviz())

    graph.validate()

    # Translate the graph
    net_irs = translator.create_net_irs_from_graph(graph=graph)

    # Assert we got a NetIR for site1
    assert "server" in net_irs
    assert "reception" in net_irs

    assert net_irs["server"] == NetIR(
        places=[
            InPlace(name="myname", kind="param", optional=False),
            InPlace(name="__start_place__", kind="special", optional=False),
        ],
        trans=[
            Transition(
                act=ActivationCondition(
                    data_places=["__start_place__"], opt=["myname"]
                ),
                body=[
                    ReadPlaceStmt(ssa_def="myname", place="myname"),
                    AdapterDiscoveryStmt(
                        prov=None,
                        nodeid=greet1.dfm_node_id.ident,
                        ssa_def="node0",
                        adapter="users_GreetMe",
                        litparams={},
                        is_async=True,
                    ),
                    SendDiscoveryStmt(),
                ],
                out=[OutPlace(job=None, to_site=None, to_place="discovery")],
            )
        ],
    )

    assert net_irs["reception"] == NetIR(
        places=[
            InPlace(name="node1_name", kind="receive", optional=False),
            InPlace(name="__start_place__", kind="special", optional=False),
        ],
        trans=[
            Transition(
                act=ActivationCondition(
                    data_places=["__start_place__"], opt=["node1_name"]
                ),
                body=[
                    ReadPlaceStmt(ssa_def="node1_name", place="node1_name"),
                    AdapterDiscoveryStmt(
                        prov=None,
                        nodeid=greet2.dfm_node_id.ident,
                        ssa_def="node1",
                        adapter="users_GreetMe",
                        litparams={},
                        is_async=True,
                    ),
                    SendDiscoveryStmt(),
                ],
                out=[OutPlace(job=None, to_site=None, to_place="discovery")],
            )
        ],
    )
