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

import logging
from unittest.mock import Mock

import pytest

from nv_dfm_core.api import BestOf, Pipeline, PlaceParam, Yield
from nv_dfm_core.gen.irgen import (
    ComputeCostInfo,
    FedInfo,
    OperationInfo,
    ProviderInfo,
    SendCostInfo,
    SiteInfo,
)
from nv_dfm_core.gen.irgen.ir import (
    OperationModel,
    SlotModel,
    YieldModel,
)
from nv_dfm_core.gen.irgen.placement import Graph
from nv_dfm_core.gen.irgen.visitors import PipelineToGraphVisitor, ValidateVisitor
from examplefed.fed.api.users import GreetMe


@pytest.fixture
def fed_info():
    fed_info = FedInfo(
        sites={
            "concierge": SiteInfo(
                interface={
                    "users.GreetMe": OperationInfo(
                        operation="examplefed.lib.customer_service.GreetMe",
                        compute_cost=ComputeCostInfo(fixed_time=1.0, fixed_size=1),
                    )
                },
                providers={},
                send_cost={"local": SendCostInfo(fixed_time=0.1)},
            ),
            "server": SiteInfo(
                interface={
                    "users.GreetMe": OperationInfo(
                        operation="examplefed.lib.customer_service.GreetMe",
                        compute_cost=ComputeCostInfo(fixed_time=1.0, fixed_size=1),
                    )
                },
                providers={
                    "some_provider": ProviderInfo(
                        interface={
                            "users.GreetMe": OperationInfo(
                                operation="examplefed.lib.customer_service.GreetMe",
                                compute_cost=ComputeCostInfo(
                                    fixed_time=1.0, fixed_size=1
                                ),
                            )
                        },
                    )
                },
                send_cost={
                    "reception": SendCostInfo(fixed_time=0.1),
                    "concierge": SendCostInfo(fixed_time=0.1),
                },
            ),
            "reception": SiteInfo(
                interface={
                    "users.GreetMe": OperationInfo(
                        operation="examplefed.lib.customer_service.GreetMe",
                        compute_cost=ComputeCostInfo(fixed_time=1.0, fixed_size=1),
                    )
                },
                providers={},
                send_cost={
                    "server": SendCostInfo(fixed_time=0.1),
                    "concierge": SendCostInfo(fixed_time=0.1),
                },
            ),
        }
    )
    return fed_info


@pytest.fixture
def mock_logger():
    return Mock(spec=logging.Logger)


@pytest.fixture
def local_visitor(fed_info, mock_logger):
    return PipelineToGraphVisitor(
        fed_info=fed_info,
        homesite="concierge",
        candidate_sites=set(["concierge"]),
        graph=Graph(validation_visitor_class=ValidateVisitor),
        logger=mock_logger,
    )


@pytest.fixture
def flare_visitor(fed_info, mock_logger):
    return PipelineToGraphVisitor(
        fed_info=fed_info,
        homesite="concierge",
        candidate_sites=set(["server", "reception"]),
        graph=Graph(validation_visitor_class=ValidateVisitor),
        logger=mock_logger,
    )


def test_create_local_graph(local_visitor):
    with Pipeline() as p:
        greet = GreetMe(site="concierge", name="test_user")
        yld = Yield(value=greet)

    p.accept(local_visitor)

    graph = local_visitor.graph
    graph.validate()

    # print("+" * 120)
    # print(graph.to_graphviz())
    # print("+" * 120)

    assert len(graph.nodes) == 3  # greetme, yield input place, yield; no alternatives
    assert isinstance(graph.nodes[0].node_model, OperationModel)
    assert graph.nodes[0].node_model.operation == greet
    # the place has been added after the yield
    assert isinstance(graph.nodes[2].node_model, SlotModel)
    assert isinstance(graph.nodes[1].node_model, YieldModel)
    assert graph.nodes[1].node_model.yield_stmt == yld


def test_local_node_in_flare_pipeline_raises(flare_visitor):
    with Pipeline() as p:
        GreetMe(site="concierge", name="test_user")

    with pytest.raises(ValueError):
        p.accept(flare_visitor)


def test_create_flare_graph(flare_visitor):
    with Pipeline() as p:
        greet = GreetMe(name="test_user")
        Yield(value=greet)

    p.accept(flare_visitor)

    graph = flare_visitor.graph
    graph.validate()

    # print("+" * 120)
    # print(graph.to_graphviz())
    # print("+" * 120)

    assert len(graph.nodes) == 6
    locations = set([node.location for node in graph.nodes])
    assert ("server",) in locations
    assert ("reception",) in locations
    assert len(locations) == 2


def test_create_param_node_graph(flare_visitor):
    with Pipeline() as p:
        greet = GreetMe(
            site=BestOf(sites=["server", "reception"]),
            name=PlaceParam(place="test_user"),
        )
        _ = Yield(value=greet)

    p.accept(flare_visitor)

    graph = flare_visitor.graph

    graph.validate()

    # print("+" * 120)
    # print(graph.to_graphviz())
    # print("+" * 120)

    assert len(graph.nodes) == 10
    for node in graph.nodes:
        assert node._next_alternative is not None
        assert node._next_alternative != node
