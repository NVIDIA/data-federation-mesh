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

from pydantic.types import JsonValue


import json
from enum import Enum
from typing import Any

import pytest
from pydantic import BaseModel, JsonValue

from nv_dfm_core.exec import Frame
from nv_dfm_core.exec._frame import FlowInfo
from nv_dfm_core.exec._token_package import any_object_to_tagged_json_value
from nv_dfm_core.gen.modgen.ir import (
    START_PLACE_NAME,
    ActivateWhenPlacesReady,
    AdapterCallStmt,
    BoundNetIR,
    InPlace,
    NetIR,
    StmtRef,
    TokenSend,
    Transition,
)


@pytest.fixture
def net() -> NetIR:
    return NetIR(
        pipeline_name="test",
        site="somesite",
        transitions=[
            Transition(
                control_place=InPlace(
                    name=START_PLACE_NAME,
                    origin="external",
                    kind="control",
                    flavor="seq_control",
                    type="nv_dfm_core.exec.FlowInfo",
                ),
                data_places=[
                    InPlace(
                        name="p1",
                        origin="external",
                        kind="data",
                        flavor="scoped",
                        type="Any",
                    ),
                    InPlace(
                        name="p2",
                        origin="external",
                        kind="data",
                        flavor="scoped",
                        type="Any",
                    ),
                ],
                try_activate_func=ActivateWhenPlacesReady(),
                fire_body=[
                    AdapterCallStmt(
                        stmt_id="x",
                        has_users=True,
                        adapter="op1",
                        literal_params={},
                        stmt_params={},
                        is_async=True,
                    ),
                    TokenSend(
                        job=None,
                        site="somesite",
                        place="p2",
                        data=StmtRef(stmt_id="x"),
                        node_id=None,
                        is_yield=False,
                        kind="data",
                    ),
                ],
                signal_error_body=[
                    TokenSend(
                        job=None,
                        site="somesite",
                        place="p2",
                        data=StmtRef(stmt_id="_error_"),
                        node_id=None,
                        is_yield=False,
                        kind="data",
                    ),
                ],
                signal_stop_body=[],
            )
        ],
    )


def test_bound_net_ir_serialization(net: NetIR):
    """Test creating a BoundNetIR, serializing to JSON, and deserializing."""

    # Create a BoundNetIR with input parameters
    input_params: list[tuple[Frame, dict[str, JsonValue]]] = [
        (
            Frame.start_frame(num=0),
            {"p1": "test_value", "p2": None, START_PLACE_NAME: FlowInfo().model_dump()},
        )
    ]
    bound_net_ir = BoundNetIR.bind_netir(net, input_params)

    # Serialize to JSON
    json_data = json.dumps(bound_net_ir.model_dump())

    # Deserialize from JSON
    loaded_data = json.loads(json_data)
    loaded_bound_net_ir = BoundNetIR.model_validate(loaded_data)

    # Verify the loaded BoundNetIR matches the original
    assert loaded_bound_net_ir.ir.transitions == bound_net_ir.ir.transitions
    assert loaded_bound_net_ir.ir == bound_net_ir.ir
    assert loaded_bound_net_ir.tagged_input_params == bound_net_ir.tagged_input_params


def test_bound_net_ir_with_required_params(net: NetIR):
    # Test with all required parameters
    input_params: list[tuple[Frame, dict[str, Any]]] = [
        (
            Frame.start_frame(num=0),
            {"p1": 123, "p2": None, START_PLACE_NAME: FlowInfo()},
        )
    ]
    tagged_input_params: list[tuple[Frame, dict[str, tuple[str, JsonValue]]]] = [
        (
            Frame.start_frame(num=0),
            {
                "p1": ("json", 123),
                "p2": ("json", None),
                START_PLACE_NAME: (
                    "BaseModel",
                    {
                        "class": "nv_dfm_core.exec._frame.FlowInfo",
                        "data": {"hint": None},
                    },
                ),
            },
        )
    ]
    bound_net = BoundNetIR.bind_netir(net, input_params)
    assert bound_net.ir == net
    assert bound_net.tagged_input_params == tagged_input_params
    assert bound_net.deserialized_input_params() == [
        (
            Frame.start_frame(num=0),
            {"p1": 123, "p2": None, START_PLACE_NAME: FlowInfo()},
        )
    ]


def test_bound_net_ir_with_missing_required_params(net: NetIR):
    # Test with all required parameters
    input_params: list[tuple[Frame, dict[str, Any]]] = [
        (Frame.start_frame(num=0), {"p1": 123})
    ]
    # missing p2
    with pytest.raises(ValueError):
        _ = BoundNetIR.bind_netir(net, input_params)


def test_bound_net_ir_with_bytestring(net: NetIR):
    # Test with a bytestring to test the PickledObject tag
    input_params = [
        (
            Frame.start_frame(num=0),
            {"p1": b"ABCD", "p2": None, START_PLACE_NAME: FlowInfo()},
        )
    ]
    tagged_input_params = [
        (
            Frame.start_frame(num=0),
            {
                "p1": (
                    "PickledObject",
                    {
                        "tag": "__dfm__pickled__",
                        "value": "gASVCAAAAAAAAABDBEFCQ0SULg==",
                    },
                ),
                "p2": ("json", None),
                START_PLACE_NAME: (
                    "BaseModel",
                    {
                        "class": "nv_dfm_core.exec._frame.FlowInfo",
                        "data": {"hint": None},
                    },
                ),
            },
        )
    ]
    bound_net = BoundNetIR.bind_netir(net, input_params)
    assert bound_net.ir == net
    assert bound_net.tagged_input_params == tagged_input_params
    assert bound_net.deserialized_input_params() == [
        (
            Frame.start_frame(num=0),
            {"p1": b"ABCD", "p2": None, START_PLACE_NAME: FlowInfo()},
        )
    ]


class TestEnum(Enum):
    A = "a"
    B = "b"


class TestModel(BaseModel):
    x: int
    y: str
    e: TestEnum


def test_bound_net_ir_with_pydantic_model(net: NetIR):
    # test with a pydantic model as input
    input_params = [
        (
            Frame.start_frame(num=0),
            {
                "p1": TestModel(x=123, y="ABCD", e=TestEnum.A),
                "p2": None,
                START_PLACE_NAME: FlowInfo(),
            },
        )
    ]
    tagged_input_params = [
        (
            Frame.start_frame(num=0),
            {
                "p1": (
                    "BaseModel",
                    {
                        "class": "test_bound_net_ir.TestModel",
                        "data": {"x": 123, "y": "ABCD", "e": "a"},
                    },
                ),
                "p2": ("json", None),
                START_PLACE_NAME: (
                    "BaseModel",
                    {
                        "class": "nv_dfm_core.exec._frame.FlowInfo",
                        "data": {"hint": None},
                    },
                ),
            },
        )
    ]
    bound_net = BoundNetIR.bind_netir(net, input_params)
    assert bound_net.ir == net
    assert bound_net.tagged_input_params == tagged_input_params
    assert bound_net.deserialized_input_params() == [
        (
            Frame.start_frame(num=0),
            {
                "p1": TestModel(x=123, y="ABCD", e=TestEnum.A),
                "p2": None,
                START_PLACE_NAME: FlowInfo(),
            },
        )
    ]
