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

from typing import Literal

import pytest

from nv_dfm_core.api import (
    NodeId,
    NodeParam,
    NodeRef,
    Operation,
    Pipeline,
    PipelineBuildHelper,
    make_auto_id,
    well_known_id,
)


def test_nodeid():
    sometuple = ("@dfm-node:reallyclosecall", 42, None)
    autoid = make_auto_id(12)
    customid1 = well_known_id("my_identifier")
    customid2 = well_known_id(12)  # type: ignore

    assert not isinstance(sometuple, NodeRef)
    assert isinstance(autoid, NodeId)
    assert not isinstance(autoid, NodeRef)
    assert isinstance(customid1, NodeId)
    assert isinstance(customid2, NodeId)

    autoid_copy = make_auto_id(12)
    customid1_copy = well_known_id("my_identifier")
    assert autoid == autoid_copy
    assert customid1 == customid1_copy
    # but no accidental aliasing between auto and user ids
    assert autoid != customid2

    assert str(autoid) == "%12"
    assert str(customid1) == "#my_identifier"
    assert str(customid2) == "#wkid12"
    assert str(autoid) == "%12"


def test_nodeid_enforces_valid_identifiers():
    with pytest.raises(ValueError):
        well_known_id("not a valid identifier")


def test_nodeid_to_noderef():
    nid = make_auto_id(42)
    assert isinstance(nid, NodeId)

    nref = nid.to_ref()
    assert isinstance(nref, NodeRef)
    assert str(nref) == "%42"
    assert nref.id == nid
    assert nref.sel is None


def test_nodeid_to_noderef_with_int_sel():
    nid = make_auto_id(42)
    assert isinstance(nid, NodeId)

    nref = nid.to_ref(sel=2)
    assert isinstance(nref, NodeRef)
    assert str(nref) == "%42.2"
    assert nref.id == nid
    assert nref.sel == 2


def test_nodeid_to_noderef_with_str_sel():
    nid = make_auto_id(42)
    assert isinstance(nid, NodeId)

    nref = nid.to_ref(sel="field")
    assert isinstance(nref, NodeRef)
    assert str(nref) == "%42.field"
    assert nref.id == nid
    assert nref.sel == "field"


class Op1(Operation):
    dfm_class_name: Literal["test_node_id.Op1"] = "test_node_id.Op1"
    __api_name__: Literal["test_node_id.Op1"] = "test_node_id.Op1"


class Op2(Operation):
    dfm_class_name: Literal["test_node_id.Op2"] = "test_node_id.Op2"
    __api_name__: Literal["test_node_id.Op2"] = "test_node_id.Op2"
    value: NodeParam


def test_setting_node_param_translates_to_noderef():
    with Pipeline():
        assert PipelineBuildHelper.build_helper_active()
        op1 = Op1()
        op2a = Op2(value=op1)
        op2b = Op2(value=op1.dfm_node_id.to_ref())
        op2c = Op2(value=op1.fieldref("someselector"))

        assert op2a.value == op1.dfm_node_id.to_ref()
        assert op2b.value == op1.dfm_node_id.to_ref()
        assert op2c.value == op1.dfm_node_id.to_ref("someselector")
