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

from nv_dfm_core.api import Pipeline, Yield
from nv_dfm_core.gen.irgen import (
    IRGen,
)
from nv_dfm_core.gen.irgen._helpers import load_fed_info_json
from nv_dfm_core.gen.modgen.ir import (
    AdapterCallStmt,
    InPlace,
    NetIR,
    StmtRef,
    Transition,
)
from nv_dfm_core.gen.modgen.ir._in_place import START_PLACE_NAME
import examplefed.fed.runtime
from examplefed.fed.api.users import GreetMe


@pytest.fixture
def mock_logger():
    return Mock(spec=logging.Logger)


def test_create_graph(mock_logger: Mock):
    with Pipeline() as p:
        greet = GreetMe(site="concierge", name="test_user")
        _ = Yield(value=greet)

    irgen = IRGen()
    prepped = irgen.prepare(
        pipeline=p,
        candidate_sites=["concierge"],
        federation_module_name="examplefed",
        fed_info=load_fed_info_json(examplefed.fed.runtime),
        homesite="concierge",
        logger=mock_logger,
        debug=False,
    )

    assert "concierge" in prepped.net_irs()
    assert len(prepped.net_irs()["concierge"].transitions) == 2
