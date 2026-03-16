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

from collections.abc import Generator
from typing import Any

import pytest

from nv_dfm_core.api import (
    Equal,
    If,
    Pipeline,
    Yield,
)
from nv_dfm_core.gen.modgen.ir import (
    START_PLACE_NAME,
    AdapterCallStmt,
    InPlace,
    NetIR,
    StmtRef,
    Transition,
)
from tests._builder_helpers import FedInfoBuilder
from tests._operations_for_testing import NullaryOp
from tests._pipeline_compilation_test_base import PipelineCompilationTestBase


class TestIfPipeline(PipelineCompilationTestBase, target="local", debug=True):
    @staticmethod
    def pipelines() -> Generator[Pipeline, Any, None]:
        with Pipeline(name="fixed site") as p:
            op1 = NullaryOp()
            with If(cond=Equal(operator="==", left=op1.dfm_node_id.to_ref(), right=1)):
                _ = Yield(value=op1, place="output")

        yield p

    @staticmethod
    def fed_infos() -> Generator[FedInfoBuilder, Any, None]:
        b = FedInfoBuilder(federation_module_name="examplefed")
        _ = b.homesite("home").op("testing.NullaryOp", fixed_sec=20.0, fixed_size=1)
        yield b

    @staticmethod
    def candidate_sites() -> Generator[list[str], Any, None]:
        yield ["home"]

    @staticmethod
    def expected_net_irs() -> Generator[dict[str, NetIR], Any, None]:
        yield {
            "home": NetIR(
                site="home",
                places=[
                    InPlace(name=START_PLACE_NAME, type="control", sender="external")
                ],
                trans=[
                    Transition(
                        act=ActivationCondition(
                            control_place=START_PLACE_NAME, data_places=[]
                        ),
                        body=[
                            AdapterCallStmt(
                                provider=None,
                                stmt_id="node0",
                                adapter="testing_NullaryOp",
                                literal_params={},
                                stmt_params={},
                                is_async=True,
                            ),
                            SendStmt(
                                stmt_id="node0",
                                job=None,
                                target=PlaceAddress(
                                    site=None,
                                    place="output",
                                ),
                                data=StmtRef(stmt_id="node0"),
                            ),
                        ],
                        out=[OutPlace(job=None, to_site=None, to_place="output")],
                    )
                ],
            )
        }

    @pytest.mark.parametrize("pipeline", pipelines())
    @pytest.mark.parametrize(
        "candidate_sites, fed_info, expected_net_irs",
        zip(candidate_sites(), fed_infos(), expected_net_irs()),
    )
    def test_compile_and_run_single_pipeline(
        self,
        pipeline: Pipeline,
        candidate_sites: list[str],
        fed_info: FedInfoBuilder,
        expected_net_irs: dict[str, NetIR],
    ) -> None:
        self.single_pipeline_test(pipeline, candidate_sites, fed_info, expected_net_irs)
