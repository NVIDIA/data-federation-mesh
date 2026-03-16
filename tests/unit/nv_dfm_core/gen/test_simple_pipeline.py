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

from nv_dfm_core.api import BestOf, Pipeline, StopToken, Yield
from nv_dfm_core.exec import FlowInfo
from nv_dfm_core.exec._frame import Frame
from nv_dfm_core.gen.modgen.ir import (
    START_PLACE_NAME,
    ActivateWhenPlacesReady,
    AdapterCallStmt,
    BodyEndMarker,
    BodyStartMarker,
    Comment,
    InPlace,
    NetIR,
    SendEndMarker,
    SendStartMarker,
    StmtRef,
    TokenSend,
    Transition,
)
from tests.assets.testfed.fed.api.ops import UnaryOp
from tests._builder_helpers import FedInfoBuilder
from tests._pipeline_compilation_test_base import (
    PipelineCompilationTestBase,
    PipelineTestCase,
    TokenResponse,
)


class TestSimplePipeline(PipelineCompilationTestBase):
    @staticmethod
    def pipelines() -> Generator[tuple[Pipeline, str], Any, None]:
        # fixed_site
        with Pipeline() as p:
            op1 = UnaryOp(site="site1", p1={"name": "test_user"})
            _ = Yield(value=op1)

        yield (p, "site1")

        # best of site1 and site2, site1 wins
        with Pipeline() as p:
            op1 = UnaryOp(
                site=BestOf(sites=["site1", "site2"]), p1={"name": "test_user"}
            )
            _ = Yield(value=op1)

        yield (p, "site1")

        # site3 wins
        with Pipeline() as p:
            op1 = UnaryOp(p1={"name": "test_user"})
            _ = Yield(value=op1)

        yield (p, "site3")

    @staticmethod
    def fed_info() -> FedInfoBuilder:
        b = FedInfoBuilder(federation_module_name="tests.assets.testfed")
        _ = (
            b.homesite("site1")
            .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
            .site("site2")
            .op("ops.UnaryOp", fixed_sec=100.0, fixed_size=100)
            .site("site3")
            .op("ops.UnaryOp", fixed_sec=1.0, fixed_size=1)
        )
        return b

    @staticmethod
    def expected_net_ir(winner: str) -> dict[str, NetIR]:
        return {
            winner: NetIR(
                pipeline_name=None,
                site=winner,
                transitions=[
                    Transition(
                        control_place=InPlace(
                            name=START_PLACE_NAME,
                            kind="control",
                            origin="external",
                            flavor="seq_control",
                            type="nv_dfm_core.exec.FlowInfo",
                        ),
                        data_places=[],
                        try_activate_func=ActivateWhenPlacesReady(
                            dfm_class_name="nv_dfm_core.gen.modgen.ir.ActivateWhenPlacesReady"
                        ),
                        fire_body=[
                            Comment(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.Comment",
                                comment="jump block",
                            ),
                            BodyStartMarker(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.BodyStartMarker"
                            ),
                            AdapterCallStmt(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.AdapterCallStmt",
                                provider=None,
                                stmt_id="node0",
                                has_users=True,
                                adapter="ops.UnaryOp",
                                literal_params={"p1": ("json", {"name": "test_user"})},
                                stmt_params={},
                                is_async=True,
                            ),
                            SendStartMarker(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.SendStartMarker"
                            ),
                            TokenSend(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.TokenSend",
                                job=None,
                                site="site1",
                                place="yield",
                                node_id=0,
                                is_yield=True,
                                kind="data",
                                data=StmtRef(stmt_id="node0", sel=None, issubs=True),
                            ),
                            TokenSend(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.TokenSend",
                                job=None,
                                site=winner,
                                place="region2___flow__",
                                node_id=None,
                                is_yield=False,
                                kind="control",
                                data=FlowInfo(hint=None),
                            ),
                            SendEndMarker(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.SendEndMarker"
                            ),
                            BodyEndMarker(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.BodyEndMarker"
                            ),
                        ],
                        signal_error_body=[
                            TokenSend(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.TokenSend",
                                job=None,
                                site="site1",
                                place="yield",
                                node_id=None,
                                is_yield=True,
                                kind="data",
                                data=StmtRef(stmt_id="_error_", sel=None, issubs=True),
                            ),
                            TokenSend(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.TokenSend",
                                job=None,
                                site=winner,
                                place="region2___flow__",
                                node_id=None,
                                is_yield=False,
                                kind="control",
                                data=StmtRef(stmt_id="_error_", sel=None, issubs=True),
                            ),
                        ],
                        signal_stop_body=[
                            TokenSend(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.TokenSend",
                                job=None,
                                site=winner,
                                place="region2___flow__",
                                node_id=None,
                                is_yield=False,
                                kind="control",
                                data=FlowInfo(hint=None),
                            ),
                        ],
                    ),
                    Transition(
                        control_place=InPlace(
                            name="region2___flow__",
                            kind="control",
                            origin="internal",
                            flavor="seq_control",
                            type="nv_dfm_core.exec.FlowInfo",
                        ),
                        data_places=[],
                        try_activate_func=ActivateWhenPlacesReady(
                            dfm_class_name="nv_dfm_core.gen.modgen.ir.ActivateWhenPlacesReady"
                        ),
                        fire_body=[
                            Comment(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.Comment",
                                comment="stop group leader block",
                            ),
                            BodyStartMarker(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.BodyStartMarker"
                            ),
                            SendStartMarker(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.SendStartMarker"
                            ),
                            Comment(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.Comment",
                                comment="Yield stop token",
                            ),
                            TokenSend(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.TokenSend",
                                job=None,
                                site="site1",
                                place="_dfm_status_",
                                node_id=None,
                                is_yield=True,
                                kind="control",
                                data=StopToken(token="@dfm-stop-token"),
                            ),
                            SendEndMarker(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.SendEndMarker"
                            ),
                            BodyEndMarker(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.BodyEndMarker"
                            ),
                        ],
                        signal_error_body=[
                            TokenSend(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.TokenSend",
                                job=None,
                                site="site1",
                                place="_dfm_status_",
                                node_id=None,
                                is_yield=True,
                                kind="data",
                                data=StmtRef(stmt_id="_error_", sel=None, issubs=True),
                            ),
                        ],
                        signal_stop_body=[
                            TokenSend(
                                dfm_class_name="nv_dfm_core.gen.modgen.ir.TokenSend",
                                job=None,
                                site="site1",
                                place="_dfm_status_",
                                node_id=None,
                                is_yield=True,
                                kind="data",
                                data=StopToken(token="@dfm-stop-token"),
                            ),
                        ],
                    ),
                ],
                fingerprint="dc82c23d8b8fb7d37d5299cd7fc8e788ec30f6e1e4eb5de9b71c4f8fdbfb53f5",
            )
        }

    @staticmethod
    def cases() -> Generator[PipelineTestCase, Any, None]:
        for pipeline, winner in TestSimplePipeline.pipelines():
            expected = TestSimplePipeline.expected_net_ir(winner)

            yield PipelineTestCase(
                test_name=None,
                test_variant=winner,
                pipeline=pipeline,
                candidate_sites=["site1", "site2", "site3"],
                fed_info=TestSimplePipeline.fed_info(),
                expected_net_irs=expected,
                run_params={},
                debug=False,
            )

    @pytest.mark.parametrize("test_case", cases())
    def test_compile_and_run_single_pipeline(
        self,
        test_case: PipelineTestCase,
    ) -> None:
        assert test_case.test_variant is not None, "Test variant is required"

        result = self.single_pipeline_test(
            test_case=test_case,
        )
        assert result, "No result"
        assert len(result) == 3, "Expected 3 result tokens"
        assert result[0] == TokenResponse(
            from_site=test_case.test_variant,
            from_node=0,
            frame=Frame(frame=[0]),
            to_place="yield",
            data="unary({'name': 'test_user'})",
        )
        assert result[1] == TokenResponse(
            from_site=test_case.test_variant,
            from_node=None,
            frame=Frame(frame=[0]),
            to_place="_dfm_status_",
            data=StopToken(),
        )
        assert result[2] == TokenResponse(
            from_site=test_case.test_variant,
            from_node=None,
            frame=Frame(frame="stop"),
            to_place="_dfm_status_",
            data=StopToken(),
        )
