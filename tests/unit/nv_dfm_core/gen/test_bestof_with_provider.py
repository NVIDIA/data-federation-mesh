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

from nv_dfm_core.api import Pipeline, Yield, StopToken
from nv_dfm_core.api._best_of import BestOf
from nv_dfm_core.api._place_param import PlaceParam
from nv_dfm_core.exec._frame import Frame

from tests.assets.testfed.fed.api.ops import NullaryOp, UnaryOp
from tests._builder_helpers import FedInfoBuilder
from tests._pipeline_compilation_test_base import (
    PipelineCompilationTestBase,
    PipelineTestCase,
    TokenResponse,
)


class TestBestOfWithProvider(PipelineCompilationTestBase):
    @staticmethod
    def pipelines() -> Generator[Pipeline, Any, None]:
        with Pipeline(name="fixed_site") as p:
            # NullaryOp doesn't exist on site3.provider1. Exists on site1 but that's not an option
            op1 = NullaryOp(site=BestOf(sites=["site2", "site3"]), provider="provider1")
            # unary op exists on all three sites. Should pick site1 perf-wise
            op2 = UnaryOp(p1=op1)
            _ = Yield(value=op2)

        yield p

    @staticmethod
    def fed_info() -> FedInfoBuilder:
        b = FedInfoBuilder(federation_module_name="tests.assets.testfed")
        _ = (
            b.homesite("site1")
            .op("ops.UnaryOp", fixed_sec=1.0, fixed_size=1)
            .prov("provider1")
            .op("ops.NullaryOp", fixed_sec=1.0, fixed_size=1)
            .site("site2")
            .op("ops.UnaryOp", fixed_sec=100.0, fixed_size=100)
            .prov("provider1")
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .site("site3")
            .op("ops.UnaryOp", fixed_sec=100.0, fixed_size=100)
        )
        b.comm("site2", "site1", fixed_sec=1.0)
        return b

    @staticmethod
    def cases() -> Generator[PipelineTestCase, Any, None]:
        for pipeline in TestBestOfWithProvider.pipelines():
            yield PipelineTestCase(
                test_name=None,
                test_variant=None,
                pipeline=pipeline,
                candidate_sites=["site1", "site2", "site3"],
                fed_info=TestBestOfWithProvider.fed_info(),
                expected_net_irs=None,
                run_params={},
                debug=False,
            )

    @pytest.mark.parametrize("test_case", cases())
    def test_compile_and_run_single_pipeline(
        self,
        test_case: PipelineTestCase,
    ) -> None:
        result = self.single_pipeline_test(
            test_case=test_case,
        )
        assert result, "No result"
        # we were sending two frames, each resulting in 3 messages (from site2, from site3, and the stop) plus the final stop
        assert len(result) == 3, "Expected 3 result tokens"
        assert (
            TokenResponse(
                from_site="site1",
                from_node=1,
                frame=Frame(token="@dfm-control-token", frame=[0]),
                to_place="yield",
                data="unary('nullary')",
            )
            in result
        ), "Expected yield token from site1 for frame 0"
        assert (
            TokenResponse(
                from_site="site1",
                from_node=None,
                frame=Frame(token="@dfm-control-token", frame=[0]),
                to_place="_dfm_status_",
                data=StopToken(token="@dfm-stop-token"),
            )
            in result
        ), "Expected stop token from site2 for frame 0"
        assert (
            result[-1]
            == TokenResponse(
                from_site="site1",
                from_node=None,
                frame=Frame(token="@dfm-control-token", frame="stop"),
                to_place="_dfm_status_",
                data=StopToken(token="@dfm-stop-token"),
            )
            in result
        ), "Expected stop token"
