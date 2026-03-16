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
from nv_dfm_core.api._place_param import PlaceParam
from nv_dfm_core.exec._frame import Frame

from tests.assets.testfed.fed.api.ops import UnaryOp
from tests._builder_helpers import FedInfoBuilder
from tests._pipeline_compilation_test_base import (
    PipelineCompilationTestBase,
    PipelineTestCase,
    TokenResponse,
)


class TestPlaceParams(PipelineCompilationTestBase):
    @staticmethod
    def pipelines() -> Generator[Pipeline, Any, None]:
        with Pipeline(name="fixed_site") as p:
            op1 = UnaryOp(site="site2", p1=PlaceParam(place="my_param", multiuse=True))
            op2 = UnaryOp(site="site3", p1=PlaceParam(place="my_param", multiuse=True))
            _ = Yield(value=op1, multiuse=True)
            _ = Yield(value=op2, multiuse=True)

        yield p

    @staticmethod
    def fed_info() -> FedInfoBuilder:
        b = FedInfoBuilder(federation_module_name="tests.assets.testfed")
        _ = (
            b.homesite("site1")
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOpTuple", fixed_sec=10.0, fixed_size=10)
            .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .prov("provider1")
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOpTuple", fixed_sec=10.0, fixed_size=10)
            .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .site("site2")
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOpTuple", fixed_sec=10.0, fixed_size=10)
            .op("ops.UnaryOp", fixed_sec=100.0, fixed_size=100)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .prov("provider1")
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOpTuple", fixed_sec=10.0, fixed_size=10)
            .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .site("site3")
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOpTuple", fixed_sec=10.0, fixed_size=10)
            .op("ops.UnaryOp", fixed_sec=1.0, fixed_size=1)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .prov("provider1")
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOpTuple", fixed_sec=10.0, fixed_size=10)
            .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
        )
        return b

    @staticmethod
    def cases() -> Generator[PipelineTestCase, Any, None]:
        for pipeline in TestPlaceParams.pipelines():
            yield PipelineTestCase(
                test_name=None,
                test_variant=None,
                pipeline=pipeline,
                candidate_sites=["site1", "site2", "site3"],
                fed_info=TestPlaceParams.fed_info(),
                expected_net_irs=None,
                run_params=[{"my_param": "first"}, {"my_param": "second"}],
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
        assert len(result) == 7, "Expected 7 result tokens"
        assert (
            TokenResponse(
                from_site="site3",
                from_node=1,
                frame=Frame(token="@dfm-control-token", frame=[0]),
                to_place="yield",
                data="unary('first')",
            )
            in result
        ), "Expected yield token from site3 for frame 0"
        assert (
            TokenResponse(
                from_site="site2",
                from_node=0,
                frame=Frame(token="@dfm-control-token", frame=[0]),
                to_place="yield",
                data="unary('first')",
            )
            in result
        ), "Expected yield token from site2 for frame 0"
        assert (
            TokenResponse(
                from_site="site3",
                from_node=1,
                frame=Frame(token="@dfm-control-token", frame=[1]),
                to_place="yield",
                data="unary('second')",
            )
            in result
        ), "Expected yield token from site3 for frame 1"
        assert (
            TokenResponse(
                from_site="site2",
                from_node=0,
                frame=Frame(token="@dfm-control-token", frame=[1]),
                to_place="yield",
                data="unary('second')",
            )
            in result
        ), "Expected yield token from site2 for frame 1"
        assert (
            TokenResponse(
                from_site="site2",
                from_node=None,
                frame=Frame(token="@dfm-control-token", frame=[0]),
                to_place="_dfm_status_",
                data=StopToken(token="@dfm-stop-token"),
            )
            in result
        ), "Expected stop token from site2 for frame 0"
        assert (
            TokenResponse(
                from_site="site2",
                from_node=None,
                frame=Frame(token="@dfm-control-token", frame=[1]),
                to_place="_dfm_status_",
                data=StopToken(token="@dfm-stop-token"),
            )
            in result
        ), "Expected stop token from site2 for frame 1"
        assert (
            result[-1]
            == TokenResponse(
                from_site="site2",
                from_node=None,
                frame=Frame(token="@dfm-control-token", frame="stop"),
                to_place="_dfm_status_",
                data=StopToken(token="@dfm-stop-token"),
            )
            in result
        ), "Expected stop token"
