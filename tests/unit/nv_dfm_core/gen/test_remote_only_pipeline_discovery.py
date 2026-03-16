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

from nv_dfm_core.api import Advise, Pipeline, StopToken
from nv_dfm_core.api._node_id import NodeId
from nv_dfm_core.api._place_param import PlaceParam
from nv_dfm_core.exec._frame import Frame
from tests.assets.testfed.fed.api.ops import UnaryOp
from tests._builder_helpers import FedInfoBuilder
from tests._pipeline_compilation_test_base import (
    PipelineCompilationTestBase,
    PipelineTestCase,
    TokenResponse,
)


class TestRemoteOnlyPipelineDiscovery(PipelineCompilationTestBase):
    @staticmethod
    def pipelines() -> Generator[Pipeline, Any, None]:
        """Make sure discovery works if there's no operation on the homesite"""
        with Pipeline(name="fixed_site", mode="discovery") as p:
            op2b = UnaryOp(site="site2", p1=PlaceParam(place="my_param"))

        yield p

    @staticmethod
    def fed_info() -> FedInfoBuilder:
        b = FedInfoBuilder(federation_module_name="tests.assets.testfed")
        _ = (
            b.homesite("site1")
            .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .prov("provider1")
            .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .site("site2")
            .op("ops.UnaryOp", fixed_sec=100.0, fixed_size=100)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .prov("provider1")
            .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .site("site3")
            .op("ops.UnaryOp", fixed_sec=1.0, fixed_size=1)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
            .prov("provider1")
            .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
            .op("ops.NullaryOp", fixed_sec=10.0, fixed_size=10)
        )
        return b

    @staticmethod
    def cases() -> Generator[PipelineTestCase, Any, None]:
        for pipeline in TestRemoteOnlyPipelineDiscovery.pipelines():
            yield PipelineTestCase(
                test_name=None,
                test_variant=None,
                pipeline=pipeline,
                candidate_sites=["site1", "site2", "site3"],
                fed_info=TestRemoteOnlyPipelineDiscovery.fed_info(),
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
        assert len(result) == 3, f"Expected 3 result tokens, but got {len(result)}"
        assert (
            TokenResponse(
                from_site="site2",
                from_node=None,
                frame=Frame(token="@dfm-control-token", frame=[0]),
                to_place="_dfm_discovery_",
                data=[(NodeId(tag="@dfm-node", ident=0), None)],
            )
            in result
        ), "Expected discovery token from site2"
        assert (
            TokenResponse(
                from_site="site2",
                from_node=None,
                frame=Frame(token="@dfm-control-token", frame=[0]),
                to_place="_dfm_status_",
                data=StopToken(token="@dfm-stop-token"),
            )
            in result
        ), "Expected local stop token from site2"
        assert (
            TokenResponse(
                from_site="site2",
                from_node=None,
                frame=Frame(token="@dfm-control-token", frame="stop"),
                to_place="_dfm_status_",
                data=StopToken(token="@dfm-stop-token"),
            )
            in result
        ), "Expected final stop token from site2"
