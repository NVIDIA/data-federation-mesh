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

from nv_dfm_core.api import Pipeline, StopToken, Yield
from nv_dfm_core.exec._frame import Frame
from tests.assets.testfed.fed.api.ops import OpWithDefault
from tests._builder_helpers import FedInfoBuilder
from tests._pipeline_compilation_test_base import (
    PipelineCompilationTestBase,
    PipelineTestCase,
    TokenResponse,
)


class TestOperationWithDefault(PipelineCompilationTestBase):
    @staticmethod
    def pipelines() -> Generator[Pipeline, Any, None]:
        with Pipeline(name="fixed_site") as p:
            op = OpWithDefault()
            _ = Yield(value=op)

        yield p

    @staticmethod
    def fed_info() -> FedInfoBuilder:
        b = FedInfoBuilder(federation_module_name="tests.assets.testfed")
        _ = b.homesite("site1").op("ops.OpWithDefault", fixed_sec=10.0, fixed_size=10)
        return b

    @staticmethod
    def cases() -> Generator[PipelineTestCase, Any, None]:
        for pipeline in TestOperationWithDefault.pipelines():
            yield PipelineTestCase(
                test_name=None,
                test_variant=None,
                pipeline=pipeline,
                candidate_sites=["site1"],
                fed_info=TestOperationWithDefault.fed_info(),
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
        assert len(result) == 3, "Expected 3 result tokens"

        assert (
            TokenResponse(
                from_site="site1",
                from_node=0,
                frame=Frame(token="@dfm-control-token", frame=[0]),
                to_place="yield",
                data="op-with-default('api-default')",
            )
            in result
        ), "Expected yield token from site1"
        assert TokenResponse(
            from_site="site1",
            from_node=None,
            frame=Frame(token="@dfm-control-token", frame=[0]),
            to_place="_dfm_status_",
            data=StopToken(token="@dfm-stop-token"),
        ), "Expected stop token for frame 0 from site1"
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
