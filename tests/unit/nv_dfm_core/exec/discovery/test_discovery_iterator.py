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

from nv_dfm_core.api.discovery import (
    BranchFieldAdvice,
    ErrorFieldAdvice,
    SingleFieldAdvice,
)
from nv_dfm_core.exec.discovery import AdvisedDateRange


def build_discovery_message() -> BranchFieldAdvice:
    # assume we asked for advise on fields 'provider' and 'simulation' and specified a date.
    # This returned an advice tree, two possible providers with two simulations each, and in each
    # simulation the date exists but the other results in an error

    tree = BranchFieldAdvice(
        field="provider",
        branches=[
            (
                "eos",
                SingleFieldAdvice(
                    field="sim",
                    value=["eos_icon", "eos_graf"],
                    edge=SingleFieldAdvice(
                        field="timestamp",
                        value=AdvisedDateRange(
                            start="today", end="tomorrow"
                        ).as_pydantic_value(),
                        edge=ErrorFieldAdvice(
                            msg="The specified date does not exist in this simulation"
                        ),
                    ),
                ),
            ),
            (
                "ganon",
                BranchFieldAdvice(
                    field="sim",
                    branches=[
                        ("ganon_icon", None),
                        (
                            "ganon_graf",
                            SingleFieldAdvice(
                                field="timestamp",
                                value=AdvisedDateRange(
                                    start="today", end="tomorrow"
                                ).as_pydantic_value(),
                                edge=ErrorFieldAdvice(
                                    msg="The specified date does not exist in this simulation"
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        ],
    )
    return tree


def test_iterator():
    provider_field = build_discovery_message()

    assert provider_field.has_good_options()
    errors = provider_field.collect_error_messages()
    assert "timestamp" in errors
    assert errors["timestamp"] == set(
        ["The specified date does not exist in this simulation"]
    )

    providers = list(prov for prov in provider_field)
    # 'eos' is not a viable option, because it doesn't lead to a non-error end
    assert providers == ["ganon"]
    sim_field = provider_field.select("ganon")
    assert sim_field
    assert sim_field.has_good_options()
    errors = sim_field.collect_error_messages()
    assert "timestamp" in errors
    sims = list(sim for sim in sim_field)
    assert sims == ["ganon_icon"]
