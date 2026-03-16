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

from collections.abc import Iterable

import pytest

from nv_dfm_core.api import Advise
from nv_dfm_core.api.discovery import ErrorFieldAdvice, PartialError, SingleFieldAdvice
from nv_dfm_core.exec.discovery import (
    AdviceBuilder,
    AdvisedDateRange,
    AdvisedDict,
    AdvisedLiteral,
    AdvisedOneOf,
    AdvisedSubsetOf,
    field_advisor,
)


class MockTextureStoreAdapter:  # type: ignore
    @field_advisor("simulation", order=0)
    async def available_simulations(self, _value, _context):
        return AdvisedOneOf(values=["sim1", "sim2"], break_on_advice=True)

    @field_advisor("location", order=1)
    async def available_locations(self, _value, _context):
        return AdvisedOneOf(values=["loc1", "loc2"], split_on_advice=True)

    @field_advisor("timestamps", order=2)
    async def available_timestamps(self, _value, context):
        if context.get("location") == "loc1":
            return AdvisedSubsetOf(values=["ts1", "ts2", "ts3"])
        return AdvisedSubsetOf(values=["ts45", "ts46"])

    @field_advisor("variables")
    async def available_variables(self, _value, context):
        if context.get("location") == "loc1":
            return AdvisedOneOf(
                [AdvisedLiteral("*"), AdvisedSubsetOf(["temp", "height"])]
            )
        return AdvisedOneOf(
            [AdvisedLiteral("*"), AdvisedSubsetOf(["u_wind", "v_wind"])],
            split_on_advice=True,
        )

    @field_advisor("selection")
    async def valid_selections(self, _value, _context):
        none_advice = AdvisedLiteral(None)
        time_advice = AdvisedDateRange(start="today", end="tomorrow")
        dict_advice = AdvisedDict({"time": time_advice}, allow_extras=True)
        return AdvisedOneOf(values=[none_advice, dict_advice])


pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_wrong_location():
    params = {"simulation": "sim1", "location": "home"}

    adapter = MockTextureStoreAdapter()
    advice = await AdviceBuilder.build_advice(adapter, **params)
    # print(advice)
    assert isinstance(advice, SingleFieldAdvice)
    assert advice.field == "location"
    assert not advice.has_good_options()
    assert (
        "Expected one of ['loc1', 'loc2'] but got home"
        in advice.collect_error_messages()["location"]
    )


@pytest.mark.asyncio
async def test_break_on_sims():
    params = {
        "simulation": Advise(),
    }

    adapter = MockTextureStoreAdapter()
    advice = await AdviceBuilder.build_advice(adapter, **params)
    # print(advice)
    assert isinstance(advice, SingleFieldAdvice)
    assert advice.field == "simulation"
    assert advice.has_good_options()
    assert isinstance(advice, Iterable)
    options = [opt for opt in advice]  # pylint: disable=not-an-iterable
    assert "sim1" in options
    assert "sim2" in options
    with pytest.raises(PartialError):
        advice.select("sim1")


@pytest.mark.asyncio
async def test_discover_timestamps():
    params = {
        "simulation": "sim1",
        "location": "loc1",
        "timestamps": Advise(),
        "selection": {"time": "today"},
    }

    adapter = MockTextureStoreAdapter()
    advice = await AdviceBuilder.build_advice(adapter, **params)
    # print(advice)
    assert isinstance(advice, SingleFieldAdvice)
    assert advice.field == "timestamps"
    assert isinstance(advice.value, list)
    assert set(advice.value) == set(["ts1", "ts2", "ts3"])


@pytest.mark.asyncio
async def test_discover_variables():
    params = {
        "simulation": "sim1",
        "location": "loc1",
        "timestamps": ["ts1", "ts3"],
        "variables": Advise(),
        "selection": {"time": "today"},
    }

    adapter = MockTextureStoreAdapter()
    advice = await AdviceBuilder.build_advice(adapter, **params)
    # print(advice)
    assert isinstance(advice, SingleFieldAdvice)
    assert advice.field == "variables"
    assert advice.value == ["*", ["temp", "height"]]


@pytest.mark.asyncio
async def test_wrong_timestamps():
    params = {
        "simulation": "sim1",
        "location": "loc1",
        "timestamps": ["ts1", "ts49"],
        "variables": Advise(),
        "selection": {"time": "today"},
    }

    adapter = MockTextureStoreAdapter()
    advice = await AdviceBuilder.build_advice(adapter, **params)
    # print(advice)
    assert isinstance(advice, SingleFieldAdvice)
    assert advice.field == "timestamps"
    assert isinstance(advice.value, list)
    assert set(advice.value) == set(["ts1", "ts49"])
    assert advice.edge == ErrorFieldAdvice(
        msg="Expected subset of values ['ts1', 'ts2', 'ts3']"
        " but got ['ts1', 'ts49']. Value ts49 is not allowed."
    )
