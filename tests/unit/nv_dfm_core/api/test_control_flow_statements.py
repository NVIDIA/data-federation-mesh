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

import pytest

from nv_dfm_core.api import (
    Equal,
    ForEach,
    If,
    Pipeline,
    TryFromCache,
    Yield,
)


@pytest.mark.skip(reason="TryFromCache is not implemented")
def test_try_from_cache_serialization():
    with Pipeline() as p:
        with TryFromCache(key="test_key") as cache:
            cache.cache(value="test_value")
            Yield(value="test_value")

    json = p.model_dump_json(exclude_defaults=False)

    # now pretend to be server-side
    pafter = Pipeline.model_validate_json(json)
    assert p == pafter
    assert type(pafter) is Pipeline


@pytest.mark.skip(reason="ForEach is not implemented")
def test_for_each_serialization():
    with Pipeline() as p:
        with ForEach(seq=["item1", "item2", "item3"]):
            Yield(value="test_value")

    json = p.model_dump_json(exclude_defaults=False)

    # now pretend to be server-side
    pafter = Pipeline.model_validate_json(json)
    assert p == pafter
    assert type(pafter) is Pipeline


def test_if_serialization():
    with Pipeline() as p:
        with If(cond=Equal(operator="==", left=1, right=1)):
            Yield(value="test_value")

    json = p.model_dump_json(exclude_defaults=False)
    # now pretend to be server-side
    pafter = Pipeline.model_validate_json(json)
    assert p == pafter
    assert type(pafter) is Pipeline
