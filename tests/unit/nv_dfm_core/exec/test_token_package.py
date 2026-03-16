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

import json
from typing import Literal

import pytest
from pydantic import BaseModel

from nv_dfm_core.api.pydantic import PolymorphicBaseModel
from nv_dfm_core.exec import (
    TokenPackage,
)
from nv_dfm_core.exec._frame import Frame


# Test models
class SimpleModel(BaseModel):
    name: str
    age: int
    optional_field: str | None = None


class NestedModel(BaseModel):
    simple: SimpleModel
    value: float


# Test cases
def test_json_serializable_data():
    """Test serialization of a token package with JSON-serializable data."""
    data = {"key": "value", "numbers": [1, 2, 3]}
    package = TokenPackage.wrap_data(
        source_job="job1",
        target_job="job1",
        is_yield=False,
        source_site="site1",
        source_node=1,
        target_site="site3",
        target_place="place1",
        frame=Frame.start_frame(0),
        data=data,
    )

    serialized = package.model_dump()
    # With the new implementation, JSON-serializable dicts are returned as-is
    # (not stringified) to avoid collision with NVFlare's {var} substitution
    assert serialized["tagged_data"] == ("json", data)

    deserialized = TokenPackage.model_validate(serialized)
    assert deserialized.unwrap_data() == data


def test_pydantic_model_data():
    """Test serialization of a token package with a Pydantic model."""
    model = SimpleModel(name="test", age=42)
    package = TokenPackage.wrap_data(
        source_job="job1",
        target_job="job1",
        is_yield=False,
        source_site="site1",
        source_node=1,
        target_site="site3",
        target_place="place1",
        frame=Frame.start_frame(0),
        data=model,
    )

    serialized = package.model_dump()
    assert serialized["tagged_data"] == (
        "BaseModel",
        {"class": "test_token_package.SimpleModel", "data": model.model_dump()},
    )

    deserialized = TokenPackage.model_validate(serialized)
    assert isinstance(deserialized.unwrap_data(), SimpleModel)
    assert deserialized.unwrap_data().name == "test"
    assert deserialized.unwrap_data().age == 42


def test_nested_pydantic_model():
    """Test serialization of a token package with nested Pydantic models."""
    simple = SimpleModel(name="test", age=42)
    nested = NestedModel(simple=simple, value=3.14)
    package = TokenPackage.wrap_data(
        source_job="job1",
        target_job="job1",
        is_yield=False,
        source_site="site1",
        source_node=1,
        target_site="site3",
        target_place="place1",
        frame=Frame.start_frame(0),
        data=nested,
    )

    serialized = package.model_dump()
    assert serialized["tagged_data"] == (
        "BaseModel",
        {"class": "test_token_package.NestedModel", "data": nested.model_dump()},
    )

    deserialized = TokenPackage.model_validate(serialized)
    assert isinstance(deserialized.unwrap_data(), NestedModel)
    assert isinstance(deserialized.unwrap_data().simple, SimpleModel)
    assert deserialized.unwrap_data().simple.name == "test"
    assert deserialized.unwrap_data().value == 3.14


class PolymorphicModel(PolymorphicBaseModel):
    """Keep this class outside of the test function because
    nesting in functions causes issues with finding the correct
    fully qualified class name."""

    dfm_class_name: Literal["test_token_package.PolymorphicModel"] = (
        "test_token_package.PolymorphicModel"
    )
    type: str
    value: int


def test_polymorphic_base_model():
    """Test serialization of a token package with a PolymorphicBaseModel."""

    model = PolymorphicModel(type="test", value=42)
    package = TokenPackage.wrap_data(
        source_job="job1",
        target_job="job1",
        is_yield=False,
        source_site="site1",
        source_node=1,
        target_site="site3",
        target_place="place1",
        frame=Frame.start_frame(0),
        data=model,
    )

    serialized = package.model_dump()
    assert serialized["tagged_data"] == (
        "BaseModel",
        {"class": "test_token_package.PolymorphicModel", "data": model.model_dump()},
    )

    deserialized = TokenPackage.model_validate(serialized)
    assert isinstance(deserialized.unwrap_data(), PolymorphicModel)
    assert deserialized.unwrap_data().type == "test"
    assert deserialized.unwrap_data().value == 42


class CustomObject:
    """Keep this class outside of the test function because
    nesting in functions causes issues with finding the correct
    fully qualified class name."""

    def __init__(self, value: int):
        self.value: int = value


def test_arbitrary_python_object():
    """Test serialization of a token package with an arbitrary Python object."""

    obj = CustomObject(value=42)
    package = TokenPackage.wrap_data(
        source_job="job1",
        target_job="job1",
        is_yield=False,
        source_site="site1",
        source_node=1,
        target_site="site3",
        target_place="place1",
        frame=Frame.start_frame(0),
        data=obj,
    )

    serialized = package.model_dump()
    assert serialized["tagged_data"][0] == "PickledObject"
    assert serialized["tagged_data"][1]["tag"] == "__dfm__pickled__"

    deserialized = TokenPackage.model_validate(serialized)
    assert isinstance(deserialized.unwrap_data(), CustomObject)
    assert deserialized.unwrap_data().value == 42


def test_invalid_class_path():
    """Test error handling for invalid class paths."""
    serialized = {
        "source_site": "site1",
        "source_node": 1,
        "source_job": "job1",
        "target_job": "job1",
        "is_yield": False,
        "target_site": "site3",
        "target_place": "place1",
        "frame": Frame.start_frame(0).model_dump(),
        "tagged_data": (
            "BaseModel",
            {"class": "non.existent.Class", "data": {"name": "test"}},
        ),
    }

    with pytest.raises(ValueError, match="Could not locate class"):
        _ = TokenPackage.model_validate(serialized).unwrap_data()


class NonPydanticClass:
    pass


def test_non_pydantic_class():
    """Test error handling for non-Pydantic classes."""

    serialized = {
        "source_site": "site1",
        "source_node": 1,
        "source_job": "job1",
        "target_job": "job1",
        "is_yield": False,
        "target_site": "site3",
        "target_place": "place1",
        "frame": Frame.start_frame(0).model_dump(),
        "tagged_data": (
            "BaseModel",
            {"class": "test_token_package.NonPydanticClass", "data": {"name": "test"}},
        ),
    }

    with pytest.raises(ValueError, match="is not a Pydantic model"):
        _ = TokenPackage.model_validate(serialized).unwrap_data()
