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
import logging

import pytest

from nv_dfm_core.api import PickledObject

# Configure logging
logging.basicConfig(level=logging.INFO)


# Define TestClass at module level so it can be pickled
class TestClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, TestClass) and self.x == other.x and self.y == other.y


# Define unpicklable function at module level
def unpicklable_function(x):
    return x + 1


def test_simple_object_serialization():
    # Test with a simple dictionary
    original = {"key": "value", "number": 42}
    obj = PickledObject(value=original)

    # Serialize to JSON
    json_str = obj.model_dump_json()

    # Deserialize back
    new_obj = PickledObject.model_validate_json(json_str)

    assert new_obj.value == original


def test_complex_object_serialization():
    # Test with a more complex nested structure
    original = {"list": [1, 2, 3], "nested": {"tuple": (1, 2, 3), "set": {1, 2, 3}}}
    obj = PickledObject(value=original)

    json_str = obj.model_dump_json()
    new_obj = PickledObject.model_validate_json(json_str)

    assert new_obj.value == original
    assert new_obj.value["nested"]["tuple"] == (1, 2, 3)
    assert new_obj.value["nested"]["set"] == {1, 2, 3}


def test_custom_class_serialization():
    # Test with a custom class instance
    original = TestClass(10, "test")
    obj = PickledObject(value=original)

    json_str = obj.model_dump_json()
    new_obj = PickledObject.model_validate_json(json_str)

    assert new_obj.value == original
    assert new_obj.value.x == 10
    assert new_obj.value.y == "test"


def test_unpicklable_object():
    # Test with a truly unpicklable object (lambda function)
    with pytest.raises(ValueError, match="Value cannot be pickled"):
        PickledObject(value=lambda x: x + 1)


def test_empty_object():
    # Test with None
    obj = PickledObject(value=None)
    json_str = obj.model_dump_json()
    new_obj = PickledObject.model_validate_json(json_str)
    assert new_obj.value is None


def test_invalid_json():
    # Test with invalid JSON
    with pytest.raises(ValueError):
        PickledObject.model_validate_json("invalid json")


def test_invalid_base64():
    # Test with invalid base64 data
    with pytest.raises(ValueError, match="Failed to decode/unpickle value"):
        PickledObject.model_validate_json('{"value": "invalid-base64-data"}')


def test_large_object():
    # Test with a large object
    original = {"data": "x" * 10000}
    obj = PickledObject(value=original)

    json_str = obj.model_dump_json()
    new_obj = PickledObject.model_validate_json(json_str)

    assert new_obj.value == original


def test_tag_in_serialized_json():
    # Test that the tag field appears in the serialized JSON
    obj = PickledObject(value={"test": "data"})
    json_str = obj.model_dump_json()
    json_data = json.loads(json_str)

    assert "tag" in json_data
    assert json_data["tag"] == "__dfm__pickled__"
    assert "value" in json_data
