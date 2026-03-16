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

import numpy as np

from nv_dfm_core.api import PickledObject


def test_complex_numpy_structures():
    """Comprehensive test for serialization/deserialization of complex numpy structures.

    This test covers:
    - Nested dictionaries containing numpy arrays
    - Multidimensional arrays of integers (2D, 3D)
    - Multidimensional arrays of floats (2D, 3D)
    - Structured arrays with multiple fields
    """
    # Create a complex nested structure with various numpy array types
    original = {
        "level1": {
            "level2": {
                # Multidimensional integer arrays
                "int_2d": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
                "int_3d": np.array(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64
                ),
                # Multidimensional float arrays
                "float_2d": np.array(
                    [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32
                ),
                "float_3d": np.array(
                    [[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]],
                    dtype=np.float64,
                ),
                # Structured array
                "structured": np.array(
                    [("Alice", 25, 55.5), ("Bob", 30, 70.2), ("Charlie", 35, 80.1)],
                    dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
                ),
            },
            # Additional arrays at level1
            "int_1d": np.array([10, 20, 30, 40], dtype=np.int32),
            "float_1d": np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
        },
    }

    # Serialize and deserialize
    obj = PickledObject(value=original)
    json_str = obj.model_dump_json()
    new_obj = PickledObject.model_validate_json(json_str)

    # Verify nested structure
    assert "level1" in new_obj.value
    assert "level2" in new_obj.value["level1"]

    # Verify integer arrays
    assert "int_2d" in new_obj.value["level1"]["level2"]
    assert np.array_equal(
        new_obj.value["level1"]["level2"]["int_2d"],
        original["level1"]["level2"]["int_2d"],
    )
    assert (
        new_obj.value["level1"]["level2"]["int_2d"].dtype
        == original["level1"]["level2"]["int_2d"].dtype
    )
    assert (
        new_obj.value["level1"]["level2"]["int_2d"].shape
        == original["level1"]["level2"]["int_2d"].shape
    )

    assert "int_3d" in new_obj.value["level1"]["level2"]
    assert np.array_equal(
        new_obj.value["level1"]["level2"]["int_3d"],
        original["level1"]["level2"]["int_3d"],
    )
    assert (
        new_obj.value["level1"]["level2"]["int_3d"].dtype
        == original["level1"]["level2"]["int_3d"].dtype
    )
    assert (
        new_obj.value["level1"]["level2"]["int_3d"].shape
        == original["level1"]["level2"]["int_3d"].shape
    )

    assert "int_1d" in new_obj.value["level1"]
    assert np.array_equal(
        new_obj.value["level1"]["int_1d"], original["level1"]["int_1d"]
    )
    assert new_obj.value["level1"]["int_1d"].dtype == original["level1"]["int_1d"].dtype
    assert new_obj.value["level1"]["int_1d"].shape == original["level1"]["int_1d"].shape

    # Verify float arrays
    assert "float_2d" in new_obj.value["level1"]["level2"]
    assert np.allclose(
        new_obj.value["level1"]["level2"]["float_2d"],
        original["level1"]["level2"]["float_2d"],
    )
    assert (
        new_obj.value["level1"]["level2"]["float_2d"].dtype
        == original["level1"]["level2"]["float_2d"].dtype
    )
    assert (
        new_obj.value["level1"]["level2"]["float_2d"].shape
        == original["level1"]["level2"]["float_2d"].shape
    )

    assert "float_3d" in new_obj.value["level1"]["level2"]
    assert np.allclose(
        new_obj.value["level1"]["level2"]["float_3d"],
        original["level1"]["level2"]["float_3d"],
    )
    assert (
        new_obj.value["level1"]["level2"]["float_3d"].dtype
        == original["level1"]["level2"]["float_3d"].dtype
    )
    assert (
        new_obj.value["level1"]["level2"]["float_3d"].shape
        == original["level1"]["level2"]["float_3d"].shape
    )

    assert "float_1d" in new_obj.value["level1"]
    assert np.allclose(
        new_obj.value["level1"]["float_1d"], original["level1"]["float_1d"]
    )
    assert (
        new_obj.value["level1"]["float_1d"].dtype
        == original["level1"]["float_1d"].dtype
    )
    assert (
        new_obj.value["level1"]["float_1d"].shape
        == original["level1"]["float_1d"].shape
    )

    # Verify structured array
    assert "structured" in new_obj.value["level1"]["level2"]
    assert (
        new_obj.value["level1"]["level2"]["structured"].dtype
        == original["level1"]["level2"]["structured"].dtype
    )
    assert (
        new_obj.value["level1"]["level2"]["structured"].shape
        == original["level1"]["level2"]["structured"].shape
    )
    assert np.array_equal(
        new_obj.value["level1"]["level2"]["structured"],
        original["level1"]["level2"]["structured"],
    )
    assert new_obj.value["level1"]["level2"]["structured"][0]["name"] == "Alice"
    assert new_obj.value["level1"]["level2"]["structured"][0]["age"] == 25
    assert np.isclose(
        new_obj.value["level1"]["level2"]["structured"][0]["weight"], 55.5
    )
