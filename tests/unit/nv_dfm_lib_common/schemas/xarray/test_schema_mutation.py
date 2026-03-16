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
import pytest

from nv_dfm_lib_common.schemas.xarray import (
    Attribute,
    Coordinate,
    DataVariable,
    XArraySchema,
)
from nv_dfm_lib_common.schemas.xarray.checks import check_dims, check_dtype


@pytest.fixture
def simple_schema():
    """Create a simple schema for testing."""
    return XArraySchema(
        data_vars={
            "temperature": DataVariable(
                dtype=np.dtype(np.float64),
                checks=[check_dims(["time"]), check_dtype(np.dtype(np.float64))],
                name="temperature",
            ),
        },
        coords={
            "time": Coordinate(
                dtype=np.dtype(np.datetime64),
                checks=[check_dims(["time"]), check_dtype(np.dtype(np.datetime64))],
                name="time",
            ),
        },
        attrs={
            "title": Attribute(dtype=str, checks=[], name="title"),
        },
    )


def test_mutable_clone(simple_schema):
    """Test that mutable_clone returns a deep copy with frozen=False."""
    clone = simple_schema.mutable_clone()
    assert clone.frozen is False
    assert clone.data_vars == simple_schema.data_vars
    assert clone.coords == simple_schema.coords
    assert clone.attrs == simple_schema.attrs


def test_freeze(simple_schema):
    """Test that freeze makes the schema immutable."""
    clone = simple_schema.mutable_clone()
    clone.freeze()
    assert clone.frozen is True
    with pytest.raises(ValueError):
        clone.add_data_var(
            "new_var",
            DataVariable(dtype=np.dtype(np.float64), checks=[], name="new_var"),
        )


def test_add_remove_data_var(simple_schema):
    """Test adding and removing a data variable."""
    clone = simple_schema.mutable_clone()
    new_var = DataVariable(dtype=np.dtype(np.float64), checks=[], name="new_var")
    clone.add_data_var("new_var", new_var)
    assert "new_var" in clone.data_vars
    clone.remove_data_var("new_var")
    assert "new_var" not in clone.data_vars


def test_add_remove_coord(simple_schema):
    """Test adding and removing a coordinate."""
    clone = simple_schema.mutable_clone()
    new_coord = Coordinate(dtype=np.dtype(np.float64), checks=[], name="new_coord")
    clone.add_coord("new_coord", new_coord)
    assert "new_coord" in clone.coords
    clone.remove_coord("new_coord")
    assert "new_coord" not in clone.coords


def test_add_remove_attr(simple_schema):
    """Test adding and removing an attribute."""
    clone = simple_schema.mutable_clone()
    new_attr = Attribute(dtype=str, checks=[], name="new_attr")
    clone.add_attr("new_attr", new_attr)
    assert "new_attr" in clone.attrs
    clone.remove_attr("new_attr")
    assert "new_attr" not in clone.attrs


def test_get_variable(simple_schema):
    """Test getting a data variable."""
    var = simple_schema.get_variable("temperature")
    assert var is not None
    assert var.name == "temperature"


def test_get_coord(simple_schema):
    """Test getting a coordinate."""
    coord = simple_schema.get_coord("time")
    assert coord is not None
    assert coord.name == "time"


def test_get_attr(simple_schema):
    """Test getting an attribute."""
    attr = simple_schema.get_attr("title")
    assert attr is not None
    assert attr.name == "title"
