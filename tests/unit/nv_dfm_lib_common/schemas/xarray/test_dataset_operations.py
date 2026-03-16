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

"""Tests for xarray-schema dataset operations."""

import numpy as np
import pytest
import xarray as xr

from nv_dfm_lib_common.schemas.xarray import (
    Attribute,
    Coordinate,
    DataVariable,
    XArraySchema,
    check_dims,
    check_dtype,
    check_max_missing,
    check_range,
)


@pytest.fixture
def simple_schema():
    """Create a simple schema for testing."""
    return XArraySchema(
        data_vars={
            "var1": DataVariable(
                dtype=np.dtype(np.float64),
                checks=[
                    check_dims(["time", "x", "y"]),
                    check_dtype(np.dtype(np.float64)),
                    check_range(0, 100),
                    check_max_missing(0.5),  # Allow up to 50% missing values
                ],
                required=True,
                name="var1",
                metadata={"units": "m"},
            ),
            "var2": DataVariable(
                dtype=np.dtype(np.float64),
                checks=[
                    check_dims(["time", "x", "y"]),
                    check_dtype(np.dtype(np.float64)),
                    check_range(-50, 50),
                ],
                required=True,
                name="var2",
                metadata={"units": "K"},
            ),
        },
        coords={
            "time": Coordinate(
                dtype=np.dtype(np.datetime64),
                checks=[check_dims(["time"])],
                required=True,
            ),
            "x": Coordinate(
                dtype=np.dtype(np.float64),
                checks=[check_dims(["x"])],
                required=True,
            ),
            "y": Coordinate(
                dtype=np.dtype(np.float64),
                checks=[check_dims(["y"])],
                required=True,
            ),
        },
        attrs={
            "title": Attribute(
                dtype=str,
                checks=[],
                required=True,
            ),
            "version": Attribute(
                dtype=str,
                checks=[],
                required=True,
            ),
        },
    )


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    time = np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]")
    x = np.linspace(0, 10, 5)
    y = np.linspace(0, 10, 5)

    return xr.Dataset(
        data_vars={
            "var1": xr.DataArray(
                np.random.uniform(0, 100, (10, 5, 5)),
                dims=["time", "x", "y"],
                attrs={"units": "m"},
            ),
            "var2": xr.DataArray(
                np.random.uniform(-50, 50, (10, 5, 5)),
                dims=["time", "x", "y"],
                attrs={"units": "K"},
            ),
        },
        coords={
            "time": xr.DataArray(time, dims=["time"]),
            "x": xr.DataArray(x, dims=["x"]),
            "y": xr.DataArray(y, dims=["y"]),
        },
        attrs={
            "title": "Test Dataset",
            "version": "1.0",
        },
    )


def test_clean_removes_extra_components(simple_dataset, simple_schema):
    """Test that clean removes extra variables, coordinates, and attributes."""
    # Add extra components
    dataset = simple_dataset.copy()
    # Create extra variable with compatible dimensions
    dataset["extra_var"] = xr.DataArray(
        np.random.rand(10, 5, 5), dims=["time", "x", "y"]
    )
    # Create extra coordinate with compatible dimension
    dataset["extra_coord"] = xr.DataArray(np.random.rand(5), dims=["x"])
    dataset.attrs["extra_attr"] = "value"

    # Clean the dataset
    cleaned = simple_schema.clean(dataset)

    # Check that extra components were removed
    assert "extra_var" not in cleaned.data_vars
    assert "extra_coord" not in cleaned.coords
    assert "extra_attr" not in cleaned.attrs

    # Check that required components were preserved
    assert "var1" in cleaned.data_vars
    assert "var2" in cleaned.data_vars
    assert "time" in cleaned.coords
    assert "title" in cleaned.attrs
    assert "version" in cleaned.attrs


def test_clean_preserves_data(simple_dataset, simple_schema):
    """Test that clean preserves data values and attributes."""
    cleaned = simple_schema.clean(simple_dataset)

    # Check data values
    assert np.array_equal(cleaned["var1"].values, simple_dataset["var1"].values)
    assert np.array_equal(cleaned["var2"].values, simple_dataset["var2"].values)

    # Check attributes
    assert cleaned["var1"].attrs == simple_dataset["var1"].attrs
    assert cleaned["var2"].attrs == simple_dataset["var2"].attrs
    assert cleaned.attrs == simple_dataset.attrs


def test_stack_to_ndarray_shape(simple_dataset, simple_schema):
    """Test that stack_to_ndarray produces correct shape."""
    stacked = simple_schema.stack_to_ndarray(simple_dataset)

    # Check shape: (time, x, y, vars)
    assert stacked.shape == (10, 5, 5, 2)


def test_stack_to_ndarray_order(simple_dataset, simple_schema):
    """Test that stack_to_ndarray preserves variable order."""
    stacked = simple_schema.stack_to_ndarray(simple_dataset)

    # Check that variables are stacked in the correct order
    assert np.array_equal(stacked[..., 0], simple_dataset["var1"].values)
    assert np.array_equal(stacked[..., 1], simple_dataset["var2"].values)


def test_stack_to_ndarray_validation(simple_dataset, simple_schema):
    """Test that stack_to_ndarray validates the dataset."""
    # Create invalid dataset
    invalid_dataset = simple_dataset.copy()
    invalid_dataset["var1"] = invalid_dataset["var1"].astype(np.int32)

    # Try to stack invalid dataset
    with pytest.raises(ValueError) as exc_info:
        simple_schema.stack_to_ndarray(invalid_dataset)
    assert "failed check" in str(exc_info.value)


def test_unstack_from_ndarray_basic(simple_dataset, simple_schema):
    """Test basic unstacking functionality."""
    # First stack the dataset
    stacked = simple_schema.stack_to_ndarray(simple_dataset)

    # Then unstack it
    unstacked = simple_schema.unstack_from_ndarray(stacked)

    # Check that the unstacked dataset matches the original
    assert "var1" in unstacked.data_vars
    assert "var2" in unstacked.data_vars
    assert np.array_equal(unstacked["var1"].values, simple_dataset["var1"].values)
    assert np.array_equal(unstacked["var2"].values, simple_dataset["var2"].values)

    # Check that metadata was preserved
    assert unstacked["var1"].attrs == simple_dataset["var1"].attrs
    assert unstacked["var2"].attrs == simple_dataset["var2"].attrs


def test_unstack_from_ndarray_custom_vars(climate_dataset, climate_schema):
    """Test unstacking with custom variable names that exist in the schema."""
    # Stack the dataset
    stacked = climate_schema.stack_to_ndarray(climate_dataset)

    # Define custom variable names that exist in the schema
    custom_vars = ["temperature", "precipitation"]

    # Unstack with custom variable names
    unstacked = climate_schema.unstack_from_ndarray(
        stacked, var_names=custom_vars, original_dataset=climate_dataset
    )

    # Check that the custom variable names are used
    assert "temperature" in unstacked
    assert "precipitation" in unstacked
    assert "temp" not in unstacked
    assert "precip" not in unstacked

    # Check that the data is preserved
    np.testing.assert_array_equal(
        unstacked["temperature"].values, climate_dataset["temperature"].values
    )
    np.testing.assert_array_equal(
        unstacked["precipitation"].values, climate_dataset["precipitation"].values
    )

    # Check that attributes are preserved
    assert unstacked["temperature"].attrs == climate_dataset["temperature"].attrs
    assert unstacked["precipitation"].attrs == climate_dataset["precipitation"].attrs


def test_unstack_from_ndarray_invalid_shape(simple_dataset, simple_schema):
    """Test unstacking with invalid array shape."""
    # Create an array with wrong number of variables
    stacked = np.random.rand(10, 5, 5, 3)  # 3 variables instead of 2

    # Try to unstack
    with pytest.raises(ValueError) as exc_info:
        simple_schema.unstack_from_ndarray(stacked)
    assert "Last dimension of array (3) doesn't match number of variables (2)" in str(
        exc_info.value
    )


def test_unstack_from_ndarray_invalid_var(simple_dataset, simple_schema):
    """Test unstacking with invalid variable name."""
    # Stack the dataset
    stacked = simple_schema.stack_to_ndarray(simple_dataset)

    # Try to unstack with invalid variable name
    custom_vars = ["temp", "invalid_var"]
    with pytest.raises(ValueError) as exc_info:
        simple_schema.unstack_from_ndarray(stacked, var_names=custom_vars)
    assert "not found in schema" in str(exc_info.value)


def test_unstack_from_ndarray_missing_dims(simple_dataset, simple_schema):
    """Test unstacking with missing dimensions."""
    # Create an array with missing dimensions
    stacked = np.random.rand(10, 5, 2)  # Missing one dimension (should be 10, 5, 5, 2)

    # Try to unstack
    with pytest.raises(ValueError) as exc_info:
        simple_schema.unstack_from_ndarray(stacked)
    assert "different number of dimensions on data and dims" in str(exc_info.value)


def test_schema_clean(climate_dataset, climate_schema):
    """Test cleaning a dataset to match schema."""
    # Add extra components to the dataset
    dataset_with_extra = climate_dataset.copy()
    dataset_with_extra["extra_var"] = xr.DataArray(
        np.random.rand(10, 5, 5), dims=["time", "latitude", "longitude"]
    )
    dataset_with_extra["extra_coord"] = xr.DataArray(
        np.random.rand(5),
        dims=["latitude"],  # Use an existing dimension name
    )

    # Clean the dataset
    cleaned = climate_schema.clean(dataset_with_extra)

    # Check that extra components are removed
    assert "extra_var" not in cleaned.data_vars
    assert "extra_coord" not in cleaned.coords

    # Check that required components are preserved
    assert "temperature" in cleaned.data_vars
    assert "precipitation" in cleaned.data_vars
    assert "time" in cleaned.coords
    assert "latitude" in cleaned.coords
    assert "longitude" in cleaned.coords

    # Check that data is preserved
    assert cleaned["temperature"].equals(climate_dataset["temperature"])
    assert cleaned["precipitation"].equals(climate_dataset["precipitation"])


def test_schema_stack_to_ndarray(climate_dataset, climate_schema):
    """Test stacking dataset variables into an ndarray."""
    # Stack the dataset
    stacked = climate_schema.stack_to_ndarray(climate_dataset)

    # Check shape
    assert stacked.shape == (10, 5, 5, 2)  # time, lat, lon, vars

    # Check that variables are stacked in the correct order
    temp_data = climate_dataset["temperature"].values
    precip_data = climate_dataset["precipitation"].values
    assert np.array_equal(stacked[..., 0], temp_data)
    assert np.array_equal(stacked[..., 1], precip_data)


def test_schema_unstack_from_ndarray(climate_dataset, climate_schema):
    """Test unstacking an ndarray back to a dataset."""
    # First stack the dataset
    stacked = climate_schema.stack_to_ndarray(climate_dataset)

    # Then unstack it
    unstacked = climate_schema.unstack_from_ndarray(
        stacked, original_dataset=climate_dataset
    )

    # Check that the unstacked dataset matches the original
    assert "temperature" in unstacked.data_vars
    assert "precipitation" in unstacked.data_vars
    assert np.array_equal(
        unstacked["temperature"].values, climate_dataset["temperature"].values
    )
    assert np.array_equal(
        unstacked["precipitation"].values, climate_dataset["precipitation"].values
    )

    # Check that metadata was preserved
    assert unstacked["temperature"].attrs == climate_dataset["temperature"].attrs
    assert unstacked["precipitation"].attrs == climate_dataset["precipitation"].attrs


def test_schema_unstack_from_ndarray_custom_vars(climate_dataset, climate_schema):
    """Test unstacking with custom variable names that exist in the schema."""
    # Stack the dataset
    stacked = climate_schema.stack_to_ndarray(climate_dataset)

    # Define custom variable names that exist in the schema
    custom_vars = ["temperature", "precipitation"]

    # Unstack with custom variable names
    unstacked = climate_schema.unstack_from_ndarray(
        stacked, var_names=custom_vars, original_dataset=climate_dataset
    )

    # Check that the custom variable names are used
    assert "temperature" in unstacked
    assert "precipitation" in unstacked
    assert "temp" not in unstacked
    assert "precip" not in unstacked

    # Check that the data is preserved
    np.testing.assert_array_equal(
        unstacked["temperature"].values, climate_dataset["temperature"].values
    )
    np.testing.assert_array_equal(
        unstacked["precipitation"].values, climate_dataset["precipitation"].values
    )

    # Check that attributes are preserved
    assert unstacked["temperature"].attrs == climate_dataset["temperature"].attrs
    assert unstacked["precipitation"].attrs == climate_dataset["precipitation"].attrs


def test_schema_unstack_from_ndarray_invalid_shape(climate_dataset, climate_schema):
    """Test unstacking with invalid array shape."""
    # Create an array with wrong number of variables
    stacked = np.random.rand(10, 5, 5, 3)  # 3 variables instead of 2

    # Try to unstack
    with pytest.raises(ValueError) as exc_info:
        climate_schema.unstack_from_ndarray(stacked)
    assert "Last dimension of array (3) doesn't match number of variables (2)" in str(
        exc_info.value
    )


def test_schema_unstack_from_ndarray_invalid_var(climate_dataset, climate_schema):
    """Test unstacking with invalid variable name."""
    # Stack the dataset
    stacked = climate_schema.stack_to_ndarray(climate_dataset)

    # Try to unstack with invalid variable name
    custom_vars = ["temp", "invalid_var"]
    with pytest.raises(ValueError) as exc_info:
        climate_schema.unstack_from_ndarray(stacked, var_names=custom_vars)
    assert "not found in schema" in str(exc_info.value)


def test_coerce_dtypes(simple_dataset, simple_schema):
    """Test coercing dataset dtypes to match schema."""
    # Create dataset with wrong dtypes
    dataset = simple_dataset.copy()
    dataset["var1"] = dataset["var1"].astype(np.int32)
    dataset["var2"] = dataset["var2"].astype(np.float32)
    dataset["x"] = dataset["x"].astype(np.int32)

    # Coerce dtypes
    coerced = simple_schema.coerce_dtypes(dataset)

    # Check that dtypes match schema
    assert coerced["var1"].dtype == np.dtype(np.float64)
    assert coerced["var2"].dtype == np.dtype(np.float64)
    assert coerced["x"].dtype == np.dtype(np.float64)

    # Check that data is preserved
    np.testing.assert_array_equal(coerced["var1"].values, dataset["var1"].values)
    np.testing.assert_array_equal(coerced["var2"].values, dataset["var2"].values)
    np.testing.assert_array_equal(coerced["x"].values, dataset["x"].values)


def test_coerce_dtypes_failure(simple_dataset, simple_schema):
    """Test dtype coercion failure."""
    # Create dataset with incompatible dtype
    dataset = simple_dataset.copy()
    dataset["var1"] = xr.DataArray(
        ["a"] * 10,  # Match the time dimension size
        dims=["time"],
        attrs=dataset["var1"].attrs,
    )

    # Try to coerce dtypes
    with pytest.raises(ValueError) as exc_info:
        simple_schema.coerce_dtypes(dataset)
    assert "Failed to coerce var1" in str(exc_info.value)


def test_clip_values(simple_dataset, simple_schema):
    """Test clipping dataset values to schema ranges."""
    # Create dataset with out-of-range values
    dataset = simple_dataset.copy()
    dataset["var1"].values[0, 0, 0] = 200  # Above range [0, 100]
    dataset["var1"].values[1, 1, 1] = -50  # Below range [0, 100]
    dataset["var2"].values[0, 0, 0] = 100  # Above range [-50, 50]
    dataset["var2"].values[1, 1, 1] = -100  # Below range [-50, 50]

    # Clip values
    clipped = simple_schema.clip_values(dataset)

    # Check that values are within ranges
    assert np.all(clipped["var1"].values >= 0)
    assert np.all(clipped["var1"].values <= 100)
    assert np.all(clipped["var2"].values >= -50)
    assert np.all(clipped["var2"].values <= 50)

    # Check that in-range values are preserved
    mask = (dataset["var1"].values >= 0) & (dataset["var1"].values <= 100)
    np.testing.assert_array_equal(
        clipped["var1"].values[mask], dataset["var1"].values[mask]
    )
    mask = (dataset["var2"].values >= -50) & (dataset["var2"].values <= 50)
    np.testing.assert_array_equal(
        clipped["var2"].values[mask], dataset["var2"].values[mask]
    )


def test_fill_missing(simple_dataset, simple_schema):
    """Test filling missing values in dataset."""
    # Create dataset with missing values
    dataset = simple_dataset.copy()
    dataset["var1"].values[0:5, :, :] = np.nan

    # Fill missing values
    filled = simple_schema.fill_missing(dataset, method="bfill")

    # Check that missing values are filled
    assert not np.any(np.isnan(filled["var1"].values))

    # Check that non-missing values are preserved
    mask = ~np.isnan(dataset["var1"].values)
    np.testing.assert_array_equal(
        filled["var1"].values[mask], dataset["var1"].values[mask]
    )


def test_fill_missing_invalid_method(simple_dataset, simple_schema):
    """Test filling missing values with invalid method."""
    # Create dataset with missing values
    dataset = simple_dataset.copy()
    dataset["var1"].values[0:5, :, :] = np.nan

    # Try to fill with invalid method
    with pytest.raises(ValueError) as exc_info:
        simple_schema.fill_missing(dataset, method="invalid")
    assert "method must be one of" in str(exc_info.value)


def test_fill_missing_methods(simple_dataset, simple_schema):
    """Test that different methods for filling missing values can be called."""
    # Create dataset with missing values
    dataset = simple_dataset.copy()
    dataset["var1"].values[0:5, :, :] = np.nan

    # Test that each method can be called without errors
    filled_ffill = simple_schema.fill_missing(dataset, method="ffill")
    filled_bfill = simple_schema.fill_missing(dataset, method="bfill")
    filled_interp = simple_schema.fill_missing(dataset, method="interpolate")

    # Verify that the methods return datasets
    assert isinstance(filled_ffill, xr.Dataset)
    assert isinstance(filled_bfill, xr.Dataset)
    assert isinstance(filled_interp, xr.Dataset)
