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

"""Tests for schema inference functionality."""

import numpy as np
import pytest
import xarray as xr

from nv_dfm_lib_common.schemas.xarray import XArraySchema


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    time = xr.DataArray(
        np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]"),
        dims=["time"],
        attrs={"standard_name": "time"},
    )
    lat = xr.DataArray(
        np.array([-45.0, 0.0, 45.0]),
        dims=["latitude"],
        attrs={"standard_name": "latitude"},
    )
    lon = xr.DataArray(
        np.array([-180.0, 0.0, 180.0]),
        dims=["longitude"],
        attrs={"standard_name": "longitude"},
    )
    temp = xr.DataArray(
        np.random.uniform(280, 320, size=(5, 3, 3)),
        dims=["time", "latitude", "longitude"],
        attrs={
            "standard_name": "air_temperature",
            "units": "K",
            "long_name": "Air Temperature",
        },
    )
    precip = xr.DataArray(
        np.random.uniform(0, 100, size=(5, 3, 3)),
        dims=["time", "latitude", "longitude"],
        attrs={
            "standard_name": "precipitation_amount",
            "units": "mm/day",
            "long_name": "Precipitation",
        },
    )
    return xr.Dataset(
        data_vars={"temperature": temp, "precipitation": precip},
        coords={"time": time, "latitude": lat, "longitude": lon},
        attrs={
            "title": "Test Dataset",
            "description": "A test dataset for schema inference",
            "institution": "Test Institution",
        },
    )


def test_basic_inference(simple_dataset):
    """Test basic schema inference."""
    schema = XArraySchema.infer_schema(simple_dataset)

    # Check that all variables and coordinates are present
    assert set(schema.data_vars.keys()) == {"temperature", "precipitation"}
    assert set(schema.coords.keys()) == {"time", "latitude", "longitude"}

    # Check data types
    assert schema.data_vars["temperature"].dtype == np.dtype(np.float64)
    assert schema.data_vars["precipitation"].dtype == np.dtype(np.float64)
    # Use np.issubdtype for datetime64
    assert np.issubdtype(schema.coords["time"].dtype, np.datetime64)

    # Check dimensions (look for check with name starting with 'check_dims(')
    for var_name in ["temperature", "precipitation"]:
        dim_check = next(
            (
                check
                for check in schema.data_vars[var_name].checks
                if hasattr(check, "name") and check.name.startswith("check_dims(")
            ),
            None,
        )
        assert dim_check is not None
    for coord_name in ["time", "latitude", "longitude"]:
        dim_check = next(
            (
                check
                for check in schema.coords[coord_name].checks
                if hasattr(check, "name") and check.name.startswith("check_dims(")
            ),
            None,
        )
        assert dim_check is not None


def test_numeric_range_inference(simple_dataset):
    """Test inference of numeric ranges."""
    schema = XArraySchema.infer_schema(simple_dataset)

    # Check that range checks are present for numeric variables
    temp_checks = schema.data_vars["temperature"].checks
    precip_checks = schema.data_vars["precipitation"].checks

    # Find range checks by name
    temp_range = next(
        (
            check
            for check in temp_checks
            if hasattr(check, "name") and check.name.startswith("check_range(")
        ),
        None,
    )
    precip_range = next(
        (
            check
            for check in precip_checks
            if hasattr(check, "name") and check.name.startswith("check_range(")
        ),
        None,
    )

    # Check that ranges match the data
    assert temp_range is not None
    assert precip_range is not None

    # Test the range checks (should return True for valid data)
    assert temp_range(simple_dataset["temperature"])
    assert precip_range(simple_dataset["precipitation"])

    # Test with out-of-range values (should return False)
    invalid_temp = simple_dataset["temperature"].copy()
    invalid_temp.values[0, 0, 0] = 400  # Outside inferred range
    assert not temp_range(invalid_temp)


def test_cf_metadata_inference(simple_dataset):
    """Test inference of CF metadata."""
    schema = XArraySchema.infer_schema(simple_dataset)

    # Check CF standard names
    temp_checks = schema.data_vars["temperature"].checks
    precip_checks = schema.data_vars["precipitation"].checks
    lat_checks = schema.coords["latitude"].checks
    lon_checks = schema.coords["longitude"].checks

    # Find standard name checks
    temp_std = next(
        (
            check
            for check in temp_checks
            if hasattr(check, "_check_fn")
            and "standard_name" in check._check_fn.__code__.co_consts
        ),
        None,
    )
    precip_std = next(
        (
            check
            for check in precip_checks
            if hasattr(check, "_check_fn")
            and "standard_name" in check._check_fn.__code__.co_consts
        ),
        None,
    )
    lat_std = next(
        (
            check
            for check in lat_checks
            if hasattr(check, "_check_fn")
            and "standard_name" in check._check_fn.__code__.co_consts
        ),
        None,
    )
    lon_std = next(
        (
            check
            for check in lon_checks
            if hasattr(check, "_check_fn")
            and "standard_name" in check._check_fn.__code__.co_consts
        ),
        None,
    )

    # Check standard names
    assert temp_std is not None
    assert precip_std is not None
    assert lat_std is not None
    assert lon_std is not None

    # Test the standard name checks
    assert temp_std(simple_dataset["temperature"])
    assert precip_std(simple_dataset["precipitation"])
    assert lat_std(simple_dataset["latitude"])
    assert lon_std(simple_dataset["longitude"])

    # Test with invalid standard names
    invalid_temp = simple_dataset["temperature"].copy()
    invalid_temp.attrs["standard_name"] = "invalid_name"
    with pytest.raises(ValueError) as exc_info:
        temp_std(invalid_temp)
    assert "incorrect standard_name" in str(exc_info.value)


def test_metadata_inference(simple_dataset):
    """Test inference of dataset metadata."""
    schema = XArraySchema.infer_schema(
        simple_dataset,
        name="test_schema",
        title="Test Schema",
        description="A test schema",
    )

    # Check schema metadata
    assert schema.name == "test_schema"
    assert schema.title == "Test Schema"
    assert schema.description == "A test schema"

    # Check variable metadata
    assert schema.data_vars["temperature"].metadata["long_name"] == "Air Temperature"
    assert schema.data_vars["precipitation"].metadata["long_name"] == "Precipitation"

    # Check dataset attributes
    assert "title" in schema.attrs
    assert schema.attrs["title"].dtype == str


def test_invalid_input():
    """Test inference with invalid input."""
    # Test with non-dataset input
    with pytest.raises(ValueError) as exc_info:
        XArraySchema.infer_schema("not a dataset")  # type: ignore
    assert "Expected Dataset" in str(exc_info.value)

    # Test with empty dataset
    empty_ds = xr.Dataset()
    schema = XArraySchema.infer_schema(empty_ds)
    assert len(schema.data_vars) == 0
    assert len(schema.coords) == 0


def test_inferred_schema_validation(simple_dataset):
    """Test that the inferred schema can validate the original dataset."""
    schema = XArraySchema.infer_schema(simple_dataset)

    # Validate the original dataset
    result = schema.validate(simple_dataset)
    assert result is True

    # Create an invalid dataset by modifying a value
    invalid_dataset = simple_dataset.copy()
    invalid_dataset["temperature"].values[0, 0, 0] = 400  # Outside inferred range

    # Validate the invalid dataset
    with pytest.raises(ValueError) as exc_info:
        schema.validate(invalid_dataset)
    # Just check for 'failed check' in the error message
    assert "failed check" in str(exc_info.value)


def test_schema_to_code(simple_dataset):
    """Test that the inferred schema can be converted to valid Python code."""
    schema = XArraySchema.infer_schema(simple_dataset)
    code = schema.to_code()

    # Basic validation of the generated code
    assert "import numpy as np" in code
    assert "from nv_dfm_lib_common.schemas.xarray import XArraySchema" in code
    assert "from nv_dfm_lib_common.schemas.xarray.checks import" in code

    # Check that all variables are included
    assert "'temperature': DataVariable(" in code
    assert "'precipitation': DataVariable(" in code

    # Check that all coordinates are included
    assert "'time': Coordinate(" in code
    assert "'latitude': Coordinate(" in code
    assert "'longitude': Coordinate(" in code

    # Check that metadata is included
    assert "'long_name': 'Air Temperature'" in code
    assert "'long_name': 'Precipitation'" in code

    # Verify the code can be executed
    namespace = {}
    exec(code, namespace)
    reconstructed_schema = namespace["schema"]

    # Verify the reconstructed schema matches the original
    assert set(reconstructed_schema.data_vars.keys()) == set(schema.data_vars.keys())
    assert set(reconstructed_schema.coords.keys()) == set(schema.coords.keys())

    # Verify the reconstructed schema can validate the original dataset
    assert reconstructed_schema.validate(simple_dataset)
