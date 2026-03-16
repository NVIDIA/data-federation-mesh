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

"""Tests for data synthesis functionality."""

import numpy as np
import xarray as xr
from hypothesis import HealthCheck, given, settings

from nv_dfm_lib_common.schemas.xarray import (
    Attribute,
    Coordinate,
    DataVariable,
    XArraySchema,
)
from nv_dfm_lib_common.schemas.xarray.synthesis import (
    generate_edge_case_dataset,
    generate_invalid_dataset,
    generate_valid_dataset,
    xarray_strategy,
)


def test_xarray_strategy(climate_schema):
    """Test xarray strategy generation."""

    @given(dataset=xarray_strategy(climate_schema, size=5))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def _test_strategy(dataset):
        # Check that the dataset matches the schema
        assert isinstance(dataset, xr.Dataset)
        assert all(var in dataset for var in climate_schema.data_vars)
        assert all(coord in dataset for coord in climate_schema.coords)

        # Check that the data types match
        for name, var in climate_schema.data_vars.items():
            assert dataset[name].dtype == var.dtype

        # Check that the dimensions match
        for name, var in climate_schema.data_vars.items():
            # Get dimensions from the variable's checks
            dims = None
            for check in var.checks:
                if hasattr(check, "dims"):
                    dims = check.dims
                    break
            if dims is None:
                # If no dims found in checks, use all available coordinates
                dims = list(dataset.coords.keys())
            assert dataset[name].dims == tuple(dims)

        # Check that the coordinates have the right size
        assert all(dataset[coord].size == 5 for coord in climate_schema.coords)

    _test_strategy()


def test_generate_valid_dataset(climate_schema):
    """Test generation of valid datasets."""
    dataset = generate_valid_dataset(climate_schema, size=5)

    # Check that the dataset matches the schema
    assert isinstance(dataset, xr.Dataset)
    assert all(var in dataset for var in climate_schema.data_vars)
    assert all(coord in dataset for coord in climate_schema.coords)

    # Check that the data types match
    for name, var in climate_schema.data_vars.items():
        assert dataset[name].dtype == var.dtype

    # Check that the dimensions match
    for name, var in climate_schema.data_vars.items():
        # Get dimensions from the variable's checks
        dims = None
        for check in var.checks:
            if hasattr(check, "dims"):
                dims = check.dims
                break
        if dims is None:
            # If no dims found in checks, use all available coordinates
            dims = list(dataset.coords.keys())
        assert dataset[name].dims == tuple(dims)

    # Check that the coordinates have the right size
    assert all(dataset[coord].size == 5 for coord in climate_schema.coords)

    # Check that the data is within the specified ranges
    for name, var in climate_schema.data_vars.items():
        for check in var.checks:
            if hasattr(check, "min_value") and hasattr(check, "max_value"):
                assert np.all(dataset[name] >= check.min_value)
                assert np.all(dataset[name] <= check.max_value)


def test_edge_case_generation():
    """Test edge case dataset generation."""
    # Create a schema with various data types
    schema = XArraySchema(
        data_vars={
            "float_var": DataVariable(
                dtype=np.dtype(np.float64),
                checks=[
                    lambda x: x >= -1000 and x <= 1000,
                ],
                required=True,
            ),
            "int_var": DataVariable(
                dtype=np.dtype(np.int32),
                checks=[
                    lambda x: x >= -100 and x <= 100,
                ],
                required=True,
            ),
        },
        coords={
            "time": Coordinate(
                dtype=np.dtype(np.datetime64),
                checks=[],
                required=True,
            ),
        },
    )

    # Generate edge case dataset
    dataset = generate_edge_case_dataset(schema, size=10)

    # Check that edge cases are present
    assert np.any(np.isnan(dataset["float_var"]))
    assert np.any(np.isinf(dataset["float_var"]))
    assert np.any(np.isneginf(dataset["float_var"]))
    assert np.any(np.isnat(dataset["time"]))

    # Check that non-edge case values are within bounds
    float_mask = ~(np.isnan(dataset["float_var"]) | np.isinf(dataset["float_var"]))
    assert np.all(dataset["float_var"][float_mask] >= -1000)
    assert np.all(dataset["float_var"][float_mask] <= 1000)
    assert np.all(dataset["int_var"] >= -100)
    assert np.all(dataset["int_var"] <= 100)


def test_invalid_dataset_generation():
    """Test invalid dataset generation."""
    # Create a schema with various constraints
    schema = XArraySchema(
        data_vars={
            "temperature": DataVariable(
                dtype=np.dtype(np.float64),
                checks=[
                    lambda x: x >= 200 and x <= 350,
                ],
                required=True,
            ),
        },
        coords={
            "time": Coordinate(
                dtype=np.dtype(np.datetime64),
                checks=[],
                required=True,
            ),
            "space": Coordinate(
                dtype=np.dtype(np.float64),
                checks=[],
                required=True,
            ),
        },
        attrs={
            "title": Attribute(
                dtype=str,
                checks=[],
                required=True,
            ),
        },
    )

    # Test missing variable
    dataset_missing = generate_invalid_dataset(schema, invalid_type="missing")
    assert "temperature" not in dataset_missing

    # Test type mismatch
    dataset_type = generate_invalid_dataset(schema, invalid_type="type")
    assert dataset_type["temperature"].dtype == np.int32

    # Test range violation
    dataset_range = generate_invalid_dataset(schema, invalid_type="range")
    assert np.any(dataset_range["temperature"] < 200) or np.any(
        dataset_range["temperature"] > 350
    )

    # Test dimension violation
    dataset_dim = generate_invalid_dataset(schema, invalid_type="dimension")
    # Get expected dimensions from the variable's checks
    expected_dims = None
    for check in schema.data_vars["temperature"].checks:
        if hasattr(check, "dims"):
            expected_dims = check.dims
            break
    if expected_dims is None:
        # If no dims found in checks, use all available coordinates
        expected_dims = list(dataset_dim.coords.keys())

    # Check that dimensions are different
    assert dataset_dim["temperature"].dims != tuple(expected_dims)

    # Test random violation
    dataset_random = generate_invalid_dataset(schema, invalid_type="random")
    # The random violation could be any of the above, so we need to check each case
    if "temperature" in dataset_random:
        # If the variable exists, it should be different from a valid dataset
        valid_dataset = generate_valid_dataset(schema)
        assert not np.array_equal(
            dataset_random["temperature"], valid_dataset["temperature"]
        )
    else:
        # If the variable is missing, that's also a valid random violation
        assert "temperature" not in dataset_random


def test_edge_case_with_custom_checks():
    """Test edge case generation with custom validation checks."""
    # Create a schema with custom checks
    schema = XArraySchema(
        data_vars={
            "temperature": DataVariable(
                dtype=np.dtype(np.float64),
                checks=[
                    lambda x: x >= 200 and x <= 350,
                ],
                required=True,
            ),
        },
        coords={
            "time": Coordinate(
                dtype=np.dtype(np.datetime64),
                checks=[],
                required=True,
            ),
        },
    )

    # Generate edge case dataset
    dataset = generate_edge_case_dataset(schema, size=10)

    # Check that edge cases are present
    assert np.any(np.isnan(dataset["temperature"]))
    assert np.any(np.isinf(dataset["temperature"]))
    assert np.any(np.isneginf(dataset["temperature"]))
    assert np.any(np.isnat(dataset["time"]))

    # Check that non-edge case values are within bounds
    temp_mask = ~(np.isnan(dataset["temperature"]) | np.isinf(dataset["temperature"]))
    assert np.all(dataset["temperature"][temp_mask] >= 200)
    assert np.all(dataset["temperature"][temp_mask] <= 350)


def test_synthesis_with_metadata():
    """Test data synthesis with metadata."""
    # Create a schema with metadata
    schema = XArraySchema(
        data_vars={
            "temperature": DataVariable(
                dtype=np.dtype(np.float64),
                metadata={
                    "units": "K",
                    "standard_name": "air_temperature",
                },
                checks=[],
                required=True,
            ),
        },
        coords={
            "time": Coordinate(
                dtype=np.dtype(np.datetime64),
                metadata={
                    "standard_name": "time",
                },
                checks=[],
                required=True,
            ),
        },
        metadata={
            "title": "Test Dataset",
            "institution": "Test Institution",
        },
    )

    # Generate a dataset
    dataset = generate_valid_dataset(schema, size=5)

    # Check that the metadata is present
    assert dataset["temperature"].attrs["units"] == "K"
    assert dataset["temperature"].attrs["standard_name"] == "air_temperature"
    assert dataset["time"].attrs["standard_name"] == "time"
    assert dataset.attrs["title"] == "Test Dataset"
    assert dataset.attrs["institution"] == "Test Institution"
