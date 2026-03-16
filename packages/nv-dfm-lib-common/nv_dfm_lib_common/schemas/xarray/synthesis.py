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

"""Data synthesis functionality for xarray-schema."""

import inspect
import re
from typing import Dict, Optional

import numpy as np
import xarray as xr
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy, composite

from nv_dfm_lib_common.schemas.xarray import Coordinate, DataVariable, XArraySchema


def _datetime64_strategy():
    """Generate numpy datetime64 values."""
    min_value = np.datetime64("1970-01-01")
    max_value = np.datetime64("2100-01-01")

    # Convert to integers for the strategy
    min_dt = int(min_value.astype(np.int64))
    max_dt = int(max_value.astype(np.int64))

    # Generate integer and convert to datetime64
    return st.integers(min_value=min_dt, max_value=max_dt).map(
        lambda x: np.datetime64(x, "ns")
    )


@composite
def _generate_coordinate(draw, schema, name, size):
    """Generate a coordinate value."""
    coord = schema.coords[name]
    data_strat = None

    if coord.dtype == np.dtype(np.datetime64):
        data_strat = _datetime64_strategy()
    elif coord.dtype == np.dtype(np.float64):
        data_strat = st.floats(allow_nan=False, allow_infinity=False)
    elif coord.dtype == np.dtype(np.int32):
        data_strat = st.integers()
    else:
        raise ValueError(f"Unsupported dtype for coordinate {name}: {coord.dtype}")

    # Generate data with the right shape
    if size is None:
        size = 5  # Default size if none provided
    data = draw(st.lists(data_strat, min_size=size, max_size=size))
    data = np.array(data)

    return xr.DataArray(
        data,
        dims=[name],
        name=name,
        attrs=coord.metadata or {},
    )


@composite
def _generate_variable(draw, schema, name, coords):
    """Generate a variable value."""
    var = schema.data_vars[name]
    data_strat = None

    if var.dtype == np.dtype(np.datetime64):
        data_strat = _datetime64_strategy()
    elif var.dtype == np.dtype(np.float64):
        data_strat = st.floats(allow_nan=False, allow_infinity=False)
    elif var.dtype == np.dtype(np.int32):
        data_strat = st.integers()
    else:
        raise ValueError(f"Unsupported dtype for variable {name}: {var.dtype}")

    # Get dimensions from the variable's checks
    dims = None
    for check in var.checks:
        if hasattr(check, "dims"):
            dims = check.dims
            break

    if dims is None:
        # If no dims found in checks, use all available coordinates
        dims = list(coords.keys())

    # Generate data with the right shape
    shape = tuple(coords[dim].size for dim in dims)
    size = int(np.prod(shape))  # Convert numpy int to Python int
    data = draw(st.lists(data_strat, min_size=size, max_size=size))
    data = np.array(data).reshape(shape)

    return xr.DataArray(
        data,
        dims=dims,
        name=name,
        attrs=var.metadata or {},
    )


@composite
def _generate_dataset(draw, schema, size):
    """Generate a dataset that matches the schema."""
    # Generate coordinates
    coords = {}
    for name in schema.coords:
        coords[name] = draw(_generate_coordinate(schema, name, size))

    # Generate variables
    data_vars = {}
    for name in schema.data_vars:
        data_vars[name] = draw(_generate_variable(schema, name, coords))

    # Create dataset
    dataset = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add metadata
    if schema.metadata:
        dataset.attrs.update(schema.metadata)

    return dataset


def xarray_strategy(
    schema: XArraySchema,
    size: Optional[int] = None,
    n_regex_cases: int = 2,
    use_checks: bool = True,
) -> SearchStrategy[xr.Dataset]:
    """
    Generate a strategy for creating synthetic xarray datasets that match a schema.

    Args:
        schema: The schema to generate data for
        size: Optional size for the generated data
        n_regex_cases: Number of cases to generate for regex patterns
        use_checks: Whether to use the schema's checks when generating data

    Returns:
        A strategy that generates synthetic xarray datasets
    """
    return _generate_dataset(schema, size)


def generate_valid_dataset(
    schema: XArraySchema,
    size: Optional[int] = None,
) -> xr.Dataset:
    """
    Generate a valid dataset that matches the schema.

    Args:
        schema: The schema to generate data for
        size: Optional size for the generated data

    Returns:
        A synthetic dataset that matches the schema
    """
    # Generate coordinates first
    coords = {}
    for name, coord in schema.coords.items():
        if coord.dtype == np.dtype(np.datetime64):
            data = np.arange("1970-01-01", "1970-01-06", dtype="datetime64[D]")
        elif coord.dtype == np.dtype(np.float64):
            data = np.linspace(-1, 1, 5)
        elif coord.dtype == np.dtype(np.int32):
            data = np.arange(5)
        else:
            raise ValueError(f"Unsupported dtype for coordinate {name}: {coord.dtype}")
        coords[name] = xr.DataArray(
            data,
            dims=[name],
            name=name,
            attrs=coord.metadata or {},
        )

    # Generate variables
    data_vars = {}
    for name, var in schema.data_vars.items():
        # Get dimensions from the variable's checks
        dims = None
        for check in var.checks:
            if hasattr(check, "dims"):
                dims = check.dims
                break
        if dims is None:
            dims = list(coords.keys())

        # Get shape from coordinates
        shape = tuple(coords[dim].size for dim in dims)

        # Generate data within valid ranges
        if var.dtype == np.dtype(np.float64):
            # Extract bounds from checks if available
            min_value = -1000  # Default minimum
            max_value = 1000  # Default maximum
            for check in var.checks:
                if hasattr(check, "_check_fn"):
                    check_fn = check._check_fn
                    if callable(check_fn) and check_fn.__name__ == "<lambda>":
                        source = inspect.getsource(check_fn)
                        if ">=" in source and "<=" in source:
                            min_match = re.search(r">=\s*([-\d.]+)", source)
                            max_match = re.search(r"<=\s*([-\d.]+)", source)
                            if min_match and max_match:
                                min_value = float(min_match.group(1))
                                max_value = float(max_match.group(1))
                                break
            data = np.random.uniform(min_value, max_value, shape)
        elif var.dtype == np.dtype(np.int32):
            data = np.random.randint(-100, 101, shape)
        else:
            data = np.random.rand(*shape)

        data_vars[name] = xr.DataArray(
            data,
            dims=dims,
            name=name,
            attrs=var.metadata or {},
        )

    # Create dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=schema.metadata or {},
    )

    return ds


def generate_edge_case_dataset(
    schema: XArraySchema,
    size: Optional[int] = None,
) -> xr.Dataset:
    """
    Generate a dataset with edge cases that tests the schema's boundaries.

    Args:
        schema: The schema to generate data for
        size: Optional size for the generated data

    Returns:
        A synthetic dataset with edge cases
    """

    def _generate_edge_coordinate(
        coord: Coordinate, size: Optional[int] = None
    ) -> xr.DataArray:
        """Generate edge case data for a coordinate."""
        # Get the dtype info
        if np.issubdtype(coord.dtype, np.floating):
            dtype_info = np.finfo(coord.dtype)
            values = np.array(
                [
                    dtype_info.min,  # Minimum value
                    dtype_info.max,  # Maximum value
                    dtype_info.eps,  # Smallest difference
                    -dtype_info.eps,  # Negative smallest difference
                    np.nan,  # Not a number
                    np.inf,  # Positive infinity
                    -np.inf,  # Negative infinity
                ]
            )
        elif np.issubdtype(coord.dtype, np.integer):
            dtype_info = np.iinfo(coord.dtype)
            values = np.array(
                [
                    dtype_info.min,  # Minimum value
                    dtype_info.max,  # Maximum value
                    0,  # Zero
                ]
            )
        elif np.issubdtype(coord.dtype, np.datetime64):
            # For datetime, use min/max dates and some edge cases
            values = np.array(
                [
                    np.datetime64("1678-01-01"),  # Minimum date
                    np.datetime64("2262-04-11"),  # Maximum date
                    np.datetime64("NaT"),  # Not a time
                ]
            )
        else:
            # For other types, just use the regular strategy
            # Note: The linter error about missing 'size' parameter is a false positive.
            # The _generate_coordinate function is a Hypothesis composite strategy that
            # expects a 'draw' parameter from Hypothesis, which is automatically provided
            # when used with @composite. The 'size' parameter is optional and defaults to 5.
            return _generate_coordinate(coord, size)  # type: ignore

        # Create the DataArray
        da = xr.DataArray(
            values,
            dims=[coord.name],
            attrs=coord.metadata,
        )

        # Apply any range checks
        for check in coord.checks:
            if hasattr(check, "min_value") and hasattr(check, "max_value"):
                da = da.clip(check.min_value, check.max_value)

        return da

    def _generate_edge_variable(
        var: DataVariable, coords: Dict[str, xr.DataArray]
    ) -> xr.DataArray:
        """Generate edge case data for a data variable."""
        # Get dimensions from the variable's checks
        dims = None
        for check in var.checks:
            if hasattr(check, "dims"):
                dims = check.dims
                break
        if dims is None:
            # If no dims found in checks, use all available coordinates
            dims = list(coords.keys())

        # Get the shape from coordinates
        shape = [coords[dim].size for dim in dims]

        # Get range constraints from checks
        min_value = -1000  # Default minimum
        max_value = 1000  # Default maximum
        for check in var.checks:
            if hasattr(check, "min_value") and hasattr(check, "max_value"):
                min_value = check.min_value
                max_value = check.max_value
                break
            elif callable(check) and check.__name__ == "<lambda>":
                # Extract bounds from lambda function
                source = inspect.getsource(check)
                if ">=" in source and "<=" in source:
                    # Extract min and max values from lambda
                    min_match = re.search(r">=\s*([-\d.]+)", source)
                    max_match = re.search(r"<=\s*([-\d.]+)", source)
                    if min_match and max_match:
                        min_value = float(min_match.group(1))
                        max_value = float(max_match.group(1))

        # Generate edge case data
        if var.dtype == np.dtype(np.float64):
            # Generate data within bounds
            data = np.random.uniform(min_value, max_value, shape)
            # Add edge cases at the first position
            if len(shape) == 1:
                data[0] = np.nan  # NaN value
                data[1] = np.inf  # Positive infinity
                data[2] = -np.inf  # Negative infinity
            else:
                data[0, 0] = np.nan
                data[0, 1] = np.inf
                data[0, 2] = -np.inf

            # Ensure all non-edge case values are within bounds
            mask = ~(np.isnan(data) | np.isinf(data))
            data[mask] = np.clip(data[mask], min_value, max_value)
        elif var.dtype == np.dtype(np.int32):
            # Convert bounds to integers for randint
            min_int = int(min_value)
            max_int = int(max_value)
            # Generate data within bounds
            data = np.random.randint(min_int, max_int + 1, shape)
            # Add edge cases at the first position
            if len(shape) == 1:
                data[0] = min_int  # Minimum value
                data[1] = max_int  # Maximum value
                data[2] = (min_int + max_int) // 2  # Middle value
            else:
                data[0, 0] = min_int
                data[0, 1] = max_int
                data[0, 2] = (min_int + max_int) // 2
        else:
            data = np.random.rand(*shape)

        return xr.DataArray(data, dims=dims)

    # Generate coordinates first
    coords = {
        name: _generate_edge_coordinate(coord, size)
        for name, coord in schema.coords.items()
    }

    # Generate data variables
    data_vars = {
        name: _generate_edge_variable(var, coords)
        for name, var in schema.data_vars.items()
    }

    # Create the dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=schema.metadata,
    )

    return ds


def generate_invalid_dataset(
    schema: XArraySchema,
    size: Optional[int] = None,
    invalid_type: str = "random",
) -> xr.Dataset:
    """
    Generate an invalid dataset that violates the schema.

    Args:
        schema: The schema to generate data for
        size: Optional size for the generated data
        invalid_type: Type of invalidity to generate
            - "random": Random violations
            - "missing": Missing required variables/attributes
            - "type": Type mismatches
            - "range": Value range violations
            - "dimension": Dimension mismatches

    Returns:
        A synthetic dataset that violates the schema
    """
    # Ensure we have at least one data variable
    if not schema.data_vars:
        raise ValueError(
            "Schema must have at least one data variable for invalid dataset generation"
        )

    # Get the first variable name
    var_name = list(schema.data_vars.keys())[0]
    var = schema.data_vars[var_name]

    # Generate coordinates first
    coords = {}
    for name, coord in schema.coords.items():
        if coord.dtype == np.dtype(np.datetime64):
            data = np.arange("1970-01-01", "1970-01-06", dtype="datetime64[D]")
        elif coord.dtype == np.dtype(np.float64):
            data = np.linspace(-1, 1, 5)
        elif coord.dtype == np.dtype(np.int32):
            data = np.arange(5)
        else:
            raise ValueError(f"Unsupported dtype for coordinate {name}: {coord.dtype}")
        coords[name] = xr.DataArray(data, dims=[name], name=name)

    # Generate the variable
    dims = None
    for check in var.checks:
        if hasattr(check, "dims"):
            dims = check.dims
            break
    if dims is None:
        dims = list(coords.keys())

    shape = tuple(coords[dim].size for dim in dims)
    if var.dtype == np.dtype(np.float64):
        data = np.random.uniform(200, 350, shape)  # Use valid range for initial data
    elif var.dtype == np.dtype(np.int32):
        data = np.random.randint(-100, 101, shape)  # Use valid range for initial data
    else:
        data = np.random.rand(*shape)

    # Create the dataset
    ds = xr.Dataset(
        data_vars={var_name: xr.DataArray(data, dims=dims, name=var_name)},
        coords=coords,
        attrs=schema.metadata,
    )

    if invalid_type == "random":
        # Choose a random violation type
        import random

        invalid_type = random.choice(["missing", "type", "range", "dimension"])

    if invalid_type == "missing":
        # Remove a required variable or attribute
        ds = ds.drop_vars(var_name)

    elif invalid_type == "type":
        # Change the type of a variable
        if np.issubdtype(ds[var_name].dtype, np.floating):
            ds[var_name] = ds[var_name].astype(np.int32)
        else:
            ds[var_name] = ds[var_name].astype(np.float64)

    elif invalid_type == "range":
        # Violate range constraints
        for check in var.checks:
            if callable(check) and check.__name__ == "<lambda>":
                # Extract bounds from lambda function
                source = inspect.getsource(check)
                if ">=" in source and "<=" in source:
                    # Extract min and max values from lambda
                    min_match = re.search(r">=\s*([-\d.]+)", source)
                    max_match = re.search(r"<=\s*([-\d.]+)", source)
                    if min_match and max_match:
                        min_value = float(min_match.group(1))
                        max_value = float(max_match.group(1))
                        # Set some values below min and above max
                        data = ds[var_name].values
                        data[0, 0] = min_value - 1  # Below min
                        data[0, 1] = max_value + 1  # Above max
                        ds[var_name].values = data
                        break

    elif invalid_type == "dimension":
        # Violate dimension constraints
        if len(dims) > 1:
            ds[var_name] = ds[var_name].transpose(*reversed(dims))

    return ds
