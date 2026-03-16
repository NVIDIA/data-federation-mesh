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

"""Tests for xarray-schema."""

import numpy as np
import pytest
import xarray as xr

from nv_dfm_lib_common.schemas.xarray import (
    Coordinate,
    DataVariable,
    XArraySchema,
)
from nv_dfm_lib_common.schemas.xarray.checks import (
    check_cf_standard_name,
    check_cf_units,
    check_coordinate_system,
    check_dims,
    check_dtype,
    check_grid_mapping,
    check_grid_mapping_reference,
    check_max_missing,
    check_time_coordinate,
)


def test_comprehensive_dataset_validation(climate_dataset, climate_schema):
    """Test comprehensive dataset validation."""
    # Validate the dataset
    result = climate_schema.validate(climate_dataset)
    assert result is True

    # Test missing required data variable
    dataset_missing_var = climate_dataset.drop_vars("temperature")
    with pytest.raises(Exception) as exc_info:
        climate_schema.validate(dataset_missing_var)
    assert "missing required data variable" in str(exc_info.value)

    # Test incorrect dimensions
    dataset_wrong_dims = climate_dataset.copy(deep=True)
    dataset_wrong_dims["temperature"] = dataset_wrong_dims["temperature"].transpose(
        "longitude", "latitude", "time"
    )
    with pytest.raises(Exception) as exc_info:
        climate_schema.validate(dataset_wrong_dims)
    assert "failed check" in str(exc_info.value)

    # Test incorrect data type
    dataset_wrong_dtype = climate_dataset.copy(deep=True)
    dataset_wrong_dtype["temperature"] = dataset_wrong_dtype["temperature"].astype(
        np.int32
    )
    with pytest.raises(Exception) as exc_info:
        climate_schema.validate(dataset_wrong_dtype)
    assert "failed check" in str(exc_info.value)

    # Test too many missing values
    dataset_missing = climate_dataset.copy(deep=True)
    dataset_missing["temperature"].values[0:5, :, :] = np.nan
    with pytest.raises(Exception) as exc_info:
        climate_schema.validate(dataset_missing)
    assert "check_max_missing" in str(exc_info.value)

    # Test missing required attribute (with no missing values)
    dataset_missing_attr = climate_dataset.copy(deep=True)
    # Ensure no missing values in the dataset
    del dataset_missing_attr.attrs["title"]
    with pytest.raises(Exception) as exc_info:
        climate_schema.validate(dataset_missing_attr)
    assert "missing required attribute" in str(exc_info.value)

    # Test out of range values
    dataset_out_of_range = climate_dataset.copy(deep=True)
    dataset_out_of_range["temperature"].values[0, 0, 0] = 400
    with pytest.raises(Exception) as exc_info:
        climate_schema.validate(dataset_out_of_range)
    assert "failed check" in str(exc_info.value)


def test_temperature_schema(climate_dataset, climate_schema):
    """Test temperature data variable schema."""
    # Get the temperature schema
    temp_schema = climate_schema.data_vars["temperature"]

    # Validate temperature data
    result = temp_schema.validate(climate_dataset["temperature"])
    assert result is True

    # Test incorrect dimensions
    temp_wrong_dims = climate_dataset["temperature"].transpose(
        "longitude", "latitude", "time"
    )
    with pytest.raises(Exception) as exc_info:
        temp_schema.validate(temp_wrong_dims)
    assert "failed check" in str(exc_info.value)

    # Test incorrect data type
    temp_wrong_dtype = climate_dataset["temperature"].astype(np.int32)
    with pytest.raises(Exception) as exc_info:
        temp_schema.validate(temp_wrong_dtype)
    assert "failed check" in str(exc_info.value)

    # Test missing values
    temp_missing = climate_dataset["temperature"].copy()
    temp_missing.values[0:5, :, :] = np.nan
    with pytest.raises(Exception) as exc_info:
        temp_schema.validate(temp_missing)
    assert "failed check" in str(exc_info.value)

    # Test missing required attribute
    temp_missing_attr = climate_dataset["temperature"].copy()
    del temp_missing_attr.attrs["units"]
    with pytest.raises(Exception) as exc_info:
        temp_schema.validate(temp_missing_attr)
    assert "missing required attribute" in str(exc_info.value)

    # Test out of range values
    temp_out_of_range = climate_dataset["temperature"].copy()
    temp_out_of_range.values[0, 0, 0] = 400
    with pytest.raises(Exception) as exc_info:
        temp_schema.validate(temp_out_of_range)
    assert "failed check" in str(exc_info.value)


def test_coordinate_schema(climate_dataset, climate_schema):
    """Test coordinate schema."""
    # Get the latitude schema
    lat_schema = climate_schema.coords["latitude"]

    # Validate latitude coordinate
    result = lat_schema.validate(climate_dataset["latitude"])
    assert result is True

    # Test incorrect dimensions
    lat_wrong_dims = climate_dataset["latitude"].expand_dims("time")
    with pytest.raises(Exception) as exc_info:
        lat_schema.validate(lat_wrong_dims)
    assert "failed check" in str(exc_info.value)

    # Test incorrect data type
    lat_wrong_dtype = climate_dataset["latitude"].astype(np.int32)
    with pytest.raises(Exception) as exc_info:
        lat_schema.validate(lat_wrong_dtype)
    assert "failed check" in str(exc_info.value)

    # Test out of range values
    lat_out_of_range = xr.DataArray(
        np.array([100.0, 45.0, 0.0, -45.0, -90.0]), dims=["latitude"], name="latitude"
    )
    with pytest.raises(Exception) as exc_info:
        lat_schema.validate(lat_out_of_range)
    assert "failed check" in str(exc_info.value)


def test_missing_value_schema(climate_dataset, climate_schema):
    """Test missing value schema."""
    # Create a dataset with missing values
    dataset_missing = climate_dataset.copy()
    dataset_missing["temperature"].values[0:5, :, :] = np.nan

    # Test with different max_missing values
    schema_high_missing = climate_schema.copy()
    schema_high_missing.data_vars["temperature"].checks[2] = check_max_missing(0.5)
    result = schema_high_missing.validate(dataset_missing)
    assert result is True

    schema_low_missing = climate_schema.copy()
    schema_low_missing.data_vars["temperature"].checks[2] = check_max_missing(0.1)
    with pytest.raises(Exception) as exc_info:
        schema_low_missing.validate(dataset_missing)
    assert "check_max_missing" in str(exc_info.value)


def test_metadata_schema(climate_dataset, climate_schema):
    """Test metadata schema."""
    # Test missing required attribute
    dataset_missing_attr = climate_dataset.copy(deep=True)
    del dataset_missing_attr.attrs["title"]
    with pytest.raises(Exception) as exc_info:
        climate_schema.validate(dataset_missing_attr)
    assert "missing required attribute" in str(exc_info.value)

    # Test incorrect attribute type
    dataset_wrong_type = climate_dataset.copy(deep=True)
    dataset_wrong_type.attrs["title"] = 123
    with pytest.raises(Exception) as exc_info:
        climate_schema.validate(dataset_wrong_type)
    assert "has incorrect type" in str(exc_info.value)


def test_cf_standard_name():
    """Test CF standard name validation."""
    # Create a DataArray with a valid CF standard name
    data = xr.DataArray(
        np.random.rand(10), dims=["time"], attrs={"standard_name": "air_temperature"}
    )

    # Test valid standard name
    check = check_cf_standard_name("air_temperature")
    assert check(data) is True

    # Test missing standard name
    data_no_std = xr.DataArray(np.random.rand(10), dims=["time"])
    with pytest.raises(ValueError) as exc_info:
        check(data_no_std)
    assert "missing standard_name attribute" in str(exc_info.value)

    # Test incorrect standard name
    data_wrong_std = xr.DataArray(
        np.random.rand(10), dims=["time"], attrs={"standard_name": "wrong_name"}
    )
    with pytest.raises(ValueError) as exc_info:
        check(data_wrong_std)
    assert "incorrect standard_name" in str(exc_info.value)

    # Test invalid CF standard name
    data_invalid_std = xr.DataArray(
        np.random.rand(10),
        dims=["time"],
        attrs={
            "standard_name": "precipitation_amount"
        },  # Valid CF name but wrong for this check
    )
    with pytest.raises(ValueError) as exc_info:
        check(data_invalid_std)
    assert "incorrect standard_name" in str(exc_info.value)


def test_cf_schema_with_standard_names(climate_dataset, climate_schema):
    """Test CF-compliant schema with standard names."""
    # Add CF standard names to the dataset
    dataset_cf = climate_dataset.copy()
    dataset_cf["temperature"].attrs["standard_name"] = "air_temperature"
    dataset_cf["precipitation"].attrs["standard_name"] = "precipitation_amount"
    dataset_cf["time"].attrs["standard_name"] = "time"
    dataset_cf["latitude"].attrs["standard_name"] = "latitude"
    dataset_cf["longitude"].attrs["standard_name"] = "longitude"

    # Add CF standard name checks to the schema
    cf_schema = climate_schema.copy()
    cf_schema.data_vars["temperature"].checks.append(
        check_cf_standard_name("air_temperature")
    )
    cf_schema.data_vars["precipitation"].checks.append(
        check_cf_standard_name("precipitation_amount")
    )
    cf_schema.coords["time"].checks.append(check_cf_standard_name("time"))
    cf_schema.coords["latitude"].checks.append(check_cf_standard_name("latitude"))
    cf_schema.coords["longitude"].checks.append(check_cf_standard_name("longitude"))

    # Validate CF dataset
    result = cf_schema.validate(dataset_cf)
    assert result is True

    # Test missing standard name
    dataset_missing_std = dataset_cf.copy()
    del dataset_missing_std["temperature"].attrs["standard_name"]
    with pytest.raises(Exception) as exc_info:
        cf_schema.validate(dataset_missing_std)
    assert "missing standard_name attribute" in str(exc_info.value)

    # Test incorrect standard name
    dataset_wrong_std = dataset_cf.copy()
    dataset_wrong_std["temperature"].attrs["standard_name"] = "wrong_name"
    with pytest.raises(Exception) as exc_info:
        cf_schema.validate(dataset_wrong_std)
    assert "incorrect standard_name" in str(exc_info.value)

    # Test invalid CF standard name
    dataset_invalid_std = dataset_cf.copy()
    # Use a valid CF standard name but for the wrong variable type
    dataset_invalid_std["temperature"].attrs["standard_name"] = "precipitation_amount"
    with pytest.raises(Exception) as exc_info:
        cf_schema.validate(dataset_invalid_std)
    assert "incorrect standard_name" in str(exc_info.value)


def test_cf_units():
    """Test CF unit validation."""
    # Create a DataArray with valid CF units
    data = xr.DataArray(np.random.rand(10), dims=["time"], attrs={"units": "K"})

    # Test exact unit match
    check = check_cf_units("K")
    assert check(data) is True

    # Test compatible units
    check_compatible = check_cf_units("K", compatible=True)
    data_celsius = xr.DataArray(
        np.random.rand(10), dims=["time"], attrs={"units": "degC"}
    )
    assert check_compatible(data_celsius) is True

    # Test missing units
    data_no_units = xr.DataArray(np.random.rand(10), dims=["time"])
    with pytest.raises(ValueError) as exc_info:
        check(data_no_units)
    assert "missing units attribute" in str(exc_info.value)

    # Test incorrect units
    data_wrong_units = xr.DataArray(
        np.random.rand(10), dims=["time"], attrs={"units": "m"}
    )
    with pytest.raises(ValueError) as exc_info:
        check(data_wrong_units)
    assert "incorrect units" in str(exc_info.value)

    # Test invalid CF units
    data_invalid_units = xr.DataArray(
        np.random.rand(10), dims=["time"], attrs={"units": "not_a_unit"}
    )
    with pytest.raises(ValueError) as exc_info:
        check(data_invalid_units)
    assert "Invalid CF units" in str(exc_info.value)

    # Test incompatible units
    data_incompatible = xr.DataArray(
        np.random.rand(10), dims=["time"], attrs={"units": "m"}
    )
    with pytest.raises(ValueError) as exc_info:
        check_compatible(data_incompatible)
    assert "Units are not compatible" in str(exc_info.value)


def test_cf_schema_with_units(climate_dataset, climate_schema):
    """Test CF-compliant schema with unit validation."""
    # Add CF units to the dataset
    dataset_cf = climate_dataset.copy()
    dataset_cf["temperature"].attrs["units"] = "K"
    dataset_cf["precipitation"].attrs["units"] = "mm/day"

    # Add CF unit checks to the schema
    cf_schema = climate_schema.copy()
    cf_schema.data_vars["temperature"].checks.append(check_cf_units("K"))
    cf_schema.data_vars["precipitation"].checks.append(check_cf_units("mm/day"))

    # Validate CF dataset
    result = cf_schema.validate(dataset_cf)
    assert result is True

    # Test missing units
    dataset_missing_units = dataset_cf.copy()
    del dataset_missing_units["temperature"].attrs["units"]
    with pytest.raises(Exception) as exc_info:
        cf_schema.validate(dataset_missing_units)
    assert "missing units attribute" in str(exc_info.value)

    # Test incorrect units
    dataset_wrong_units = dataset_cf.copy()
    dataset_wrong_units["temperature"].attrs["units"] = "m"
    with pytest.raises(Exception) as exc_info:
        cf_schema.validate(dataset_wrong_units)
    assert "incorrect units" in str(exc_info.value)

    # Test invalid CF units
    dataset_invalid_units = dataset_cf.copy()
    dataset_invalid_units["temperature"].attrs["units"] = "not_a_unit"
    with pytest.raises(Exception) as exc_info:
        cf_schema.validate(dataset_invalid_units)
    assert "Invalid CF units" in str(exc_info.value)

    # Test compatible units
    dataset_compatible = dataset_cf.copy()
    dataset_compatible["temperature"].attrs["units"] = "degC"
    cf_schema_compatible = climate_schema.copy()
    cf_schema_compatible.data_vars["temperature"].checks.append(
        check_cf_units("K", compatible=True)
    )
    result = cf_schema_compatible.validate(dataset_compatible)
    assert result is True


def test_geographic_coordinates_valid():
    """Test valid geographic coordinates."""
    lon = xr.DataArray(
        np.linspace(-180, 180, 5),
        dims=["longitude"],
        attrs={
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        },
    )
    lat = xr.DataArray(
        np.linspace(-90, 90, 5),
        dims=["latitude"],
        attrs={
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        },
    )

    check_geo = check_coordinate_system("geographic")
    assert check_geo(lon) is True
    assert check_geo(lat) is True


def test_geographic_coordinates_missing_attribute():
    """Test geographic coordinates with missing required attribute."""
    lon = xr.DataArray(
        np.linspace(-180, 180, 5),
        dims=["longitude"],
        attrs={
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        },
    )
    lon_missing = lon.copy()
    del lon_missing.attrs["standard_name"]

    check_geo = check_coordinate_system("geographic")
    with pytest.raises(ValueError) as exc_info:
        check_geo(lon_missing)
    assert "Missing required attribute" in str(exc_info.value)


def test_geographic_coordinates_invalid_attribute():
    """Test geographic coordinates with invalid attribute value."""
    lon = xr.DataArray(
        np.linspace(-180, 180, 5),
        dims=["longitude"],
        attrs={
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        },
    )
    lon_invalid = lon.copy()
    lon_invalid.attrs["units"] = "m"

    check_geo = check_coordinate_system("geographic")
    with pytest.raises(ValueError) as exc_info:
        check_geo(lon_invalid)
    assert "Invalid value for units" in str(exc_info.value)


def test_projected_coordinates_valid():
    """Test valid projected coordinates."""
    x = xr.DataArray(
        np.linspace(0, 1000, 5),
        dims=["x"],
        attrs={
            "standard_name": "projection_x_coordinate",
            "units": "m",
            "axis": "X",
            "grid_mapping_name": "transverse_mercator",
            "crs": "EPSG:32633",
        },
    )
    y = xr.DataArray(
        np.linspace(0, 1000, 5),
        dims=["y"],
        attrs={
            "standard_name": "projection_y_coordinate",
            "units": "m",
            "axis": "Y",
            "grid_mapping_name": "transverse_mercator",
            "crs": "EPSG:32633",
        },
    )

    check_proj = check_coordinate_system("projected")
    assert check_proj(x) is True
    assert check_proj(y) is True


def test_projected_coordinates_missing_grid_mapping():
    """Test projected coordinates with missing grid mapping."""
    x = xr.DataArray(
        np.linspace(0, 1000, 5),
        dims=["x"],
        attrs={
            "standard_name": "projection_x_coordinate",
            "units": "m",
            "axis": "X",
            "grid_mapping_name": "transverse_mercator",
            "crs": "EPSG:32633",
        },
    )
    x_missing = x.copy()
    del x_missing.attrs["grid_mapping_name"]

    check_proj = check_coordinate_system("projected")
    with pytest.raises(ValueError) as exc_info:
        check_proj(x_missing)
    assert (
        "Missing required attribute for projected coordinate system: grid_mapping_name"
        in str(exc_info.value)
    )


def test_vertical_coordinates_valid():
    """Test valid vertical coordinates."""
    height = xr.DataArray(
        np.linspace(0, 1000, 5),
        dims=["height"],
        attrs={
            "standard_name": "height",
            "units": "m",
            "axis": "Z",
            "positive": "up",
        },
    )

    check_vert = check_coordinate_system("vertical")
    assert check_vert(height) is True


def test_vertical_coordinates_missing_positive():
    """Test vertical coordinates with missing positive attribute."""
    height = xr.DataArray(
        np.linspace(0, 1000, 5),
        dims=["height"],
        attrs={
            "standard_name": "height",
            "units": "m",
            "axis": "Z",
            "positive": "up",
        },
    )
    height_missing = height.copy()
    del height_missing.attrs["positive"]

    check_vert = check_coordinate_system("vertical")
    with pytest.raises(ValueError) as exc_info:
        check_vert(height_missing)
    assert "Missing required attribute for vertical coordinate system: positive" in str(
        exc_info.value
    )


def test_temporal_coordinates_valid():
    """Test valid temporal coordinates."""
    time = xr.DataArray(
        np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
        dims=["time"],
        attrs={
            "standard_name": "time",
            "axis": "T",
        },
    )

    check_time = check_coordinate_system("temporal")
    assert check_time(time) is True


def test_coordinate_system_custom_requirements():
    """Test coordinate system with custom requirements."""
    lon = xr.DataArray(
        np.linspace(-180, 180, 5),
        dims=["longitude"],
        attrs={
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        },
    )
    custom_geo = check_coordinate_system(
        "geographic",
        required_attrs={"long_name": "custom longitude"},
    )
    lon_custom = lon.copy()
    lon_custom.attrs["long_name"] = "custom longitude"
    assert custom_geo(lon_custom) is True


def test_coordinate_system_invalid_type():
    """Test coordinate system with invalid type."""
    with pytest.raises(ValueError) as exc_info:
        check_coordinate_system("invalid_type")  # type: ignore
    assert "Unknown coordinate system type" in str(exc_info.value)


def test_cf_schema_with_coordinate_systems(climate_dataset, climate_schema):
    """Test CF-compliant schema with coordinate system validation."""
    # Add coordinate system attributes to the dataset
    dataset_cf = climate_dataset.copy()
    dataset_cf["longitude"].attrs.update(
        {
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        }
    )
    dataset_cf["latitude"].attrs.update(
        {
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        }
    )
    dataset_cf["time"].attrs.update(
        {
            "standard_name": "time",
            "axis": "T",
        }
    )

    # Add coordinate system checks to the schema
    cf_schema = climate_schema.copy()
    cf_schema.coords["longitude"].checks.append(check_coordinate_system("geographic"))
    cf_schema.coords["latitude"].checks.append(check_coordinate_system("geographic"))
    cf_schema.coords["time"].checks.append(check_coordinate_system("temporal"))

    # Validate CF dataset
    result = cf_schema.validate(dataset_cf)
    assert result is True

    # Test missing coordinate system attributes
    dataset_missing = dataset_cf.copy()
    del dataset_missing["longitude"].attrs["standard_name"]
    with pytest.raises(Exception) as exc_info:
        cf_schema.validate(dataset_missing)
    assert "Missing required attribute" in str(exc_info.value)

    # Test invalid coordinate system attributes
    dataset_invalid = dataset_cf.copy()
    dataset_invalid["longitude"].attrs["units"] = "m"
    with pytest.raises(Exception) as exc_info:
        cf_schema.validate(dataset_invalid)
    assert "Invalid value for units" in str(exc_info.value)


# Grid mapping tests
def test_grid_mapping_valid():
    """Test valid grid mapping definition."""
    grid_mapping = xr.DataArray(
        0,  # Dummy value
        attrs={
            "grid_mapping_name": "transverse_mercator",
            "scale_factor_at_central_meridian": 0.9996,
            "longitude_of_central_meridian": -117.0,
            "latitude_of_projection_origin": 0.0,
            "false_easting": 500000.0,
            "false_northing": 0.0,
            "crs": "EPSG:32633",
        },
    )

    # Test with required parameters
    required_params = {
        "scale_factor_at_central_meridian": 0.9996,
        "longitude_of_central_meridian": -117.0,
        "latitude_of_projection_origin": 0.0,
        "false_easting": 500000.0,
        "false_northing": 0.0,
        "crs": "EPSG:32633",
    }
    check = check_grid_mapping("transverse_mercator", required_params)
    assert check(grid_mapping) is True

    # Test without required parameters
    check = check_grid_mapping("transverse_mercator")
    assert check(grid_mapping) is True


def test_grid_mapping_missing_name():
    """Test grid mapping with missing name."""
    grid_mapping = xr.DataArray(
        0,
        attrs={
            "scale_factor_at_central_meridian": 0.9996,
            "longitude_of_central_meridian": -117.0,
        },
    )

    check = check_grid_mapping("transverse_mercator")
    with pytest.raises(ValueError) as exc_info:
        check(grid_mapping)
    assert "DataArray missing grid_mapping_name attribute" in str(exc_info.value)


def test_grid_mapping_incorrect_name():
    """Test grid mapping with incorrect name."""
    grid_mapping = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "lambert_conformal",
            "scale_factor_at_central_meridian": 0.9996,
        },
    )

    check = check_grid_mapping("transverse_mercator")
    with pytest.raises(ValueError) as exc_info:
        check(grid_mapping)
    assert "DataArray has incorrect grid_mapping_name" in str(exc_info.value)


def test_grid_mapping_missing_parameter():
    """Test grid mapping with missing required parameter."""
    grid_mapping = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "transverse_mercator",
            "scale_factor_at_central_meridian": 0.9996,
            # Missing longitude_of_central_meridian
        },
    )

    required_params = {
        "scale_factor_at_central_meridian": 0.9996,
        "longitude_of_central_meridian": -117.0,
    }
    check = check_grid_mapping("transverse_mercator", required_params)
    with pytest.raises(ValueError) as exc_info:
        check(grid_mapping)
    assert "Grid mapping missing required parameter" in str(exc_info.value)


def test_grid_mapping_incorrect_parameter():
    """Test grid mapping with incorrect parameter value."""
    grid_mapping = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "transverse_mercator",
            "scale_factor_at_central_meridian": 0.9996,
            "longitude_of_central_meridian": -120.0,  # Incorrect value
        },
    )

    required_params = {
        "scale_factor_at_central_meridian": 0.9996,
        "longitude_of_central_meridian": -117.0,
    }
    check = check_grid_mapping("transverse_mercator", required_params)
    with pytest.raises(ValueError) as exc_info:
        check(grid_mapping)
    assert "Grid mapping has incorrect value for" in str(exc_info.value)


def test_grid_mapping_missing_crs():
    """Test grid mapping with missing CRS."""
    grid_mapping = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "transverse_mercator",
            "scale_factor_at_central_meridian": 0.9996,
        },
    )

    check = check_grid_mapping("transverse_mercator")
    with pytest.raises(ValueError) as exc_info:
        check(grid_mapping)
    assert "Grid mapping requires either crs or grid_mapping attribute" in str(
        exc_info.value
    )


def test_grid_mapping_reference_valid():
    """Test valid grid mapping reference."""
    data = xr.DataArray(
        np.random.rand(5, 5),
        dims=["y", "x"],
        attrs={
            "grid_mapping": "transverse_mercator",
            "standard_name": "air_temperature",
            "units": "K",
        },
    )

    check = check_grid_mapping_reference("transverse_mercator")
    assert check(data) is True


def test_grid_mapping_reference_missing():
    """Test missing grid mapping reference."""
    data = xr.DataArray(
        np.random.rand(5, 5),
        dims=["y", "x"],
        attrs={
            "standard_name": "air_temperature",
            "units": "K",
        },
    )

    check = check_grid_mapping_reference("transverse_mercator")
    with pytest.raises(ValueError) as exc_info:
        check(data)
    assert "DataArray missing grid_mapping attribute" in str(exc_info.value)


def test_grid_mapping_reference_incorrect():
    """Test incorrect grid mapping reference."""
    data = xr.DataArray(
        np.random.rand(5, 5),
        dims=["y", "x"],
        attrs={
            "grid_mapping": "lambert_conformal",  # Incorrect reference
            "standard_name": "air_temperature",
            "units": "K",
        },
    )

    check = check_grid_mapping_reference("transverse_mercator")
    with pytest.raises(ValueError) as exc_info:
        check(data)
    assert "DataArray references incorrect grid mapping variable" in str(exc_info.value)


def test_grid_mapping_in_schema():
    """Test grid mapping validation in a complete schema."""
    # Create a dataset with projected coordinates and grid mapping
    dataset = xr.Dataset(
        data_vars={
            "temperature": xr.DataArray(
                np.random.rand(5, 5),
                dims=["y", "x"],
                attrs={
                    "grid_mapping": "transverse_mercator",
                    "standard_name": "air_temperature",
                    "units": "K",
                },
            ),
            "transverse_mercator": xr.DataArray(
                0,
                attrs={
                    "grid_mapping_name": "transverse_mercator",
                    "scale_factor_at_central_meridian": 0.9996,
                    "longitude_of_central_meridian": -117.0,
                    "latitude_of_projection_origin": 0.0,
                    "false_easting": 500000.0,
                    "false_northing": 0.0,
                    "crs": "EPSG:32633",
                },
            ),
        },
        coords={
            "x": xr.DataArray(
                np.linspace(0, 1000, 5),
                dims=["x"],
                attrs={
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                    "axis": "X",
                    "grid_mapping_name": "transverse_mercator",
                    "crs": "EPSG:32633",
                },
            ),
            "y": xr.DataArray(
                np.linspace(0, 1000, 5),
                dims=["y"],
                attrs={
                    "standard_name": "projection_y_coordinate",
                    "units": "m",
                    "axis": "Y",
                    "grid_mapping_name": "transverse_mercator",
                    "crs": "EPSG:32633",
                },
            ),
        },
    )

    # Create a schema that validates the grid mapping
    schema = XArraySchema(
        data_vars={
            "temperature": DataVariable(
                dtype=np.dtype(np.float64),
                checks=[
                    check_dims(["y", "x"]),
                    check_dtype(np.dtype(np.float64)),
                    check_grid_mapping_reference("transverse_mercator"),
                ],
                required=True,
            ),
            "transverse_mercator": DataVariable(
                dtype=np.dtype(np.int32),
                checks=[
                    check_grid_mapping(
                        "transverse_mercator",
                        {
                            "scale_factor_at_central_meridian": 0.9996,
                            "longitude_of_central_meridian": -117.0,
                            "latitude_of_projection_origin": 0.0,
                            "false_easting": 500000.0,
                            "false_northing": 0.0,
                            "crs": "EPSG:32633",
                        },
                    ),
                ],
                required=True,
            ),
        },
        coords={
            "x": Coordinate(
                dtype=np.dtype(np.float64),
                checks=[
                    check_dims(["x"]),
                    check_coordinate_system("projected"),
                ],
                required=True,
            ),
            "y": Coordinate(
                dtype=np.dtype(np.float64),
                checks=[
                    check_dims(["y"]),
                    check_coordinate_system("projected"),
                ],
                required=True,
            ),
        },
    )

    # Validate the dataset
    result = schema.validate(dataset)
    assert result is True

    # Test missing grid mapping reference
    dataset_missing_ref = dataset.copy()
    del dataset_missing_ref["temperature"].attrs["grid_mapping"]
    with pytest.raises(Exception) as exc_info:
        schema.validate(dataset_missing_ref)
    assert "DataArray missing grid_mapping attribute" in str(exc_info.value)

    # Test incorrect grid mapping reference
    dataset_wrong_ref = dataset.copy()
    dataset_wrong_ref["temperature"].attrs["grid_mapping"] = "wrong_mapping"
    with pytest.raises(Exception) as exc_info:
        schema.validate(dataset_wrong_ref)
    assert "DataArray references incorrect grid mapping variable" in str(exc_info.value)

    # Test missing grid mapping variable
    dataset_missing_var = dataset.drop_vars("transverse_mercator")
    with pytest.raises(Exception) as exc_info:
        schema.validate(dataset_missing_var)
    assert "missing required data variable" in str(exc_info.value)

    # Test incorrect grid mapping parameters
    dataset_wrong_params = dataset.copy()
    dataset_wrong_params["transverse_mercator"].attrs[
        "longitude_of_central_meridian"
    ] = -120.0
    with pytest.raises(Exception) as exc_info:
        schema.validate(dataset_wrong_params)
    assert "Grid mapping has incorrect value for" in str(exc_info.value)


def test_time_coordinate_valid():
    """Test valid time coordinate."""
    time = xr.DataArray(
        np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
        dims=["time"],
        attrs={
            "standard_name": "time",
            "calendar": "standard",
            "units": "days since 2020-01-01",
            "timezone": "UTC",
        },
    )

    # Test with all parameters
    check = check_time_coordinate(
        calendar="standard",
        units="days since 2020-01-01",
        timezone="UTC",
    )
    assert check(time) is True

    # Test with no parameters
    check = check_time_coordinate()
    assert check(time) is True


def test_time_coordinate_invalid_standard_name():
    """Test time coordinate with invalid standard name."""
    time = xr.DataArray(
        np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
        dims=["time"],
        attrs={
            "standard_name": "wrong_name",
            "calendar": "standard",
            "units": "days since 2020-01-01",
            "timezone": "UTC",
        },
    )

    check = check_time_coordinate()
    with pytest.raises(ValueError) as exc_info:
        check(time)
    assert "DataArray is not a time coordinate" in str(exc_info.value)


def test_time_coordinate_invalid_dtype():
    """Test time coordinate with invalid dtype."""
    time = xr.DataArray(
        np.arange(10),
        dims=["time"],
        attrs={
            "standard_name": "time",
            "calendar": "standard",
            "units": "days since 2020-01-01",
            "timezone": "UTC",
        },
    )

    check = check_time_coordinate()
    with pytest.raises(ValueError) as exc_info:
        check(time)
    assert "Time coordinate must have datetime64 dtype" in str(exc_info.value)


def test_time_coordinate_missing_calendar():
    """Test time coordinate with missing calendar."""
    time = xr.DataArray(
        np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
        dims=["time"],
        attrs={
            "standard_name": "time",
            "units": "days since 2020-01-01",
            "timezone": "UTC",
        },
    )

    check = check_time_coordinate(calendar="standard")
    with pytest.raises(ValueError) as exc_info:
        check(time)
    assert "Time coordinate missing calendar attribute" in str(exc_info.value)


def test_time_coordinate_incorrect_calendar():
    """Test time coordinate with incorrect calendar."""
    time = xr.DataArray(
        np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
        dims=["time"],
        attrs={
            "standard_name": "time",
            "calendar": "gregorian",
            "units": "days since 2020-01-01",
            "timezone": "UTC",
        },
    )

    check = check_time_coordinate(calendar="standard")
    with pytest.raises(ValueError) as exc_info:
        check(time)
    assert "Time coordinate has incorrect calendar" in str(exc_info.value)


def test_time_coordinate_missing_units():
    """Test time coordinate with missing units."""
    time = xr.DataArray(
        np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
        dims=["time"],
        attrs={
            "standard_name": "time",
            "calendar": "standard",
            "timezone": "UTC",
        },
    )

    check = check_time_coordinate(units="days since 2020-01-01")
    with pytest.raises(ValueError) as exc_info:
        check(time)
    assert "Time coordinate missing units attribute" in str(exc_info.value)


def test_time_coordinate_incorrect_units():
    """Test time coordinate with incorrect units."""
    time = xr.DataArray(
        np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
        dims=["time"],
        attrs={
            "standard_name": "time",
            "calendar": "standard",
            "units": "days since 2019-01-01",
            "timezone": "UTC",
        },
    )

    check = check_time_coordinate(units="days since 2020-01-01")
    with pytest.raises(ValueError) as exc_info:
        check(time)
    assert "Time coordinate has incorrect units" in str(exc_info.value)


def test_time_coordinate_missing_timezone():
    """Test time coordinate with missing timezone."""
    time = xr.DataArray(
        np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
        dims=["time"],
        attrs={
            "standard_name": "time",
            "calendar": "standard",
            "units": "days since 2020-01-01",
        },
    )

    check = check_time_coordinate(timezone="UTC")
    with pytest.raises(ValueError) as exc_info:
        check(time)
    assert "Time coordinate missing timezone attribute" in str(exc_info.value)


def test_time_coordinate_incorrect_timezone():
    """Test time coordinate with incorrect timezone."""
    time = xr.DataArray(
        np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
        dims=["time"],
        attrs={
            "standard_name": "time",
            "calendar": "standard",
            "units": "days since 2020-01-01",
            "timezone": "GMT",
        },
    )

    check = check_time_coordinate(timezone="UTC")
    with pytest.raises(ValueError) as exc_info:
        check(time)
    assert "Time coordinate has incorrect timezone" in str(exc_info.value)


def test_time_coordinate_non_monotonic():
    """Test time coordinate that is not monotonic."""
    time = xr.DataArray(
        np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-03"),
                np.datetime64("2020-01-02"),  # Out of order
                np.datetime64("2020-01-04"),
            ]
        ),
        dims=["time"],
        attrs={
            "standard_name": "time",
            "calendar": "standard",
            "units": "days since 2020-01-01",
            "timezone": "UTC",
        },
    )

    check = check_time_coordinate()
    with pytest.raises(ValueError) as exc_info:
        check(time)
    assert "Time coordinate must be monotonic" in str(exc_info.value)
