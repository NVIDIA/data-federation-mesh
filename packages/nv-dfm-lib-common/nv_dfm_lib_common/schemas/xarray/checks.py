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

"""Common validation checks for xarray."""

from typing import Any, Callable, Dict, Hashable, List, Literal, Optional

import numpy as np
import xarray as xr
from pandera.api.checks import Check
from pandera.backends.base import BaseCheckBackend


class XArrayCheckBackend(BaseCheckBackend):
    """Backend for xarray DataArray checks."""

    def __init__(self, check: Check):
        """Initialize the backend.

        Args:
            check: The check to run.
        """
        self.check = check

    def __call__(self, check_obj: xr.DataArray, column: Optional[str] = None) -> bool:
        """Run the check on the data.

        Args:
            check_obj: The DataArray to check.
            column: Ignored for DataArray checks.

        Returns:
            True if the check passes, False otherwise.
        """
        return self.check._check_fn(check_obj)


# Register the backend
Check.register_backend(xr.DataArray, XArrayCheckBackend)


def check_dims(dims: List[Hashable]) -> Callable[[xr.DataArray], bool]:
    """Check if DataArray has expected dimensions in the correct order.

    Args:
        dims: expected dimensions in order

    Returns:
        Function that returns True if dimensions match in order
    """

    def _check_dims(dataarray: xr.DataArray) -> bool:
        # Check if all required dimensions are present
        if not all(dim in dataarray.dims for dim in dims):
            return False
        # Check if dimensions are in the correct order
        return list(dataarray.dims) == list(dims)

    _check_dims.__name__ = f"check_dims({dims})"
    return _check_dims


def check_dtype(dtype: np.dtype) -> Callable[[xr.DataArray], bool]:
    """Check if DataArray has expected dtype.

    Args:
        dtype: expected dtype

    Returns:
        Function that returns True if dtype matches
    """

    def _check_dtype(dataarray: xr.DataArray) -> bool:
        return np.issubdtype(dataarray.dtype, dtype)

    _check_dtype.__name__ = f"check_dtype({dtype})"
    return _check_dtype


def check_max_missing(max_missing: float) -> Callable[[xr.DataArray], bool]:
    """Check if the proportion of missing values is below a threshold.

    Args:
        max_missing: Maximum allowed proportion of missing values (0-1)

    Returns:
        Check function that returns True if the proportion of missing values is below the threshold
    """

    def _check_max_missing(data: xr.DataArray) -> bool:
        """Check if the proportion of missing values is below a threshold.

        Args:
            data: DataArray to check

        Returns:
            True if the proportion of missing values is below the threshold
        """
        missing = data.isnull().sum() / data.size
        if missing > max_missing:
            return False
        return True

    _check_max_missing.__name__ = f"check_max_missing({max_missing})"
    return _check_max_missing


def check_attrs(attrs: Dict[str, Any]) -> Check:
    """Check if DataArray has expected attributes.

    Args:
        attrs: dictionary of expected attributes

    Returns:
        Check object
    """

    def _check_attrs(dataarray: xr.DataArray) -> bool:
        for key, value in attrs.items():
            if key not in dataarray.attrs:
                return False
            if dataarray.attrs[key] != value:
                return False
        return True

    return Check(_check_attrs, name="check_attrs")


def check_range(min_val: float, max_val: float) -> Callable[[xr.DataArray], bool]:
    """Check if DataArray values are within range.

    Args:
        min_val: minimum allowed value
        max_val: maximum allowed value

    Returns:
        Function that returns True if values are within range
    """

    def _check_range(dataarray: xr.DataArray) -> bool:
        # Use xarray's min and max with skipna=True to ignore NaN values
        min_check = dataarray.min(skipna=True).item() >= min_val
        max_check = dataarray.max(skipna=True).item() <= max_val
        return bool(min_check and max_check)

    _check_range.__name__ = f"check_range({min_val}, {max_val})"
    return _check_range


def check_cf_standard_name(standard_name: str) -> Callable[[xr.DataArray], bool]:
    """Check if DataArray has expected CF standard name.

    Args:
        standard_name: expected CF standard name

    Returns:
        Function that returns True if standard name matches
    """

    def _check_cf_standard_name(dataarray: xr.DataArray) -> bool:
        if "standard_name" not in dataarray.attrs:
            raise ValueError("DataArray missing standard_name attribute")

        actual_name = dataarray.attrs["standard_name"]
        if actual_name != standard_name:
            raise ValueError(
                f"DataArray has incorrect standard_name: {actual_name}, expected: {standard_name}"
            )
        return True

    _check_cf_standard_name.__name__ = f"check_cf_standard_name({standard_name})"
    return _check_cf_standard_name


def check_cf_units(
    units: str, compatible: bool = False
) -> Callable[[xr.DataArray], bool]:
    """Check if DataArray has expected CF units.

    Args:
        units: expected CF units
        compatible: if True, check for unit compatibility instead of exact match

    Returns:
        Function that returns True if units match or are compatible
    """
    # Define temperature unit compatibility
    TEMPERATURE_UNITS = {
        "K": ["K", "kelvin", "degK"],
        "degC": ["degC", "celsius", "C"],
        "degF": ["degF", "fahrenheit", "F"],
    }

    # Common CF units and their components
    CF_UNITS = {
        # Basic units
        "K",
        "kelvin",
        "degK",
        "degC",
        "celsius",
        "C",
        "degF",
        "fahrenheit",
        "F",  # Temperature
        "m",
        "meter",
        "meters",
        "km",
        "kilometer",
        "kilometers",  # Length
        "cm",
        "centimeter",
        "centimeters",
        "mm",
        "millimeter",
        "millimeters",
        "s",
        "second",
        "seconds",
        "min",
        "minute",
        "minutes",  # Time
        "h",
        "hour",
        "hours",
        "d",
        "day",
        "days",
        "kg",
        "kilogram",
        "kilograms",
        "g",
        "gram",
        "grams",  # Mass
        "m3",
        "cubic meter",
        "cubic meters",
        "L",
        "liter",
        "liters",  # Volume
        "Pa",
        "pascal",
        "pascals",
        "hPa",
        "hectopascal",
        "hectopascals",  # Pressure
        "J",
        "joule",
        "joules",
        "kJ",
        "kilojoule",
        "kilojoules",  # Energy
        "W",
        "watt",
        "watts",
        "kW",
        "kilowatt",
        "kilowatts",  # Power
        "rad",
        "radian",
        "radians",
        "deg",
        "degree",
        "degrees",  # Angle
        "m/s",
        "meters per second",
        "km/h",
        "kilometers per hour",  # Speed
        "m2",
        "square meter",
        "square meters",
        "ha",
        "hectare",
        "hectares",  # Area
        "mol/m3",
        "moles per cubic meter",
        "mol/kg",
        "moles per kilogram",  # Concentration
    }

    # Common compound units and their components
    COMPOUND_UNITS = {
        # Rate units
        "mm/day": ["mm", "day"],
        "mm/hour": ["mm", "hour"],
        "mm/s": ["mm", "s"],
        "m/s": ["m", "s"],
        "km/h": ["km", "h"],
        "kg/m2/s": ["kg", "m2", "s"],
        "W/m2": ["W", "m2"],
        "J/kg": ["J", "kg"],
        "K/s": ["K", "s"],
        "Pa/s": ["Pa", "s"],
        # Area units
        "m2": ["m", "m"],
        "km2": ["km", "km"],
        # Volume units
        "m3": ["m", "m", "m"],
        "km3": ["km", "km", "km"],
        # Concentration units
        "mol/m3": ["mol", "m3"],
        "mol/kg": ["mol", "kg"],
    }

    def _validate_unit(unit: str) -> bool:
        """Validate if a unit is a valid CF unit or compound unit."""
        # Check if it's a basic unit
        if unit in CF_UNITS:
            return True

        # Check if it's a compound unit
        if unit in COMPOUND_UNITS:
            # Validate each component
            return all(comp in CF_UNITS for comp in COMPOUND_UNITS[unit])

        # Check if it's a rate unit (e.g., "mm/day")
        if "/" in unit:
            parts = unit.split("/")
            return all(part in CF_UNITS for part in parts)

        return False

    def _check_cf_units(dataarray: xr.DataArray) -> bool:
        if "units" not in dataarray.attrs:
            raise ValueError("DataArray missing units attribute")

        actual_units = dataarray.attrs["units"]

        # First validate that both units are valid CF units
        if not _validate_unit(units):
            raise ValueError(f"Invalid CF units: {units}")
        if not _validate_unit(actual_units):
            raise ValueError(f"Invalid CF units: {actual_units}")

        if compatible:
            # Special handling for temperature units
            # Check if both units are temperature units
            is_expected_temp = any(
                units in temp_units for temp_units in TEMPERATURE_UNITS.values()
            )
            is_actual_temp = any(
                actual_units in temp_units for temp_units in TEMPERATURE_UNITS.values()
            )

            if is_expected_temp and is_actual_temp:
                # Both are temperature units, they're compatible
                pass
            elif actual_units != units:
                # For non-temperature units, use exact match for now
                # TODO: Add proper unit compatibility checking using a unit conversion library
                raise ValueError(f"Units are not compatible: {actual_units} vs {units}")
        else:
            if actual_units != units:
                raise ValueError(
                    f"DataArray has incorrect units: {actual_units}, expected: {units}"
                )
        return True

    _check_cf_units.__name__ = f"check_cf_units({units}, compatible={compatible})"
    return _check_cf_units


def check_coordinate_system(
    system_type: Literal["geographic", "projected", "vertical", "temporal"],
    required_attrs: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Check if a DataArray has a properly defined coordinate system.

    Args:
        system_type: Type of coordinate system to validate
        required_attrs: Additional required attributes specific to this coordinate system

    Returns:
        A check function that validates the coordinate system

    Raises:
        ValueError: If the coordinate system is not properly defined
    """
    # Define required attributes for each coordinate system type
    system_requirements = {
        "geographic": {
            "standard_name": ["longitude", "latitude"],
            "units": ["degrees_east", "degrees_north"],
            "axis": ["X", "Y"],
        },
        "projected": {
            "standard_name": ["projection_x_coordinate", "projection_y_coordinate"],
            "units": ["m"],
            "axis": ["X", "Y"],
            "grid_mapping_name": None,  # Will be validated separately
        },
        "vertical": {
            "standard_name": ["height", "depth", "altitude"],
            "units": ["m"],
            "axis": ["Z"],
            "positive": ["up", "down"],
        },
        "temporal": {
            "standard_name": ["time"],
            "axis": ["T"],
        },
    }

    if system_type not in system_requirements:
        raise ValueError(f"Unknown coordinate system type: {system_type}")

    # Merge default requirements with user-specified requirements
    requirements = system_requirements[system_type].copy()
    if required_attrs:
        for key, value in required_attrs.items():
            if key in requirements:
                if isinstance(requirements[key], list):
                    requirements[key] = value if isinstance(value, list) else [value]
                else:
                    requirements[key] = value
            else:
                requirements[key] = value

    def _check_coordinate_system(data: xr.DataArray) -> bool:
        """Check if a DataArray has a properly defined coordinate system.

        Args:
            data: DataArray to check

        Returns:
            True if the coordinate system is valid

        Raises:
            ValueError: If the coordinate system is not properly defined
        """
        # Check for required attributes
        for attr, allowed_values in requirements.items():
            if attr not in data.attrs:
                raise ValueError(
                    f"Missing required attribute for {system_type} coordinate system: {attr}"
                )

            if allowed_values is not None:
                if isinstance(allowed_values, list):
                    if data.attrs[attr] not in allowed_values:
                        raise ValueError(
                            f"Invalid value for {attr} in {system_type} coordinate system. "
                            f"Expected one of {allowed_values}, got {data.attrs[attr]}"
                        )
                else:
                    if data.attrs[attr] != allowed_values:
                        raise ValueError(
                            f"Invalid value for {attr} in {system_type} coordinate system. "
                            f"Expected {allowed_values}, got {data.attrs[attr]}"
                        )

        # Additional checks for specific coordinate system types
        if system_type == "projected":
            if "grid_mapping_name" not in data.attrs:
                raise ValueError(
                    "Projected coordinate system requires grid_mapping_name attribute"
                )
            if "crs" not in data.attrs and "grid_mapping" not in data.attrs:
                raise ValueError(
                    "Projected coordinate system requires either crs or grid_mapping attribute"
                )

        elif system_type == "vertical":
            if "positive" not in data.attrs:
                raise ValueError(
                    "Vertical coordinate system requires positive attribute (up/down)"
                )

        return True

    _check_coordinate_system.__name__ = f"check_coordinate_system({system_type})"
    return _check_coordinate_system


def check_grid_mapping(
    grid_mapping_name: str,
    required_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Check if a DataArray has a valid grid mapping definition.

    Args:
        grid_mapping_name: The expected grid mapping name (e.g., "transverse_mercator")
        required_params: Dictionary of required projection parameters and their expected values

    Returns:
        A check function that validates the grid mapping

    Raises:
        ValueError: If the grid mapping is not properly defined
    """

    def _check_grid_mapping(data: xr.DataArray) -> bool:
        """Check if a DataArray has a valid grid mapping definition.

        Args:
            data: DataArray to check

        Returns:
            True if the grid mapping is valid

        Raises:
            ValueError: If the grid mapping is not properly defined
        """
        # Check if this is a grid mapping variable
        if "grid_mapping_name" not in data.attrs:
            raise ValueError("DataArray missing grid_mapping_name attribute")

        actual_name = data.attrs["grid_mapping_name"]
        if actual_name != grid_mapping_name:
            raise ValueError(
                f"DataArray has incorrect grid_mapping_name: {actual_name}, "
                f"expected: {grid_mapping_name}"
            )

        # Check required parameters
        if required_params:
            for param, expected_value in required_params.items():
                if param not in data.attrs:
                    raise ValueError(
                        f"Grid mapping missing required parameter: {param}"
                    )
                if data.attrs[param] != expected_value:
                    raise ValueError(
                        f"Grid mapping has incorrect value for {param}: "
                        f"{data.attrs[param]}, expected: {expected_value}"
                    )

        # Check for CRS if not in required_params
        if not required_params or "crs" not in required_params:
            if "crs" not in data.attrs and "grid_mapping" not in data.attrs:
                raise ValueError(
                    "Grid mapping requires either crs or grid_mapping attribute"
                )

        return True

    _check_grid_mapping.__name__ = f"check_grid_mapping({grid_mapping_name})"
    return _check_grid_mapping


def check_grid_mapping_reference(grid_mapping_var: str) -> Callable:
    """Check if a DataArray references a valid grid mapping variable.

    Args:
        grid_mapping_var: The name of the expected grid mapping variable

    Returns:
        A check function that validates the grid mapping reference

    Raises:
        ValueError: If the grid mapping reference is not properly defined
    """

    def _check_grid_mapping_reference(data: xr.DataArray) -> bool:
        """Check if a DataArray references a valid grid mapping variable.

        Args:
            data: DataArray to check

        Returns:
            True if the grid mapping reference is valid

        Raises:
            ValueError: If the grid mapping reference is not properly defined
        """
        if "grid_mapping" not in data.attrs:
            raise ValueError("DataArray missing grid_mapping attribute")

        actual_var = data.attrs["grid_mapping"]
        if actual_var != grid_mapping_var:
            raise ValueError(
                f"DataArray references incorrect grid mapping variable: {actual_var}, "
                f"expected: {grid_mapping_var}"
            )

        return True

    _check_grid_mapping_reference.__name__ = (
        f"check_grid_mapping_reference({grid_mapping_var})"
    )
    return _check_grid_mapping_reference


def check_time_coordinate(
    calendar: Optional[str] = None,
    units: Optional[str] = None,
    timezone: Optional[str] = None,
) -> Callable[[xr.DataArray], bool]:
    """
    Check if a DataArray has a properly defined time coordinate according to CF conventions.

    Args:
        calendar: Expected calendar type (e.g., "standard", "gregorian", "proleptic_gregorian",
                 "noleap", "365_day", "all_leap", "366_day", "360_day", "julian", "none")
        units: Expected time units (e.g., "days since 2000-01-01")
        timezone: Expected timezone (e.g., "UTC", "GMT")

    Returns:
        A function that validates a DataArray's time coordinate.

    Raises:
        ValueError: If the time coordinate is invalid.
    """

    def _check_time_coordinate(data: xr.DataArray) -> bool:
        # Check if the data is a time coordinate
        if "standard_name" in data.attrs and data.attrs["standard_name"] != "time":
            raise ValueError("DataArray is not a time coordinate")

        # Check if the data type is datetime64
        if not np.issubdtype(data.dtype, np.datetime64):
            raise ValueError("Time coordinate must have datetime64 dtype")

        # Check calendar if specified
        if calendar is not None:
            if "calendar" not in data.attrs:
                raise ValueError("Time coordinate missing calendar attribute")
            if data.attrs["calendar"] != calendar:
                raise ValueError(
                    f"Time coordinate has incorrect calendar: {data.attrs['calendar']}, "
                    f"expected {calendar}"
                )

        # Check units if specified
        if units is not None:
            if "units" not in data.attrs:
                raise ValueError("Time coordinate missing units attribute")
            if data.attrs["units"] != units:
                raise ValueError(
                    f"Time coordinate has incorrect units: {data.attrs['units']}, "
                    f"expected {units}"
                )

        # Check timezone if specified
        if timezone is not None:
            if "timezone" not in data.attrs:
                raise ValueError("Time coordinate missing timezone attribute")
            if data.attrs["timezone"] != timezone:
                raise ValueError(
                    f"Time coordinate has incorrect timezone: {data.attrs['timezone']}, "
                    f"expected {timezone}"
                )

        # Check for monotonicity using numpy's diff
        time_diffs = np.diff(data.values)
        is_increasing = np.all(time_diffs > np.timedelta64(0))
        is_decreasing = np.all(time_diffs < np.timedelta64(0))
        if not (is_increasing or is_decreasing):
            raise ValueError("Time coordinate must be monotonic")

        return True

    return _check_time_coordinate
