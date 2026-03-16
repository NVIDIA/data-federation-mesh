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

from nv_dfm_lib_common.schemas.xarray import (
    Attribute,
    Coordinate,
    DataVariable,
    XArraySchema,
)
from nv_dfm_lib_common.schemas.xarray.checks import (
    check_dims,
    check_dtype,
    check_range,
)


class ConvertToUint8InputSchema(XArraySchema):
    """
    Input validation schema for uint8 conversion operations.

    This schema defines the expected structure and data types for xarray datasets
    that will be converted to uint8 format. It validates that the input dataset
    has the proper coordinates (time, lat, lon or similar spatial dimensions) with
    correct data types before conversion.

    The schema ensures:
    - Time coordinate exists with datetime64 dtype
    - Spatial coordinates (e.g., lat/lon) exist with floating point dtypes
    - All required dimensions are present with proper structure

    This validation prevents errors during the uint8 conversion process and ensures
    consistent data format across the pipeline.
    """

    def __init__(self, time_dimension: str, xydims: list[str]):
        """
        Initialize the input validation schema for uint8 conversion.

        Creates a schema that validates the presence and types of required coordinates
        in the input dataset. The schema is configurable to support different coordinate
        naming conventions (e.g., "lat"/"lon" vs "latitude"/"longitude").

        Args:
            time_dimension: Name of the time dimension coordinate (e.g., "time")
            xydims: List of two spatial dimension names in order [x, y] or [lat, lon]
                   Example: ["lat", "lon"] or ["latitude", "longitude"]
        """
        super().__init__(frozen=False)
        self._time_dimension = time_dimension
        self._xydims = xydims

        # Add time coordinate validation
        # Requires datetime64 type for temporal data
        self.add_coord(
            self._time_dimension,
            Coordinate(
                dtype=np.datetime64,
                checks=[
                    check_dtype(np.datetime64),
                    check_dims([self._time_dimension]),
                ],
                required=True,
                name=self._time_dimension,
            ),
        )

        # Add first spatial coordinate validation (e.g., latitude)
        # Requires floating point type for geographic coordinates
        self.add_coord(
            self._xydims[0],
            Coordinate(
                dtype=np.floating,
                checks=[
                    check_dtype(np.floating),
                    check_dims([self._xydims[0]]),
                ],
                required=True,
                name=self._xydims[0],
            ),
        )

        # Add second spatial coordinate validation (e.g., longitude)
        # Requires floating point type for geographic coordinates
        self.add_coord(
            self._xydims[1],
            Coordinate(
                dtype=np.floating,
                checks=[
                    check_dtype(np.floating),
                    check_dims([self._xydims[1]]),
                ],
                required=True,
                name=self._xydims[1],
            ),
        )

        # Freeze the schema to prevent further modifications
        self.freeze()


class ConvertToUint8OutputSchema(XArraySchema):
    """
    Output validation schema for uint8 conversion operations.

    This schema defines the expected structure and data types for xarray datasets
    after conversion to uint8 format. It validates that the output dataset has:
    - Preserved coordinate structure (time and spatial dimensions)
    - Converted data variables with uint8 dtype and valid range [0, 255]
    - Required metadata attributes (data_min, data_max) for value reconstruction

    The schema ensures:
    - All data variables are properly converted to uint8 format
    - Value ranges are within valid uint8 bounds [0, 255]
    - Normalization metadata is preserved for future denormalization
    - Coordinate structure remains consistent with input

    This validation ensures the uint8 conversion maintains data integrity and
    includes necessary information for reversing the conversion when needed.
    """

    def __init__(self, time_dimension: str, xydims: list[str], data_vars):
        """
        Initialize the output validation schema for uint8 conversion.

        Creates a schema that validates the structure of uint8-converted datasets,
        ensuring all data variables are properly converted and metadata is preserved
        for future denormalization.

        Args:
            time_dimension: Name of the time dimension coordinate (e.g., "time")
            xydims: List of two spatial dimension names in order [x, y] or [lat, lon]
                   Example: ["lat", "lon"] or ["latitude", "longitude"]
            data_vars: List of data variable names that should be present in uint8 format
                      Each variable will be validated for uint8 dtype and range [0, 255]
        """
        super().__init__(frozen=False)

        # Add time coordinate validation (preserved from input)
        # Maintains datetime64 type for temporal consistency
        self.add_coord(
            time_dimension,
            Coordinate(
                dtype=np.datetime64,
                checks=[
                    check_dtype(np.datetime64),
                    check_dims([time_dimension]),
                ],
                required=True,
                name=time_dimension,
            ),
        )

        # Add first spatial coordinate validation (preserved from input)
        # Maintains floating point type for geographic coordinates
        self.add_coord(
            xydims[0],
            Coordinate(
                dtype=np.floating,
                checks=[
                    check_dtype(np.floating),
                    check_dims([xydims[0]]),
                ],
                required=True,
                name=xydims[0],
            ),
        )

        # Add second spatial coordinate validation (preserved from input)
        # Maintains floating point type for geographic coordinates
        self.add_coord(
            xydims[1],
            Coordinate(
                dtype=np.floating,
                checks=[
                    check_dtype(np.floating),
                    check_dims([xydims[1]]),
                ],
                required=True,
                name=xydims[1],
            ),
        )

        # Validate presence of data_min attribute
        # Stores minimum values per variable for denormalization
        self.add_attr(
            "data_min",
            Attribute(
                dtype=np.ndarray,
                checks=[],
                required=True,
                name="data_min",
            ),
        )

        # Validate presence of data_max attribute
        # Stores maximum values per variable for denormalization
        self.add_attr(
            "data_max",
            Attribute(
                dtype=np.ndarray,
                checks=[],
                required=True,
                name="data_max",
            ),
        )

        # Add validation for each data variable
        # Each variable should be converted to uint8 with values in [0, 255]
        for var in data_vars:
            self.add_data_var(
                var,
                DataVariable(
                    dtype=np.uint8,
                    checks=[
                        check_dtype(np.uint8),  # Ensure uint8 conversion
                        check_dims(
                            [time_dimension] + xydims
                        ),  # Verify dimensions preserved
                        check_range(0, 255),  # Validate value range
                    ],
                    required=True,
                    name=var,
                ),
            )
