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

"""Test configuration for xarray-schema."""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def climate_dataset():
    """Create a sample climate dataset."""
    return xr.Dataset(
        data_vars={
            "temperature": xr.DataArray(
                np.random.uniform(200, 350, (10, 5, 5)),
                dims=["time", "latitude", "longitude"],
                attrs={"units": "K"},
            ),
            "precipitation": xr.DataArray(
                np.random.uniform(0, 1000, (10, 5, 5)),
                dims=["time", "latitude", "longitude"],
                attrs={"units": "mm/day"},
            ),
        },
        coords={
            "time": xr.DataArray(
                np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
                dims=["time"],
            ),
            "latitude": xr.DataArray(
                np.linspace(-90, 90, 5),
                dims=["latitude"],
            ),
            "longitude": xr.DataArray(
                np.linspace(-180, 180, 5),
                dims=["longitude"],
            ),
        },
        attrs={
            "title": "Climate Dataset",
            "institution": "Example University",
        },
    )


@pytest.fixture
def climate_schema():
    """Create a sample climate schema."""
    from nv_dfm_lib_common.schemas.xarray import (
        Attribute,
        Coordinate,
        DataVariable,
        XArraySchema,
        check_attrs,
        check_dims,
        check_dtype,
        check_max_missing,
        check_range,
    )

    return XArraySchema(
        data_vars={
            "temperature": DataVariable(
                dtype=np.dtype(np.float64),
                checks=[
                    check_dims(["time", "latitude", "longitude"]),
                    check_dtype(np.dtype(np.float64)),
                    check_max_missing(0.1),
                    check_attrs({}),
                    check_range(200, 350),
                ],
                nullable=False,
                required=True,
                name="temperature",
                title="Temperature",
                description="Surface air temperature",
                metadata={"standard_name": "air_temperature"},
                required_attrs=["units"],
            ),
            "precipitation": DataVariable(
                dtype=np.dtype(np.float64),
                checks=[
                    check_dims(["time", "latitude", "longitude"]),
                    check_dtype(np.dtype(np.float64)),
                    check_max_missing(0.1),
                    check_attrs({}),
                    check_range(0, 1000),
                ],
                nullable=False,
                required=True,
                name="precipitation",
                title="Precipitation",
                description="Daily precipitation",
                metadata={"standard_name": "precipitation_amount"},
                required_attrs=["units"],
            ),
        },
        coords={
            "time": Coordinate(
                dtype=np.dtype(np.datetime64),
                checks=[
                    check_dims(["time"]),
                    check_dtype(np.dtype(np.datetime64)),
                ],
                nullable=False,
                required=True,
                name="time",
                title="Time",
                description="Time coordinate",
                metadata={"standard_name": "time"},
            ),
            "latitude": Coordinate(
                dtype=np.dtype(np.float64),
                checks=[
                    check_dims(["latitude"]),
                    check_dtype(np.dtype(np.float64)),
                    check_range(-90, 90),
                ],
                nullable=False,
                required=True,
                name="latitude",
                title="Latitude",
                description="Latitude coordinate",
                metadata={"standard_name": "latitude"},
            ),
            "longitude": Coordinate(
                dtype=np.dtype(np.float64),
                checks=[
                    check_dims(["longitude"]),
                    check_dtype(np.dtype(np.float64)),
                    check_range(-180, 180),
                ],
                nullable=False,
                required=True,
                name="longitude",
                title="Longitude",
                description="Longitude coordinate",
                metadata={"standard_name": "longitude"},
            ),
        },
        attrs={
            "title": Attribute(
                dtype=str,
                checks=[],
                nullable=False,
                required=True,
                name="title",
                title="Title",
                description="Dataset title",
                metadata={},
            ),
            "institution": Attribute(
                dtype=str,
                checks=[],
                nullable=False,
                required=True,
                name="institution",
                title="Institution",
                description="Institution name",
                metadata={},
            ),
        },
        name="climate_dataset",
        title="Climate Dataset",
        description="A climate dataset with temperature and precipitation",
        metadata={},
    )
