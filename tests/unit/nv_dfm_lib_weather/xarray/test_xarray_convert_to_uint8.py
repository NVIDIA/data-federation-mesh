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

import pytest
import xarray
import numpy as np
from unittest.mock import Mock

from nv_dfm_lib_weather.xarray import ConvertToUint8


@pytest.mark.asyncio
async def test_convert_to_uint8_adapter():
    """Test the ConvertToUint8 adapter with proper mocking."""
    # Create proper mocks for dependencies
    mock_site = Mock()
    mock_site.cache_storage.return_value = Mock()
    mock_provider = Mock()

    # Create test data
    data = xarray.Dataset(
        data_vars={
            "t2m": xarray.DataArray(
                np.random.rand(10, 10, 10),
                dims=["time", "lat", "lon"],
            ),
        },
        coords={
            "time": xarray.DataArray(
                np.arange("2023-01-01", "2023-01-11", dtype="datetime64[D]"),
                dims=["time"],
            ),
            "lat": xarray.DataArray(
                np.random.rand(10),
                dims=["lat"],
            ),
            "lon": xarray.DataArray(
                np.random.rand(10),
                dims=["lon"],
            ),
        },
    )

    # Create adapter with proper mocks
    adapter = ConvertToUint8(site=mock_site, provider=mock_provider)

    # Test the conversion
    output = await adapter.body(
        data=data,
        time_dimension="time",
        xydims=["lat", "lon"],
        min_value=None,
        max_value=None,
    )

    # Verify the output
    assert output is not None
    assert isinstance(output, xarray.Dataset)
    assert "t2m" in output.data_vars

    # Verify the data type conversion to uint8
    assert output["t2m"].dtype == np.uint8

    # Verify the data range is 0-255
    assert output["t2m"].min() >= 0
    assert output["t2m"].max() <= 255

    # Verify the attributes were set
    assert "data_min" in output.attrs
    assert "data_max" in output.attrs
