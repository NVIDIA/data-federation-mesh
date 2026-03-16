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
import xarray as xr
import numpy as np
from unittest.mock import Mock

from nv_dfm_lib_weather.xarray import VariableNorm


class TestVariableNorm:
    @pytest.fixture
    def mock_site(self):
        site = Mock()
        site.dfm_context.logger = Mock()
        site.cache_storage.return_value = Mock()
        return site

    @pytest.fixture
    def mock_provider(self):
        return Mock()

    @pytest.fixture
    def variable_norm(self, mock_site, mock_provider):
        return VariableNorm(mock_site, mock_provider)

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample xarray dataset for testing."""
        data = np.random.rand(10, 5, 3)
        coords = {
            "time": np.arange(10),
            "lat": np.linspace(0, 1, 5),
            "lon": np.linspace(0, 1, 3),
        }

        ds = xr.Dataset(
            {
                "temperature": (["time", "lat", "lon"], data),
                "humidity": (["time", "lat", "lon"], data * 0.5),
                "pressure": (["time", "lat", "lon"], data * 2.0),
            },
            coords=coords,
        )

        return ds

    def test_init(self, variable_norm):
        """Test that VariableNorm initializes correctly."""
        assert variable_norm is not None
        assert hasattr(variable_norm, "_site")
        assert hasattr(variable_norm, "_provider")
        assert hasattr(variable_norm, "_logger")
        assert hasattr(variable_norm, "_cache")

    @pytest.mark.asyncio
    async def test_body_impl_basic_norm(self, variable_norm, sample_dataset):
        """Test basic norm computation."""
        variables = ["temperature", "humidity"]
        result = variable_norm.body_impl(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)
        assert "norm" in result.data_vars
        assert result["norm"].shape == sample_dataset["temperature"].shape
        assert result["norm"].attrs["p_value"] == 2.0
        assert result["norm"].attrs["variables_used"] == variables

    @pytest.mark.asyncio
    async def test_body_impl_custom_p_value(self, variable_norm, sample_dataset):
        """Test norm computation with custom p value."""
        variables = ["temperature", "humidity"]
        p_value = 1.0
        result = variable_norm.body_impl(sample_dataset, variables, p=p_value)

        assert result["norm"].attrs["p_value"] == p_value

    @pytest.mark.asyncio
    async def test_body_impl_custom_output_name(self, variable_norm, sample_dataset):
        """Test norm computation with custom output name."""
        variables = ["temperature", "humidity"]
        output_name = "custom_norm"
        result = variable_norm.body_impl(
            sample_dataset, variables, output_name=output_name
        )

        assert output_name in result.data_vars
        assert "norm" not in result.data_vars

    @pytest.mark.asyncio
    async def test_body_impl_missing_variables(self, variable_norm, sample_dataset):
        """Test that missing variables raise an error."""
        variables = ["temperature", "nonexistent_var"]

        with pytest.raises(ValueError, match="Variables .* not found in dataset"):
            variable_norm.body_impl(sample_dataset, variables)

    @pytest.mark.asyncio
    async def test_body_async_wrapper(self, variable_norm, sample_dataset):
        """Test the async body wrapper."""
        variables = ["temperature", "humidity"]
        result = await variable_norm.body(
            sample_dataset, variables, p=2.0, output_name="norm"
        )

        assert isinstance(result, xr.Dataset)
        assert "norm" in result.data_vars

    @pytest.mark.asyncio
    async def test_field_advisors(self, variable_norm):
        """Test that field advisors are properly defined."""
        # Test variables advisor
        variables_advice = await variable_norm.available_variables(None, None)
        assert variables_advice is not None

        # Test output_name advisor
        output_name_advice = await variable_norm.valid_output_names(None, None)
        assert output_name_advice is not None

        # Test p advisor
        p_advice = await variable_norm.valid_p_values(None, None)
        assert p_advice is not None

    def test_norm_calculation_accuracy(self, variable_norm):
        """Test that the norm calculation is mathematically correct."""
        # Create a simple dataset with known values
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        ds = xr.Dataset({"var1": (["x", "y"], data), "var2": (["x", "y"], data * 2)})

        variables = ["var1", "var2"]
        result = variable_norm.body_impl(ds, variables, p=2.0)

        # For p=2, the L2 norm should be sqrt(var1^2 + var2^2)
        expected = np.sqrt(data**2 + (data * 2) ** 2)
        np.testing.assert_array_almost_equal(result["norm"].values, expected)

    def test_single_variable_norm(self, variable_norm):
        """Test norm computation with a single variable."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        ds = xr.Dataset({"var1": (["x", "y"], data)})

        variables = ["var1"]
        result = variable_norm.body_impl(ds, variables, p=2.0)

        # For a single variable, the norm should equal the absolute value
        expected = np.abs(data)
        np.testing.assert_array_almost_equal(result["norm"].values, expected)
