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

"""
Smoke tests for the LoadEcmwfEra5Data class.

These tests verify the basic functionality of the ECMWF ERA5 data loader
without making actual network calls to external data sources.
"""

import pytest
from unittest.mock import Mock, patch
import xarray as xr
import numpy as np
import pandas as pd

from nv_dfm_lib_weather.load_data._load_ecmwf_era5_data import LoadEcmwfEra5Data


class TestLoadEcmwfEra5Data:
    """Test cases for the LoadEcmwfEra5Data class."""

    @pytest.fixture
    def mock_site(self):
        """Create a mock Site object for testing."""
        from upath import UPath
        import tempfile
        import shutil

        # Create a temporary directory for cache storage
        temp_dir = tempfile.mkdtemp()

        site = Mock()
        site.dfm_context.logger = Mock()
        site.cache_storage.return_value = UPath(temp_dir)

        # Clean up after test
        yield site

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_provider(self):
        """Create a mock Provider object for testing."""
        return Mock()

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample xarray Dataset for testing."""
        # Create a simple dataset with some ERA5-like variables
        coords = {
            "time": pd.date_range("2020-01-01", periods=10, freq="h"),
            "latitude": np.linspace(-90, 90, 73),
            "longitude": np.linspace(-180, 180, 144),
        }

        data_vars = {
            "2m_temperature": xr.DataArray(
                np.random.randn(10, 73, 144),
                dims=["time", "latitude", "longitude"],
                coords=coords,
            ),
            "10m_u_component_of_wind": xr.DataArray(
                np.random.randn(10, 73, 144),
                dims=["time", "latitude", "longitude"],
                coords=coords,
            ),
            "10m_v_component_of_wind": xr.DataArray(
                np.random.randn(10, 73, 144),
                dims=["time", "latitude", "longitude"],
                coords=coords,
            ),
            "mean_sea_level_pressure": xr.DataArray(
                np.random.randn(10, 73, 144),
                dims=["time", "latitude", "longitude"],
                coords=coords,
            ),
        }

        return xr.Dataset(data_vars, coords=coords)

    def test_init(self, mock_site, mock_provider):
        """Test that the LoadEcmwfEra5Data class initializes correctly."""
        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        assert loader._site == mock_site
        assert loader._provider == mock_provider
        assert loader._logger == mock_site.dfm_context.logger
        assert loader._cache is not None

    def test_init_with_none_provider(self, mock_site):
        """Test that the class initializes correctly with None provider."""
        loader = LoadEcmwfEra5Data(mock_site, None)

        assert loader._site == mock_site
        assert loader._provider is None
        assert loader._logger == mock_site.dfm_context.logger
        assert loader._cache is not None

    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.xarray.open_dataset")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.time.sleep")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.random.uniform")
    def test_body_impl_basic(
        self,
        mock_random,
        mock_sleep,
        mock_open_dataset,
        mock_site,
        mock_provider,
        sample_dataset,
    ):
        """Test the basic functionality of body_impl method."""
        # Setup mocks
        mock_random.return_value = 0.5
        mock_open_dataset.return_value = sample_dataset

        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test with basic parameters
        variables = ["2m_temperature", "10m_u_component_of_wind"]
        selection = {"time": "2020-01-01T12:00:00"}

        result = loader.body_impl(variables, selection)

        # Verify the result
        assert isinstance(result, xr.Dataset)
        assert "2m_temperature" in result.data_vars
        assert "10m_u_component_of_wind" in result.data_vars
        assert (
            "10m_v_component_of_wind" not in result.data_vars
        )  # Should be filtered out

        # Verify mocks were called correctly
        mock_random.assert_called_once_with(0.01, 0.5)
        mock_sleep.assert_called_once_with(0.5)
        mock_open_dataset.assert_called_once()

    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.xarray.open_dataset")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.time.sleep")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.random.uniform")
    def test_body_impl_with_selection(
        self,
        mock_random,
        mock_sleep,
        mock_open_dataset,
        mock_site,
        mock_provider,
        sample_dataset,
    ):
        """Test body_impl with selection parameters."""
        # Setup mocks
        mock_random.return_value = 0.5
        mock_open_dataset.return_value = sample_dataset

        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test with selection
        variables = ["2m_temperature"]
        selection = {
            "time": "2020-01-01T12:00:00",
            "latitude": 45.0,
            "longitude": -120.0,
        }

        result = loader.body_impl(variables, selection)

        # Verify the result
        assert isinstance(result, xr.Dataset)

        # Verify that sel was called on the dataset
        # Note: We can't easily verify the exact sel call due to the way xarray works,
        # but we can verify the method was called on the dataset
        assert result is not None

    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.xarray.open_dataset")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.time.sleep")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.random.uniform")
    def test_body_impl_with_empty_selection(
        self,
        mock_random,
        mock_sleep,
        mock_open_dataset,
        mock_site,
        mock_provider,
        sample_dataset,
    ):
        """Test body_impl with empty selection."""
        # Setup mocks
        mock_random.return_value = 0.5
        mock_open_dataset.return_value = sample_dataset

        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test with empty selection
        variables = ["2m_temperature"]
        selection = {}

        result = loader.body_impl(variables, selection)

        # Verify the result
        assert isinstance(result, xr.Dataset)
        assert "2m_temperature" in result.data_vars

    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.xarray.open_dataset")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.time.sleep")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.random.uniform")
    def test_body_impl_with_all_variables(
        self,
        mock_random,
        mock_sleep,
        mock_open_dataset,
        mock_site,
        mock_provider,
        sample_dataset,
    ):
        """Test body_impl with all variables (no filtering)."""
        # Setup mocks
        mock_random.return_value = 0.5
        mock_open_dataset.return_value = sample_dataset

        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test with all variables
        variables = ["*"]
        selection = {}

        result = loader.body_impl(variables, selection)

        # Verify the result - should contain all variables
        assert isinstance(result, xr.Dataset)
        assert "2m_temperature" in result.data_vars
        assert "10m_u_component_of_wind" in result.data_vars
        assert "10m_v_component_of_wind" in result.data_vars
        assert "mean_sea_level_pressure" in result.data_vars

    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.xarray.open_dataset")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.time.sleep")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.random.uniform")
    def test_body_impl_with_custom_url(
        self,
        mock_random,
        mock_sleep,
        mock_open_dataset,
        mock_site,
        mock_provider,
        sample_dataset,
    ):
        """Test body_impl with custom URL."""
        # Setup mocks
        mock_random.return_value = 0.5
        mock_open_dataset.return_value = sample_dataset

        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test with custom URL - URL is now hardcoded in __init__, so we test the default URL
        variables = ["2m_temperature"]
        selection = {}

        result = loader.body_impl(variables, selection)

        # Verify the result
        assert isinstance(result, xr.Dataset)

        # Verify that open_dataset was called with the default URL
        mock_open_dataset.assert_called_once()
        call_args = mock_open_dataset.call_args
        expected_url = "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
        assert call_args[0][0] == expected_url

    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.xarray.open_dataset")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.time.sleep")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.random.uniform")
    def test_body_impl_with_engine_kwargs(
        self,
        mock_random,
        mock_sleep,
        mock_open_dataset,
        mock_site,
        mock_provider,
        sample_dataset,
    ):
        """Test body_impl with custom engine kwargs (excluding decode_times to avoid conflict)."""
        # Setup mocks
        mock_random.return_value = 0.5
        mock_open_dataset.return_value = sample_dataset

        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test with default engine kwargs (empty dict)
        variables = ["2m_temperature"]
        selection = {}

        result = loader.body_impl(variables, selection)

        # Verify the result
        assert isinstance(result, xr.Dataset)

        # Verify that open_dataset was called with the expected parameters
        mock_open_dataset.assert_called_once()
        call_kwargs = mock_open_dataset.call_args[1]
        assert call_kwargs["cache"] is False
        assert call_kwargs["decode_times"] is True
        assert call_kwargs["engine"] == "zarr"
        assert call_kwargs == {
            "cache": False,
            "decode_times": True,
            "engine": "zarr",
        }

    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.xarray.open_dataset")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.time.sleep")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.random.uniform")
    def test_body_impl_error_handling(
        self, mock_random, mock_sleep, mock_open_dataset, mock_site, mock_provider
    ):
        """Test body_impl error handling."""
        # Setup mocks to raise an exception
        mock_random.return_value = 0.5
        mock_open_dataset.side_effect = Exception("Network error")

        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test that the exception is properly raised
        variables = ["2m_temperature"]
        selection = {}

        with pytest.raises(Exception, match="Network error"):
            loader.body_impl(variables, selection)

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.asyncio.to_thread")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.xarray.open_dataset")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.time.sleep")
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.random.uniform")
    async def test_body_async(
        self,
        mock_random,
        mock_sleep,
        mock_open_dataset,
        mock_to_thread,
        mock_site,
        mock_provider,
        sample_dataset,
    ):
        """Test the async body method."""
        # Setup mocks
        mock_random.return_value = 0.5
        mock_open_dataset.return_value = sample_dataset
        mock_to_thread.return_value = sample_dataset

        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test async method
        variables = ["2m_temperature"]
        selection = "{}"

        result = await loader.body(variables, selection)

        # Verify the result
        assert isinstance(result, xr.Dataset)

        # Verify that to_thread was called
        mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_ecmwf_era5_data.asyncio.to_thread")
    async def test_body_async_error_handling(
        self, mock_to_thread, mock_site, mock_provider
    ):
        """Test async body method error handling."""
        # Setup mock to raise an exception
        mock_to_thread.side_effect = Exception("Async error")

        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test that the exception is properly raised and logged
        variables = ["2m_temperature"]
        selection = "{}"

        with pytest.raises(Exception, match="Async error"):
            await loader.body(variables, selection)

        # Verify that the error was logged
        loader._logger.error.assert_called_once()

    def test_all_variables_constant(self):
        """Test that ALL_VARIABLES contains expected variables."""
        from nv_dfm_lib_weather.load_data._load_ecmwf_era5_data import (
            ALL_VARIABLES,
        )

        # Check that essential variables are present
        essential_vars = [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "latitude",
            "longitude",
            "time",
        ]

        for var in essential_vars:
            assert var in ALL_VARIABLES, f"Variable {var} not found in ALL_VARIABLES"

        # Check that the list is not empty
        assert len(ALL_VARIABLES) > 0

        # Check that all variables are strings
        for var in ALL_VARIABLES:
            assert isinstance(var, str), f"Variable {var} is not a string"

    @pytest.mark.asyncio
    async def test_available_variables_advisor(self, mock_site, mock_provider):
        """Test the available_variables field advisor."""
        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test the advisor method
        advice = await loader.available_variables(None, None)

        # Verify the advice structure
        assert hasattr(advice, "_values"), "Advice should have _values attribute"

        # Check that the advice contains expected values
        advice_values = advice._values
        assert len(advice_values) == 2, "Should have exactly 2 advice options"

        # Check for the wildcard option
        wildcard_found = False
        subset_found = False

        for value in advice_values:
            if hasattr(value, "_value") and value._value == "*":
                wildcard_found = True
            elif hasattr(value, "_values"):  # This should be the AdvisedSubsetOf
                subset_found = True
                # Check that coordinate variables are excluded
                excluded_vars = ["latitude", "longitude", "level", "time"]
                for excluded_var in excluded_vars:
                    assert excluded_var not in value._values, (
                        f"Coordinate variable {excluded_var} should not be in advice"
                    )

                # Check that some data variables are included
                data_vars = [
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "mean_sea_level_pressure",
                ]
                for data_var in data_vars:
                    assert data_var in value._values, (
                        f"Data variable {data_var} should be in advice"
                    )

        assert wildcard_found, "Wildcard option (*) should be in advice"
        assert subset_found, "Subset option should be in advice"

    @pytest.mark.asyncio
    async def test_valid_selections_advisor(self, mock_site, mock_provider):
        """Test the valid_selections field advisor."""
        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test the advisor method
        advice = await loader.valid_selections(None, None)

        # Verify the advice structure
        assert hasattr(advice, "_values"), "Advice should have _values attribute"

        # Check that the advice contains expected values
        advice_values = advice._values
        assert len(advice_values) == 2, "Should have exactly 2 advice options"

        # Check for the None option and dict option
        none_found = False
        dict_found = False

        for value in advice_values:
            if hasattr(value, "_value") and value._value is None:
                none_found = True
            elif hasattr(value, "_dictionary"):  # This should be the AdvisedDict
                dict_found = True
                # Check that the dict has the expected time field structure
                assert "time" in value._dictionary, (
                    "Time field should be in dict advice"
                )

                time_field = value._dictionary["time"]
                expected_time_keys = ["first_date", "last_date", "frequency"]
                for key in expected_time_keys:
                    assert key in time_field, f"Time field should have {key}"

                # Check the specific values
                assert time_field["first_date"] == "1959-01-01"
                assert time_field["last_date"] == "2021-12-31"
                assert time_field["frequency"] == 1

                # Check that allow_extras is True
                assert value._allow_extras is True, (
                    "Dict advice should allow extra fields"
                )

        assert none_found, "None option should be in advice"
        assert dict_found, "Dict option should be in advice"

    @pytest.mark.asyncio
    async def test_available_variables_advisor_excludes_coordinates(
        self, mock_site, mock_provider
    ):
        """Test that available_variables advisor excludes coordinate variables."""
        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        advice = await loader.available_variables(None, None)

        # Find the subset advice
        subset_advice = None
        for value in advice._values:
            if hasattr(value, "_values"):
                subset_advice = value
                break

        assert subset_advice is not None, "Should have subset advice"

        # Check that coordinate variables are excluded
        coordinate_vars = ["latitude", "longitude", "level", "time"]
        for coord_var in coordinate_vars:
            assert coord_var not in subset_advice._values, (
                f"Coordinate variable {coord_var} should be excluded"
            )

    @pytest.mark.asyncio
    async def test_available_variables_advisor_includes_data_variables(
        self, mock_site, mock_provider
    ):
        """Test that available_variables advisor includes data variables."""
        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        advice = await loader.available_variables(None, None)

        # Find the subset advice
        subset_advice = None
        for value in advice._values:
            if hasattr(value, "_values"):
                subset_advice = value
                break

        assert subset_advice is not None, "Should have subset advice"

        # Check that some key data variables are included
        data_vars = [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "total_precipitation",
            "sea_surface_temperature",
        ]

        for data_var in data_vars:
            assert data_var in subset_advice._values, (
                f"Data variable {data_var} should be included"
            )

    @pytest.mark.asyncio
    async def test_valid_selections_advisor_time_constraints(
        self, mock_site, mock_provider
    ):
        """Test that valid_selections advisor provides correct time constraints."""
        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        advice = await loader.valid_selections(None, None)

        # Find the dict advice
        dict_advice = None
        for value in advice._values:
            if hasattr(value, "_dictionary"):
                dict_advice = value
                break

        assert dict_advice is not None, "Should have dict advice"
        assert "time" in dict_advice._dictionary, (
            "Should have time field in dict advice"
        )

        time_field = dict_advice._dictionary["time"]

        # Test the time constraints
        assert time_field["first_date"] == "1959-01-01", (
            "First date should be 1959-01-01"
        )
        assert time_field["last_date"] == "2021-12-31", "Last date should be 2021-12-31"
        assert time_field["frequency"] == 1, "Frequency should be 1"

    @pytest.mark.asyncio
    async def test_field_advisors_are_async(self, mock_site, mock_provider):
        """Test that field advisor methods are properly async."""
        loader = LoadEcmwfEra5Data(mock_site, mock_provider)

        # Test that both methods are async and return advice
        variables_advice = await loader.available_variables(None, None)
        selections_advice = await loader.valid_selections(None, None)

        # Both should return advice objects
        assert variables_advice is not None, "Variables advice should not be None"
        assert selections_advice is not None, "Selections advice should not be None"

        # Both should have the expected structure
        assert hasattr(variables_advice, "_values"), (
            "Variables advice should have _values"
        )
        assert hasattr(selections_advice, "_values"), (
            "Selections advice should have _values"
        )

    def test_field_advisors_have_decorators(self):
        """Test that field advisor methods are properly decorated."""
        from nv_dfm_lib_weather.load_data._load_ecmwf_era5_data import (
            LoadEcmwfEra5Data,
        )

        # Check that the methods exist and are decorated
        assert hasattr(LoadEcmwfEra5Data, "available_variables"), (
            "available_variables method should exist"
        )
        assert hasattr(LoadEcmwfEra5Data, "valid_selections"), (
            "valid_selections method should exist"
        )

        # Check that they are async methods
        import inspect

        available_variables_method = getattr(LoadEcmwfEra5Data, "available_variables")
        valid_selections_method = getattr(LoadEcmwfEra5Data, "valid_selections")

        assert inspect.iscoroutinefunction(available_variables_method), (
            "available_variables should be async"
        )
        assert inspect.iscoroutinefunction(valid_selections_method), (
            "valid_selections should be async"
        )

        # Check that they have the expected signatures (decorated with *args, **kwargs)
        available_vars_sig = inspect.signature(available_variables_method)
        valid_selections_sig = inspect.signature(valid_selections_method)

        # Both should have *args, **kwargs signature due to decorator
        assert str(available_vars_sig) == "(*args, **kwargs)", (
            "available_variables should have *args, **kwargs signature"
        )
        assert str(valid_selections_sig) == "(*args, **kwargs)", (
            "valid_selections should have *args, **kwargs signature"
        )

        # Check that the methods are callable
        assert callable(available_variables_method), (
            "available_variables should be callable"
        )
        assert callable(valid_selections_method), "valid_selections should be callable"


# Import pandas for the sample dataset creation
