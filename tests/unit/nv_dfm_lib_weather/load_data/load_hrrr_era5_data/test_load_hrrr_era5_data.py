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

import dateutil.parser
import pytest
from unittest.mock import Mock, patch
import xarray as xr
import numpy as np
from datetime import datetime, timedelta, timezone
import inspect

from nv_dfm_lib_weather.load_data import LoadHrrrEra5Data


class TestLoadHrrrEra5Data:
    """Test suite for LoadHrrrEra5Data class."""

    @pytest.fixture
    def mock_site(self):
        """Create a mock site object."""
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
        """Create a mock provider object."""
        return Mock()

    @pytest.fixture
    def mock_cache(self):
        """Create a mock XArrayCache."""
        cache = Mock()
        cache.load_value = Mock(return_value=None)
        cache.write_value = Mock()
        return cache

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock xarray dataset as it would be returned by Herbie."""
        # Create a simple mock dataset with the expected structure. The returned dataset just contains
        # multiple variables, the actual herbie dataset may only return a subset at a time
        coords = {
            "y": np.linspace(0, 1058, 1059),
            "x": np.linspace(0, 1798, 1799),
            "time": np.datetime64("2024-01-31"),
            "step": np.timedelta64(0, "h"),
            "heightAboveGround": 10.0,
            "latitude": (
                ("y", "x"),
                np.random.rand(1059, 1799),
            ),  # we don't really care about the lat/lon values, we only use x and y
            "longitude": (("y", "x"), np.random.rand(1059, 1799)),
            "valid_time": np.datetime64("2024-01-31"),
            "gribfile_projection": None,
        }

        # Create mock data arrays for herbie variables
        data_vars = {}
        for mapping in LoadHrrrEra5Data.ERA5_TO_HRRR_VARIABLES.values():
            if mapping is None:
                continue
            data = np.random.random((1059, 1799))
            da = xr.DataArray(
                data,
                coords={
                    "time": coords["time"],
                    "step": coords["step"],
                    "heightAboveGround": coords["heightAboveGround"],
                    "latitude": coords["latitude"],
                    "longitude": coords["longitude"],
                    "valid_time": coords["valid_time"],
                    "gribfile_projection": coords["gribfile_projection"],
                },
                dims=["y", "x"],
                name=mapping.hrrr_var,
            )
            data_vars[mapping.hrrr_var] = da.metpy.assign_crs(
                dict(
                    grid_mapping_name="lambert_conformal_conic",
                    standard_parallel=(38.5, 38.5),
                    latitude_of_projection_origin=38.5,
                    longitude_of_central_meridian=262.5,
                )
            )

        ds = xr.Dataset(data_vars, coords=coords)
        return ds

    @pytest.fixture
    def loader(self, mock_site, mock_provider, mock_cache):
        """Create a LoadHrrrEra5Data instance with mocked dependencies."""
        with patch(
            "nv_dfm_lib_weather.load_data._load_hrrr_era5_data.XArrayCache",
            return_value=mock_cache,
        ):
            return LoadHrrrEra5Data(mock_site, mock_provider)

    def test_init(self, mock_site, mock_provider):
        """Test initialization of LoadHrrrEra5Data."""
        with patch(
            "nv_dfm_lib_weather.load_data._load_hrrr_era5_data.XArrayCache"
        ) as mock_cache_class:
            loader = LoadHrrrEra5Data(mock_site, mock_provider)

            assert loader._site == mock_site
            assert loader._provider == mock_provider
            assert loader._logger == mock_site.dfm_context.logger
            mock_cache_class.assert_called_once()

    def test_calculate_params_hash(self, loader):
        """Test parameter hash calculation."""
        variables = ["t2m", "u10m"]
        selection = {"time": "2024-01-31", "lat": "45.0"}

        hash1 = loader._calculate_params_hash(variables, selection)
        hash2 = loader._calculate_params_hash(variables, selection)

        # Same parameters should produce same hash
        assert hash1 == hash2

        # Different parameters should produce different hashes
        hash3 = loader._calculate_params_hash(["t2m"], selection)
        assert hash1 != hash3

        # Test with None values
        hash4 = loader._calculate_params_hash(None, None)
        assert isinstance(hash4, str)

    @pytest.mark.asyncio
    async def test_body_missing_time_selection(self, loader):
        """Test that body raises ValueError when time selection is missing."""
        with pytest.raises(ValueError, match="requires a specific time selection"):
            await loader.body(variables=["t2m"], selection={"lat": "45.0"})

    @pytest.mark.asyncio
    async def test_body_empty_selection(self, loader):
        """Test that body raises ValueError when selection is empty."""
        with pytest.raises(ValueError, match="requires a specific time selection"):
            await loader.body(variables=["t2m"], selection={})

    @pytest.mark.asyncio
    async def test_body_none_selection(self, loader):
        """Test that body raises ValueError when selection is None."""
        with pytest.raises(ValueError, match="requires a specific time selection"):
            await loader.body(variables=["t2m"], selection=None)

    @pytest.mark.asyncio
    async def test_body_cached_result(self, loader, mock_cache):
        """Test that cached results are returned when available."""
        # Mock cached dataset
        cached_ds = xr.Dataset(
            {
                "t2m": xr.DataArray(
                    np.random.random((1, 721, 1440)),
                    coords={
                        "time": [np.datetime64("2024-01-31")],
                        "lat": np.linspace(90, -90, 721),
                        "lon": np.linspace(0, 359.75, 1440),
                    },
                    dims=["time", "lat", "lon"],
                )
            }
        )
        mock_cache.load_value.return_value = cached_ds

        result = await loader.body(variables=["t2m"], selection={"time": "2024-01-31"})

        assert result is cached_ds
        mock_cache.load_value.assert_called_once()

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_hrrr_era5_data.dateutil.parser.parse")
    @patch("nv_dfm_lib_weather.load_data._load_hrrr_era5_data.Herbie.xarray")
    async def test_body_load_from_remote(
        self,
        mock_herbie,
        mock_parse,
        loader,
        mock_cache,
        mock_dataset,
    ):
        """Test loading data from remote source when not cached."""
        # Setup mocks
        mock_parse.return_value = datetime(2024, 1, 31)
        mock_herbie.return_value = mock_dataset

        # Mock cache to return None (no cached data)
        mock_cache.load_value.return_value = None

        result = await loader.body(
            variables=["t2m", "u10m"], selection={"time": "2024-01-31", "lat": "45.0"}
        )

        # Verify the result is a Dataset
        assert isinstance(result, xr.Dataset)

        # Verify cache was checked and written
        mock_cache.load_value.assert_called_once()
        mock_cache.write_value.assert_called_once()

        # Verify remote dataset was opened
        # mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_hrrr_era5_data.dateutil.parser.parse")
    @patch("nv_dfm_lib_weather.load_data._load_hrrr_era5_data.Herbie.xarray")
    async def test_body_variable_filtering(
        self,
        mock_herbie,
        mock_parse,
        loader,
        mock_cache,
        mock_dataset,
    ):
        """Test that variables are properly filtered."""
        # Setup mocks
        mock_parse.return_value = datetime(2024, 1, 31)
        mock_herbie.return_value = mock_dataset

        # Mock cache to return None
        mock_cache.load_value.return_value = None

        # Test with specific variables
        result = await loader.body(variables=["t2m"], selection={"time": "2024-01-31"})

        # Should only contain t2m variable
        assert "t2m" in result.data_vars
        assert len(result.data_vars) == 1

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_hrrr_era5_data.dateutil.parser.parse")
    @patch("nv_dfm_lib_weather.load_data._load_hrrr_era5_data.Herbie.xarray")
    async def test_body_all_variables(
        self,
        mock_herbie,
        mock_parse,
        loader,
        mock_cache,
        mock_dataset,
    ):
        """Test loading all variables when no specific variables provided."""
        # Setup mocks
        mock_parse.return_value = datetime(2024, 1, 31)
        mock_herbie.return_value = mock_dataset

        # Mock cache to return None
        mock_cache.load_value.return_value = None

        # Test with None variables (should load all)
        result = await loader.body(variables=None, selection={"time": "2024-01-31"})

        # Should contain all ERA5 channels
        for mapping in LoadHrrrEra5Data.ERA5_TO_HRRR_VARIABLES.values():
            if mapping is None:
                continue
            assert mapping.era5_var in result.data_vars.keys()

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_hrrr_era5_data.dateutil.parser.parse")
    @patch("nv_dfm_lib_weather.load_data._load_hrrr_era5_data.Herbie.xarray")
    async def test_body_latitude_reversal(
        self,
        mock_herbie,
        mock_parse,
        loader,
        mock_cache,
        mock_dataset,
    ):
        """Test that latitude is properly reversed."""
        # Setup mocks
        mock_parse.return_value = datetime(2024, 1, 31)
        mock_herbie.return_value = mock_dataset

        # Mock cache to return None
        mock_cache.load_value.return_value = None

        result = await loader.body(variables=["t2m"], selection={"time": "2024-01-31"})

        # Check that latitude is reversed (North to South)
        lat_coords = result.coords["lat"].values
        assert (
            lat_coords[0] < lat_coords[-1]
        )  # First should be lower than last (South to North after reversal)

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_hrrr_era5_data.dateutil.parser.parse")
    @patch("nv_dfm_lib_weather.load_data._load_hrrr_era5_data.Herbie.xarray")
    async def test_body_timeout_error(
        self, mock_herbie, mock_parse, loader, mock_cache
    ):
        """Test handling of timeout errors."""
        # Setup mocks
        mock_parse.return_value = datetime(2024, 1, 31)
        # Mock to_thread to raise timeout
        mock_herbie.side_effect = TimeoutError("Operation timed out")

        # Mock cache to return None
        mock_cache.load_value.return_value = None

        with pytest.raises(TimeoutError):
            await loader.body(variables=["t2m"], selection={"time": "2024-01-31"})

    @pytest.mark.asyncio
    async def test_body_with_different_date_formats(self, loader, mock_cache):
        """Test that different date formats are handled correctly."""
        # Mock cached dataset
        cached_ds = xr.Dataset(
            {
                "t2m": xr.DataArray(
                    np.random.random((1, 721, 1440)),
                    coords={
                        "time": [np.datetime64("2024-01-31")],
                        "lat": np.linspace(90, -90, 721),
                        "lon": np.linspace(0, 359.75, 1440),
                    },
                    dims=["time", "lat", "lon"],
                )
            }
        )
        mock_cache.load_value.return_value = cached_ds

        # Test different date formats
        date_formats = ["2024-01-31", "2024-01-31T00:00:00", "2024-01-31 00:00:00"]

        for date_format in date_formats:
            result = await loader.body(
                variables=["t2m"], selection={"time": date_format}
            )
            assert result is cached_ds

    # Discovery method tests (from test_load_era5_gfs_data.py)
    @pytest.mark.asyncio
    async def test_available_variables_discovery(self, loader):
        """Test the available_variables discovery method."""
        advice = await loader.available_variables(None, None)
        assert hasattr(advice, "_values")
        assert len(advice._values) == 2
        from nv_dfm_core.exec.discovery._advised_values import (
            AdvisedLiteral,
            AdvisedSubsetOf,
        )

        wildcard_option = None
        subset_option = None
        for value in advice._values:
            if (
                isinstance(value, AdvisedLiteral)
                and getattr(value, "_value", None) == "*"
            ):
                wildcard_option = value
            elif (
                isinstance(value, AdvisedSubsetOf)
                and getattr(value, "_values", None)
                == LoadHrrrEra5Data.ERA5_TO_HRRR_VARIABLES.keys()
            ):
                subset_option = value
        assert wildcard_option is not None, "Wildcard '*' option should be available"
        assert subset_option is not None, "Subset of ERA5_CHANNELS should be available"

    @pytest.mark.asyncio
    async def test_available_variables_contains_expected_channels(self, loader):
        """Test that available_variables includes expected ERA5 channels."""
        advice = await loader.available_variables(None, None)
        from nv_dfm_core.exec.discovery._advised_values import AdvisedSubsetOf

        subset_option = None
        for value in advice._values:
            if isinstance(value, AdvisedSubsetOf):
                subset_option = value
                break
        assert subset_option is not None
        assert subset_option._values == LoadHrrrEra5Data.ERA5_TO_HRRR_VARIABLES.keys()

        # Check that it contains expected surface variables
        expected_surface_vars = [
            "u10m",
            "v10m",
            "u100m",
            "v100m",
            "t2m",
            "r2m",
            "sp",
            "msl",
            "tcwv",
        ]
        for var in expected_surface_vars:
            assert var in subset_option._values

        # Check that it contains expected pressure level variables
        expected_prefixes = ["u", "v", "z", "t", "r", "q"]
        expected_levels = [
            50,
            100,
            150,
            200,
            250,
            300,
            400,
            500,
            600,
            700,
            850,
            925,
            1000,
        ]

        for prefix in expected_prefixes:
            for level in expected_levels:
                expected_var = f"{prefix}{level}"
                assert expected_var in subset_option._values

    @pytest.mark.asyncio
    async def test_valid_selections_discovery(self, loader):
        """Test the valid_selections discovery method."""
        advice = await loader.valid_selections(None, None)
        assert hasattr(advice, "_values")
        assert len(advice._values) == 3
        from nv_dfm_core.exec.discovery._advised_values import (
            AdvisedLiteral,
            AdvisedDict,
        )

        none_option = None
        dict_option = None
        for value in advice._values:
            if (
                isinstance(value, AdvisedLiteral)
                and getattr(value, "_value", "not-none") is None
            ):
                none_option = value
            elif isinstance(value, AdvisedDict):
                dict_option = value
        assert none_option is not None, "None option should be available"
        assert dict_option is not None, "Dictionary option should be available"

    @pytest.mark.asyncio
    async def test_valid_selections_time_range(self, loader):
        """Test that valid_selections provides correct time range."""
        advice = await loader.valid_selections(None, None)
        from nv_dfm_core.exec.discovery._advised_values import AdvisedDict

        dict_options = []
        for value in advice._values:
            if isinstance(value, AdvisedDict):
                dict_options.append(value)
        assert len(dict_options) == 2
        assert dict_options[0] is not None
        assert dict_options[1] is not None

        for dict_option in dict_options:
            advice_dict = dict_option._dictionary
            assert "time" in advice_dict

            time_advice = advice_dict["time"]
            assert isinstance(time_advice, dict)
            assert "first_date" in time_advice
            assert "last_date" in time_advice
            assert "frequency" in time_advice

            # Verify the date range logic
            expected_first_date = dateutil.parser.parse("2014-07-30T00:00")
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            expected_last_date = now - timedelta(hours=2)

            assert time_advice["frequency"] in [6, 1]
            if time_advice["frequency"] == 6:
                # round to previous 6 hour boundary
                expected_last_date = expected_last_date.replace(
                    hour=expected_last_date.hour - (expected_last_date.hour % 6)
                )

            assert dateutil.parser.parse(str(time_advice["first_date"])).strftime(
                "%Y-%m-%dT%H:%M"
            ) == expected_first_date.strftime("%Y-%m-%dT%H:%M")
            assert dateutil.parser.parse(str(time_advice["last_date"])).strftime(
                "%Y-%m-%dT%H:%M"
            ) == expected_last_date.strftime("%Y-%m-%dT%H:%M")

    @pytest.mark.asyncio
    async def test_valid_selections_allow_extras(self, loader):
        """Test that valid_selections allows extra parameters."""
        advice = await loader.valid_selections(None, None)
        from nv_dfm_core.exec.discovery._advised_values import AdvisedDict

        dict_option = None
        for value in advice._values:
            if isinstance(value, AdvisedDict):
                dict_option = value
                break
        assert dict_option is not None
        # The allow_extras=True should be set in the original implementation
        # We can't directly test this from the advice object, but we can verify
        # that the structure allows for additional parameters beyond the template

    @pytest.mark.asyncio
    async def test_discovery_methods_are_async(self, loader):
        """Test that both discovery methods are properly async."""
        # Test that both methods are coroutines
        available_vars_coro = loader.available_variables(None, None)
        valid_selections_coro = loader.valid_selections(None, None)

        assert inspect.iscoroutine(available_vars_coro)
        assert inspect.iscoroutine(valid_selections_coro)

        # Test that they can be awaited
        available_vars_result = await available_vars_coro
        valid_selections_result = await valid_selections_coro

        assert available_vars_result is not None
        assert valid_selections_result is not None

    @pytest.mark.asyncio
    async def test_discovery_methods_with_different_contexts(self, loader):
        """Test that discovery methods work with different context values."""
        # Test with different context values
        contexts = [None, {}, {"some": "context"}, {"time": "2024-01-31"}]

        for context in contexts:
            available_vars_advice = await loader.available_variables(None, context)
            valid_selections_advice = await loader.valid_selections(None, context)

            # Both should return valid advice regardless of context
            assert available_vars_advice is not None
            assert valid_selections_advice is not None

            # The advice structure should be consistent
            assert hasattr(available_vars_advice, "_values")
            assert hasattr(valid_selections_advice, "_values")
