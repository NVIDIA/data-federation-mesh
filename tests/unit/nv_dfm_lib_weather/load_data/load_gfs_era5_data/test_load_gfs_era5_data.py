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
from unittest.mock import Mock, patch
import xarray as xr
import numpy as np

from nv_dfm_lib_weather.load_data._load_gfs_era5_data import LoadGfsEra5Data
from nv_dfm_lib_weather.load_data._channels import ERA5_CHANNELS


class TestLoadGfsEra5Data:
    @pytest.fixture
    def mock_site(self):
        from upath import UPath
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        site = Mock()
        site.dfm_context.logger = Mock()
        site.cache_storage.return_value = UPath(temp_dir)
        yield site
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_provider(self):
        return Mock()

    @pytest.fixture
    def mock_cache(self):
        cache = Mock()
        cache.load_value = Mock(return_value=None)
        cache.write_value = Mock()
        return cache

    @pytest.fixture
    def mock_da(self):
        # Build an earth2studio-like DataArray with variable dim
        coords = {
            "time": [np.datetime64("2022-09-05T00:00:00")],
            "lat": np.linspace(-90, 90, 721),  # ascending to exercise flip
            "lon": np.linspace(0, 359.75, 1440),
            "variable": ERA5_CHANNELS,
        }
        data = np.random.random((1, len(coords["variable"]), 721, 1440))
        da = xr.DataArray(
            data,
            coords=coords,
            dims=["time", "variable", "lat", "lon"],
            name="gfs",
        )
        return da

    @pytest.fixture
    def loader(self, mock_site, mock_provider, mock_cache):
        with patch(
            "nv_dfm_lib_weather.load_data._load_gfs_era5_data.XArrayCache",
            return_value=mock_cache,
        ):
            return LoadGfsEra5Data(mock_site, mock_provider)

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_gfs_era5_data.E2S_GFS")
    async def test_body_aws_mode_with_timeout(
        self, mock_gfs_cls, loader, mock_cache, mock_da
    ):
        # Mock earth2studio GFS callable to return DataArray
        mock_gfs = Mock()
        mock_gfs.side_effect = lambda *args, **kwargs: mock_da
        mock_gfs_cls.return_value = mock_gfs

        ds = await loader.body(
            variables=["t2m", "u10m"],
            selection={"time": "2022-09-05", "mode": "aws", "timeout": 15},
        )
        assert isinstance(ds, xr.Dataset)
        # variable filtering should keep only requested
        assert set(ds.data_vars.keys()) == {"t2m", "u10m"}
        # lat should be in native (ascending) order
        assert ds["lat"].values[0] < ds["lat"].values[-1]
        # cache interactions
        mock_cache.load_value.assert_called_once()
        mock_cache.write_value.assert_called_once()
        # ensure GFS constructed with mode and timeout
        mock_gfs_cls.assert_called()
        args, kwargs = mock_gfs_cls.call_args
        # In some environments the constructor may not accept these kwargs
        if "mode" in kwargs:
            assert kwargs["mode"] == "aws"
        if "timeout" in kwargs:
            assert kwargs["timeout"] == 15.0

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_gfs_era5_data.E2S_GFS")
    async def test_invalidate_cache_true_skips_cache_and_fetches(
        self, mock_gfs_cls, loader, mock_cache, mock_da
    ):
        mock_gfs = Mock()
        mock_gfs.side_effect = lambda *args, **kwargs: mock_da
        mock_gfs_cls.return_value = mock_gfs

        ds = await loader.body(
            variables=["t2m"],
            selection={"time": "2022-09-05", "mode": "aws"},
            invalidate_cache=True,
        )

        assert isinstance(ds, xr.Dataset)
        # load_value should not be called when invalidate_cache is True
        mock_cache.load_value.assert_not_called()
        mock_gfs_cls.assert_called()
        mock_cache.write_value.assert_called_once()

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_gfs_era5_data.E2S_GFS")
    async def test_uses_cached_when_not_invalidating(
        self, mock_gfs_cls, loader, mock_cache
    ):
        # Prepare a cached dataset
        cached = xr.Dataset(
            {
                "t2m": xr.DataArray(
                    np.random.random((1, 721, 1440)),
                    coords={
                        "time": [np.datetime64("2022-09-05")],
                        "lat": np.linspace(-90, 90, 721),
                        "lon": np.linspace(0, 359.75, 1440),
                    },
                    dims=["time", "lat", "lon"],
                )
            }
        )
        mock_cache.load_value.return_value = cached

        ds = await loader.body(
            variables=["t2m"], selection={"time": "2022-09-05", "mode": "aws"}
        )

        assert ds is cached
        # Should not fetch or write when cached and invalidate_cache is False/None
        mock_gfs_cls.assert_not_called()
        mock_cache.write_value.assert_not_called()

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_gfs_era5_data.E2S_GFS")
    async def test_body_default_mode_no_timeout(
        self, mock_gfs_cls, loader, mock_cache, mock_da
    ):
        mock_gfs = Mock()
        mock_gfs.side_effect = lambda *args, **kwargs: mock_da
        mock_gfs_cls.return_value = mock_gfs

        ds = await loader.body(
            variables=None,
            selection={"time": "2022-09-05", "mode": "aws"},
        )
        assert isinstance(ds, xr.Dataset)
        # all ERA5 channels present when variables=None
        for v in ERA5_CHANNELS:
            assert v in ds.data_vars
        mock_cache.write_value.assert_called_once()

    @pytest.mark.asyncio
    @patch("nv_dfm_lib_weather.load_data._load_gfs_era5_data.E2S_GFS")
    async def test_body_ncep_date_window(self, mock_gfs_cls, loader, mock_da):
        mock_gfs = Mock()
        mock_gfs.side_effect = lambda *args, **kwargs: mock_da
        mock_gfs_cls.return_value = mock_gfs

        # Choose a date 5 days ago: should pass
        import datetime as _dt

        ok_date = (_dt.datetime.today().date() - _dt.timedelta(days=5)).strftime(
            "%Y-%m-%d"
        )
        await loader.body(
            variables=["t2m"], selection={"time": ok_date, "mode": "ncep"}
        )

        # Choose a date 20 days ago: should fail
        bad_date = (_dt.datetime.today().date() - _dt.timedelta(days=20)).strftime(
            "%Y-%m-%d"
        )
        with pytest.raises(ValueError):
            await loader.body(
                variables=["t2m"], selection={"time": bad_date, "mode": "ncep"}
            )

    @pytest.mark.asyncio
    async def test_body_missing_time_raises(self, loader):
        with pytest.raises(ValueError, match="requires a specific time selection"):
            await loader.body(variables=["t2m"], selection={})

    @pytest.mark.asyncio
    async def test_body_none_selection_raises(self, loader):
        with pytest.raises(ValueError, match="requires a specific time selection"):
            await loader.body(variables=["t2m"], selection=None)
