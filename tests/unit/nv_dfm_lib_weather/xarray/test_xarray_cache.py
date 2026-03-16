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

"""Tests for the XArrayCache class."""

import pytest  # type: ignore
import tempfile
import shutil
from upath import UPath
from unittest.mock import Mock
import logging

import xarray  # type: ignore
import numpy as np  # type: ignore

from nv_dfm_lib_weather.xarray.cache import XArrayCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary UPath directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield UPath(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset():
    """Create a sample xarray dataset for testing."""
    return xarray.Dataset(  # type: ignore
        data_vars={
            "temperature": xarray.DataArray(  # type: ignore
                np.random.rand(10, 5, 5),
                dims=["time", "lat", "lon"],
                attrs={"units": "celsius"},
            ),
            "humidity": xarray.DataArray(  # type: ignore
                np.random.rand(10, 5, 5),
                dims=["time", "lat", "lon"],
                attrs={"units": "percent"},
            ),
        },
        coords={
            "time": xarray.DataArray(  # type: ignore
                np.arange("2023-01-01", "2023-01-11", dtype="datetime64[D]"),
                dims=["time"],
            ),
            "lat": xarray.DataArray(  # type: ignore
                np.linspace(0, 90, 5), dims=["lat"]
            ),
            "lon": xarray.DataArray(  # type: ignore
                np.linspace(-180, 180, 5), dims=["lon"]
            ),
        },
        attrs={"title": "Test Dataset"},
    )


class TestXArrayCache:
    """Test cases for the XArrayCache class."""

    def test_init_defaults(self, temp_cache_dir):
        """Test XArrayCache initialization with default parameters."""
        cache = XArrayCache(temp_cache_dir)

        assert cache._cache_storage == temp_cache_dir
        assert cache._file_prefix == "dataset"
        assert isinstance(cache._logger, logging.Logger)
        assert cache._logger.name == "nv_dfm_lib_weather.xarray.cache._xarray_cache"

    def test_init_custom_parameters(self, temp_cache_dir):
        """Test XArrayCache initialization with custom parameters."""
        custom_logger = logging.getLogger("custom_logger")
        cache = XArrayCache(
            cache_storage=temp_cache_dir,
            file_prefix="custom_prefix",
            logger=custom_logger,
        )

        assert cache._cache_storage == temp_cache_dir
        assert cache._file_prefix == "custom_prefix"
        assert cache._logger == custom_logger

    def test_write_value_success(self, temp_cache_dir, sample_dataset):
        """Test successful writing of xarray dataset to cache."""
        cache = XArrayCache(temp_cache_dir)
        key = "test_key"

        # Write dataset to cache
        cache.write_value(key, sample_dataset)

        # Verify file was created
        expected_file = temp_cache_dir / f"dataset_{key}.nc"
        assert expected_file.exists()

        # Verify file can be read back
        loaded_dataset = xarray.open_dataset(expected_file)  # type: ignore
        assert isinstance(loaded_dataset, xarray.Dataset)  # type: ignore
        assert "temperature" in loaded_dataset.data_vars
        assert "humidity" in loaded_dataset.data_vars
        assert "time" in loaded_dataset.coords
        assert "lat" in loaded_dataset.coords
        assert "lon" in loaded_dataset.coords

    def test_write_value_custom_prefix(self, temp_cache_dir, sample_dataset):
        """Test writing with custom file prefix."""
        cache = XArrayCache(temp_cache_dir, file_prefix="custom")
        key = "test_key"

        cache.write_value(key, sample_dataset)

        expected_file = temp_cache_dir / f"custom_{key}.nc"
        assert expected_file.exists()

    def test_write_value_error_handling(self, temp_cache_dir):
        """Test that write errors are swallowed (best-effort cache)."""
        cache = XArrayCache(temp_cache_dir)

        mock_dataset = Mock()
        mock_dataset.to_netcdf.side_effect = Exception("Write error")

        cache.write_value("test_key", mock_dataset)

        expected_file = temp_cache_dir / "dataset_test_key.nc"
        assert not expected_file.exists()

    def test_load_value_success(self, temp_cache_dir, sample_dataset):
        """Test successful loading of xarray dataset from cache."""
        cache = XArrayCache(temp_cache_dir)
        key = "test_key"

        # First write a dataset
        cache.write_value(key, sample_dataset)

        # Then load it back
        loaded_dataset = cache.load_value(key)

        assert loaded_dataset is not None
        assert isinstance(loaded_dataset, xarray.Dataset)  # type: ignore
        assert "temperature" in loaded_dataset.data_vars
        assert "humidity" in loaded_dataset.data_vars
        assert "time" in loaded_dataset.coords
        assert "lat" in loaded_dataset.coords
        assert "lon" in loaded_dataset.coords

    def test_load_value_not_found(self, temp_cache_dir):
        """Test loading when file doesn't exist."""
        cache = XArrayCache(temp_cache_dir)

        result = cache.load_value("nonexistent_key")
        assert result is None

    def test_load_value_error_handling(self, temp_cache_dir):
        """Test that a corrupted file is treated as a cache miss."""
        cache = XArrayCache(temp_cache_dir)
        key = "test_key"

        corrupted_file = temp_cache_dir / f"dataset_{key}.nc"
        corrupted_file.write_text("This is not a valid netCDF file")

        result = cache.load_value(key)
        assert result is None

    def test_write_and_load_roundtrip(self, temp_cache_dir, sample_dataset):
        """Test complete write and load roundtrip."""
        cache = XArrayCache(temp_cache_dir)
        key = "roundtrip_test"

        # Write dataset
        cache.write_value(key, sample_dataset)

        # Load dataset
        loaded_dataset = cache.load_value(key)

        # Verify data integrity
        assert loaded_dataset is not None
        assert loaded_dataset.attrs["title"] == sample_dataset.attrs["title"]

        # Verify data arrays are equal
        np.testing.assert_array_equal(
            loaded_dataset["temperature"].values, sample_dataset["temperature"].values
        )
        np.testing.assert_array_equal(
            loaded_dataset["humidity"].values, sample_dataset["humidity"].values
        )

    def test_multiple_keys(self, temp_cache_dir, sample_dataset):
        """Test caching multiple datasets with different keys."""
        cache = XArrayCache(temp_cache_dir)

        # Create second dataset
        dataset2 = sample_dataset.copy()
        dataset2.attrs["title"] = "Second Dataset"

        # Write both datasets
        cache.write_value("key1", sample_dataset)
        cache.write_value("key2", dataset2)

        # Load both datasets
        loaded1 = cache.load_value("key1")
        loaded2 = cache.load_value("key2")

        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded1.attrs["title"] == "Test Dataset"
        assert loaded2.attrs["title"] == "Second Dataset"

    def test_logging(self, temp_cache_dir, sample_dataset, caplog):
        """Test that logging works correctly."""
        # Ensure logger is set to INFO and propagates
        logger = logging.getLogger("nv_dfm_lib_weather.xarray.cache._xarray_cache")
        logger.setLevel(logging.INFO)
        logger.propagate = True
        cache = XArrayCache(temp_cache_dir, logger=logger)
        key = "logging_test"

        # Write and load to trigger logging
        cache.write_value(key, sample_dataset)
        cache.load_value(key)

        # Print all log records for debugging
        for record in caplog.records:
            print(f"LOG [{record.levelname}] {record.name}: {record.getMessage()}")
        log_messages = [record.getMessage() for record in caplog.records]
        assert any("Caching xarray dataset" in msg for msg in log_messages)
        assert any("Loaded CACHED xarray dataset" in msg for msg in log_messages)

    def test_file_path_construction(self, temp_cache_dir):
        """Test that file paths are constructed correctly."""
        cache = XArrayCache(temp_cache_dir, file_prefix="test_prefix")
        key = "test_key"

        expected_file = temp_cache_dir / f"test_prefix_{key}.nc"
        assert cache._cache_storage / f"{cache._file_prefix}_{key}.nc" == expected_file

    def test_special_characters_in_key(self, temp_cache_dir, sample_dataset):
        """Test handling of special characters in keys."""
        cache = XArrayCache(temp_cache_dir)
        key = "test_key_with_special_chars_!@#$%^&*()"

        cache.write_value(key, sample_dataset)
        loaded_dataset = cache.load_value(key)

        assert loaded_dataset is not None
        assert isinstance(loaded_dataset, xarray.Dataset)  # type: ignore

    def test_empty_dataset(self, temp_cache_dir):
        """Test caching an empty dataset."""
        cache = XArrayCache(temp_cache_dir)
        key = "empty_test"

        # Create empty dataset
        empty_dataset = xarray.Dataset()  # type: ignore

        cache.write_value(key, empty_dataset)
        loaded_dataset = cache.load_value(key)

        assert loaded_dataset is not None
        assert isinstance(loaded_dataset, xarray.Dataset)  # type: ignore
        assert len(loaded_dataset.data_vars) == 0

    def test_large_dataset(self, temp_cache_dir):
        """Test caching a large dataset."""
        cache = XArrayCache(temp_cache_dir)
        key = "large_test"

        # Create large dataset
        large_dataset = xarray.Dataset(  # type: ignore
            data_vars={
                "data": xarray.DataArray(  # type: ignore
                    np.random.rand(100, 100, 100), dims=["x", "y", "z"]
                )
            }
        )

        cache.write_value(key, large_dataset)
        loaded_dataset = cache.load_value(key)

        assert loaded_dataset is not None
        assert loaded_dataset["data"].shape == (100, 100, 100)

    def test_multiple_loads(self, temp_cache_dir, sample_dataset):
        """Test multiple loads of the same cached dataset."""
        cache = XArrayCache(temp_cache_dir)
        key = "multiple_loads_test"

        # Write dataset
        cache.write_value(key, sample_dataset)

        # Load the same dataset multiple times
        results = []
        for _ in range(5):
            result = cache.load_value(key)
            results.append(result)

        # All should return the same dataset
        for result in results:
            assert result is not None
            assert isinstance(result, xarray.Dataset)  # type: ignore
            assert "temperature" in result.data_vars
