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

import logging

import xarray
from upath import UPath

from nv_dfm_lib_weather._netcdf_lock import netcdf_lock as _netcdf_lock


class XArrayCache:
    """
    A cache manager for xarray datasets using NetCDF format.

    This class provides functionality to cache xarray datasets to disk using
    NetCDF4 format, with support for configurable storage locations and logging.
    """

    def __init__(
        self,
        cache_storage: UPath,
        file_prefix: str = "dataset",
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the xarray cache.

        Args:
            cache_storage: UPath for cache storage location
            file_prefix: Prefix for cached files (default: "dataset")
            logger: Optional logger instance (uses module logger if None)
        """
        self._cache_storage: UPath = cache_storage
        self._file_prefix: str = file_prefix
        self._logger: logging.Logger = logger or logging.getLogger(__name__)

    def write_value(self, key: str, item: xarray.Dataset):
        """
        Write an xarray dataset to cache storage.

        Best-effort: failures are logged but never raised, so a cache write
        failure does not interrupt the rest of the application.

        Args:
            key: Unique key for the cached item
            item: xarray dataset to cache
        """
        file = self._cache_storage / f"{self._file_prefix}_{key}.nc"
        try:
            if file.exists():
                self._logger.info(
                    f"<cache> Deleting existing xarray dataset from {file}"
                )
                file.unlink()
        except Exception as e:
            self._logger.warning(
                f"<cache> Failed to delete existing cache file {file}: {e}"
            )

        self._logger.info(f"<cache> Writing xarray dataset to {file}")
        try:
            with _netcdf_lock:
                item.to_netcdf(file.path, "w", format="NETCDF4")
            self._logger.info(f"<cache> Written xarray dataset to {file}")
        except Exception as e:
            self._logger.warning(
                f"<cache> Failed to write xarray dataset to {file}: {e}"
            )
            return
        self._logger.info("Caching xarray dataset %s to %s", key, self._cache_storage)

    def load_value(self, key: str) -> xarray.Dataset | None:
        """
        Load an xarray dataset from cache storage.

        Args:
            key: Unique key for the cached item

        Returns:
            xarray.Dataset if found in cache, None if not found or unreadable
        """
        filename = self._cache_storage / f"{self._file_prefix}_{key}.nc"
        self._logger.info(f"<cache> Trying to load xarray dataset from {filename}")
        if not filename.exists():
            self._logger.info(f"<cache> Xarray dataset not found {filename}")
            return None
        try:
            with _netcdf_lock:
                ds = xarray.open_dataset(filename.path, chunks=None, engine="netcdf4")
        except Exception as e:
            self._logger.warning(
                f"<cache> Failed to load xarray dataset from {filename}, treating as cache miss: {e}"
            )
            return None
        self._logger.info(
            "Loaded CACHED xarray dataset %s from %s", key, self._cache_storage
        )
        self._logger.info(f"<cache> Loaded xarray dataset {ds}")
        return ds
