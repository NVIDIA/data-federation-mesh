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

import asyncio
import hashlib
import json
import random
import time
from typing import Any

import xarray

from nv_dfm_core.exec import Provider, Site
from nv_dfm_core.exec.discovery import (
    AdvisedDict,
    AdvisedLiteral,
    AdvisedOneOf,
    AdvisedSubsetOf,
    field_advisor,
)
from nv_dfm_lib_weather.xarray.cache import XArrayCache

from ._utils import should_filter_variables

ALL_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "geopotential",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "land_sea_mask",
    "latitude",
    "level",
    "longitude",
    "low_vegetation_cover",
    "mean_sea_level_pressure",
    "sea_ice_cover",
    "sea_surface_temperature",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "specific_humidity",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "surface_pressure",
    "temperature",
    "time",
    "toa_incident_solar_radiation",
    "total_cloud_cover",
    "total_column_water_vapour",
    "total_precipitation",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]


class LoadEcmwfEra5Data:
    """
    A data loader for ECMWF ERA5 weather data from Google Cloud Storage.

    This class provides functionality to load ERA5 weather data from a Zarr store,
    with caching support and parameter validation. It supports variable selection
    and time-based filtering of the dataset.
    """

    def __init__(self, site: Site, provider: Provider | None):
        """
        Initialize the ECMWF ERA5 data loader.

        Args:
            site: DFM site configuration
            provider: Optional provider configuration
        """
        self._site = site
        self._provider = provider
        self._logger = site.dfm_context.logger
        cache_storage = site.cache_storage(subpath=self.__class__.__name__)
        self._cache = XArrayCache(cache_storage, file_prefix="era5_ecmwf_dataset")
        # If we ever want to test with other data sources, this should be a parameter
        self._url = "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
        self._engine = "zarr"
        # Empty for now, but we can add more parameters here if needed
        self._engine_kwargs: dict[str, Any] = {}

    @field_advisor("variables")
    async def available_variables(self, _value, _context):
        """
        Provide advice on available variables for the ECMWF ERA5 dataset.

        Args:
            _value: The current value being validated
            _context: The validation context

        Returns:
            AdvisedOneOf containing available variable options
        """
        advice_list = [
            v
            for v in ALL_VARIABLES
            if v not in ["latitude", "longitude", "level", "time"]
        ]
        return AdvisedOneOf([AdvisedLiteral("*"), AdvisedSubsetOf(advice_list)])

    @field_advisor("selection")
    async def valid_selections(self, _value, _context):
        """
        Provide advice on valid selection parameters for the ECMWF ERA5 dataset.

        Args:
            _value: The current value being validated
            _context: The validation context

        Returns:
            AdvisedOneOf containing valid selection options
        """
        none_advice = AdvisedLiteral(None)
        dict_advice = AdvisedDict(
            {
                "time": {
                    "first_date": "1959-01-01",
                    "last_date": "2021-12-31",
                    "frequency": 1,
                }
            },
            allow_extras=True,
        )
        return AdvisedOneOf(values=[none_advice, dict_advice])

    def _calculate_params_hash(
        self, variables: list[str], selection: dict[str, str]
    ) -> str:
        """Calculate a hash of the function parameters for caching purposes.

        Args:
            variables: List of variable names
            selection: Dictionary of selection parameters

        Returns:
            A hex string representing the hash of the parameters
        """
        # Create a dictionary with the parameters
        params_dict = {
            "variables": sorted(variables) if variables else None,
            "selection": dict(sorted(selection.items())) if selection else None,
        }

        # Convert to JSON string and hash it
        params_json = json.dumps(params_dict, sort_keys=True)
        return hashlib.sha256(params_json.encode("utf-8")).hexdigest()

    def body_impl(
        self,
        variables: list[str],
        selection: dict[str, str],
    ) -> xarray.Dataset:
        """
        Load ECMWF ERA5 data with caching and filtering.

        This method loads ERA5 weather data from the configured Zarr store,
        applies variable selection and time filtering, and caches the result.

        Args:
            variables: List of variable names to load, or empty list for all
            selection: Dictionary of selection parameters (e.g., time range)

        Returns:
            xarray.Dataset containing the loaded and filtered weather data
        """
        # randomize the access so we don't hit the servier all at once
        params_hash = self._calculate_params_hash(variables, selection)
        self._logger.debug(f"LoadEcmwfEra5Data: params_hash: {params_hash}")
        cached_ds = self._cache.load_value(params_hash)
        if cached_ds:
            self._logger.info(f"Loaded cached dataset {cached_ds}")
            return cached_ds

        time_delay = random.uniform(0.01, 0.5)
        time.sleep(time_delay)
        self._logger.info(
            f"Opening ECMWF ERA5 dataset from {self._url} with engine {self._engine} and engine_kwargs {self._engine_kwargs}"
        )
        print(
            f"Opening ECMWF ERA5 dataset from {self._url} with engine {self._engine} and engine_kwargs {self._engine_kwargs}"
        )
        ds = xarray.open_dataset(
            self._url,
            cache=False,
            decode_times=True,
            engine=self._engine,
            **self._engine_kwargs,
        )
        self._logger.info(f"Opened ECMWF ERA5 dataset: {ds}")
        print(f"Opened ECMWF ERA5 dataset: {ds}")
        if selection:
            selection_dict: dict[Any, Any] = {
                k: (v if isinstance(v, list) else [v]) for k, v in selection.items()
            }
            ds = ds.sel(method="nearest", **selection_dict)

        # pick the selected variables (or all, if no explicit list or '*')
        # This works better than "drop_variables", because it keeps the
        # coordinates intact, whereas drop_variables
        # may also drop labels
        if should_filter_variables(variables):
            ds = ds[variables]
        self._logger.info(f"Filtered ECMWF ERA5 dataset: {ds}")
        print(f"Filtered ECMWF ERA5 dataset: {ds}")
        # Write the dataset to the cache
        self._cache.write_value(params_hash, ds)
        return ds

    async def body(
        self,
        variables: list[str],
        selection: dict[str, str],
    ) -> xarray.Dataset:
        """
        Async wrapper for loading ECMWF ERA5 data.

        This method provides an async interface to the data loading functionality,
        parsing the selection parameter from JSON and handling errors gracefully.

        Args:
            variables: List of variable names to load, or empty list for all
            selection: JSON string containing selection parameters

        Returns:
            xarray.Dataset containing the loaded and filtered weather data

        Raises:
            Exception: If there's an error loading the data or parsing parameters
        """
        try:
            # TODO: this is temporary, DFM sometimes has issues with dict params
            # selection_dict = json.loads(selection)
            result = await asyncio.to_thread(self.body_impl, variables, selection)
            self._logger.info(f"Loaded ECMWF ERA5 dataset: {result}")
            return result
        except Exception as e:
            self._logger.error(f"Error loading ECMWF ERA5 data: {e}")
            raise e
