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
GFS loader adapter producing ERA5-compatible datasets.

Uses earth2studio.data.GFS as the data source and maps to ERA5 variable names
and coordinates.
"""

import hashlib
import inspect
import json
import logging
from datetime import datetime, timedelta

import dateutil.parser
import xarray
from earth2studio.data import GFS as E2S_GFS

from nv_dfm_core.exec import Provider, Site
from nv_dfm_core.exec.discovery import (
    AdvisedDict,
    AdvisedLiteral,
    AdvisedOneOf,
    AdvisedSubsetOf,
    field_advisor,
)

from ..xarray.cache import XArrayCache
from ._channels import ERA5_CHANNELS
from ._utils import should_filter_variables


class LoadGfsEra5Data:
    """
    GFS weather data loader with ERA5-compatible output.

    This class provides functionality to load Global Forecast System (GFS) weather data
    through the Earth2Studio library and converts it to ERA5-compatible format. It serves
    as an adapter that bridges GFS data sources with ERA5 variable naming conventions and
    coordinate systems, enabling seamless integration with workflows expecting ERA5 data.

    The loader supports two operational modes:
    - 'aws': Historical GFS data from AWS (2021-01-01 onwards)
    - 'ncep': Recent GFS data from NCEP servers (past 10 days)

    Key features:
    - Automatic data caching for improved performance
    - Variable name mapping from GFS to ERA5 conventions
    - Flexible variable selection (individual or all channels)
    - Discovery system for available variables and time ranges

    Attributes:
        _site: Site instance for accessing infrastructure and storage
        _provider: DFM Provider instance, or None if running standalone
        _logger: Logger instance for tracking operations and debugging
        _cache: XArray cache for storing and retrieving processed datasets
        _source: Earth2Studio GFS data source instance (initialized on demand)
    """

    def __init__(self, site: Site, provider: Provider | None):
        """
        Initialize the GFS ERA5 data loader.

        Args:
            site: Site instance providing access to caching and infrastructure
            provider: DFM Provider instance, or None if running standalone
        """
        self._site: Site = site
        self._provider: Provider | None = provider
        self._logger: logging.Logger = site.dfm_context.logger
        # Set up dedicated cache storage for this loader class
        cache_storage = site.cache_storage(subpath=self.__class__.__name__)
        self._cache: XArrayCache = XArrayCache(
            cache_storage, file_prefix="era5_gfs_dataset"
        )
        self._source: E2S_GFS | None = None

    def _calculate_params_hash(
        self, variables: list[str], selection: dict[str, str]
    ) -> str:
        """
        Calculate a deterministic hash for cache key generation.

        Creates a SHA-256 hash based on the requested variables and selection parameters
        to uniquely identify each data loading request. This enables efficient caching
        by ensuring identical requests produce identical cache keys.

        Args:
            variables: List of ERA5-compatible variable names requested
            selection: Dictionary of selection parameters (time, mode, etc.)

        Returns:
            str: Hexadecimal SHA-256 hash string for use as cache key
        """
        # Create a deterministic dictionary by sorting all elements
        params_dict = {
            "variables": sorted(variables) if variables else None,
            "selection": dict(sorted(selection.items())) if selection else None,
        }
        # Convert to JSON with sorted keys for consistency
        params_json = json.dumps(params_dict, sort_keys=True)
        # Generate and return SHA-256 hash
        return hashlib.sha256(params_json.encode("utf-8")).hexdigest()

    @field_advisor("variables")
    async def available_variables(self, _value: object, _context: object):
        """
        Field advisor that provides available ERA5-compatible variables for GFS data.

        This method is used by the DFM discovery system to inform clients about which
        ERA5-compatible climate variables are available from the GFS data source. Clients
        can either request all variables using "*" or select specific variables from the
        ERA5 channel list.

        Args:
            _value: The current value (unused)
            _context: The discovery context (unused)

        Returns:
            AdvisedOneOf: Either "*" (all variables) or a subset of ERA5_CHANNELS
        """
        self._logger.info("Discovery started for variables (GFS)")
        result = AdvisedOneOf([AdvisedLiteral("*"), AdvisedSubsetOf(ERA5_CHANNELS)])  # pyright: ignore[reportArgumentType]
        self._logger.info(f"Discovery finished for variables (GFS): {result}")
        return result

    @field_advisor("selection")
    async def valid_selections(self, _value: object, context: object):
        """
        Field advisor that provides valid selection parameters for GFS data loading.

        This method informs clients about available selection options based on the data
        source mode. It dynamically adjusts the valid time range depending on whether
        AWS (historical) or NCEP (recent) mode is selected, ensuring clients only request
        data within available ranges.

        Selection parameters include:
        - time: Date/datetime for data retrieval (range depends on mode)
        - mode: Data source ("aws" for historical, "ncep" for recent)
        - timeout: Optional timeout in seconds for data access

        Args:
            _value: The current value (unused)
            context: Discovery context potentially containing the selected mode

        Returns:
            AdvisedOneOf: Either None or a dictionary of selection parameters with constraints
        """
        self._logger.info("Discovery started for selection (GFS)")
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

        # Extract the selected mode from context to provide mode-specific time ranges
        # Default to 'aws' if not specified or invalid
        selected_mode = None
        try:
            selected_mode = context.get("mode")  # type: ignore[attr-defined]
        except Exception:
            selected_mode = None
        if selected_mode not in ("aws", "ncep"):
            selected_mode = "aws"

        # AWS mode provides historical data from 2022 onwards
        if selected_mode == "aws":
            time_advice = {
                "first_date": "2022-01-01",
                "last_date": today.strftime("%Y-%m-%d"),
                "frequency": 1,
            }
        else:
            # NCEP mode provides recent data for the past 10 days (inclusive of today)
            time_advice = {
                "first_date": (today - timedelta(days=9)).strftime("%Y-%m-%d"),
                "last_date": today.strftime("%Y-%m-%d"),
                "frequency": 1,
            }

        # Build the advice dictionary with all selection parameters
        advice_d = {
            "time": time_advice,
            "mode": {
                "one_of": ["aws", "ncep"],
                "default": "aws",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds for data access",
                "optional": True,
            },
        }
        none_advice = AdvisedLiteral(None)
        dict_advice = AdvisedDict(advice_d, allow_extras=True)  # pyright: ignore[reportArgumentType]
        result = AdvisedOneOf(values=[none_advice, dict_advice])
        self._logger.info(f"Discovery finished for selection (GFS): {result}")
        return result

    async def body(
        self,
        variables: list[str],
        selection: dict[str, str],
        invalidate_cache: bool | None = None,
    ) -> xarray.Dataset:
        """
        Load GFS weather data and convert to ERA5-compatible format.

        This method fetches Global Forecast System (GFS) data from either AWS or NCEP
        sources using the Earth2Studio library and transforms it into ERA5-compatible
        format with proper variable naming and coordinate systems. The data is cached
        to improve performance for repeated requests.

        The method supports two operational modes:
        - 'aws': Historical GFS data from 2021-01-01 onwards
        - 'ncep': Recent GFS data from the past 10 days

        Args:
            variables: List of ERA5-compatible variable names to load. Can use ["*"] to
                      load all available ERA5 channels, or specify individual variables
                      (e.g., ["t2m", "u10", "v10"]).
            selection: Dictionary containing selection parameters:
                      - time (required): Date or datetime string (e.g., "2024-01-31" or
                        "2024-01-31T12:00:00")
                      - mode (optional): "aws" for historical data or "ncep" for recent data
                        (default: "aws")
                      - timeout (optional): Timeout in seconds for data access
            invalidate_cache: If True, bypass cache and force fresh data loading. If False
                            or None, use cached data when available. (default: None)

        Returns:
            xarray.Dataset: GFS data in ERA5-compatible format with dimensions:
                - time: temporal dimension
                - lat: latitude dimension
                - lon: longitude dimension
                Each requested variable appears as a data variable in the dataset.

        Raises:
            ValueError: If selection is missing, lacks 'time' parameter, or specifies
                       a time outside the valid range for the selected mode.

        Example:
            >>> # Load temperature and wind data for a specific date
            >>> ds = await loader.body(
            ...     variables=["t2m", "u10", "v10"],
            ...     selection={"time": "2024-01-31", "mode": "aws"},
            ...     invalidate_cache=False
            ... )
        """
        self._logger.info(
            f"Loading GFS ERA5 data with variables: {variables}, selection: {selection}"
        )

        # Validate that required time parameter is present
        if not selection or "time" not in selection:
            raise ValueError(
                "The LoadGfsEra5Data adapter requires a specific time selection. "
                "Please supply something like {'time': '2024-01-31', ...}"
            )

        # Convert all selection values to strings for hash calculation
        selection_for_hash: dict[str, str] = {k: str(v) for k, v in selection.items()}

        # Generate cache key from parameters
        params_hash = self._calculate_params_hash(variables, selection_for_hash)

        # Check cache for existing data unless invalidate_cache is requested
        cached_ds = None
        if self._cache and not invalidate_cache:
            cached_ds = self._cache.load_value(params_hash)
            if cached_ds:
                self._logger.info(f"Loaded cached dataset {cached_ds}")
                return cached_ds

        # Parse the time parameter into a datetime object
        # Supports ISO date/time strings or numeric timestamps
        when_dt = dateutil.parser.parse(str(selection["time"]))

        # Validate that the requested time falls within the valid range for the selected mode
        mode = str(selection.get("mode", "aws"))
        when_date = when_dt.date()
        today = datetime.today().date()

        if mode == "aws":
            # AWS mode: data available from 2021-01-01 onwards
            min_date = datetime(2021, 1, 1).date()
            if when_date < min_date:
                raise ValueError("For aws mode, 'time' must be on/after 2021-01-01")
        elif mode == "ncep":
            # NCEP mode: data available for the past 10 days (inclusive)
            if when_date < (today - timedelta(days=10)) or when_date > today:
                raise ValueError(
                    "For ncep mode, 'time' must be within the past 10 days"
                )

        # Determine which variables to load
        # If specific variables requested (not "*"), use them; otherwise use all ERA5 channels
        candidates: list[str]
        if should_filter_variables(variables):
            candidates = variables
        else:
            candidates = ERA5_CHANNELS

        # Configure the Earth2Studio GFS data source with appropriate parameters
        # Build kwargs dynamically based on what the E2S_GFS constructor accepts
        timeout_val_obj = selection.get("timeout")
        sig_params = set(inspect.signature(E2S_GFS).parameters.keys())
        e2s_kwargs: dict[str, str | float] = {}

        if "mode" in sig_params:
            e2s_kwargs["mode"] = mode
        if timeout_val_obj is not None and "timeout" in sig_params:
            e2s_kwargs["timeout"] = float(timeout_val_obj)
        if invalidate_cache is not None:
            e2s_kwargs["cache"] = not invalidate_cache

        # Initialize the Earth2Studio GFS data source
        source = E2S_GFS(**e2s_kwargs)

        # Fetch data from Earth2Studio as a DataArray
        da = source(time=when_dt, variable=candidates)

        # Convert DataArray to Dataset format with ERA5-compatible variable names
        # This preserves the native latitude order from the source
        ds = da.to_dataset(dim="variable")

        # Select only the requested variables from the dataset
        ds = ds[candidates]

        # Cache the processed dataset for future requests
        self._logger.info(f"Writing dataset to cache {params_hash}")
        if self._cache:
            self._cache.write_value(params_hash, ds)
        self._logger.info("Loaded GFS dataset")
        return ds
