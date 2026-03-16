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
import datetime
import hashlib
import json
import logging

import aiohttp
import dateutil.parser
import intake_esgf
import xarray
from earth2studio.data.cmip6 import CMIP6, CMIP6Lexicon

from nv_dfm_core.api import Advise
from nv_dfm_core.exec import Provider, Site
from nv_dfm_core.exec.discovery import (
    AdvisedDateRange,
    AdvisedDict,
    AdvisedError,
    AdvisedLiteral,
    AdvisedOneOf,
    AdvisedSubsetOf,
    AdvisedValue,
    Okay,
    field_advisor,
)

from ..xarray.cache import XArrayCache


class LoadCmip6Data:
    """
    A data loader for CMIP6 (Coupled Model Intercomparison Project Phase 6) data.

    This class provides functionality to load climate model data from the CMIP6 archive
    using the ESGF (Earth System Grid Federation) catalog. It supports caching, parameter
    validation, and discovery of available datasets through field advisors.

    The loader integrates with Earth2Studio's CMIP6 implementation and provides
    additional caching capabilities through DFM's cache storage system.

    Attributes:
        _site: DFM site configuration for caching and logging
        _provider: Optional provider configuration
        _cmip6_vars: List of available CMIP6 variable names from the lexicon
        _logger: Logger instance for the class
        _has_cache: Boolean indicating if caching is available
        _cache_storage: Cache storage path for CMIP6 datasets
        _esgf_cache_storage: Cache storage path for ESGF catalog data
        _cache: XArrayCache instance for dataset caching
        _esgf_catalog: ESGF catalog instance for data discovery
    """

    def __init__(self, site: Site | None, provider: Provider | None):
        """
        Initialize the CMIP6 data loader.

        Args:
            site: DFM site configuration
            provider: Optional provider configuration
        """
        self._site = site
        self._provider = provider
        self._cmip6_vars = [k for k in CMIP6Lexicon.build_vocab().keys()]
        if site is not None:
            self._logger = site.dfm_context.logger
            self._has_cache = True
            self._cache_storage = site.cache_storage(subpath=self.__class__.__name__)
            self._esgf_cache_storage = site.cache_storage(subpath="intake_esgf")
            self._cache = XArrayCache(self._cache_storage, file_prefix="cmip6_dataset")
        else:
            self._logger = logging.getLogger(__name__)
            self._has_cache = False

        # We need to do it every time before calling the ESGF catalog, E2Studio can reset the cache config
        self._configure_esgf_cache()
        self._esgf_catalog = intake_esgf.ESGFCatalog()

    def _configure_esgf_cache(self):
        """
        Configure ESGF cache settings for intake-esgf.

        This method configures the intake-esgf library to use DFM's cache storage
        for ESGF catalog data. It sets up both local file cache and requests cache
        with appropriate expiration times. The configuration is applied every time
        before making ESGF queries since Earth2Studio may reset the cache config.

        The method only configures caching if:
        - Caching is available (_has_cache is True)
        - Cache storage uses file protocol (ESGF requires local filesystem)

        If caching cannot be configured, it logs a warning and disables caching.
        """
        # CMIP6 in Earth2Studio is using intake-esgf, but we will use ours
        # and replace E2S' one with it, because we want to configure cache.
        if self._has_cache and self._esgf_cache_storage.protocol == "file":
            # ESGF requires local filesystem cache, so we only configure it if the cache storage is a file protocol.
            self._logger.info(
                f"Setting ESGF local cache to {self._esgf_cache_storage.path}"
            )
            intake_esgf.conf.set(
                local_cache=self._esgf_cache_storage.path,
                requests_cache=dict(
                    expire_after=datetime.timedelta(hours=6).seconds,
                    cache_name=(
                        self._esgf_cache_storage / "requests-cache.sqlite"
                    ).path,
                ),
            )
        else:
            self._logger.warning(
                "ESGF cache is not configured - adapter running without cache or with non-local cache storage"
            )
            intake_esgf.conf.set(local_cache=None, requests_cache=None)

    @staticmethod
    async def download_to_memory(url: str, retries: int = 3) -> dict | None:
        """
        Download JSON data from a URL with retry logic.

        This static method downloads JSON data from the specified URL with
        configurable retry attempts. It's used to fetch experiment metadata
        from the CMIP6 controlled vocabulary.

        Args:
            url: The URL to download JSON data from
            retries: Number of retry attempts on failure (default: 3)

        Returns:
            dict | None: Downloaded JSON data as a dictionary, or None if all retries fail
        """
        for _ in range(retries):
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
            except TimeoutError:
                continue  # Retry on timeout
        return None

    def _calculate_params_hash(
        self,
        variables: list[str],
        selection: dict[str, str],
        experiment_id: str,
        source_id: str,
        table_id: str,
        variant_label: str,
    ) -> str:
        """Calculate a hash of the function parameters for caching purposes.

        This method creates a deterministic hash from all the parameters used to
        load CMIP6 data. The hash is used as a cache key to store and retrieve
        datasets, ensuring that different parameter combinations get separate cache entries.

        Args:
            variables: List of variable names to load
            selection: Dictionary of selection parameters (e.g., time range)
            experiment_id: CMIP6 experiment identifier
            source_id: CMIP6 source identifier (climate model)
            table_id: CMIP6 table identifier (frequency)
            variant_label: CMIP6 variant label (ensemble member)

        Returns:
            str: A hex string representing the SHA-256 hash of the parameters
        """
        # Create a dictionary with the parameters
        params_dict = {
            "variables": sorted(variables) if variables else None,
            "selection": dict(sorted(selection.items())) if selection else None,
            "experiment_id": experiment_id,
            "source_id": source_id,
            "table_id": table_id,
            "variant_label": variant_label,
        }
        # Convert to JSON string and hash it
        params_json = json.dumps(params_dict, sort_keys=True)
        return hashlib.sha256(params_json.encode("utf-8")).hexdigest()

    @field_advisor("variables", order=1)
    async def available_variables(self, _value, _context):
        """
        Field advisor for discovering available CMIP6 variables.

        This method provides a list of all available CMIP6 variable names from the
        CMIP6Lexicon that can be used for data loading. It's used by the DFM framework
        to provide parameter discovery and validation.

        Args:
            _value: The current value being validated (not used in discovery mode)
            _context: Context dictionary containing other parameter values

        Returns:
            AdvisedSubsetOf: Advisor containing the list of available CMIP6 variables
        """
        self._logger.info("Disovery started for variables")
        result = AdvisedSubsetOf(self._cmip6_vars, break_on_advice=True)  # type: ignore
        self._logger.info(f"Disovery finished for variables: {result}")
        return result

    async def _cache_experiment_data(self) -> dict:
        """
        Get experiment metadata with caching support.

        This method retrieves CMIP6 experiment metadata from the controlled vocabulary.
        It implements a 31-day cache to reduce network requests and improve performance.
        If the cache is expired or unavailable, it downloads fresh data from the web.

        Returns:
            dict: Dictionary containing experiment metadata with experiment IDs as keys

        Raises:
            AssertionError: If the experiment data cannot be downloaded
        """
        experiment_dict = None
        if self._has_cache:
            cache_file = self._cache_storage / "CMIP6_experiment_id.json"
            if cache_file.exists() and cache_file.is_file():
                cache_age = (
                    datetime.datetime.now()
                    - datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
                ).days
                if cache_age < 31:
                    with cache_file.open() as f:
                        experiment_dict = json.load(f)
                    self._logger.info(f"Loaded cached experiment_ids from {cache_file}")

        # If we don't have a cached experiment data or cache is older than 31 days, download
        if not experiment_dict:
            experiment_dict = await self.download_to_memory(
                "https://wcrp-cmip.github.io/CMIP6_CVs/CMIP6_experiment_id.json"
            )
            assert experiment_dict is not None, ValueError(
                "Failed to download experiment_ids"
            )
            if self._has_cache:
                with cache_file.open("w") as f:
                    json.dump(experiment_dict, f)
                    self._logger.info(f"Wrote cached experiment_ids to {cache_file}")

        return experiment_dict["experiment_id"]

    @field_advisor("experiment_id", order=2)
    async def available_experiment_ids(self, _value, _context):
        """
        Field advisor for discovering available CMIP6 experiment IDs.

        This method fetches the list of available experiment IDs from the CMIP6
        controlled vocabulary. The data is cached locally for 31 days to improve
        performance and reduce network requests.

        Args:
            _value: The current value being validated (not used in discovery mode)
            _context: Context dictionary containing other parameter values

        Returns:
            AdvisedOneOf: Advisor containing the list of available experiment IDs

        Raises:
            ValueError: If the experiment data cannot be downloaded
        """
        self._logger.info("Disovery started for experiment_ids")
        # Get experiment data from cache or download from web
        experiment_dict = await self._cache_experiment_data()
        experiment_ids = [AdvisedLiteral(v) for v in experiment_dict.keys()]
        result = AdvisedOneOf(experiment_ids, break_on_advice=True)  # type: ignore
        self._logger.info(f"Disovery finished for experiment_ids: {result}")
        return result

    @staticmethod
    def _from_context(context, key: str) -> str | None:
        """
        Safely extract a value from the context dictionary.

        This helper method safely retrieves values from the context dictionary
        used by field advisors, returning None if the key doesn't exist or
        if a ValueError is raised.

        Args:
            context: Context dictionary containing parameter values
            key: The key to retrieve from the context

        Returns:
            str | None: The value from the context, or None if not found
        """
        try:
            return context.get(key)
        except ValueError:
            return None

    async def _advisor_helper(self, value, context, parameter: str) -> AdvisedValue:
        """
        Helper method for field advisors that require ESGF catalog queries.

        This method provides common functionality for field advisors that need to
        query the ESGF catalog to discover available values for parameters like
        source_id, table_id, and variant_label. It handles both discovery and
        validation modes.

        Args:
            value: The current value being validated or discovered
            context: Context dictionary containing other parameter values
            parameter: The name of the parameter to discover/validate

        Returns:
            AdvisedValue: Either Okay() for validation or AdvisedOneOf for discovery

        Raises:
            AdvisedError: If required context parameters are missing
        """
        is_validation = not isinstance(value, Advise)
        self._logger.info(
            f"Disovery started for {parameter}, is_validation: {is_validation}"
        )

        variables = self._from_context(context, "variables")
        experiment_id = self._from_context(context, "experiment_id")

        if not variables or not experiment_id:
            raise AdvisedError("Variables and experiment_id are required")

        cmip6_variables = [CMIP6Lexicon.get_item(v)[0][0] for v in variables]

        # Add current value to the search arguments if it is a validation
        extra_search_args = {parameter: value} if is_validation else {}

        # We need to do it every time before calling the ESGF catalog, E2Studio can reset the cache config
        self._configure_esgf_cache()
        self._esgf_catalog.search(
            variable_id=cmip6_variables,
            experiment_id=experiment_id,
            **extra_search_args,
        )

        # Get all unique values from the respective column
        unique_values = self._esgf_catalog.df[parameter].unique()
        result = Okay() if is_validation else AdvisedOneOf(unique_values)
        self._logger.info(f"Disovery finished for {parameter}: {result}")

        return result

    @field_advisor("source_id", order=3)
    async def available_source_ids(self, _value, _context):
        """
        Field advisor for discovering available CMIP6 source IDs.

        This method queries the ESGF catalog to find all available source IDs
        (climate models) that have data for the specified variables and experiment.

        Args:
            _value: The current value being validated (not used in discovery mode)
            _context: Context dictionary containing variables and experiment_id

        Returns:
            AdvisedValue: Advisor containing available source IDs or validation result
        """
        return await self._advisor_helper(_value, _context, "source_id")

    @field_advisor("table_id", order=3)
    async def available_table_ids(self, _value, _context):
        """
        Field advisor for discovering available CMIP6 table IDs.

        This method queries the ESGF catalog to find all available table IDs
        (frequency of data output) that have data for the specified variables,
        experiment, and source.

        Args:
            _value: The current value being validated (not used in discovery mode)
            _context: Context dictionary containing variables, experiment_id, and source_id

        Returns:
            AdvisedValue: Advisor containing available table IDs or validation result
        """
        return await self._advisor_helper(_value, _context, "table_id")

    @field_advisor("variant_label", order=3)
    async def available_variant_labels(self, _value, _context):
        """
        Field advisor for discovering available CMIP6 variant labels.

        This method queries the ESGF catalog to find all available variant labels
        (ensemble members) that have data for the specified variables, experiment,
        source, and table. The method maps variant_label to member_id in the ESGF catalog.

        Args:
            _value: The current value being validated (not used in discovery mode)
            _context: Context dictionary containing variables, experiment_id, source_id, and table_id

        Returns:
            AdvisedValue: Advisor containing available variant labels or validation result
        """
        # member_id is equivalent to variant_label
        return await self._advisor_helper(_value, _context, "member_id")

    @field_advisor("selection")
    async def valid_selections(self, value, context):
        """
        Field advisor for validating time selections against experiment date ranges.

        This method validates that the provided time selection falls within the
        valid date range for the specified experiment. It fetches experiment metadata
        to determine the start and end years for the experiment.

        Args:
            value: The time selection value to validate (JSON string in validation mode)
            context: Context dictionary containing experiment_id

        Returns:
            AdvisedValue: Either Okay() for valid selections or AdvisedDateRange/AdvisedDict for discovery

        Raises:
            AdvisedError: If experiment_id is invalid or no data is available
        """
        # Are we in validation mode?
        is_validation = not isinstance(value, Advise)
        self._logger.info(
            f"Disovery started for selection, is_validation: {is_validation}"
        )

        # Get experiment_id from context and experiment data from cache
        experiment_id = self._from_context(context, "experiment_id")
        experiment_dict = await self._cache_experiment_data()

        if not experiment_id or experiment_id not in experiment_dict.keys():
            raise AdvisedError("Valid experiment_id is required")

        # Get time range for the experiment data

        # A helper function to convert year string to int or list of ints
        def parse_year(year: str) -> str | list[str] | None:
            if year == "":
                return None
            if " or " in year:
                return [y for y in year.split(" or ")]
            return year

        start_year = parse_year(experiment_dict[experiment_id]["start_year"])
        end_year = parse_year(experiment_dict[experiment_id]["end_year"])

        if start_year is None or end_year is None:
            raise AdvisedError("No data available")

        assert isinstance(start_year, str), "We expect start_year to be a string"

        if is_validation:
            # First, convert all strings to dates
            if value is None:
                return AdvisedError("Value is required for validation")
            value = dateutil.parser.parse(value["time"])
            # Parse year string (e.g., "1856") to first day of that year
            start_year = datetime.datetime(int(start_year), 1, 1)
            if isinstance(end_year, list):
                end_year = str(max([int(i) for i in end_year]))
            end_year = datetime.datetime(int(end_year), 12, 31)
            if value < start_year or value > end_year:
                return AdvisedError("Value is not within the valid date range")
            return Okay()

        if isinstance(end_year, list):
            range_result = AdvisedOneOf(
                values=[AdvisedDateRange(start_year, end) for end in end_year]
            )
        else:
            range_result = AdvisedDateRange(start_year, end_year)

        result = AdvisedDict(
            {
                "time": range_result,
            },
            allow_extras=True,
        )

        self._logger.info(f"Disovery finished for selection: {result}")
        return result

    def body_impl(
        self,
        variables: list[str],
        selection: dict[str, str],
        experiment_id: str,
        source_id: str,
        table_id: str,
        variant_label: str,
        invalidate_cache: bool,
    ) -> xarray.Dataset:
        """
        Implementation method for loading CMIP6 data.

        This method loads CMIP6 climate model data for the specified parameters.
        It handles caching, data retrieval from ESGF, and dataset preprocessing.
        The method uses Earth2Studio's CMIP6 implementation with custom caching.

        Args:
            variables: List of CMIP6 variable names to load
            selection: Dictionary containing time selection parameters
            experiment_id: CMIP6 experiment identifier (e.g., "historical")
            source_id: CMIP6 source identifier (climate model, e.g., "MPI-ESM1-2-LR")
            table_id: CMIP6 table identifier (frequency, e.g., "day")
            variant_label: CMIP6 variant label (ensemble member, e.g., "r1i1p1f1")
            invalidate_cache: Whether to bypass cache and reload data

        Returns:
            xarray.Dataset: Loaded CMIP6 dataset with variables as separate data variables

        Raises:
            ValueError: If time selection is missing or invalid
        """

        self._logger.info(
            f"Loading CMIP6 data with variables: {variables}, selection: {selection} \
              ({experiment_id}, {source_id}, {table_id}, {variant_label})"
        )
        if not selection or "time" not in selection:
            raise ValueError(
                "The CMIP6Data adapter requires a specific time selection. \
                            Please supply something like {'time': '2024-01-31', ...}"
            )
        params_hash = self._calculate_params_hash(
            variables, selection, experiment_id, source_id, table_id, variant_label
        )
        if self._has_cache and not invalidate_cache:
            cached_ds = self._cache.load_value(params_hash)
            if cached_ds:
                self._logger.info(f"Loaded cached dataset {cached_ds}")
                return cached_ds

        self._logger.info(
            f"Opening CMIP6 dataset from {experiment_id}, {source_id}, {table_id}, {variant_label}"
        )
        date = dateutil.parser.parse(selection["time"])

        # Load the data using Earth2Studio. Use available date range for the dataset to shorten download time
        cmip6 = CMIP6(
            experiment_id,
            source_id,
            table_id,
            variant_label,
            file_start=selection["time"],
            file_end=selection["time"],
        )
        # Switch to our catalog (caching-related) and cache configuration (E2Studio can reset the cache config)
        self._configure_esgf_cache()
        cmip6.catalog = self._esgf_catalog
        da = cmip6(time=date, variable=variables)

        # Create a dataset with each variable as a separate data variable
        # This is needed because Earth2Studio returns a DataArray with a variable dimension
        ds = da.to_dataset(dim="variable")
        # Display ESGF session log
        self._logger.info(f"ESGF session log: {self._esgf_catalog.session_log()}")
        # Rename the data variable to use the actual variable names
        ds = ds.rename({var: var for var in variables})

        # Flip latitude dimension from -90 to 90 to 90 to -90
        if "lat" in ds.dims:
            ds = ds.reindex(lat=ds.lat[::-1])
        elif "latitude" in ds.dims:
            ds = ds.reindex(latitude=ds.latitude[::-1])

        if self._cache:
            # Write to cache
            self._logger.info(f"Writing dataset to cache {params_hash}")
            self._cache.write_value(params_hash, ds)

        self._logger.info(f"LoadedCMIP6 dataset: {ds}")

        return ds

    async def body(
        self,
        variables: list[str],
        selection: dict[str, str],
        experiment_id: str | None,
        source_id: str | None,
        table_id: str | None,
        variant_label: str | None,
        invalidate_cache: bool | None,
    ) -> xarray.Dataset:
        """
        Main entry point for loading CMIP6 data.

        This is the primary method called by the DFM framework to load CMIP6 data.
        It provides default values for optional parameters and handles the async
        execution of the data loading implementation.

        Args:
            variables: List of CMIP6 variable names to load
            selection: JSON string containing time selection parameters
            experiment_id: CMIP6 experiment identifier (defaults to "historical")
            source_id: CMIP6 source identifier (defaults to "MPI-ESM1-2-LR")
            table_id: CMIP6 table identifier (defaults to "day")
            variant_label: CMIP6 variant label (defaults to "r1i1p1f1")
            invalidate_cache: Whether to bypass cache (defaults to False)

        Returns:
            xarray.Dataset: Loaded CMIP6 dataset

        Raises:
            Exception: Re-raises any exception from the implementation method
        """
        # Handle default values
        if experiment_id is None:
            experiment_id = "historical"
        if source_id is None:
            source_id = "MPI-ESM1-2-LR"
        if table_id is None:
            table_id = "day"
        if variant_label is None:
            variant_label = "r1i1p1f1"
        if invalidate_cache is None:
            invalidate_cache = False
        try:
            return await asyncio.to_thread(
                self.body_impl,
                variables,
                selection,
                experiment_id,
                source_id,
                table_id,
                variant_label,
                invalidate_cache,
            )
        except Exception as e:
            self._logger.error(f"Error loading CMIP6 data: {e}")
            raise e
