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

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import dateutil.parser
import metpy.xarray  # noqa: F401
import numpy as np

# needs to be imported to load rio and metpy
import rioxarray  # noqa: F401
import xarray
from herbie.core import Herbie

from nv_dfm_core.exec import Provider, Site
from nv_dfm_core.exec.discovery import (
    AdvisedDict,
    AdvisedLiteral,
    AdvisedOneOf,
    AdvisedSubsetOf,
    field_advisor,
)
from nv_dfm_lib_weather.xarray.cache import XArrayCache


class HrrrProduct(Enum):
    """
    Enum representing different HRRR product types:
    - SFC: Surface products (2m temperature, 10m winds, etc)
    - PRS: Pressure level products (temperature, winds at different pressure levels)
    - NAT: Native model level products
    - SUBH: Sub-hourly products
    """

    SFC = "sfc"
    PRS = "prs"
    NAT = "nat"
    SUBH = "subh"


@dataclass
class HrrrDataRequest:
    """
    Data class representing mapping from ERA5-style variable to HRRR-style variable.

    Attributes:
        era5_var: The ERA5 variable name
        product: The HRRR product type (surface, pressure level, etc)
        search_string: String pattern to search for in HRRR variable names
        fxx: Forecast hour (0 for analysis, >0 for forecast)
        hrrr_var: The HRRR variable name
    """

    era5_var: str
    product: HrrrProduct
    search_string: str
    hrrr_var: str


def generate_era5_to_hrrr_variables() -> dict[str, HrrrDataRequest]:
    # Surface variables
    variables = {
        "u10m": HrrrDataRequest("u10m", HrrrProduct.SFC, "UGRD:10 m above", "u10"),
        "v10m": HrrrDataRequest("v10m", HrrrProduct.SFC, "VGRD:10 m above", "v10"),
        "u100m": None,
        "v100m": None,
        "t2m": HrrrDataRequest("t2m", HrrrProduct.SFC, "TMP:2 m above", "t2m"),
        "r2m": HrrrDataRequest("r2m", HrrrProduct.SFC, "RH:2 m above", "r2m"),
        "sp": HrrrDataRequest("sp", HrrrProduct.SFC, "PRES:surface", "sp"),
        "msl": HrrrDataRequest("msl", HrrrProduct.SFC, "MSLMA", "mslma"),
        "tcwv": HrrrDataRequest("tcwv", HrrrProduct.SFC, "PWAT", "pwat"),
    }

    # Pressure level variables
    pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    var_mappings = {
        "u": ("UGRD", "u"),  # (HRRR search string prefix, HRRR variable name)
        "v": ("VGRD", "v"),
        "z": ("HGT", "gh"),
        "t": ("TMP", "t"),
        "q": ("SPFH", "q"),
        "r": ("RH", "r"),
    }

    # Generate pressure level variables
    for var, (search_prefix, hrrr_var) in var_mappings.items():
        for level in pressure_levels:
            era5_name = f"{var}{level}"
            search_string = f"{search_prefix}:{level} mb"
            variables[era5_name] = HrrrDataRequest(
                era5_name, HrrrProduct.PRS, search_string, hrrr_var
            )

    return variables


class LoadHrrrEra5Data:
    DATE_FORMAT = "%Y-%m-%dT%H:00"
    ERA5_TO_HRRR_VARIABLES: dict[str, HrrrDataRequest] = (
        generate_era5_to_hrrr_variables()
    )

    def __init__(  # pylint: disable=useless-parent-delegation
        self, site: Site, provider: Provider | None
    ):
        self._site = site
        self._provider = provider
        self._logger = site.dfm_context.logger
        cache_storage = site.cache_storage(subpath=self.__class__.__name__)
        self._cache = XArrayCache(cache_storage, file_prefix="era5_hrrr_dataset")

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

    @field_advisor("variables", order=0)
    async def available_variables(self, _value, _context):
        return AdvisedOneOf(
            [
                AdvisedLiteral("*"),
                AdvisedSubsetOf(LoadHrrrEra5Data.ERA5_TO_HRRR_VARIABLES.keys()),  # type: ignore
            ]
        )

    @field_advisor("selection")
    async def valid_selections(self, _value, _context):
        self._logger.info("Disovery started for selection")
        # Conservatively advise time two hors earlier
        first_date = (dateutil.parser.parse("2014-07-30T00:00")).strftime(
            self.DATE_FORMAT
        )
        last_date = datetime.now(timezone.utc) - timedelta(hours=2)
        advice_d1 = {
            # we have 0..48 steps every 6 hours
            "time": {
                "first_date": first_date,
                "last_date": last_date.replace(
                    hour=last_date.hour - (last_date.hour % 6)
                ).strftime(self.DATE_FORMAT),
                "frequency": 6,
            },
            "step": {"min": 0, "max": 48},
        }
        advice_d2 = {
            # or hourly, but then only 18 steps
            "time": {
                "first_date": first_date,
                "last_date": last_date.strftime(self.DATE_FORMAT),
                "frequency": 1,
            },
            "step": {"min": 0, "max": 18},
        }
        result = AdvisedOneOf(
            values=[
                AdvisedLiteral(None),
                AdvisedDict(advice_d1, allow_extras=True),
                AdvisedDict(advice_d2, allow_extras=True),
            ]
        )
        self._logger.info(f"Disovery finished for selection: {result}")
        return result

    async def body(
        self,
        variables: list[str] | Literal["*"] | None,
        selection: dict[str, str],
        invalidate_cache: bool | None = None,
    ) -> xarray.Dataset:
        """
        Load HRRR data from the HRRR model.

        Args:
            variables: List of variables to load
            selection: Selection parameters. Must include a 'time' parameter. May include a 'step' parameter.
            invalidate_cache: Whether to bypass cache and reload data
        """
        self._logger.info(
            f"Loading HRRR data with variables: {variables}, selection: {selection}"
        )

        ##########################################################
        # Retrieve and validate input params
        ##########################################################

        # Get list of variables to load - either all valid variables if '*' is specified,
        # or use the explicitly provided list of variables
        if variables is None or "*" in variables or variables == "*":
            candidates = [
                var
                for var, mapping in LoadHrrrEra5Data.ERA5_TO_HRRR_VARIABLES.items()
                if mapping is not None
            ]
        else:
            candidates = variables

        # Parse the time parameter
        if not selection or "time" not in selection:
            raise ValueError(
                "The LoadHrrrEra5Data adapter requires a specific time selection. \
                             Please supply something like {'time': '2024-01-31', ...}"
            )
        try:
            time = dateutil.parser.parse(selection["time"])
        except dateutil.parser.ParserError:
            raise ValueError(f"Invalid start time {selection['time']}.")

        # validate forecast step is within allowed range
        step = int(selection.get("step", 0))
        max_step = 48 if time.hour % 6 == 0 else 18
        if step < 0 or step > max_step:
            raise ValueError(
                f"Step value {step} outside of required range [0, {max_step}] for time {time}."
            )

        ##########################################################
        # Check Cache
        ##########################################################
        params_hash = self._calculate_params_hash(candidates, selection)
        if not invalidate_cache:
            cached_ds = self._cache.load_value(params_hash)
            if cached_ds:
                self._logger.info(f"Loaded cached dataset {cached_ds}")
                return cached_ds

        ##########################################################
        # Load Data
        ##########################################################
        # sort variables into their products
        hrrr_mappings_by_products: dict[HrrrProduct, list[HrrrDataRequest]] = {}
        for var in candidates:
            hrrr_mapping = LoadHrrrEra5Data.ERA5_TO_HRRR_VARIABLES.get(var)
            if not hrrr_mapping:
                raise ValueError(f"No HRRR mapping found for ERA5 variable: {var}")
            if hrrr_mapping.product not in hrrr_mappings_by_products:
                hrrr_mappings_by_products[hrrr_mapping.product] = []
            hrrr_mappings_by_products[hrrr_mapping.product].append(hrrr_mapping)

        if not hrrr_mappings_by_products:
            raise ValueError("No valid variables specified")

        # Log the variables for each product
        for product, mappings_list in hrrr_mappings_by_products.items():
            self._logger.info(
                "product %s, era5 variables: %s",
                product.value,
                [v.era5_var for v in mappings_list],
            )

        # Now group variables by their search strings to minimize file reads
        # and load the data for each search string into a separate data array.
        # We do it to avoid running into issues with different hypercubes.
        dataarrays: list[xarray.DataArray] = []

        herbie_save_dir = self._site.cache_storage(
            subpath=Path(self.__class__.__name__) / "herbie"
        )
        for product, mappings_list in hrrr_mappings_by_products.items():
            assert mappings_list
            # Open file for product and step
            herbie_args = {
                "date": time,
                "model": "hrrr",
                "product": product.value,
                "fxx": step,
                # make sure to get rid of the file:// prefix
                "save_dir": Path(herbie_save_dir.path).absolute(),
            }
            hr = Herbie(**herbie_args)
            # Group all mappings by search string
            search_groups: dict[str, list[HrrrDataRequest]] = {}
            for hrrr_mapping in mappings_list:
                search = hrrr_mapping.search_string
                if search not in search_groups:
                    search_groups[search] = []
                search_groups[search].append(hrrr_mapping)
            # Read data for each search string into a separate data array.
            for search, mappings_list in search_groups.items():
                self._logger.info("Using search group: %s", search)
                ds = hr.xarray(
                    search=search,
                    remove_grib=False,
                )
                if isinstance(ds, list):
                    raise ValueError(
                        "Selected variables belong to different hypercubes."
                    )
                for hrrr_mapping in mappings_list:
                    # rename the data array to the era5 variable name
                    da = ds.get(hrrr_mapping.hrrr_var)
                    assert da is not None
                    dataarrays.append(da.rename(hrrr_mapping.era5_var))

        # Merge data arrays into a single dataset
        # Create new dataset with coordinates from first data array
        first_da = dataarrays[0]
        ds = xarray.Dataset(
            coords={
                "time": first_da.time + first_da.step,
                "step": first_da.step,
                "latitude": first_da.latitude,
                "longitude": first_da.longitude,
                "valid_time": first_da.time + first_da.step,
                "gribfile_projection": first_da.gribfile_projection,
            }
        )

        # Add each data array to the dataset
        for da in dataarrays:
            ds[da.name] = da

        ##########################################################
        # Reproject Data
        ##########################################################

        ds.rio.write_crs(ds.herbie.crs, inplace=True)
        assert ds.rio.crs == ds.herbie.crs, (
            f"CRS Mismatch: rio.crs: {ds.rio.crs}, herbie.crs: {ds.herbie.crs}"
        )

        # set x and y as explicit coordinates. The herbie dataset mentions x and y dimensions but doesn't have coordinate arrays for them
        # Those numbers are "semi-magic", they are what the hrrr data contains when you download it without herbie. Since HRRR is very
        # localized it's okay to hard-code I think
        assert ds.x.size == 1799
        assert ds.y.size == 1059
        xcoords = np.linspace(-2697520.142521929, 2696479.857478071, 1799)
        ycoords = np.linspace(-1587306.152556665, 1586693.847443335, 1059)
        ds.coords["x"] = xcoords
        ds.coords["y"] = ycoords

        # shape of ds at this point is something like:
        # <xarray.Dataset> Size: 38MB
        # Dimensions:              (y: 1059, x: 1799)
        # Coordinates:
        #     time                 datetime64[ns] 8B 2024-01-31
        #     step                 timedelta64[ns] 8B 00:00:00
        #     heightAboveGround    float64 8B 10.0
        #     valid_time           datetime64[ns] 8B 2024-01-31
        #     gribfile_projection  object 8B None
        #     latitude             (y, x) float64 15MB 21.14 21.15 21.15 ... 47.85 47.84
        #     longitude            (y, x) float64 15MB 237.3 237.3 237.3 ... 299.0 299.1
        #   * y                    (y) float64 8kB -1.587e+06 -1.584e+06 ... 1.587e+06
        #   * x                    (x) float64 14kB -2.698e+06 -2.695e+06 ... 2.696e+06
        # Data variables:
        #     u10m                 (y, x) float32 8MB -2.898 -2.898 ... 0.8524 0.7899

        # reproject to WGS84; note that this will reproject x and y but drop latitude and longitude
        as_wgs84: xarray.Dataset = ds.rio.reproject(4326)
        # Rename x and y to lon and lat
        as_wgs84 = as_wgs84.rename({"x": "lon", "y": "lat"})
        # Remove some variables that are not needed
        as_wgs84 = as_wgs84.drop_vars(
            [
                "time",
                "heightAboveGround",
                "atmosphereSingleLayer",
                "valid_time",
                "gribfile_projection",
                "step",
            ],
            errors="ignore",
        )

        # Expand the dataset along the 'time' dimension
        # And shift longitudes into positive space
        as_wgs84 = as_wgs84.assign_coords(
            time=ds["time"].data, lon=(as_wgs84.lon % 360)
        ).expand_dims(dim="time")

        # apply the rest of the selector
        xarray_selection: dict[str, Any] = {
            k: (v if isinstance(v, list) else [v])
            for k, v in selection.items()
            if k not in ["time", "step"]
        }
        if len(xarray_selection) > 0:
            as_wgs84 = as_wgs84.sel(method="nearest", **xarray_selection)

        # reverse latitude index to mirror North-South
        as_wgs84 = as_wgs84.isel(lat=slice(None, None, -1))

        self._logger.info("Loaded dataset %s", as_wgs84)

        ##########################################################
        # Write to Cache
        ##########################################################
        self._cache.write_value(params_hash, as_wgs84)

        return as_wgs84
