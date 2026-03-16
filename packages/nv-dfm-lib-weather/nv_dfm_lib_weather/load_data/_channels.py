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

import numpy as np


def gen_era5_channels():
    """
    Generate a list of ERA5 channel names for weather data.

    This function creates a comprehensive list of ERA5 weather channels including
    surface-level variables and pressure-level variables at multiple altitudes.

    Returns:
        List of ERA5 channel names including:
        - Surface variables: u10m, v10m, u100m, v100m, t2m, r2m, sp, msl, tcwv
        - Pressure-level variables: u, v, z, t, r, q at levels 50-1000 hPa
    """
    base_channels = [
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
    prefix = ["u", "v", "z", "t", "r", "q"]
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    channels = [f"{p}{lev}" for p in prefix for lev in levels]
    return base_channels + channels


def gen_gfs_channels():
    """
    Generate a list of GFS channel names and their parameters for weather data.

    This function creates a comprehensive list of GFS weather channels including
    surface-level variables and pressure-level variables. Returns tuples where
    the first element is the channel name and the second is optional parameters.

    Returns:
        List of tuples containing (channel_name, parameters) where:
        - Surface variables: ugrd10m, vgrd10m, ugrd100m, vgrd100m, tmp2m, rh2m, pressfc, prmslmsl, pwatclm
        - Pressure-level variables: ugrdprs, vgrdprs, hgtprs, tmpprs, rhprs, spfhprs at levels 50-1000 hPa
    """
    base_channels = [
        "ugrd10m",
        "vgrd10m",
        "ugrd100m",
        "vgrd100m",
        "tmp2m",
        "rh2m",
        "pressfc",
        "prmslmsl",
        "pwatclm",
    ]
    base_channel_pairs = [(c, None) for c in base_channels]
    prefix = ["ugrdprs", "vgrdprs", "hgtprs", "tmpprs", "rhprs", "spfhprs"]
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    channel_pairs = [(p, {"lev": lev}) for p in prefix for lev in levels]
    return base_channel_pairs + channel_pairs


def gen_gfs_s3_channels():
    """
    Generate a list of GFS S3 channel names for weather data.

    This function creates a comprehensive list of GFS weather channels in S3 format
    including surface-level variables and pressure-level variables at multiple altitudes.
    The S3 format uses descriptive names with level information.

    Returns:
        List of GFS S3 channel names including:
        - Surface variables: UGRD, VGRD, TMP, RH, PRES, PRMSL, PWAT at various heights
        - Pressure-level variables: UGRD, VGRD, HGT, TMP, RH, SPFH at levels 50-1000 mb
    """
    base_channels = [
        "UGRD::10 m above ground",
        "VGRD::10 m above ground",
        "UGRD::100 m above ground",
        "VGRD::100 m above ground",
        "TMP::2 m above ground",
        "RH::2 m above ground",
        "PRES::surface",
        "PRMSL::mean sea level",
        "PWAT::entire atmosphere (considered as a single layer)",
    ]
    prefix = ["UGRD", "VGRD", "HGT", "TMP", "RH", "SPFH"]
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    channels = [f"{p}::{lev} mb" for p in prefix for lev in levels]
    return base_channels + channels


# ERA5 coordinate grids
ERA5_CHANNEL_LON = np.arange(
    0, 360, 0.25
).tolist()  # Longitude grid: 0 to 359.75 degrees
ERA5_CHANNEL_LAT = np.arange(
    90, -90.25, -0.25
).tolist()  # Latitude grid: 90 to -90 degrees

# Channel lists
ERA5_CHANNELS = gen_era5_channels()
ERA5_CHANNELS_PBSS = [
    c for c in ERA5_CHANNELS if (c != "r2m" and not c.startswith("q"))
]
GFS_CHANNELS = gen_gfs_channels()
GFS_S3_CHANNELS = gen_gfs_s3_channels()

# Mapping dictionaries between different channel formats
ERA5_TO_GFS_MAP = dict(zip(ERA5_CHANNELS, GFS_CHANNELS))
ERA5_TO_GFS_S3_MAP = dict(zip(ERA5_CHANNELS, GFS_S3_CHANNELS))

# Validation: ensure all channel lists have the same length
# easy to screw up the zips by adding vars to one but not the other
assert len(ERA5_CHANNELS) == len(GFS_CHANNELS)
assert len(ERA5_CHANNELS) == len(ERA5_TO_GFS_MAP)
assert len(ERA5_CHANNELS) == len(ERA5_TO_GFS_S3_MAP)
