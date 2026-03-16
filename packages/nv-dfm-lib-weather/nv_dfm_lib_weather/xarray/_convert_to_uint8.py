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

import numpy as np
import xarray

from nv_dfm_core.exec import Provider, Site
from nv_dfm_lib_weather.xarray.cache import XArrayCache
from nv_dfm_lib_weather.xarray.schema import (
    ConvertToUint8InputSchema,
    ConvertToUint8OutputSchema,
)


class ConvertToUint8:
    """
    A processor for converting xarray datasets to uint8 format.

    This class provides functionality to convert xarray datasets to uint8 format
    with normalization, dimension handling, and caching support.
    """

    def __init__(self, site: Site, provider: Provider | None):
        """
        Initialize the ConvertToUint8 processor.

        Args:
            site: DFM site configuration
            provider: Optional provider configuration
        """
        self._site = site
        self._provider = provider
        self._logger = logging.getLogger(__name__)
        cache_storage = site.cache_storage(subpath=self.__class__.__name__)
        self._cache = XArrayCache(cache_storage, file_prefix="dataset")

    async def body_impl(
        self,
        data: xarray.Dataset,
        time_dimension: str,
        xydims: list[str],
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> xarray.Dataset:
        """
        Convert xarray dataset to uint8 format with normalization.

        This method processes the input dataset by validating it, handling dimensions,
        normalizing values, and converting to uint8 format.

        Args:
            data: Input xarray dataset to convert
            time_dimension: Name of the time dimension
            xydims: List of x and y dimension names
            min_value: Optional minimum value for normalization
            max_value: Optional maximum value for normalization

        Returns:
            xarray.Dataset converted to uint8 format with normalization metadata
        """
        self._logger.info("Converting to uint8", data)

        # validate the input data against the schema
        schema = ConvertToUint8InputSchema(time_dimension, xydims)
        schema.validate(data, strict=False)  # Allow extra dimensions

        # sum all dimensions except for the xydims requested
        sum_over_dims = [
            dim
            for dim in data.dims
            if dim not in (time_dimension, xydims[0], xydims[1])
        ]

        if len(sum_over_dims) > 0:
            data = data.sum(sum_over_dims)

        assert len(data.dims) <= 3

        # transpose to the requested x and y format, if not already
        data = data.transpose(
            time_dimension, xydims[0], xydims[1], missing_dims="ignore"
        )

        # now, normalize the dataset
        as_arr = data.to_array().to_numpy()
        the_min = min_value if min_value is not None else as_arr.min().item()
        the_max = max_value if max_value is not None else as_arr.max().item()
        data = ((data - the_min) / (the_max - the_min)).clip(min=0, max=1)

        data_uint8 = (data * 255.0).astype(np.uint8)
        data_uint8.attrs["data_min"] = np.array([the_min])
        data_uint8.attrs["data_max"] = np.array([the_max])

        # validate the output data against the schema
        schema = ConvertToUint8OutputSchema(
            time_dimension, xydims, data_uint8.data_vars
        )
        schema.validate(data_uint8, strict=False)

        self._logger.info("Returning data", data_uint8)

        return data_uint8

    async def body(
        self,
        data: xarray.Dataset,
        time_dimension: str,
        xydims: list[str],
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> xarray.Dataset:
        """
        Async wrapper for converting xarray dataset to uint8 format.

        This method provides an async interface to the uint8 conversion functionality
        with error handling and logging.

        Args:
            data: Input xarray dataset to convert
            time_dimension: Name of the time dimension
            xydims: List of x and y dimension names
            min_value: Optional minimum value for normalization
            max_value: Optional maximum value for normalization

        Returns:
            xarray.Dataset converted to uint8 format

        Raises:
            Exception: If there's an error during conversion
        """
        try:
            return await self.body_impl(
                data,
                time_dimension,
                xydims,
                min_value,
                max_value,
            )
        except Exception as e:
            self._logger.error(f"Error converting to uint8: {e}")
            raise e
