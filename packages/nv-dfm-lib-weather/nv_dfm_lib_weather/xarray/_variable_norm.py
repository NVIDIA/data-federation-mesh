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

import xarray

from nv_dfm_core.exec import Provider, Site
from nv_dfm_core.exec.discovery import (
    AdvisedLiteral,
    AdvisedOneOf,
    AdvisedSubsetOf,
    field_advisor,
)

from ..xarray.cache import XArrayCache


class VariableNorm:
    """
    A class for computing p-norm of variables in xarray datasets.

    This class provides functionality to compute various p-norms (L1, L2, L3, etc.)
    of specified variables in an xarray dataset, with caching support.
    """

    def __init__(self, site: Site, provider: Provider | None):
        """
        Initialize the VariableNorm processor.

        Args:
            site: DFM site configuration
            provider: Optional provider configuration
        """
        self._site = site
        self._provider = provider
        self._logger = site.dfm_context.logger
        cache_storage = site.cache_storage(subpath=self.__class__.__name__)
        self._cache = XArrayCache(cache_storage, file_prefix="variable_norm_dataset")

    @field_advisor("variables")
    async def available_variables(self, _value, _context):
        """Provide advice for available variables in the dataset."""
        # This would typically be discovered from the actual dataset
        # For now, return a generic subset pattern
        return AdvisedSubsetOf([])

    @field_advisor("output_name")
    async def valid_output_names(self, _value, _context):
        """Provide advice for output variable names."""
        return AdvisedLiteral("norm")

    @field_advisor("p")
    async def valid_p_values(self, _value, _context):
        """Provide advice for p values in the norm calculation."""
        return AdvisedOneOf([AdvisedLiteral(1), AdvisedLiteral(2), AdvisedLiteral(3)])

    def body_impl(
        self,
        data: xarray.Dataset,
        variables: list[str],
        p: float = 2.0,
        output_name: str = "norm",
    ) -> xarray.Dataset:
        """Compute the p-norm of specified variables in the dataset."""
        self._logger.info(f"Computing {p}-norm of variables {variables}")

        # Validate input variables exist
        missing_vars = [var for var in variables if var not in data.data_vars]
        if missing_vars:
            raise ValueError(f"Variables {missing_vars} not found in dataset")

        # Stack variables and compute norm
        stacked_vars = xarray.concat([data[var] for var in variables], dim="variable")
        norm_values = (abs(stacked_vars) ** p).sum(dim="variable") ** (1.0 / p)
        norm_values.name = output_name
        output_ds = norm_values.to_dataset()

        # Compute min and max values
        # Note that norm_values can be a dask array, which does not support .item(), so we use .data
        data_min = norm_values.min().data
        data_max = norm_values.max().data

        # Add some metadata just in case
        output_ds[output_name].attrs.update(
            {
                "description": f"P-norm (p={p}) of variables {variables}",
                "variables_used": variables,
                "p_value": p,
                "data_min": data_min,
                "data_max": data_max,
            }
        )

        self._logger.info(f"Computed norm with shape {output_ds[output_name].shape}")
        return output_ds

    async def body(
        self,
        data: xarray.Dataset,
        variables: list[str],
        p: float = 2.0,
        output_name: str = "norm",
    ) -> xarray.Dataset:
        """Async wrapper for the norm computation."""
        try:
            result = await asyncio.to_thread(
                self.body_impl, data, variables, p, output_name
            )
            self._logger.info(f"Successfully computed variable norm: {result}")
            return result
        except Exception as e:
            self._logger.error(f"Error computing variable norm: {e}")
            raise e
