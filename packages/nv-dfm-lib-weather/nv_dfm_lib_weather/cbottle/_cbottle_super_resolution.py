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

import numpy as np
import torch
import xarray
from earth2studio.data.utils import fetch_data
from earth2studio.models.dx.cbottle_sr import CBottleSR
from earth2studio.utils.coords import map_coords

from nv_dfm_core.exec import Provider
from nv_dfm_core.exec.discovery import AdvisedOneOf, field_advisor

from ..site import ModelCacheSite
from ..utils._device_utils import (
    available_devices,
    setup_device,
)


class CbottleSuperResolution:
    """
    CBottle super-resolution model for climate data upscaling.

    This class provides functionality to perform super-resolution (spatial upscaling) on
    low-resolution climate datasets using the CBottleSR diffusion model. The model enhances
    spatial resolution while preserving physical consistency and generating realistic
    fine-scale features.

    Super-resolution is useful for:
    - Enhancing coarse climate model outputs to higher resolutions
    - Downscaling global climate data for regional studies
    - Generating high-resolution details from low-resolution observations
    - Creating detailed climate visualizations
    - Bridging the gap between different resolution datasets

    The model uses a diffusion-based approach to generate physically plausible high-resolution
    climate fields that are consistent with the low-resolution input data.

    Attributes:
        _site: Model cache site for managing model storage and retrieval
        _provider: DFM provider instance for the service
        _logger: Logger instance for tracking operations and debugging
    """

    def __init__(self, site: ModelCacheSite, provider: Provider | None):
        """
        Initialize the CBottle super-resolution model.

        Args:
            site: ModelCacheSite instance for model caching and management
            provider: DFM Provider instance, or None if running standalone
        """
        self._site: ModelCacheSite = site
        self._provider = provider
        self._logger = site.dfm_context.logger

    @field_advisor("device")
    async def valid_device(self, _value, _context):
        """
        Field advisor that provides available compute devices for model inference.

        This method queries the system for available compute devices (CPU and CUDA GPUs)
        and informs clients about which devices can be used for running the CBottle super-resolution model.

        Args:
            _value: The current value (unused)
            _context: The discovery context (unused)

        Returns:
            AdvisedOneOf: A list of valid device identifiers (e.g., "cpu", "cuda", "cuda:0")
        """
        return AdvisedOneOf(available_devices())

    async def body(
        self,
        dataset: xarray.Dataset,
        output_resolution: tuple[int, int] = (2161, 4320),
        super_resolution_window: tuple[int, int, int, int] | None = None,
        sampler_steps: int = 18,
        sigma_max: float = 800.0,
        seed: int | None = None,
        device: str | None = None,
        lat_lon: bool = True,
    ) -> xarray.Dataset:
        """
        Perform climate data super-resolution using the CBottleSR diffusion model.

        This method takes a low-resolution climate dataset and generates a high-resolution version
        using the CBottleSR model. The model employs a diffusion-based approach to enhance spatial
        resolution while generating physically plausible fine-scale details and features that are
        consistent with the low-resolution input.

        The super-resolution process uses an iterative denoising diffusion approach, starting from
        noise and gradually refining it to produce realistic high-resolution climate data conditioned
        on the low-resolution input. The sampler_steps and sigma_max parameters control the quality
        and characteristics of this diffusion process.

        Optionally, super-resolution can be applied to a specific geographic window (subregion)
        rather than the entire global domain, which can improve performance for regional applications.

        The generated data is cached using the model cache system to optimize performance across
        multiple requests with the same model configuration.

        Args:
            dataset: Input low-resolution xarray.Dataset containing climate data. Must have dimensions:
                    - time: temporal dimension (one or more timesteps)
                    - lat: latitude dimension
                    - lon: longitude dimension
                    The dataset must contain all variables required by the CBottleSR model.
            output_resolution: Target high-resolution grid size as (nlat, nlon) tuple.
                             Common values:
                             - (2161, 4320): 0.0833° resolution (~9.25 km at equator)
                             - (1441, 2880): 0.125° resolution (~13.9 km at equator)
                             - (721, 1440): 0.25° resolution (~27.8 km at equator)
                             (default: (2161, 4320))
            super_resolution_window: Optional geographic bounding box to limit super-resolution to
                                    a specific region, specified as (lat_south, lon_west, lat_north, lon_east)
                                    in degrees. For example: (-20, -180, 20, 180) for a tropical band.
                                    If None, super-resolution is applied globally. (default: None)
            sampler_steps: Number of denoising steps in the diffusion sampling process. More steps
                         generally produce higher quality results but take longer to compute.
                         Typical range: 10-50 steps. (default: 18)
            sigma_max: Maximum noise level (variance) at the start of the diffusion process.
                      Higher values allow for more fine-scale detail generation but may introduce
                      artifacts. Typical range: 400-1200 for super-resolution. (default: 800.0)
            seed: Random seed for reproducibility. (default: None)
            device: Compute device to use for model inference. Options:
                   - "cuda" or "cuda:0": Use default CUDA device (requires GPU)
                   - "cuda:N": Use specific CUDA device N (e.g., "cuda:1" for second GPU)
                   - "cpu": Force CPU usage (slower but doesn't require GPU)
                   - None: Auto-select (CUDA if available, CPU otherwise)
                   Use the valid_device field advisor to discover available options.
                   (default: None)
            lat_lon: Whether to return data on a regular lat/lon grid (True) or a nested
                    HealPix grid (False). Currently, only lat/lon is supported for super-resolution.
                    (default: True)

        Returns:
            xarray.Dataset: High-resolution climate dataset with the same variables as the input
                but at the enhanced spatial resolution.

        Raises:
            ValueError: If lat_lon is False (HealPix not yet supported), dataset is None, empty,
                       missing required coordinates, or missing variables required by the model.
            RuntimeError: If model loading or inference fails.

        Example:
            >>> # Upscale a coarse 1° dataset to high-resolution 0.0833°
            >>> low_res_ds = xr.Dataset({...})  # 180x360 grid (1° resolution)
            >>> result = await sr.body(
            ...     dataset=low_res_ds,
            ...     output_resolution=(2161, 4320),  # 0.0833° resolution
            ...     super_resolution_window=None,  # Global
            ...     sampler_steps=20,
            ...     sigma_max=800.0,
            ...     seed=42,
            ...     device="cuda:0"
            ... )
            >>> # Result is now a 2161x4320 high-resolution dataset
        """
        cbottle_sr_model = None
        try:
            # Validate inputs
            if not lat_lon:
                raise ValueError("HPX is not yet supported for CBottle SR")

            if dataset is None:
                raise ValueError("Input dataset cannot be None")

            if len(dataset.data_vars) == 0:
                raise ValueError(
                    "Input dataset must contain at least one data variable"
                )

            if "time" not in dataset.coords:
                raise ValueError("Input dataset must have a 'time' coordinate")

            times = dataset.coords["time"].values
            if len(times) == 0:
                raise ValueError("Input dataset must have at least one time step")

            if not np.issubdtype(times.dtype, np.datetime64):
                times = times.astype("datetime64[ns]")

            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                self._logger.info(f"Set CBottle SR seed to: {seed}")

            local_device = setup_device(device, self._logger)

            # Define model loader function for the cache system
            # This function is called only if the model is not already cached
            def _loader():
                package = CBottleSR.load_default_package()
                return CBottleSR.load_model(
                    package,
                    output_resolution=output_resolution,
                    super_resolution_window=super_resolution_window,
                    sampler_steps=sampler_steps,
                    sigma_max=sigma_max,
                )

            # Load the CBottleSR model from cache or initialize it if not cached
            # The cache key includes resolution, window, and diffusion parameters for proper configuration
            cache = self._site.model_cache
            cache_key = (
                f"cbottle.sr:res={output_resolution}:window={super_resolution_window}:"
                f"steps={sampler_steps}:sigma={sigma_max}"
            )
            cbottle_sr_model = cache.get_or_load(
                model_id=cache_key,
                loader=_loader,
                device=str(local_device),
            )
            self._logger.info(
                f"CBottleSR model loaded on {local_device}; output_resolution={cbottle_sr_model.output_resolution}, window={super_resolution_window}"
            )

            # Query the model for its required input variables and verify they exist in the dataset
            required_variables = list(cbottle_sr_model.input_coords()["variable"])  # type: ignore[index]
            missing_vars = set(required_variables) - set(dataset.data_vars.keys())
            if missing_vars:
                raise ValueError(
                    "Dataset is missing variables required for super-resolution: "
                    f"{sorted(missing_vars)}"
                )

            # Define the expected dimension order based on the grid type (lat/lon vs HealPix)
            dimensions = (
                ["time", "variable", "lat", "lon"]
                if lat_lon
                else ["time", "variable", "hpx"]
            )

            # Define the super-resolution computation function
            # This performs the actual diffusion-based upscaling conditioned on low-resolution input
            def perform_super_resolution():
                # Convert the xarray Dataset to a DataArray and reorder dimensions to match expected format
                data_array = dataset.to_array(dim="variable").transpose(*dimensions)

                # Convert to PyTorch tensor format on the target device, selecting only required variables
                x, coords = fetch_data(
                    lambda timestamps, variable: data_array,  # type: ignore[arg-type]
                    times,
                    required_variables,
                    device=local_device,
                )

                # Map the data coordinates to match the model's expected input coordinate system
                x, coords = map_coords(x, coords, cbottle_sr_model.input_coords())

                # Execute the super-resolution model
                output, output_coords = cbottle_sr_model(x, coords)
                return output, output_coords

            # Run super-resolution in a background thread since it's a synchronous, compute-intensive operation
            output_tensor, output_coords = await asyncio.to_thread(
                perform_super_resolution
            )

            # Post-process the model output tensor to convert it to xarray format

            # Average over the batch dimension if present (multiple samples are averaged for stability)
            if "batch" in output_coords:
                idx = list(output_coords.keys()).index("batch")
                output_tensor = torch.mean(output_tensor, dim=idx, keepdim=False)

            # Remove the lead_time dimension as it's not needed in the final output
            output_tensor = output_tensor.squeeze(
                dim=list(output_coords.keys()).index("lead_time")
            )

            # Build coordinate dictionary based on the grid type (lat/lon or HealPix)
            coords = (
                {
                    "time": output_coords["time"],
                    "variable": output_coords["variable"],
                    "lat": output_coords["lat"],
                    "lon": output_coords["lon"],
                }
                if lat_lon
                else {
                    "time": output_coords["time"],
                    "variable": output_coords["variable"],
                    "hpx": output_coords["hpx"],
                }
            )

            # Create an xarray DataArray from the high-resolution output tensor
            sr_da = xarray.DataArray(
                output_tensor.detach().cpu().numpy(),
                coords=coords,
                dims=dimensions,
            )

            sr_ds = sr_da.to_dataset(dim="variable")
            self._logger.info(
                f"Super-resolution completed. Output variables: {list(sr_ds.data_vars.keys())}"
            )
            return sr_ds
        except Exception as e:
            self._logger.error(f"Error performing CBottle super-resolution: {e}")
            raise e
