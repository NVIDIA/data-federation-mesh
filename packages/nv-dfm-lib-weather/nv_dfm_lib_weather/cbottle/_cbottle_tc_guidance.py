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

from datetime import datetime

import torch
import xarray
from earth2studio.models.dx import CBottleTCGuidance

from nv_dfm_core.exec import Provider, Site
from nv_dfm_core.exec.discovery import AdvisedOneOf, field_advisor

from ..site import ModelCacheSite
from ..utils._device_utils import (
    available_devices,
    setup_device,
)


class CBottleTropicalCycloneGuidance:
    """
    CBottle Tropical Cyclone (TC) Guidance model for hurricane/typhoon-conditioned climate generation.

    This class provides functionality to generate climate fields conditioned on tropical cyclone
    track information using the CBottleTCGuidance model. Given the spatial and temporal coordinates
    of a tropical cyclone (hurricane, typhoon, cyclone), the model generates physically consistent
    climate fields that reflect the presence and influence of the tropical cyclone system.

    TC Guidance is useful for:
    - Generating climate fields around observed or forecasted tropical cyclone tracks
    - Creating realistic hurricane/typhoon environments for analysis and visualization
    - Studying the atmospheric conditions associated with tropical cyclones
    - Producing training data for cyclone-related models
    - Visualizing tropical cyclone impacts on regional climate

    The model uses a diffusion-based approach conditioned on the tropical cyclone position and
    timing to generate complete climate states that are physically consistent with the presence
    of an active tropical cyclone system.

    Attributes:
        _site: Model cache site for managing model storage and retrieval
        _provider: DFM provider instance for the service
        _logger: Logger instance for tracking operations and debugging
    """

    def __init__(self, site: Site, provider: Provider | None):
        """
        Initialize the CBottle Tropical Cyclone Guidance model.

        Args:
            site: Site instance for model caching and management
            provider: DFM Provider instance, or None if running standalone
        """
        self._site: ModelCacheSite = site  # type: ignore[assignment]
        self._provider = provider
        self._logger = site.dfm_context.logger

    @field_advisor("device")
    async def valid_device(self, _value, _context):
        """
        Field advisor that provides available compute devices for model inference.

        This method queries the system for available compute devices (CPU and CUDA GPUs)
        and informs clients about which devices can be used for running the CBottle TC Guidance model.

        Args:
            _value: The current value (unused)
            _context: The discovery context (unused)

        Returns:
            AdvisedOneOf: A list of valid device identifiers (e.g., "cpu", "cuda", "cuda:0")
        """
        return AdvisedOneOf(available_devices())

    async def body(
        self,
        lat_coords: list[float],
        lon_coords: list[float],
        times: list[datetime],
        sampler_steps: int = 18,
        sigma_max: float = 200.0,
        seed: int | None = None,
        device: str | None = None,
        lat_lon: bool = True,
    ) -> xarray.Dataset:
        """
        Generate climate fields conditioned on tropical cyclone track information.

        This method uses the CBottleTCGuidance model to generate complete climate states that
        reflect the presence and influence of a tropical cyclone at specified locations and times.
        The model takes a tropical cyclone track (sequence of positions and times) as input and
        produces physically consistent climate fields showing the atmospheric conditions associated
        with the cyclone system.

        The generation process uses diffusion-based conditioning, where the tropical cyclone
        position acts as a constraint that guides the generation of realistic climate patterns
        around the storm system. This includes features like wind circulation, pressure gradients,
        moisture fields, and other phenomena associated with tropical cyclones.

        The sampler_steps and sigma_max parameters control the quality and characteristics of
        the diffusion process used to generate the climate fields.

        The generated data is cached using the model cache system to optimize performance
        across multiple requests with the same configuration.

        Args:
            lat_coords: List of latitude coordinates (in degrees, range: -90 to 90) for the
                       tropical cyclone positions along its track. One value per timestep.
                       Positive values are north, negative are south.
                       Example: [25.0, 26.5, 28.0] for a northward-moving storm.
            lon_coords: List of longitude coordinates (in degrees, range: -180 to 180 or 0 to 360)
                       for the tropical cyclone positions along its track. Must have the same
                       length as lat_coords. One value per timestep.
                       Example: [-80.0, -79.5, -79.0] for a westward-moving storm.
            times: List of datetime objects specifying when the tropical cyclone is at each
                  position. Must have the same length as lat_coords and lon_coords. The times
                  define the temporal evolution of the cyclone track.
                  Example: [datetime(2024, 9, 15, 0, 0), datetime(2024, 9, 15, 6, 0), ...]
            sampler_steps: Number of denoising steps in the diffusion sampling process. More steps
                         generally produce higher quality results but take longer to compute.
                         Typical range: 10-50 steps. (default: 18)
            sigma_max: Maximum noise level (variance) at the start of the diffusion process.
                      Controls the level of variability in the generated climate fields.
                      Typical range: 100-300. (default: 200.0)
            seed: Random seed for reproducibility. If provided, the same seed with identical
                 track information will produce identical climate fields. (default: None)
            device: Compute device to use for model inference. Options:
                   - "cuda" or "cuda:0": Use default CUDA device (requires GPU)
                   - "cuda:N": Use specific CUDA device N (e.g., "cuda:1" for second GPU)
                   - "cpu": Force CPU usage (slower but doesn't require GPU)
                   - None: Auto-select (CUDA if available, CPU otherwise)
                   Use the valid_device field advisor to discover available options.
                   (default: None)
            lat_lon: Whether to return data on a regular lat/lon grid (True) or a nested
                    HealPix grid (False). Most use cases should use lat/lon format.
                    (default: True)

        Returns:
            xarray.Dataset: Climate fields conditioned on the tropical cyclone track.

        Raises:
            RuntimeError: If model loading or generation fails.

        Example:
            >>> # Generate climate fields for Hurricane track
            >>> lats = [25.0, 26.5, 28.0, 29.5]  # Moving northward
            >>> lons = [-80.0, -79.5, -79.0, -78.5]  # Moving slightly eastward
            >>> times = [
            ...     datetime(2024, 9, 15, 0, 0),
            ...     datetime(2024, 9, 15, 6, 0),
            ...     datetime(2024, 9, 15, 12, 0),
            ...     datetime(2024, 9, 15, 18, 0),
            ... ]
            >>> result = await tc_guidance.body(
            ...     lat_coords=lats,
            ...     lon_coords=lons,
            ...     times=times,
            ...     sampler_steps=18,
            ...     sigma_max=200.0,
            ...     seed=42,
            ...     device="cuda:0"
            ... )
            >>> # Result contains climate fields showing the cyclone at each position
        """
        model: CBottleTCGuidance | None = None
        try:
            self._logger.info(
                f"CBottleTCGuidance parameters: sampler_steps={sampler_steps}, sigma_max={sigma_max}, seed={seed}"
            )

            # Determine the target device (CPU or CUDA) based on availability and user preference
            local_device = setup_device(device, self._logger)

            # Define model loader function for the cache system
            # This function is called only if the model is not already cached
            def _loader():
                package = CBottleTCGuidance.load_default_package()
                return CBottleTCGuidance.load_model(
                    package,
                    sampler_steps=sampler_steps,
                    sigma_max=sigma_max,
                    lat_lon=lat_lon,
                )

            # Load the CBottleTCGuidance model from cache or initialize it if not cached
            # The cache key includes lat_lon and diffusion parameters for proper configuration
            cache = self._site.model_cache
            cache_key = f"cbottle.tc_guidance:latlon={bool(lat_lon)}:steps={sampler_steps}:sigma={sigma_max}"
            model = cache.get_or_load(
                model_id=cache_key,
                loader=_loader,
                device=str(local_device),
            )
            self._logger.info(
                f"CBottleTCGuidance model loaded and moved to device: {local_device}"
            )

            # Define the expected dimension order based on the grid type (lat/lon vs HealPix)
            dimensions = (
                ["time", "variable", "lat", "lon"]
                if lat_lon
                else ["time", "variable", "hpx"]
            )

            # Convert tropical cyclone track coordinates to PyTorch tensors
            lat_coords_tensor = torch.Tensor(lat_coords)
            lon_coords_tensor = torch.Tensor(lon_coords)

            # Create the guidance tensor that encodes the tropical cyclone track information
            # This tensor acts as the conditioning signal for the diffusion model
            guidance_tensor, coords = model.create_guidance_tensor(
                lat_coords=lat_coords_tensor, lon_coords=lon_coords_tensor, times=times
            )

            guidance_tensor = guidance_tensor.to(local_device)

            # Set random seed for reproducibility if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                self._logger.info(f"Set CBottleTCGuidance seed to: {seed}")

            # Generate climate fields conditioned on the tropical cyclone track
            output, output_coords = model(guidance_tensor, coords)

            # Post-process the model output tensor to convert it to xarray format

            idx = list(output_coords.keys()).index("lead_time")
            output = output.squeeze(dim=idx)

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

            tc_da = xarray.DataArray(
                output.detach().cpu().numpy(),
                coords=coords,
                dims=dimensions,
            )

            tc_ds = tc_da.to_dataset(dim="variable")
            self._logger.info(
                f"Tropical cyclone guidance completed. Output variables: {list(tc_ds.data_vars.keys())}"
            )
            return tc_ds
        except Exception as e:
            self._logger.error(
                f"Error performing CBottle tropical cyclone guidance: {e}"
            )
            raise e
