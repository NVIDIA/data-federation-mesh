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
from typing import Any

import numpy as np
import torch
import xarray
from earth2studio.data.utils import fetch_data

# SFNO lives under px (prognostic)
from earth2studio.models.px.sfno import SFNO as SFNOModel
from earth2studio.utils.coords import map_coords

from nv_dfm_core.exec import Provider, Site
from nv_dfm_core.exec.discovery import AdvisedOneOf, field_advisor

from ..site import ModelCacheSite
from ..utils._device_utils import (
    available_devices,
    setup_device,
)


class SfnoPrognostic:
    """
    Spherical Fourier Neural Operator (SFNO) model for global weather forecasting.

    This class provides functionality to generate medium-range weather forecasts using the
    SFNO prognostic model. SFNO is a data-driven model that operates on spherical harmonics
    to efficiently predict global atmospheric evolution. Starting from an initial weather state,
    the model generates future timesteps by iteratively rolling forward in time.

    SFNO is particularly well-suited for:
    - Medium-range global weather forecasting (up to 10 days)
    - High-resolution atmospheric predictions
    - Efficient computation on spherical domains (no polar singularities)
    - Operational weather prediction workflows
    - Ensemble forecasting with different seeds

    The model uses a prognostic approach where each forecast step conditions on the
    previous state to generate the next timestep, maintaining physical consistency
    throughout the trajectory.

    Attributes:
        _site: Model cache site for managing model storage and retrieval
        _provider: DFM provider instance for the service
        _logger: Logger instance for tracking operations and debugging
    """

    def __init__(self, site: Site, provider: Provider | None):
        """
        Initialize the SFNO prognostic forecasting model.

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
        and informs clients about which devices can be used for running the SFNO model.

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
        n_steps: int = 12,
        seed: int | None = None,
        device: str | None = None,
    ) -> xarray.Dataset:
        """
        Generate global weather forecasts using the SFNO prognostic model.

        This method performs iterative forecasting (time-stepping) from an initial atmospheric state,
        generating future timesteps by repeatedly applying the SFNO prognostic model. The model
        creates a temporal trajectory by conditioning each forecast step on the previous state,
        producing physically consistent weather evolution.

        SFNO uses spherical harmonic representations to efficiently model global atmospheric
        dynamics without polar singularities, making it particularly effective for medium-range
        weather prediction. The forecasting uses an autoregressive approach where:
        1. The first timestep from the input dataset serves as the initial condition
        2. The model generates the next timestep (typically 6 hours ahead) conditioned on the current state
        3. This process repeats for n_steps, creating a temporal sequence

        The generated data is cached using the model cache system to optimize performance
        across multiple requests.

        Args:
            dataset: Input xarray.Dataset containing the initial atmospheric state. Must have dimensions:
                    - time: temporal dimension (at least one timestep)
                    - lat: latitude dimension (typically 721 points for 0.25° resolution)
                    - lon: longitude dimension (typically 1440 points for 0.25° resolution)
                    The dataset must contain all variables required by the SFNO model (typically
                    20 atmospheric variables including temperature, pressure, winds, humidity, etc.).
                    Only the first timestep (time=0) is used as the initial condition for forecasting.
            n_steps: Number of forecast steps to generate into the future. Each step typically
                    represents a 6-hour interval. More steps produce longer forecasts (e.g.,
                    n_steps=40 produces a 10-day forecast) but may accumulate more uncertainty.
                    Typical range: 4-40 steps (1-10 days). (default: 12)
            seed: Random seed for reproducibility. If provided, the same seed with identical
                 initial conditions will produce identical forecast trajectories. If None,
                 non-deterministic forecasting is used. Note: SFNO is largely deterministic,
                 so seed primarily affects initialization. (default: None)
            device: Compute device to use for model inference. Options:
                   - "cuda" or "cuda:0": Use default CUDA device (requires GPU)
                   - "cuda:N": Use specific CUDA device N (e.g., "cuda:1" for second GPU)
                   - "cpu": Force CPU usage (much slower but doesn't require GPU)
                   - None: Auto-select (CUDA if available, CPU otherwise)
                   Use the valid_device field advisor to discover available options.
                   (default: None)

        Returns:
            xarray.Dataset: Temporal forecast sequence with dimensions:
                - time: temporal dimension (length = n_steps + 1, including initial condition)
                - lat: latitude dimension (same as input, typically 721 points)
                - lon: longitude dimension (same as input, typically 1440 points)
                Each variable required by the model appears in the output, showing the
                evolution of atmospheric fields over the forecast period.

        Raises:
            ValueError: If dataset is None, empty, missing required coordinates,
                       or missing variables required by the model.
            RuntimeError: If model loading, iterator creation, or forecasting fails.

        Example:
            >>> # Generate 10-day (240-hour) forecast from initial conditions
            >>> initial_state = xr.Dataset({...})  # Contains ERA5 or GFS initial state
            >>> result = await sfno.body(
            ...     dataset=initial_state,
            ...     n_steps=40,  # 40 steps * 6 hours = 240 hours (10 days)
            ...     seed=42,
            ...     device="cuda:0"
            ... )
            >>> # Result contains initial state + 40 forecast timesteps
            >>> print(result.dims["time"])  # 41 (initial + 40 forecast)
        """
        sfno_model = None
        try:
            # Validate that the input dataset is not None
            if dataset is None:
                raise ValueError("Input dataset cannot be None")

            # Ensure the dataset contains at least one data variable
            if len(dataset.data_vars) == 0:
                raise ValueError(
                    "Input dataset must contain at least one data variable"
                )

            # Verify the dataset has a time coordinate (required for temporal processing)
            if "time" not in dataset.coords:
                raise ValueError("Input dataset must have a 'time' coordinate")

            # Extract time coordinates - the first timestep will serve as initial condition
            times = dataset.coords["time"].values
            if len(times) == 0:
                raise ValueError("Input dataset must have at least one time step")

            # Set random seed for reproducibility if provided
            # This ensures deterministic forecast trajectories across runs with the same seed
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                self._logger.info(f"Set SFNO seed to: {seed}")

            # Determine the target device (CPU or CUDA) based on availability and user preference
            local_device = setup_device(device, self._logger)

            # Define model loader function for the cache system
            # This function is called only if the model is not already cached
            def _loader():
                package = SFNOModel.load_default_package()
                # Some model wrappers accept device in load_model; we call .to() afterward for safety
                return SFNOModel.load_model(package, device=str(local_device))

            # Load the SFNO model from cache or initialize it if not cached
            cache = self._site.model_cache
            sfno_model = cache.get_or_load(
                model_id="sfno.prognostic",
                loader=_loader,
                device=str(local_device),
            )
            # Ensure all model buffers and parameters are on the selected device
            sfno_model = sfno_model.to(local_device)
            self._logger.info(f"SFNO model loaded on {local_device}; seed={seed}")

            # Query the model for its required input variables and verify they exist in the dataset
            required_variables = list(sfno_model.input_coords()["variable"])  # type: ignore[index]
            missing_vars = set(required_variables) - set(dataset.data_vars.keys())
            if missing_vars:
                raise ValueError(
                    "Dataset is missing variables required for SFNO: "
                    f"{sorted(missing_vars)}"
                )

            # Prepare the initial condition for forecasting
            # Extract only the first timestep to use as the starting point for the forecast
            init_ds = dataset.isel(time=0)
            dimensions = ["time", "variable", "lat", "lon"]

            # Convert the initial condition to a DataArray with required variables only
            data_array = (
                init_ds[required_variables]
                .to_array(dim="variable")
                .transpose(*dimensions[1:])
            )

            # Create a single-element time array for the initial condition
            init_time = np.array([dataset.coords["time"].values[0]]).astype(
                "datetime64[ns]"
            )

            # Define a provider function that returns the initial condition with proper dimensions
            def provide_initial_da(_timestamps, _variables):
                # fetch_data expects a DataArray with time dimension; expand and transpose to match
                return data_array.expand_dims({"time": init_time}, axis=0).transpose(
                    *dimensions
                )

            # Convert the initial condition to PyTorch tensor format on the target device
            x, coords = fetch_data(
                provide_initial_da,
                init_time,
                required_variables,
                device=local_device,
            )

            # Map the data coordinates to match the model's expected input coordinate system
            # This ensures proper alignment of dimensions (batch, time, variable, lat, lon)
            x, coords = map_coords(x, coords, sfno_model.input_coords())
            # Re-assert device placement in case coordinate mapping changed the device
            x = x.to(local_device)

            # Define the iterative forecasting function
            # This performs the autoregressive time-stepping to generate the forecast trajectory
            def run_unroll():
                # Create an iterator that generates successive forecast timesteps
                iterator = sfno_model.create_iterator(x, coords)
                outputs: list[torch.Tensor] = []
                last_coords: dict[str, Any] | None = None
                times = []

                # Iterate through forecast steps, generating one timestep at a time
                for step, (x_step, coords_step) in enumerate(iterator):
                    # Collect each forecast timestep, moving to CPU to free GPU memory
                    outputs.append(x_step.detach().to("cpu"))
                    # Combine base time with lead time to get actual forecast times
                    times.extend(coords_step["time"] + coords_step["lead_time"])
                    last_coords = coords_step
                    # Stop after generating the requested number of steps
                    if step >= n_steps:
                        break

                if not outputs:
                    raise RuntimeError("SFNO iterator produced no outputs")

                # Concatenate all timesteps along the time dimension to create the full trajectory
                stacked = torch.cat(outputs, dim=1)
                last_coords["time"] = times
                return stacked, last_coords

            # Run forecasting in a background thread since it's a synchronous, compute-intensive operation
            # This prevents blocking the async event loop during the iterative forecast generation
            output_tensor, output_coords = await asyncio.to_thread(run_unroll)

            # Post-process the model output tensor to convert it to xarray format

            # Verify that output coordinates were returned from the iterator
            if output_coords is None:
                raise RuntimeError("Missing output coordinates from SFNO iterator")

            # Average over the batch dimension if present (multiple samples are averaged for stability)
            if "batch" in output_coords:
                batch_idx = list(output_coords.keys()).index("batch")
                output_tensor = torch.mean(output_tensor, dim=batch_idx, keepdim=False)

            # Remove the base time dimension (lead times are used as the actual time coordinate)
            output_tensor = output_tensor.squeeze(
                dim=list(output_coords.keys()).index("time")
            )

            # Extract coordinate information for constructing the output dataset
            # Use output coords when available, fall back to input dataset coords
            variable_coord = output_coords.get("variable", required_variables)
            lat_coord = output_coords.get("lat", dataset.coords.get("lat"))
            lon_coord = output_coords.get("lon", dataset.coords.get("lon"))
            time_coord = output_coords["time"]
            coords = {
                "time": time_coord,
                "variable": variable_coord,
                "lat": lat_coord,
                "lon": lon_coord,
            }

            # Create an xarray DataArray from the forecast trajectory
            da = xarray.DataArray(
                output_tensor.detach().cpu().numpy(),
                coords=coords,
                dims=dimensions,
            )

            # Convert the DataArray to a Dataset with each variable as a separate data variable
            # This makes it easier to work with individual atmospheric variables
            ds = da.to_dataset(dim="variable")
            self._logger.info(
                f"SFNO unroll completed. Steps: {n_steps}. Output variables: {list(ds.data_vars.keys())}"
            )
            return ds
        except Exception as e:
            self._logger.error(f"Error performing SFNO unroll: {e}")
            raise
