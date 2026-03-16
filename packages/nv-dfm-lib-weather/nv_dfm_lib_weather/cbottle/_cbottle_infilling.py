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
from earth2studio.lexicon.cbottle import CBottleLexicon
from earth2studio.models.dx import CBottleInfill

from nv_dfm_core.exec import Provider
from nv_dfm_core.exec.discovery import AdvisedOneOf, AdvisedSubsetOf, field_advisor

from ..site import ModelCacheSite
from ..utils._device_utils import available_devices, setup_device


class CbottleInfilling:
    """
    CBottle infilling model for climate data completion.

    This class provides functionality to perform infilling (gap-filling) on climate datasets
    using the CBottleInfill diffusion model.

    Attributes:
        _site: Model cache site for managing model storage and retrieval
        _provider: DFM provider instance for the service
        _logger: Logger instance for tracking operations and debugging
    """

    def __init__(self, site: ModelCacheSite, provider: Provider | None):
        """
        Initialize the CBottle infilling model.

        Args:
            site: ModelCacheSite instance for model caching and management
            provider: DFM Provider instance, or None if running standalone
        """
        self._site: ModelCacheSite = site
        self._provider = provider
        self._logger = site.dfm_context.logger

    @field_advisor("input_variables")
    async def available_variables(self, _value, _context):
        """
        Field advisor that provides available climate variables for CBottle infilling.

        This method is used by the DFM discovery system to inform clients about which
        climate variables can be used as input for the infilling process. The available
        variables are defined by the CBottle lexicon vocabulary.

        Args:
            _value: The current value (unused)
            _context: The discovery context (unused)

        Returns:
            AdvisedSubsetOf: A set of valid variable names that can be selected as inputs
        """
        self._logger.info("Discovery started for variables")
        result = AdvisedSubsetOf(CBottleLexicon.VOCAB.keys())  # type: ignore
        self._logger.info(f"Discovery finished for variables: {result}")
        return result

    @field_advisor("device")
    async def valid_device(self, _value, _context):
        """
        Field advisor that provides available compute devices for model inference.

        This method queries the system for available compute devices (CPU and CUDA GPUs)
        and informs clients about which devices can be used for running the CBottle infilling model.

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
        input_variables: list[str],
        sampler_steps: int = 18,
        sigma_max: float = 80.0,
        seed: int | None = None,
        device: str | None = None,
    ) -> xarray.Dataset:
        """
        Perform climate data infilling using the CBottleInfill diffusion model.

        This method takes an input dataset containing a subset of climate variables and uses
        the CBottleInfill model to generate all climate variables in the CBottle vocabulary.
        The model uses a diffusion-based approach to generate physically consistent climate
        fields conditioned on the provided input variables.

        The infilling process employs an iterative denoising diffusion approach, starting from
        random noise and gradually refining it to produce realistic climate data that is
        consistent with the input variables. The sampler_steps and sigma_max parameters control
        the quality and characteristics of this diffusion process.

        The generated data is cached using the model cache system to optimize performance
        across multiple requests with the same model configuration.

        Args:
            dataset: Input xarray.Dataset containing climate data. Must have dimensions:
                    - time: temporal dimension (one or more timesteps)
                    - lat: latitude dimension
                    - lon: longitude dimension
                    The dataset must contain all variables specified in input_variables.
            input_variables: List of climate variable names to use as conditioning inputs for
                           the infilling process. Must be valid CBottle lexicon variables and
                           must exist in the input dataset. Use the available_variables field
                           advisor to discover valid options.
            sampler_steps: Number of denoising steps in the diffusion sampling process. More steps
                         generally produce higher quality results but take longer to compute.
                         Typical range: 10-50 steps. (default: 18)
            sigma_max: Maximum noise level (variance) at the start of the diffusion process.
                      Higher values allow for more variation in the generated data but may
                      reduce consistency with inputs. Typical range: 50-200. (default: 80.0)
            seed: Random seed for reproducibility. (default: None)
            device: Compute device to use for model inference. Options:
                   - "cuda" or "cuda:0": Use default CUDA device (requires GPU)
                   - "cuda:N": Use specific CUDA device N (e.g., "cuda:1" for second GPU)
                   - "cpu": Force CPU usage (slower but doesn't require GPU)
                   - None: Auto-select (CUDA if available, CPU otherwise)
                   Use the valid_device field advisor to discover available options.
                   (default: None)

        Returns:
            xarray.Dataset: Generated climate dataset containing all CBottle vocabulary variables.
                The output has dimensions:
                - time: single timestep (averaged over input timesteps)
                - lat: latitude dimension (same as input)
                - lon: longitude dimension (same as input)
                Each variable in the CBottle vocabulary appears as a data variable, including
                both the original input variables and the newly generated variables.

        Raises:
            ValueError: If dataset is None, empty, missing required coordinates,
                       input_variables is empty, or specified variables are not found in dataset.
            RuntimeError: If model loading or inference fails.

        Example:
            >>> # Start with a dataset containing only a few variables
            >>> input_ds = xr.Dataset({
            ...     "msl": ...,  # mean sea level pressure
            ...     "tcwv": ..., # total column water vapor
            ... })
            >>> # Infill to generate all climate variables
            >>> result = await infiller.body(
            ...     dataset=input_ds,
            ...     input_variables=["msl", "tcwv"],
            ...     sampler_steps=20,
            ...     sigma_max=80.0,
            ...     seed=42,
            ...     device="cuda:0"
            ... )
            >>> # Result now contains all CBottle vocabulary variables
        """
        cbottle_infill_model = None
        try:
            # Validate inputs
            if dataset is None:
                raise ValueError("Input dataset cannot be None")

            if len(dataset.data_vars) == 0:
                raise ValueError(
                    "Input dataset must contain at least one data variable"
                )

            if not input_variables:
                raise ValueError("At least one input variable must be specified")

            missing_vars = set(input_variables) - set(dataset.data_vars.keys())
            if missing_vars:
                raise ValueError(
                    f"Input variables not found in dataset: {missing_vars}"
                )

            # Check if input variables are recognized by the CBottle lexicon
            # This is a warning rather than an error to allow flexibility
            invalid_vars = set(input_variables) - set(CBottleLexicon.VOCAB.keys())
            if invalid_vars:
                self._logger.warning(
                    f"Some input variables are not in CBottle vocabulary: {invalid_vars}"
                )

            self._logger.info(
                f"Performing CBottle infilling with input variables: {input_variables}"
            )
            self._logger.info(
                f"Input dataset variables: {list(dataset.data_vars.keys())}"
            )

            # Set random seed for reproducibility if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                self._logger.info(f"Set CBottle infill seed to: {seed}")

            # Determine the target device (CPU or CUDA) based on availability and user preference
            local_device = setup_device(device, self._logger)

            # Define model loader function for the cache system
            # This function is called only if the model is not already cached
            def _loader():
                package = CBottleInfill.load_default_package()
                return CBottleInfill.load_model(
                    package,
                    input_variables=input_variables,
                    sampler_steps=sampler_steps,
                    sigma_max=sigma_max,
                )

            # Load the CBottleInfill model from cache or initialize it if not cached
            # The cache key includes input variables and model parameters to ensure proper configuration
            cache = self._site.model_cache
            cache_key = (
                f"cbottle.infill:vars={','.join(sorted(input_variables))}:"
                f"steps={sampler_steps}:sigma={sigma_max}"
            )
            cbottle_infill_model = cache.get_or_load(
                model_id=cache_key,
                loader=_loader,
                device=str(local_device),
            )
            self._logger.info(
                f"CBottleInfill model loaded and moved to device: {local_device}"
            )

            # Extract and validate time coordinates from the input dataset
            # The time dimension is required for the infilling process
            if "time" not in dataset.coords:
                raise ValueError("Input dataset must have a 'time' coordinate")

            times = dataset.coords["time"].values
            if len(times) == 0:
                raise ValueError("Input dataset must have at least one time step")

            if not np.issubdtype(times.dtype, np.datetime64):
                times = times.astype("datetime64[ns]")

            # Define the infilling computation function
            # This performs the actual diffusion-based generation conditioned on input variables
            def perform_infilling():
                data_array = dataset.to_array(dim="variable")
                data_array = data_array.transpose("time", "variable", "lat", "lon")
                x, coords = fetch_data(
                    lambda timestamps, variable: data_array,
                    times,
                    input_variables,
                    device=local_device,
                )

                output, output_coords = cbottle_infill_model(x, coords)

                return output, output_coords

            # Run infilling in a background thread since it's a synchronous, compute-intensive operation
            output_tensor, output_coords = await asyncio.to_thread(perform_infilling)

            # Post-process the model output tensor to convert it to xarray format

            # Average over the batch dimension if present (multiple samples are averaged for stability)
            if "batch" in output_coords:
                batch_idx = list(output_coords.keys()).index("batch")
                output_tensor = torch.mean(output_tensor, dim=batch_idx, keepdim=False)

            # Average over the time dimension to produce a single representative timestep
            # Multiple input timesteps are condensed into one output timestep
            if "time" in output_coords:
                time_idx = list(output_coords.keys()).index("time")
                output_tensor = torch.mean(output_tensor, dim=time_idx, keepdim=True)

            # Remove the lead_time dimension as it's not needed in the final output
            output_tensor = output_tensor.squeeze(
                dim=list(output_coords.keys()).index("lead_time")
            )

            # Create an xarray DataArray from the output tensor with proper coordinates
            infilled_da = xarray.DataArray(
                output_tensor.cpu().numpy(),
                coords={
                    "time": output_coords["time"][:1],
                    "variable": output_coords["variable"],
                    "lat": output_coords["lat"],
                    "lon": output_coords["lon"],
                },
                dims=["time", "variable", "lat", "lon"],
            )

            infilled_ds = infilled_da.to_dataset(dim="variable")

            self._logger.info(
                f"Infilling completed. Output dataset variables: {list(infilled_ds.data_vars.keys())}"
            )
            return infilled_ds
        except Exception as e:
            self._logger.error(f"Error performing CBottle infilling: {e}")
            raise e
