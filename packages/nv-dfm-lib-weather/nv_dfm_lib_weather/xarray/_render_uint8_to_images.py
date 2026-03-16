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

import base64
import io
import json
import logging
from uuid import uuid4

import numpy as np
import xarray
from PIL import Image

from nv_dfm_core.exec import Provider, Site
from nv_dfm_lib_common.schemas import TextureFile, TextureFileList
from nv_dfm_lib_weather.xarray.schema import RenderUint8ToImagesInputSchema


def _sanitize_for_json(obj: dict) -> dict:
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: A dictionary that may contain numpy types

    Returns:
        Dictionary with numpy types converted to native Python types
    """

    def _convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        return value

    return {k: _convert(v) for k, v in obj.items()}


class RenderUint8ToImages:
    """
    A processor for rendering uint8 xarray datasets to image files.

    This class provides functionality to convert uint8 xarray datasets into image files
    (PNG, JPEG, etc.) with support for multiple timesteps and various output formats.
    """

    def __init__(self, site: Site, provider: Provider | None):
        """
        Initialize the RenderUint8ToImages processor.

        Args:
            site: DFM site configuration
            provider: Optional provider configuration
        """
        self._site = site
        self._provider = provider
        if site is not None:
            self._logger = site.dfm_context.logger
            self._can_save = True
            # Use cache storage to save images and metadata if requested
            self._storage = site.cache_storage(subpath=self.__class__.__name__)
        else:
            self._logger = logging.getLogger(__name__)
            self._can_save = False

        self._logger = logging.getLogger(__name__)

    async def body_impl(
        self,
        data: xarray.Dataset,
        variable: str,
        xydims: list[str],
        time_dimension: str,
        additional_meta_data: dict | None,
        return_meta_data: bool,
        return_image_data: bool,
        quality: int,
        format: str,
    ):
        """
        Render uint8 xarray dataset to image files.

        This method converts a uint8 xarray dataset into image files for each timestep,
        with support for various image formats and quality settings.

        Args:
            data: Input uint8 xarray dataset to render
            variable: Name of the variable to render (optional if dataset has only one)
            xydims: List of x and y dimension names
            time_dimension: Name of the time dimension
            additional_meta_data: Optional additional metadata to include
            return_meta_data: Whether to return metadata
            return_image_data: Whether to return image data as base64
            quality: Image quality for JPEG format (1-100)
            format: Image format ("png", "jpeg", etc.)

        Returns:
            TextureFileList containing rendered images for each timestep

        Raises:
            ValueError: If variable doesn't exist or dataset has multiple variables without specification
        """

        # Use unique call ID to generate unique directory name for saving images and metadata
        call_id = str(uuid4())

        self._logger.info(
            f"Rendering uint8 xarray dataset to image files for variable {variable} with call ID {call_id}"
        )
        self._logger.debug(f"Additional metadata: {additional_meta_data}")
        self._logger.debug(f"Return metadata: {return_meta_data}")
        self._logger.debug(f"Return image data: {return_image_data}")
        self._logger.debug(f"Quality: {quality}")
        self._logger.debug(f"Format: {format}")
        self._logger.debug(f"Data: {data}")
        self._logger.debug(f"Xydims: {xydims}")
        self._logger.debug(f"Time dimension: {time_dimension}")

        if self._can_save:
            save_dir = self._storage.joinpath(call_id)
            # Don't use exist_ok=True here, because we want to raise an error if the directory already exists
            save_dir.mkdir(parents=True)
            self._logger.info(f"Save directory: {save_dir}")
        else:
            self._logger.warning(
                "No save directory available, will not save images and metadata"
            )

        # validate the input data against the schema
        schema = RenderUint8ToImagesInputSchema(time_dimension, xydims, data.data_vars)
        schema.validate(data, strict=False)

        # Find out which variable to render; either there is only one or we require the
        # user to pass the name
        if variable:
            if variable not in data:
                raise ValueError(
                    f"Variable {variable} selected for rendering does not exist in dataset {data.dims}"
                )
            one_variable = data[variable]
        else:
            data_vars_names = [v for v in data.data_vars]
            if len(data_vars_names) != 1:
                raise ValueError(
                    f"Data has multiple variables {data_vars_names}. Need an explicit 'variable' parameter."
                )
            one_variable = data[data_vars_names[0]]

        # Compute the metadata
        # y dim is lat, x dim is lon
        lat_min = data[xydims[1]].min().item()
        lat_max = data[xydims[1]].max().item()
        lon_min = data[xydims[0]].min().item()
        lon_max = data[xydims[0]].max().item()

        # correct lon_max by one step size
        longitudes = data[xydims[0]]
        lon_step_size = (
            (longitudes[1] - longitudes[0]).item() if len(longitudes) > 1 else 0.0
        )
        lon_max += abs(lon_step_size)

        additional_meta_data = additional_meta_data if additional_meta_data else {}

        # Sanitize xarray attrs to ensure JSON serializable (numpy arrays -> lists)
        sanitized_attrs = _sanitize_for_json(one_variable.attrs)

        meta = (
            {  # write lon and lat info plus all the xarray attributes
                "lon_minmax": [lon_min, lon_max],
                "lat_minmax": [lat_min, lat_max],
            }
            | sanitized_attrs
            | additional_meta_data
        )

        if self._can_save:  # and not return_meta_data:
            save_path = save_dir.joinpath("metadata.json")
            save_path.write_text(json.dumps(meta))
            metadata_url = save_path.as_posix()
            self._logger.info(f"Saved metadata to {metadata_url}")
        else:
            metadata_url = None

        metadata = meta if return_meta_data else None

        # now render each timestep (inside a coroutine)
        num_timesteps = data.sizes[time_dimension]

        results = []

        for i in range(num_timesteps):
            one_timestep = one_variable.isel({time_dimension: i}, missing_dims="ignore")
            np_arr = one_timestep.to_numpy().astype(np.uint8)
            # apparently it can sometimes happen that we get all-black textures;
            the_sum = np_arr.sum()
            if the_sum == 0 or the_sum == 1:
                self._logger.warning(
                    f"Image was all {the_sum}, possibly bad normalization?: {one_timestep}"
                )

            # create the image
            img_pil = Image.fromarray(np_arr.transpose())
            img_options = {"quality": quality}
            img_format = format.upper()
            with io.BytesIO() as bio:
                img_pil.save(bio, format=img_format, **img_options)  # type: ignore
                img_bytes = bio.getvalue()

                # in case we want to send it back to the client
                img_str = (
                    base64.b64encode(img_bytes).decode() if return_image_data else None
                )

                if self._can_save:
                    save_path = save_dir.joinpath(f"image_{i}.{img_format.lower()}")
                    save_path.write_bytes(img_bytes)
                    image_url = save_path.as_posix()
                    self._logger.info(f"Saved image to {image_url}")
                else:
                    image_url = None

            time_str = (
                one_timestep[time_dimension].time.dt.strftime("%Y-%m-%dT%H:%M").item(0)
            )

            # create the result object and yield
            result = TextureFile(
                metadata_url=metadata_url,
                url=image_url,
                format=img_format,
                timestamp=time_str,
                metadata=metadata,
                base64_image_data=img_str,
            )

            results.append(result)

        self._logger.info(f"Returning {len(results)} texture files")
        return TextureFileList(texture_files=results)

    async def body(
        self,
        data: xarray.Dataset,
        variable: str,
        xydims: list[str],
        time_dimension: str,
        additional_meta_data: dict | None = None,
        return_meta_data: bool | None = None,
        return_image_data: bool | None = None,
        quality: int | None = None,
        format: str | None = None,
    ):
        """
        Async wrapper for rendering uint8 xarray dataset to image files.

        This method provides an async interface to the image rendering functionality.

        Args:
            data: Input uint8 xarray dataset to render
            variable: Name of the variable to render (optional if dataset has only one)
            xydims: List of x and y dimension names
            time_dimension: Name of the time dimension
            additional_meta_data: Optional additional metadata to include
            return_meta_data: Whether to return metadata
            return_image_data: Whether to return image data as base64
            quality: Image quality for JPEG format (1-100)
            format: Image format ("png", "jpeg", etc.)

        Returns:
            TextureFileList containing rendered images for each timestep
        """
        # Set defaults. Federation configuration will override any defaults set for function
        # parameters, so we need to make sure we can still work if defaults are set to None.
        additional_meta_data = additional_meta_data or {}
        return_meta_data = return_meta_data if return_meta_data is not None else True
        return_image_data = return_image_data if return_image_data is not None else True
        quality = quality or 99
        format = format or "png"

        # Call the implementation
        return await self.body_impl(
            data,
            variable,
            xydims,
            time_dimension,
            additional_meta_data,
            return_meta_data,
            return_image_data,
            quality,
            format,
        )
