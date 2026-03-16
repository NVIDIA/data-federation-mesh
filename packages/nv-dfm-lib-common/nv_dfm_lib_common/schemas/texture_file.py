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

from typing import Any

from pydantic import BaseModel


class TextureFile(BaseModel):
    """
    A response object containing information about a single texture file.

    For example, produced as the output of RenderUint8ToImages.
    The contents of TextureFile depends on the configuration of the provider and
    the parameters to the function producing the TextureFile response.
    A texture file may be returned as the URL of the storage location if the provider
    is configured to provide storage.
    And/or a texture file may be returned inline as a base64 encoded image if
    the clients indictates so in the function parameters.
    Similarly for metadata. Metadata may be returned as the URL of the storage location
    if metadata is present and the provider is configured to provide storage.
    And/or the metadata may be returned inline as a Dict if the client indicates
    so in the function parameters.

    Args:
        metadata_url: The URL of the metadata file, if a metadata file exists.
        url: The URL of the texture file, if a texture file exists.
        format: The format (jpeg, png, etc) of the texture file.
        timestamp: The timestamp of the texture, if the texture was associated with a time.
        base64_image_data: Base64 encoded inlined image, if requested.
        metadata: The inlined metadata dict, if requested.
    """

    metadata_url: str | None = None
    # if the dfm is configured to write the images, this will be the URL
    url: str | None = None
    format: str = "png"
    timestamp: str | None = None
    base64_image_data: str | None = None
    metadata: dict[str, Any] | None = None

    def __repr_args__(self):
        """
        Custom representation arguments for the TextureFile model.

        This method truncates long base64_image_data strings in the representation
        to avoid overwhelming output with large image data.

        Returns:
            List of tuples containing field names and values for representation
        """
        # start from the default args provided by BaseModel
        args = super().__repr_args__()  # list[tuple[str, Any]]
        out = []
        for k, v in args:
            if k == "base64_image_data" and isinstance(v, str) and len(v) > 40:
                out.append((k, v[:40] + "..."))
            else:
                out.append((k, v))
        return out


class TextureFileList(BaseModel):
    """
    A response object containing a collection of texture files.

    This class serves as a container for multiple TextureFile objects, typically used
    when a function produces multiple texture outputs. Common use cases include:
    - Multiple timesteps in a temporal sequence
    - Multiple frames in a video or animation
    - Multiple variables rendered at the same time
    - Tiled or multi-region outputs

    Each TextureFile in the list may contain its own metadata, URLs, inline data,
    and timestamps, depending on the provider configuration and client parameters.

    Args:
        texture_files: List of TextureFile objects representing the collection of textures.
    """

    texture_files: list[TextureFile] = []
