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


class GeoJsonFile(BaseModel):
    """
    A response object containing information about a single geojson file.

    The contents of GeoJsonFile depends on the configuration of the provider and
    the parameters to the function producing the GeoJsonFile response.
    A geojson file may be returned as the URL of the storage location if the provider
    is configured to provide storage.
    And/or a geojson file may be returned inline as string if the client indicates
    so in the function parameters.
    Similarly for metadata. Metadata may be returned as the URL of the storage location
    if metadata is present and the provider is configured to provide storage.
    And/or the metadata may be returned inline as a Dict if the client indicates
    so in the function parameters.

    Args:
        metadata_url: The URL of the metadata file, if a metadata file exists.
        url: The URL of the geojson file, if a geojson file exists.
        timestamp: The timestamp of the geojson file, if the geojson was associated with a time.
        data: GeoJSON string, if requested.
        metadata: The inlined metadata dict, if requested.
    """

    metadata_url: str | None = None
    # if the dfm is configured to write the data, this will be the URL
    url: str | None = None
    timestamp: str
    data: str | None = None
    metadata: dict[str, Any] | None = None
