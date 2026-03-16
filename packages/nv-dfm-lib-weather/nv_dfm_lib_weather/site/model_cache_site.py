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

from __future__ import annotations

import nv_dfm_core.exec  # type: ignore

from ..utils._model_cache import TorchModelCache


class ModelCacheSite(nv_dfm_core.exec.Site):  # type: ignore[misc]
    """
    Extended DFM Site with PyTorch model caching capabilities.

    This class extends the standard DFM Site to provide built-in support for caching
    PyTorch models. It maintains a shared model cache that can be used across different
    service instances to avoid redundant model loading and improve performance.

    The model cache enables:
    - Efficient model sharing across multiple inference requests
    - Automatic device management (CPU/GPU) for cached models
    - Memory-efficient model storage and retrieval
    - Reduced startup time for services using large ML models

    This is particularly useful for Earth2 models (SFNO, CBottle, etc.) which can be
    large and time-consuming to load from disk.

    Attributes:
        _model_cache: TorchModelCache instance managing PyTorch model storage and retrieval
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the ModelCacheSite with model caching support.

        Creates a new site instance with an integrated PyTorch model cache. All
        arguments are passed through to the parent dfm.exec.Site class.

        Args:
            *args: Positional arguments for dfm.exec.Site initialization
            **kwargs: Keyword arguments for dfm.exec.Site initialization
        """
        # Initialize the parent Site class with all provided arguments
        super().__init__(*args, **kwargs)
        # Create a dedicated model cache for PyTorch models
        self._model_cache = TorchModelCache()

    @property
    def model_cache(self) -> TorchModelCache:
        """
        Access the site's PyTorch model cache.

        This property provides access to the shared model cache that can be used to
        store and retrieve PyTorch models efficiently across service calls.

        Returns:
            TorchModelCache: The model cache instance for this site
        """
        return self._model_cache
