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

import gc
import threading
from typing import Callable

import torch


class TorchModelCache:
    """Per-device cache for `torch.nn.Module` with OOM-aware eviction and
    duplicate-load prevention.

    Expected behavior
    - Per-device namespacing: Models are cached independently per device
      string (e.g., "cuda:0", "cpu"). The cache key is `(device, model_id)`.
    - Loader on miss: `get_or_load(model_id, loader, device)` returns an
      existing model if present; otherwise it calls `loader()` to construct
      the model and then moves it to the requested device via `.to(device)`.
    - OOM-aware retry: If an out-of-memory error occurs while loading or
      moving the model, all models on that device are evicted and the load is
      retried once. Non-OOM errors are propagated immediately.
    - Concurrency control: All cache operations are serialized by a single
      global lock. This prevents duplicate instantiation across all keys but
      does not allow parallel loads.
    - No implicit eviction: There is no LRU/TTL. Entries remain until process
      exit or explicit eviction. Call `unload_device(device)` to drop all
      models for a device and trigger `gc.collect()` and, when applicable,
      `torch.cuda.empty_cache()`.

    Usage guidance
    - Choose a stable `model_id` that uniquely identifies the architecture and
      weights (e.g., a name plus weights hash or version).
    - Implement `loader()` to construct and fully initialize the model,
      including loading weights. Device placement in the loader is optional;
      the cache will call `.to(device)` to ensure correct placement.
    - The returned object is the same cached instance on subsequent calls. If
      you require `eval()` mode, autocast, or specific precision, set that in
      the loader or immediately after retrieval. The cache does not clone or
      wrap models.
    - Caching is in-process only; there is no cross-process sharing and no
      per-model unload API beyond clearing a whole device via
      `unload_device(device)`.
    """

    def __init__(self) -> None:
        """
        Initialize the TorchModelCache.

        Creates an empty cache dictionary and a global re-entrant lock for
        serializing cache operations.
        """
        # device_str -> (model_id -> model)
        self._models: dict[str, dict[str, torch.nn.Module]] = {}
        # Single global re-entrant lock to serialize cache operations
        self._global_lock: threading.RLock = threading.RLock()

    def get_or_load(
        self,
        model_id: str,
        loader: Callable[[], torch.nn.Module],
        device: str,
    ) -> torch.nn.Module:
        """
        Get or load a model from the cache.

        Args:
            model_id: The unique identifier for the model
            loader: A callable that constructs and returns the model
            device: The device string (e.g., "cuda:0", "cpu")
        """
        device_str = str(torch.device(device))
        with self._global_lock:
            # Fast path under the global lock
            device_cache = self._models.get(device_str)
            if device_cache is not None and model_id in device_cache:
                return device_cache[model_id]

            def _attempt_load() -> torch.nn.Module:
                model = loader()
                # Ensure the model is actually on the desired device
                desired_device = torch.device(device_str)
                model = model.to(desired_device)
                return model

            # First attempt
            try:
                model = _attempt_load()
            except (
                torch.cuda.OutOfMemoryError,
                RuntimeError,
            ) as e:  # pragma: no cover - device specific
                if self._is_oom_error(e):
                    # Evict all on this device and retry once
                    self.unload_device(device_str)
                    model = _attempt_load()
                else:
                    raise

            # Store in cache
            device_cache = self._models.setdefault(device_str, {})
            device_cache[model_id] = model
            return model

    def unload_device(self, device: str) -> None:
        """
        Unload all models for the given device and free memory.

        Args:
            device: The device string (e.g., "cuda:0", "cpu")
        """
        device_str = str(torch.device(device))
        with self._global_lock:
            to_delete = self._models.pop(device_str, None)

            if to_delete:
                # Drop strong references held by the popped dict eagerly
                to_delete.clear()

            # Free CPU and, if applicable, GPU allocator caches
            _ = gc.collect()
            if (
                device_str.startswith("cuda") and torch.cuda.is_available()
            ):  # pragma: no cover - device specific
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    # Best-effort; ignore device-specific issues
                    pass

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        """
        Check if the exception is an out-of-memory error.

        Args:
            exc: The exception to check

        Returns:
            True if the exception is an out-of-memory error, False otherwise
        """
        msg = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg
