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
import sys
import types

import numpy as np
import pytest
import torch
import xarray as xr
from nv_dfm_lib_weather.utils._model_cache import TorchModelCache


class DummyLogger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - stub
        pass

    def error(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - stub
        pass


class DummyDfmContext:
    def __init__(self) -> None:
        self.logger = DummyLogger()


class DummySite:
    def __init__(self) -> None:
        self.dfm_context = DummyDfmContext()
        self._model_cache = TorchModelCache()

    @property
    def model_cache(self) -> TorchModelCache:
        return self._model_cache


def ensure_fake_earth2studio_modules() -> None:
    if "earth2studio" in sys.modules:
        return
    e2s = types.ModuleType("earth2studio")
    e2s.models = types.ModuleType("earth2studio.models")
    e2s.models.dx = types.ModuleType("earth2studio.models.dx")
    sys.modules["earth2studio"] = e2s
    sys.modules["earth2studio.models"] = e2s.models
    sys.modules["earth2studio.models.dx"] = e2s.models.dx


@pytest.mark.asyncio
async def test_cbottle_tc_guidance_latlon_and_hpx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_fake_earth2studio_modules()

    lat_coords = [10.0, 12.0]
    lon_coords = [100.0, 102.0]
    times = [np.datetime64("2025-01-01T00:00:00", "ns")]

    class FakeTCGuidanceModel:
        def to(self, device: str) -> "FakeTCGuidanceModel":
            return self

        def create_guidance_tensor(self, lat_coords, lon_coords, times):
            # Return a dummy tensor and coords
            batch = 1
            lead = 1
            variables = ["msl", "tcwv"]
            lat = np.linspace(-90, 90, 2)
            lon = np.linspace(0, 360, 3, endpoint=False)
            x = torch.rand(
                batch, lead, len(variables), len(lat), len(lon), dtype=torch.float32
            )
            coords = {
                "batch": np.array([0]),
                "time": np.array(times),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(variables),
                "lat": lat,
                "lon": lon,
            }
            return x, coords

        def __call__(self, x: torch.Tensor, coords: dict[str, Any]):
            # Echo back tensor but remove batch dimension to match adapter's expected squeeze
            x = x.squeeze(0)
            return x, {k: v for k, v in coords.items() if k != "batch"}

    class FakeTCGuidance:
        @staticmethod
        def load_default_package():
            return object()

        @staticmethod
        def load_model(package, sampler_steps: int, sigma_max: float, lat_lon: bool):
            return FakeTCGuidanceModel()

    # Patch device and model
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_tc_guidance.CBottleTCGuidance",
        FakeTCGuidance,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_tc_guidance.setup_device",
        lambda device, logger: "cpu",
        raising=True,
    )

    from nv_dfm_lib_weather.cbottle import CBottleTropicalCycloneGuidance

    site = DummySite()
    adapter = CBottleTropicalCycloneGuidance(site=site, provider=None)

    # lat/lon path
    ds = await adapter.body(
        lat_coords=lat_coords,
        lon_coords=lon_coords,
        times=list(times),
        sampler_steps=18,
        sigma_max=200.0,
        seed=1,
        device="cpu",
        lat_lon=True,
    )
    assert isinstance(ds, xr.Dataset)
    for v in ["msl", "tcwv"]:
        assert v in ds.data_vars
        assert (
            "lat" in ds[v].coords
            and "lon" in ds[v].coords
            and "hpx" not in ds[v].coords
        )

    # hpx path
    class FakeTCGuidanceModelHPX(FakeTCGuidanceModel):
        def create_guidance_tensor(self, lat_coords, lon_coords, times):
            batch = 1
            lead = 1
            variables = ["msl", "tcwv"]
            hpx = np.arange(6)
            x = torch.rand(batch, lead, len(variables), len(hpx), dtype=torch.float32)
            coords = {
                "batch": np.array([0]),
                "time": np.array(times),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(variables),
                "hpx": hpx,
            }
            return x, coords

    class FakeTCGuidanceHPX:
        @staticmethod
        def load_default_package():
            return object()

        @staticmethod
        def load_model(package, sampler_steps: int, sigma_max: float, lat_lon: bool):
            return FakeTCGuidanceModelHPX()

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_tc_guidance.CBottleTCGuidance",
        FakeTCGuidanceHPX,
        raising=True,
    )

    ds_hpx = await adapter.body(
        lat_coords=lat_coords,
        lon_coords=lon_coords,
        times=list(times),
        sampler_steps=18,
        sigma_max=200.0,
        seed=1,
        device="cpu",
        lat_lon=False,
    )
    assert isinstance(ds_hpx, xr.Dataset)
    for v in ["msl", "tcwv"]:
        assert v in ds_hpx.data_vars
        assert (
            "hpx" in ds_hpx[v].coords
            and "lat" not in ds_hpx[v].coords
            and "lon" not in ds_hpx[v].coords
        )
