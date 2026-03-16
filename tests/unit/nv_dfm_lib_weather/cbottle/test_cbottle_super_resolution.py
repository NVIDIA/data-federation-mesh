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
    def info(
        self, *_args: Any, **_kwargs: Any
    ) -> None:  # pragma: no cover - simple stub
        pass

    def error(
        self, *_args: Any, **_kwargs: Any
    ) -> None:  # pragma: no cover - simple stub
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
    e2s.data = types.ModuleType("earth2studio.data")
    e2s.models = types.ModuleType("earth2studio.models")
    e2s.models.dx = types.ModuleType("earth2studio.models.dx")
    e2s.models.dx.cbottle_sr = types.ModuleType("earth2studio.models.dx.cbottle_sr")
    e2s.data.utils = types.ModuleType("earth2studio.data.utils")
    e2s.utils = types.ModuleType("earth2studio.utils")
    e2s.utils.coords = types.ModuleType("earth2studio.utils.coords")

    class _Placeholder:
        pass

    e2s.models.dx.cbottle_sr.CBottleSR = _Placeholder
    # Also stub TC guidance symbol import path used elsewhere
    e2s.models.dx.CBottleTCGuidance = _Placeholder

    def _fetch_data_stub(*_args, **_kwargs):
        raise RuntimeError("fetch_data should be monkeypatched in tests")

    def _map_coords_stub(*_args, **_kwargs):
        raise RuntimeError("map_coords should be monkeypatched in tests")

    e2s.data.utils.fetch_data = _fetch_data_stub
    e2s.utils.coords.map_coords = _map_coords_stub

    sys.modules["earth2studio"] = e2s
    sys.modules["earth2studio.data"] = e2s.data
    sys.modules["earth2studio.models"] = e2s.models
    sys.modules["earth2studio.models.dx"] = e2s.models.dx
    sys.modules["earth2studio.models.dx.cbottle_sr"] = e2s.models.dx.cbottle_sr
    sys.modules["earth2studio.data.utils"] = e2s.data.utils
    sys.modules["earth2studio.utils"] = e2s.utils
    sys.modules["earth2studio.utils.coords"] = e2s.utils.coords


@pytest.mark.asyncio
async def test_cbottle_super_resolution_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_fake_earth2studio_modules()
    # Arrange: low-res input dataset with variables required by model
    variables = ["msl", "tcwv"]
    lat = np.linspace(-90, 90, 2)
    lon = np.linspace(0, 360, 3, endpoint=False)
    time_vals = np.array([np.datetime64("2025-01-01T00:00:00", "ns")])

    data_vars = {}
    for v in variables:
        vals = np.random.rand(len(time_vals), len(lat), len(lon)).astype(np.float32)
        data_vars[v] = ("time", "lat", "lon"), vals
    ds = xr.Dataset(data_vars, coords=dict(time=time_vals, lat=lat, lon=lon))

    class FakeCBottleSRModel:
        def __init__(self) -> None:
            self._device = "cpu"
            self.output_resolution = (4, 6)

        def to(self, device: str) -> "FakeCBottleSRModel":
            self._device = device
            return self

        def input_coords(self) -> dict[str, Any]:
            return {
                "batch": [0],
                "time": time_vals,
                "variable": variables,
                "lat": lat,
                "lon": lon,
            }

        def __call__(self, x: torch.Tensor, coords: dict[str, Any]):
            # Output dims [batch, lead_time(=1), time(=1), variable, lat, lon]
            batch = x.shape[0]
            variable = x.shape[2]
            out_lat, out_lon = self.output_resolution
            out = torch.rand(
                batch, 1, 1, variable, out_lat, out_lon, dtype=torch.float32
            )
            out_coords = {
                "batch": np.arange(batch),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "time": time_vals,
                "variable": np.array(variables),
                "lat": np.linspace(-90, 90, out_lat),
                "lon": np.linspace(0, 360, out_lon, endpoint=False),
            }
            return out, out_coords

    class FakeCBottleSR:
        @staticmethod
        def load_default_package():  # pragma: no cover - trivial
            return object()

        @staticmethod
        def load_model(
            _package,
            output_resolution,
            super_resolution_window,
            sampler_steps,
            sigma_max,
        ):  # pragma: no cover
            model = FakeCBottleSRModel()
            # Respect provided resolution if given
            if output_resolution is not None:
                model.output_resolution = tuple(output_resolution)
            return model

    def fake_fetch_data(provider, times, required_variables, device="cpu"):
        da = provider(times, required_variables)
        da = da.transpose("time", "variable", "lat", "lon")
        x = torch.from_numpy(da.values).unsqueeze(0)
        coords = {
            "batch": np.array([0]),
            "time": da.coords["time"].values,
            "variable": da.coords["variable"].values,
            "lat": da.coords["lat"].values,
            "lon": da.coords["lon"].values,
        }
        return x, coords

    def fake_map_coords(
        x: torch.Tensor, coords: dict[str, Any], _target: dict[str, Any]
    ):
        # For test purposes, return inputs as-is
        return x, coords

    def fake_setup_device(device: str | None, _logger) -> str:
        return device or "cpu"

    # Patch module dependencies
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_super_resolution.CBottleSR",
        FakeCBottleSR,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_super_resolution.fetch_data",
        fake_fetch_data,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_super_resolution.map_coords",
        fake_map_coords,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_super_resolution.setup_device",
        fake_setup_device,
        raising=True,
    )

    from nv_dfm_lib_weather.cbottle import CbottleSuperResolution

    site = DummySite()
    adapter = CbottleSuperResolution(site=site, provider=None)

    out = await adapter.body(
        dataset=ds,
        output_resolution=[4, 6],
        sampler_steps=18,
        sigma_max=800.0,
        seed=7,
        device="cpu",
        lat_lon=True,
    )

    assert isinstance(out, xr.Dataset)
    # Expect same variables as input
    for v in variables:
        assert v in out.data_vars
        assert out[v].sizes["lat"] == 4
        assert out[v].sizes["lon"] == 6
        assert out[v].sizes["time"] == 1
        assert "hpx" not in out[v].coords

    # SR adapter does not support lat_lon=False; HPX tests intentionally omitted


@pytest.mark.asyncio
async def test_cbottle_sr_missing_vars_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_fake_earth2studio_modules()

    # Patch model to avoid touching real earth2studio
    class FakeCBottleSR:
        @staticmethod
        def load_default_package():
            return object()

        @staticmethod
        def load_model(_package, **kwargs):
            class M:
                def to(self, device: str):
                    return self

                def input_coords(self):
                    return {"variable": ["msl", "tcwv"]}

                output_resolution = (4, 6)

            return M()

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_super_resolution.CBottleSR",
        FakeCBottleSR,
        raising=True,
    )
    lat = np.linspace(-90, 90, 2)
    lon = np.linspace(0, 360, 3, endpoint=False)
    time_vals = np.array([np.datetime64("2025-01-01T00:00:00", "ns")])
    ds = xr.Dataset(
        {"msl": (("time", "lat", "lon"), np.random.rand(1, 2, 3).astype(np.float32))},
        coords=dict(time=time_vals, lat=lat, lon=lon),
    )
    from nv_dfm_lib_weather.cbottle import CbottleSuperResolution

    with pytest.raises(ValueError):
        await CbottleSuperResolution(DummySite(), None).body(
            dataset=ds,
            sampler_steps=18,
            sigma_max=800.0,
            seed=None,
            device="cpu",
            lat_lon=True,
        )


@pytest.mark.asyncio
async def test_cbottle_sr_lat_lon_false_raises() -> None:
    # Directly assert adapter raises on lat_lon=False without mocking
    lat = np.linspace(-90, 90, 2)
    lon = np.linspace(0, 360, 3, endpoint=False)
    time_vals = np.array([np.datetime64("2025-01-01T00:00:00", "ns")])
    ds = xr.Dataset(
        {"msl": (("time", "lat", "lon"), np.random.rand(1, 2, 3).astype(np.float32))},
        coords=dict(time=time_vals, lat=lat, lon=lon),
    )
    from nv_dfm_lib_weather.cbottle import CbottleSuperResolution

    with pytest.raises(ValueError):
        await CbottleSuperResolution(DummySite(), None).body(
            dataset=ds,
            sampler_steps=18,
            sigma_max=800.0,
            seed=None,
            device="cpu",
            lat_lon=False,
        )
