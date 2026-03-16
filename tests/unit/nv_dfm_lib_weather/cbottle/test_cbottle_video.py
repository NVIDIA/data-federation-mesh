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
    e2s.models = types.ModuleType("earth2studio.models")
    e2s.models.px = types.ModuleType("earth2studio.models.px")
    e2s.models.px.cbottle_video = types.ModuleType(
        "earth2studio.models.px.cbottle_video"
    )
    e2s.data = types.ModuleType("earth2studio.data")
    e2s.data.utils = types.ModuleType("earth2studio.data.utils")
    e2s.utils = types.ModuleType("earth2studio.utils")
    e2s.utils.coords = types.ModuleType("earth2studio.utils.coords")
    e2s.lexicon = types.ModuleType("earth2studio.lexicon")
    e2s.lexicon.cbottle = types.ModuleType("earth2studio.lexicon.cbottle")

    class _Placeholder:
        pass

    e2s.models.px.cbottle_video.CBottleVideo = _Placeholder
    # Stub symbol used by other adapters to avoid import-time failures
    e2s.models = e2s.models
    e2s.models.dx = types.ModuleType("earth2studio.models.dx")
    e2s.models.dx.CBottleTCGuidance = _Placeholder
    e2s.models.dx.CBottleInfill = _Placeholder
    # Stub cbottle_sr submodule and symbol
    e2s.models.dx.cbottle_sr = types.ModuleType("earth2studio.models.dx.cbottle_sr")
    e2s.models.dx.cbottle_sr.CBottleSR = _Placeholder
    sys.modules["earth2studio.models.dx"] = e2s.models.dx
    sys.modules["earth2studio.models.dx.cbottle_sr"] = e2s.models.dx.cbottle_sr
    # Stub CBottle3D used by cbottle imports
    e2s.data.CBottle3D = _Placeholder

    # Stub CBottleLexicon
    class _FakeLex:
        VOCAB = {"msl": 1, "tcwv": 1, "u10m": 1, "v10m": 1}

    e2s.lexicon.cbottle.CBottleLexicon = _FakeLex

    def _fetch_data_stub(*_args, **_kwargs):
        raise RuntimeError("fetch_data should be monkeypatched in tests")

    def _map_coords_stub(*_args, **_kwargs):
        raise RuntimeError("map_coords should be monkeypatched in tests")

    e2s.data.utils.fetch_data = _fetch_data_stub
    e2s.utils.coords.map_coords = _map_coords_stub

    sys.modules["earth2studio"] = e2s
    sys.modules["earth2studio.models"] = e2s.models
    sys.modules["earth2studio.models.px"] = e2s.models.px
    sys.modules["earth2studio.models.px.cbottle_video"] = e2s.models.px.cbottle_video
    sys.modules["earth2studio.data"] = e2s.data
    sys.modules["earth2studio.data.utils"] = e2s.data.utils
    sys.modules["earth2studio.utils"] = e2s.utils
    sys.modules["earth2studio.utils.coords"] = e2s.utils.coords
    sys.modules["earth2studio.lexicon"] = e2s.lexicon
    sys.modules["earth2studio.lexicon.cbottle"] = e2s.lexicon.cbottle


@pytest.mark.asyncio
async def test_cbottle_video_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_fake_earth2studio_modules()
    variables = ["msl", "tcwv"]
    lat = np.linspace(-90, 90, 2)
    lon = np.linspace(0, 360, 3, endpoint=False)
    time_vals = np.array([np.datetime64("2025-01-01T00:00:00", "ns")])

    data_vars = {}
    for v in variables:
        vals = np.random.rand(len(time_vals), len(lat), len(lon)).astype(np.float32)
        data_vars[v] = ("time", "lat", "lon"), vals
    ds = xr.Dataset(data_vars, coords=dict(time=time_vals, lat=lat, lon=lon))

    class FakeCBottleVideoModel:
        def __init__(self) -> None:
            self._device = "cpu"

        def to(self, device: str) -> "FakeCBottleVideoModel":
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

        def create_iterator(self, x: torch.Tensor, coords: dict[str, Any]):
            # Yield n_steps entries; we simulate two steps
            for i in range(2):
                # x: expect [batch, time(=1), variable, lat, lon]
                out = torch.rand(
                    x.shape[0],
                    1,
                    len(variables),
                    len(lat),
                    len(lon),
                    dtype=torch.float32,
                )
                step_coords = {
                    "batch": np.array([0]),
                    "time": coords["time"],
                    "lead_time": np.array([np.timedelta64(i, "h")]),
                    "variable": coords["variable"],
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                }
                yield out, step_coords

    class FakeCBottleVideo:
        @staticmethod
        def load_default_package():  # pragma: no cover - trivial
            return object()

        @staticmethod
        def load_model(_package, seed=None, lat_lon=True):  # pragma: no cover - trivial
            return FakeCBottleVideoModel()

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
        return x, coords

    def fake_setup_device(device: str | None, _logger) -> str:
        return device or "cpu"

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.CBottleVideoModel",
        FakeCBottleVideo,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.fetch_data",
        fake_fetch_data,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.map_coords",
        fake_map_coords,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.setup_device",
        fake_setup_device,
        raising=True,
    )

    from nv_dfm_lib_weather.cbottle import CbottleVideo

    site = DummySite()
    adapter = CbottleVideo(site=site, provider=None)

    out = await adapter.body(
        dataset=ds,
        n_steps=2,
        seed=1,
        device="cpu",
        lat_lon=True,
    )

    assert isinstance(out, xr.Dataset)
    for v in variables:
        assert v in out.data_vars
        assert out[v].sizes["time"] == 2
        assert "hpx" not in out[v].coords


@pytest.mark.asyncio
async def test_cbottle_video_hpx_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_fake_earth2studio_modules()

    variables = ["msl", "tcwv"]
    hpx = np.arange(6)
    time_vals = np.array([np.datetime64("2025-01-01T00:00:00", "ns")])
    data_vars = {}
    for v in variables:
        vals = np.random.rand(len(time_vals), len(hpx)).astype(np.float32)
        data_vars[v] = ("time", "hpx"), vals
    ds = xr.Dataset(data_vars, coords=dict(time=time_vals, hpx=hpx))

    class FakeCBottleVideoModel:
        def __init__(self) -> None:
            self._device = "cpu"

        def to(self, device: str) -> "FakeCBottleVideoModel":
            self._device = device
            return self

        def input_coords(self) -> dict[str, Any]:
            return {
                "batch": [0],
                "time": time_vals,
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": variables,
                "hpx": hpx,
            }

        def create_iterator(self, x: torch.Tensor, coords: dict[str, Any]):
            for i in range(2):
                out = torch.rand(
                    x.shape[0], 1, len(variables), len(hpx), dtype=torch.float32
                )
                step_coords = {
                    "batch": np.array([0]),
                    "time": coords["time"],
                    "lead_time": np.array([np.timedelta64(i, "h")]),
                    "variable": coords["variable"],
                    "hpx": coords["hpx"],
                }
                yield out, step_coords

    class FakeCBottleVideo:
        @staticmethod
        def load_default_package():  # pragma: no cover - trivial
            return object()

        @staticmethod
        def load_model(_package, seed=None, lat_lon=True):  # pragma: no cover - trivial
            assert lat_lon is False
            return FakeCBottleVideoModel()

    def fake_fetch_data(provider, times, required_variables, device="cpu"):
        da = provider(times, required_variables)
        da = da.transpose("time", "variable", "hpx")
        x = torch.from_numpy(da.values).unsqueeze(0)
        coords = {
            "batch": np.array([0]),
            "time": da.coords["time"].values,
            "variable": da.coords["variable"].values,
            "hpx": da.coords["hpx"].values,
        }
        return x, coords

    def fake_map_coords(
        x: torch.Tensor, coords: dict[str, Any], _target: dict[str, Any]
    ):
        return x, coords

    def fake_setup_device(device: str | None, _logger) -> str:
        return device or "cpu"

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.CBottleVideoModel",
        FakeCBottleVideo,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.fetch_data",
        fake_fetch_data,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.map_coords",
        fake_map_coords,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.setup_device",
        fake_setup_device,
        raising=True,
    )

    from nv_dfm_lib_weather.cbottle import CbottleVideo

    site = DummySite()
    adapter = CbottleVideo(site=site, provider=None)

    out = await adapter.body(
        dataset=ds,
        n_steps=2,
        seed=1,
        device="cpu",
        lat_lon=False,
    )

    assert isinstance(out, xr.Dataset)
    for v in variables:
        assert v in out.data_vars
        assert out[v].sizes["time"] == 2
        assert (
            "hpx" in out[v].coords
            and "lat" not in out[v].coords
            and "lon" not in out[v].coords
        )


@pytest.mark.asyncio
async def test_cbottle_video_n_steps_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_fake_earth2studio_modules()

    variables = ["msl", "tcwv"]
    lat = np.linspace(-90, 90, 2)
    lon = np.linspace(0, 360, 3, endpoint=False)
    time_vals = np.array([np.datetime64("2025-01-01T00:00:00", "ns")])
    ds = xr.Dataset(
        {
            v: (("time", "lat", "lon"), np.random.rand(1, 2, 3).astype(np.float32))
            for v in variables
        },
        coords=dict(time=time_vals, lat=lat, lon=lon),
    )

    class FakeCBottleVideoModel:
        def __init__(self, steps: int) -> None:
            self._steps = steps

        def to(self, device: str) -> "FakeCBottleVideoModel":
            return self

        def input_coords(self) -> dict[str, Any]:
            return {
                "batch": [0],
                "time": time_vals,
                "variable": variables,
                "lat": lat,
                "lon": lon,
            }

        def create_iterator(self, x: torch.Tensor, coords: dict[str, Any]):
            for i in range(self._steps):
                out = torch.rand(
                    x.shape[0],
                    1,
                    len(variables),
                    len(lat),
                    len(lon),
                    dtype=torch.float32,
                )
                step_coords = {
                    "batch": np.array([0]),
                    "time": coords["time"],
                    "lead_time": np.array([np.timedelta64(i, "h")]),
                    "variable": coords["variable"],
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                }
                yield out, step_coords

    class FakeCBottleVideo:
        @staticmethod
        def load_default_package():
            return object()

    def set_model(monkeypatch, steps: int):
        def load_model(_package, seed=None, lat_lon=True):
            return FakeCBottleVideoModel(steps)

        monkeypatch.setattr(
            "nv_dfm_lib_weather.cbottle._cbottle_video.CBottleVideoModel",
            FakeCBottleVideo,
            raising=True,
        )
        monkeypatch.setattr(
            "nv_dfm_lib_weather.cbottle._cbottle_video.CBottleVideoModel.load_model",
            load_model,
            raising=False,
        )
        # Ensure dependent functions are stubbed
        monkeypatch.setattr(
            "nv_dfm_lib_weather.cbottle._cbottle_video.fetch_data",
            lambda provider, times, required_variables, device="cpu": (
                torch.rand(1, 1, len(variables), len(lat), len(lon)),
                {
                    "batch": np.array([0]),
                    "time": time_vals,
                    "variable": np.array(variables),
                    "lat": lat,
                    "lon": lon,
                },
            ),
            raising=True,
        )
        monkeypatch.setattr(
            "nv_dfm_lib_weather.cbottle._cbottle_video.map_coords",
            lambda x, c, t: (x, c),
            raising=True,
        )
        monkeypatch.setattr(
            "nv_dfm_lib_weather.cbottle._cbottle_video.setup_device",
            lambda device, logger: "cpu",
            raising=True,
        )

    from nv_dfm_lib_weather.cbottle import CbottleVideo

    adapter = CbottleVideo(DummySite(), None)

    # n_steps = 1
    set_model(monkeypatch, 1)
    out = await adapter.body(
        dataset=ds, n_steps=1, seed=None, device="cpu", lat_lon=True
    )
    for v in variables:
        assert out[v].sizes["time"] == 1

    # n_steps = 0 -> RuntimeError
    set_model(monkeypatch, 0)
    # Use a fresh adapter/site to avoid cached model reuse
    fresh_adapter = CbottleVideo(DummySite(), None)
    with pytest.raises(RuntimeError):
        await fresh_adapter.body(
            dataset=ds, n_steps=0, seed=None, device="cpu", lat_lon=True
        )


@pytest.mark.asyncio
async def test_cbottle_video_uses_first_time(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_fake_earth2studio_modules()
    variables = ["msl", "tcwv"]
    lat = np.linspace(-90, 90, 2)
    lon = np.linspace(0, 360, 3, endpoint=False)
    time_vals = np.array(
        [
            np.datetime64("2025-01-01T00:00:00", "ns"),
            np.datetime64("2025-01-02T00:00:00", "ns"),
        ]
    )
    vals1 = np.random.rand(len(time_vals), len(lat), len(lon)).astype(np.float32)
    vals2 = np.random.rand(len(time_vals), len(lat), len(lon)).astype(np.float32)
    ds = xr.Dataset(
        {
            "msl": (("time", "lat", "lon"), vals1),
            "tcwv": (("time", "lat", "lon"), vals2),
        },
        coords=dict(time=time_vals, lat=lat, lon=lon),
    )

    class FakeCBottleVideoModel:
        def to(self, device: str) -> "FakeCBottleVideoModel":
            return self

        def input_coords(self) -> dict[str, Any]:
            return {
                "batch": [0],
                "time": np.array([time_vals[0]]),
                "variable": variables,
                "lat": lat,
                "lon": lon,
            }

        def create_iterator(self, x: torch.Tensor, coords: dict[str, Any]):
            for i in range(2):
                out = torch.rand(
                    x.shape[0],
                    1,
                    len(variables),
                    len(lat),
                    len(lon),
                    dtype=torch.float32,
                )
                step_coords = {
                    "batch": np.array([0]),
                    "time": coords["time"],
                    "lead_time": np.array([np.timedelta64(i, "h")]),
                    "variable": coords["variable"],
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                }
                yield out, step_coords

    class FakeCBottleVideo:
        @staticmethod
        def load_default_package():
            return object()

        @staticmethod
        def load_model(_package, seed=None, lat_lon=True):
            return FakeCBottleVideoModel()

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.CBottleVideoModel",
        FakeCBottleVideo,
        raising=True,
    )
    # fetch_data returns x with dims [batch, time, variable, lat, lon]
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.fetch_data",
        lambda p, t, v, device="cpu": (
            torch.rand(1, 1, len(variables), len(lat), len(lon)),
            {
                "batch": np.array([0]),
                "time": np.array([time_vals[0]]),
                "variable": np.array(variables),
                "lat": lat,
                "lon": lon,
            },
        ),
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.map_coords",
        lambda x, c, t: (x, c),
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_video.setup_device",
        lambda device, logger: "cpu",
        raising=True,
    )

    from nv_dfm_lib_weather.cbottle import CbottleVideo

    out = await CbottleVideo(DummySite(), None).body(
        dataset=ds, n_steps=2, seed=None, device="cpu", lat_lon=True
    )
    # First time should be the dataset's first time
    assert np.datetime64(out.coords["time"].values[0]) == time_vals[0]
