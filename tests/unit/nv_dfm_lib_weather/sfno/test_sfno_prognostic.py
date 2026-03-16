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
from nv_dfm_lib_weather.utils import TorchModelCache


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
    # Ensure module tree exists
    if "earth2studio" not in sys.modules:
        sys.modules["earth2studio"] = types.ModuleType("earth2studio")
    if "earth2studio.models" not in sys.modules:
        sys.modules["earth2studio.models"] = types.ModuleType("earth2studio.models")
    if "earth2studio.models.px" not in sys.modules:
        sys.modules["earth2studio.models.px"] = types.ModuleType(
            "earth2studio.models.px"
        )
    if "earth2studio.models.px.sfno" not in sys.modules:
        sys.modules["earth2studio.models.px.sfno"] = types.ModuleType(
            "earth2studio.models.px.sfno"
        )
    if "earth2studio.data" not in sys.modules:
        sys.modules["earth2studio.data"] = types.ModuleType("earth2studio.data")
    if "earth2studio.data.utils" not in sys.modules:
        sys.modules["earth2studio.data.utils"] = types.ModuleType(
            "earth2studio.data.utils"
        )
    if "earth2studio.utils" not in sys.modules:
        sys.modules["earth2studio.utils"] = types.ModuleType("earth2studio.utils")
    if "earth2studio.utils.coords" not in sys.modules:
        sys.modules["earth2studio.utils.coords"] = types.ModuleType(
            "earth2studio.utils.coords"
        )

    # Provide required symbols
    class _SFNO:
        @staticmethod
        def load_default_package():
            return object()

        @staticmethod
        def load_model(package, device: str = "cpu"):
            return object()

    sys.modules["earth2studio.models.px.sfno"].SFNO = _SFNO

    def _fetch_data_stub(*_args, **_kwargs):
        raise RuntimeError("fetch_data should be monkeypatched in tests")

    def _map_coords_stub(*_args, **_kwargs):
        raise RuntimeError("map_coords should be monkeypatched in tests")

    sys.modules["earth2studio.data.utils"].fetch_data = _fetch_data_stub
    sys.modules["earth2studio.utils.coords"].map_coords = _map_coords_stub


@pytest.mark.asyncio
async def test_sfno_prognostic_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_fake_earth2studio_modules()

    # Arrange an input dataset for two variables required by SFNO
    variables = ["msl", "tcwv"]
    lat = np.linspace(-90, 90, 2)
    lon = np.linspace(0, 360, 3, endpoint=False)
    time_vals = np.array([np.datetime64("2025-01-01T00:00:00", "ns")])
    data_vars = {}
    for v in variables:
        vals = np.random.rand(len(time_vals), len(lat), len(lon)).astype(np.float32)
        data_vars[v] = ("time", "lat", "lon"), vals
    ds = xr.Dataset(data_vars, coords=dict(time=time_vals, lat=lat, lon=lon))

    class FakeSFNOModel:
        def __init__(self) -> None:
            self._device = "cpu"

        def to(self, device: str) -> "FakeSFNOModel":
            self._device = device
            return self

        @staticmethod
        def load_default_package():  # pragma: no cover - trivial
            return object()

        @staticmethod
        def load_model(
            _package, device: str = "cpu"
        ) -> "FakeSFNOModel":  # pragma: no cover - trivial
            return FakeSFNOModel()

        def input_coords(self) -> dict[str, Any]:
            return {
                "batch": [0],
                "time": time_vals,
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": variables,
                "lat": lat,
                "lon": lon,
            }

        def create_iterator(self, x: torch.Tensor, coords: dict[str, Any]):
            # Yield n_steps entries; simulate two steps with 1-time slices
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

    # Patch external dependencies used by the adapter
    monkeypatch.setattr(
        "nv_dfm_lib_weather.sfno._sfno.SFNOModel",
        FakeSFNOModel,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.sfno._sfno.setup_device",
        lambda device, logger: "cpu",
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.sfno._sfno.fetch_data",
        fake_fetch_data,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.sfno._sfno.map_coords",
        fake_map_coords,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.sfno._sfno.setup_device",
        fake_setup_device,
        raising=True,
    )

    from nv_dfm_lib_weather.sfno._sfno import SfnoPrognostic

    site = DummySite()
    adapter = SfnoPrognostic(site=site, provider=None)

    out = await adapter.body(
        dataset=ds,
        n_steps=2,
        seed=1,
        device="cpu",
    )

    assert isinstance(out, xr.Dataset)
    for v in variables:
        assert v in out.data_vars
        assert out[v].sizes["time"] == 2
        assert set(out[v].dims) == {"time", "lat", "lon"}


@pytest.mark.asyncio
async def test_sfno_n_steps_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
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

    class FakeSFNOModel:
        def to(self, device: str) -> "FakeSFNOModel":
            return self

        @staticmethod
        def load_default_package():
            return object()

        @staticmethod
        def load_model(_package, device: str = "cpu") -> "FakeSFNOModel":
            return FakeSFNOModel()

        def input_coords(self) -> dict[str, Any]:
            return {
                "batch": [0],
                "time": time_vals,
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": variables,
                "lat": lat,
                "lon": lon,
            }

        def create_iterator(self, x: torch.Tensor, coords: dict[str, Any]):
            # yield zero steps -> no outputs
            if self._n == 0:
                if False:
                    yield  # pragma: no cover
            else:
                for i in range(self._n):
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

    def set_model(monkeypatch, n_steps: int):
        def load_model(_package, device: str = "cpu"):
            m = FakeSFNOModel()
            m._n = n_steps
            return m

        monkeypatch.setattr(
            "nv_dfm_lib_weather.sfno._sfno.SFNOModel.load_model",
            load_model,
            raising=False,
        )

    from nv_dfm_lib_weather.sfno import SfnoPrognostic

    site = DummySite()
    adapter = SfnoPrognostic(site=site, provider=None)

    # n_steps = 1 should work
    set_model(monkeypatch, 1)
    # stub fetch_data and map_coords for this call
    monkeypatch.setattr(
        "nv_dfm_lib_weather.sfno._sfno.fetch_data",
        lambda provider, times, vars, device="cpu": (
            torch.rand(1, 1, len(variables), len(lat), len(lon)),
            {
                "batch": np.array([0]),
                "time": times,
                "variable": np.array(variables),
                "lat": lat,
                "lon": lon,
            },
        ),
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.sfno._sfno.map_coords",
        lambda x, c, t: (x, c),
        raising=True,
    )
    out = await adapter.body(
        dataset=ds,
        n_steps=1,
        seed=None,
        device="cpu",
    )
    for v in variables:
        assert out[v].sizes["time"] == 1

    # n_steps = 0 should raise due to empty iterator
    set_model(monkeypatch, 0)
    # Use fresh adapter/site to avoid cached model reuse
    fresh_adapter = SfnoPrognostic(site=DummySite(), provider=None)
    with pytest.raises(RuntimeError):
        await fresh_adapter.body(
            dataset=ds,
            n_steps=0,
            seed=None,
            device="cpu",
        )


@pytest.mark.asyncio
async def test_sfno_missing_required_vars_raise() -> None:
    lat = np.linspace(-90, 90, 2)
    lon = np.linspace(0, 360, 3, endpoint=False)
    time_vals = np.array([np.datetime64("2025-01-01T00:00:00", "ns")])
    ds = xr.Dataset(
        {"msl": (("time", "lat", "lon"), np.random.rand(1, 2, 3).astype(np.float32))},
        coords=dict(time=time_vals, lat=lat, lon=lon),
    )
    # Ensure stubs exist and patch adapter-level functions to avoid touching real models
    ensure_fake_earth2studio_modules()
    import nv_dfm_lib_weather.sfno._sfno as sfno_mod

    # Provide a fake model with .to
    class _FakeModel:
        def to(self, _device: str):
            return self

        def input_coords(self) -> dict[str, Any]:
            return {"variable": np.array(["msl", "tcwv"])}

    sfno_mod.SFNOModel.load_default_package = staticmethod(lambda: object())
    sfno_mod.SFNOModel.load_model = staticmethod(
        lambda package, device="cpu": _FakeModel()
    )
    sfno_mod.setup_device = lambda device, logger: "cpu"
    from nv_dfm_lib_weather.sfno._sfno import SfnoPrognostic

    with pytest.raises(ValueError):
        await SfnoPrognostic(DummySite(), None).body(
            dataset=ds,
            n_steps=1,
            seed=None,
            device="cpu",
        )
