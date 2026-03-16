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

    def warning(
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
    e2s.data.utils = types.ModuleType("earth2studio.data.utils")
    e2s.lexicon = types.ModuleType("earth2studio.lexicon")
    e2s.lexicon.cbottle = types.ModuleType("earth2studio.lexicon.cbottle")

    class _FakeLex:
        VOCAB = {"msl": 1, "tcwv": 1, "u10m": 1, "v10m": 1}

    e2s.lexicon.cbottle.CBottleLexicon = _FakeLex

    class _Placeholder:
        pass

    e2s.models.dx.CBottleInfill = _Placeholder
    e2s.models.dx.CBottleTCGuidance = _Placeholder

    def _fetch_data_stub(*_args, **_kwargs):
        raise RuntimeError("fetch_data should be monkeypatched in tests")

    e2s.data.utils.fetch_data = _fetch_data_stub

    sys.modules["earth2studio"] = e2s
    sys.modules["earth2studio.data"] = e2s.data
    sys.modules["earth2studio.models"] = e2s.models
    sys.modules["earth2studio.models.dx"] = e2s.models.dx
    sys.modules["earth2studio.data.utils"] = e2s.data.utils
    sys.modules["earth2studio.lexicon"] = e2s.lexicon
    sys.modules["earth2studio.lexicon.cbottle"] = e2s.lexicon.cbottle


@pytest.mark.asyncio
async def test_cbottle_infilling_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_fake_earth2studio_modules()
    # Arrange a tiny input dataset with CBottle-like variables
    input_variables = ["u10m", "v10m"]
    lat = np.linspace(-90, 90, 2)
    lon = np.linspace(0, 360, 3, endpoint=False)
    time_vals = np.array(
        [
            np.datetime64("2025-01-01T00:00:00", "ns"),
            np.datetime64("2025-01-02T00:00:00", "ns"),
        ]
    )
    _ = np.random.rand(len(input_variables), len(time_vals), len(lat), len(lon)).astype(
        np.float32
    )
    ds = xr.Dataset(
        {
            v: ("time", np.random.rand(len(time_vals)).astype(np.float32))
            for v in input_variables
        }
    )
    # Convert variables to 2D fields per time using broadcasting
    for v in input_variables:
        vals = np.random.rand(len(time_vals), len(lat), len(lon)).astype(np.float32)
        ds[v] = ("time", "lat", "lon"), vals
    ds = ds.assign_coords(time=time_vals, lat=lat, lon=lon)

    class FakeCBottleInfillModel:
        def __init__(self) -> None:
            self._device = "cpu"

        def to(self, device: str) -> "FakeCBottleInfillModel":
            self._device = device
            return self

        def __call__(self, x: torch.Tensor, coords: dict[str, Any]):
            # x dims expected [batch, time, variable, lat, lon]
            assert isinstance(x, torch.Tensor)
            time_len = x.shape[1]
            lat_len = x.shape[3]
            lon_len = x.shape[4]
            from earth2studio.lexicon.cbottle import CBottleLexicon as RealLex

            all_vars = list(RealLex.VOCAB.keys())
            # Output dims [time, lead_time, variable, lat, lon]
            out = torch.rand(
                time_len, 1, len(all_vars), lat_len, lon_len, dtype=torch.float32
            )
            out_coords = {
                "time": coords["time"],
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(all_vars),
                "lat": coords["lat"],
                "lon": coords["lon"],
            }
            return out, out_coords

    class FakeCBottleInfill:
        @staticmethod
        def load_default_package():  # pragma: no cover - trivial
            return object()

        @staticmethod
        def load_model(
            _package, input_variables, sampler_steps, sigma_max
        ):  # pragma: no cover - trivial
            assert isinstance(input_variables, list)
            assert isinstance(sampler_steps, int)
            assert isinstance(sigma_max, float)
            return FakeCBottleInfillModel()

    def fake_fetch_data(provider, times, variables, device="cpu"):
        # provider returns a DataArray when called; we simulate it here directly
        da = provider(times, variables)
        # Convert to torch tensor with expected dims [batch, time, variable, lat, lon]
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

    def fake_setup_device(device: str | None, _logger) -> str:
        return device or "cpu"

    # Patch dependencies in module under test
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_infilling.CBottleInfill",
        FakeCBottleInfill,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_infilling.fetch_data",
        fake_fetch_data,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_infilling.setup_device",
        lambda device, logger: "cpu",
        raising=True,
    )

    from nv_dfm_lib_weather.cbottle import CbottleInfilling

    site = DummySite()
    adapter = CbottleInfilling(site=site, provider=None)

    # Act
    out = await adapter.body(
        dataset=ds,
        input_variables=input_variables,
        sampler_steps=18,
        sigma_max=80.0,
        seed=42,
        device="cpu",
    )

    # Assert
    assert isinstance(out, xr.Dataset)
    # Should contain all CBottle variables (from lexicon)
    from earth2studio.lexicon.cbottle import CBottleLexicon as RealLex

    expected_vars = list(RealLex.VOCAB.keys())
    for v in expected_vars:
        assert v in out.data_vars
        assert set(out[v].dims) == {"time", "lat", "lon"}
    # output has one time due to averaging
    assert out.sizes["time"] == 1


@pytest.mark.asyncio
async def test_cbottle_infilling_missing_vars_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_fake_earth2studio_modules()
    # Avoid loading models in adapter for this negative case
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_infilling.CBottleInfill.load_default_package",
        lambda: object(),
        raising=False,
    )

    class _FakeModel:
        def to(self, _device: str):
            return self

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_infilling.CBottleInfill.load_model",
        lambda *a, **k: _FakeModel(),
        raising=False,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_infilling.setup_device",
        lambda device, logger: "cpu",
        raising=True,
    )
    # Dataset with only 'u10m'
    lat = np.linspace(-90, 90, 2)
    lon = np.linspace(0, 360, 3, endpoint=False)
    time_vals = np.array([np.datetime64("2025-01-01T00:00:00", "ns")])
    ds = xr.Dataset(
        {"u10m": (("time", "lat", "lon"), np.random.rand(1, 2, 3).astype(np.float32))},
        coords=dict(time=time_vals, lat=lat, lon=lon),
    )
    from nv_dfm_lib_weather.cbottle import CbottleInfilling

    with pytest.raises(ValueError):
        await CbottleInfilling(DummySite(), None).body(
            dataset=ds,
            input_variables=["u10m", "v10m"],
            sampler_steps=18,
            sigma_max=80.0,
            seed=None,
            device="cpu",
        )


@pytest.mark.asyncio
async def test_cbottle_infilling_no_time_coord_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Avoid model calls before the adapter checks for 'time'
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_infilling.CBottleInfill.load_default_package",
        lambda: object(),
        raising=False,
    )

    class _FakeModel3:
        def to(self, _device: str):
            return self

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_infilling.CBottleInfill.load_model",
        lambda *a, **k: _FakeModel3(),
        raising=False,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_infilling.setup_device",
        lambda device, logger: "cpu",
        raising=True,
    )
    ds = xr.Dataset(
        {"u10m": (("lat", "lon"), np.random.rand(2, 3).astype(np.float32))},
        coords=dict(
            lat=np.linspace(-90, 90, 2), lon=np.linspace(0, 360, 3, endpoint=False)
        ),
    )
    from nv_dfm_lib_weather.cbottle import CbottleInfilling

    with pytest.raises(ValueError):
        await CbottleInfilling(DummySite(), None).body(
            dataset=ds,
            input_variables=["u10m"],
            sampler_steps=18,
            sigma_max=80.0,
            seed=None,
            device="cpu",
        )
