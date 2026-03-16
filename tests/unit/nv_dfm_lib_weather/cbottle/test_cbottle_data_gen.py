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

from datetime import datetime
from typing import Any
import sys
import types

import numpy as np
import pytest
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
    # Create a minimal fake earth2studio module tree to satisfy imports in adapters
    if "earth2studio" in sys.modules:
        return
    e2s = types.ModuleType("earth2studio")
    e2s.data = types.ModuleType("earth2studio.data")
    e2s.models = types.ModuleType("earth2studio.models")
    e2s.models.dx = types.ModuleType("earth2studio.models.dx")
    e2s.models.px = types.ModuleType("earth2studio.models.px")
    e2s.models.dx.cbottle_sr = types.ModuleType("earth2studio.models.dx.cbottle_sr")
    e2s.models.px.cbottle_video = types.ModuleType(
        "earth2studio.models.px.cbottle_video"
    )
    e2s.data.utils = types.ModuleType("earth2studio.data.utils")
    e2s.utils = types.ModuleType("earth2studio.utils")
    e2s.utils.coords = types.ModuleType("earth2studio.utils.coords")
    e2s.lexicon = types.ModuleType("earth2studio.lexicon")
    e2s.lexicon.cbottle = types.ModuleType("earth2studio.lexicon.cbottle")

    class _FakeLex:
        VOCAB = {"msl": 1, "tcwv": 1, "u10m": 1, "v10m": 1}

    # Populate placeholders referenced at import time
    e2s.lexicon.cbottle.CBottleLexicon = _FakeLex

    class _Placeholder:  # generic placeholder class for imports
        pass

    e2s.data.CBottle3D = _Placeholder
    e2s.models.dx.CBottleInfill = _Placeholder
    e2s.models.dx.CBottleTCGuidance = _Placeholder
    e2s.models.dx.cbottle_sr.CBottleSR = _Placeholder
    e2s.models.px.cbottle_video.CBottleVideo = _Placeholder

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
    sys.modules["earth2studio.models.px"] = e2s.models.px
    sys.modules["earth2studio.models.dx.cbottle_sr"] = e2s.models.dx.cbottle_sr
    sys.modules["earth2studio.models.px.cbottle_video"] = e2s.models.px.cbottle_video
    sys.modules["earth2studio.data.utils"] = e2s.data.utils
    sys.modules["earth2studio.utils"] = e2s.utils
    sys.modules["earth2studio.utils.coords"] = e2s.utils.coords
    sys.modules["earth2studio.lexicon"] = e2s.lexicon
    sys.modules["earth2studio.lexicon.cbottle"] = e2s.lexicon.cbottle


@pytest.mark.asyncio
async def test_cbottle_datagen_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_fake_earth2studio_modules()
    # Arrange: create a simple dataset behavior for the mocked CBottle3D model
    variables = ["msl", "tcwv"]
    time_str = "2022-09-05T00:00:00"
    timestamp = datetime.fromisoformat(time_str)
    n_samples = 2

    class FakeCBottleModel:
        def __init__(self) -> None:
            self._device = "cpu"

        def to(self, device: str) -> "FakeCBottleModel":
            self._device = device
            return self

        def __call__(self, timestamps, variables_param):
            # Return a DataArray [time, variable, lat, lon]
            assert variables_param == variables
            assert len(timestamps) == n_samples
            lat = np.linspace(-90, 90, 3)
            lon = np.linspace(0, 360, 4, endpoint=False)
            time_vals = np.array(
                [np.datetime64(timestamp, "ns") for _ in range(n_samples)]
            )
            data = np.random.rand(n_samples, len(variables), len(lat), len(lon)).astype(
                np.float32
            )
            return xr.DataArray(
                data,
                coords={
                    "time": time_vals,
                    "variable": variables,
                    "lat": lat,
                    "lon": lon,
                },
                dims=["time", "variable", "lat", "lon"],
            )

    class FakeCBottle3D:
        @staticmethod
        def load_default_package():  # pragma: no cover - trivial
            return object()

        @staticmethod
        def load_model(_package, seed=None, lat_lon=True):  # pragma: no cover - trivial
            assert lat_lon in (True, False)
            return FakeCBottleModel()

    def fake_setup_device(device: str | None, _logger) -> str:
        return device or "cpu"

    # Patch external dependencies used by the adapter
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.CBottle3D",
        FakeCBottle3D,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.setup_device",
        fake_setup_device,
        raising=True,
    )

    # Import after monkeypatch so the adapter picks up our fakes
    from nv_dfm_lib_weather.cbottle import CbottleDataGen

    site = DummySite()
    adapter = CbottleDataGen(site=site, provider=None)

    # Act
    ds = await adapter.body(
        variables=variables,
        time=time_str,
        n_samples=n_samples,
        seed=123,
        device="cpu",
        lat_lon=True,
    )

    # Assert: dataset structure and coords
    assert isinstance(ds, xr.Dataset)
    # Variables should be split into dataset variables
    for v in variables:
        assert v in ds.data_vars
        assert "time" in ds[v].coords
        assert "lat" in ds[v].coords
        assert "lon" in ds[v].coords
    # time dtype should be datetime64
    assert np.issubdtype(ds.coords["time"].dtype, np.datetime64)
    # Ensure HPX is not present when lat_lon=True
    for v in variables:
        assert "hpx" not in ds[v].coords


@pytest.mark.asyncio
async def test_cbottle_datagen_seed_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_fake_earth2studio_modules()
    variables = ["msl", "tcwv"]
    time_str = "2022-09-05T00:00:00"

    class FakeCBottleModel:
        def __init__(self, seed: int | None) -> None:
            self._device = "cpu"
            self._seed = seed if seed is not None else 0

        def to(self, device: str) -> "FakeCBottleModel":
            self._device = device
            return self

        def __call__(self, timestamps, variables_param):
            rs = np.random.RandomState(self._seed)
            lat = np.linspace(-90, 90, 3)
            lon = np.linspace(0, 360, 4, endpoint=False)
            time_vals = np.array([np.datetime64(timestamps[0], "ns")])
            data = rs.rand(1, len(variables_param), len(lat), len(lon)).astype(
                np.float32
            )
            return xr.DataArray(
                data,
                coords={
                    "time": time_vals,
                    "variable": variables_param,
                    "lat": lat,
                    "lon": lon,
                },
                dims=["time", "variable", "lat", "lon"],
            )

    class FakeCBottle3D:
        @staticmethod
        def load_default_package():
            return object()

        @staticmethod
        def load_model(_package, seed=None, lat_lon=True):
            return FakeCBottleModel(seed)

    def fake_setup_device(device: str | None, _logger) -> str:
        return device or "cpu"

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.CBottle3D",
        FakeCBottle3D,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.setup_device",
        fake_setup_device,
        raising=True,
    )

    from nv_dfm_lib_weather.cbottle import CbottleDataGen

    site = DummySite()
    adapter = CbottleDataGen(site=site, provider=None)

    ds1 = await adapter.body(
        variables=variables,
        time=time_str,
        n_samples=1,
        seed=111,
        device="cpu",
        lat_lon=True,
    )
    ds2 = await adapter.body(
        variables=variables,
        time=time_str,
        n_samples=1,
        seed=111,
        device="cpu",
        lat_lon=True,
    )
    ds3 = await adapter.body(
        variables=variables,
        time=time_str,
        n_samples=1,
        seed=222,
        device="cpu",
        lat_lon=True,
    )

    for v in variables:
        np.testing.assert_allclose(ds1[v].values, ds2[v].values)
        assert not np.allclose(ds1[v].values, ds3[v].values)


@pytest.mark.asyncio
async def test_cbottle_datagen_multiple_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_fake_earth2studio_modules()
    variables = ["msl"]
    time_str = "2022-09-05T00:00:00Z"

    class FakeCBottleModel:
        def to(self, device: str) -> "FakeCBottleModel":
            return self

        def __call__(self, timestamps, variables_param):
            lat = np.linspace(-90, 90, 2)
            lon = np.linspace(0, 360, 3, endpoint=False)
            data = np.random.rand(
                len(timestamps), len(variables_param), len(lat), len(lon)
            ).astype(np.float32)
            return xr.DataArray(
                data,
                coords={
                    "time": np.array([np.datetime64(t, "ns") for t in timestamps]),
                    "variable": variables_param,
                    "lat": lat,
                    "lon": lon,
                },
                dims=["time", "variable", "lat", "lon"],
            )

    class FakeCBottle3D:
        @staticmethod
        def load_default_package():
            return object()

        @staticmethod
        def load_model(_package, seed=None, lat_lon=True):
            return FakeCBottleModel()

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.CBottle3D",
        FakeCBottle3D,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.setup_device",
        lambda device, logger: "cpu",
        raising=True,
    )

    from nv_dfm_lib_weather.cbottle import CbottleDataGen

    ds = await CbottleDataGen(DummySite(), None).body(
        variables=variables,
        time=time_str,
        n_samples=3,
        seed=None,
        device=None,
        lat_lon=True,
    )
    assert ds.sizes["time"] == 3


@pytest.mark.asyncio
async def test_cbottle_datagen_invalid_variable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_fake_earth2studio_modules()

    # Patch early to avoid attempting to load real package
    class FakeCBottle3D:
        @staticmethod
        def load_default_package():
            return object()

        @staticmethod
        def load_model(_package, seed=None, lat_lon=True):
            return object()

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.CBottle3D",
        FakeCBottle3D,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.setup_device",
        lambda device, logger: "cpu",
        raising=True,
    )
    from nv_dfm_lib_weather.cbottle import CbottleDataGen

    with pytest.raises(Exception):
        await CbottleDataGen(DummySite(), None).body(
            variables=["bogus"],
            time="2022-09-05T00:00:00",
            n_samples=1,
            seed=None,
            device=None,
            lat_lon=True,
        )


@pytest.mark.asyncio
async def test_cbottle_datagen_time_parsing_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_fake_earth2studio_modules()

    class FakeCBottleModel:
        def to(self, device: str) -> "FakeCBottleModel":
            return self

        def __call__(self, timestamps, variables_param):
            lat = np.linspace(-90, 90, 1)
            lon = np.linspace(0, 360, 1, endpoint=False)
            data = np.random.rand(len(timestamps), len(variables_param), 1, 1).astype(
                np.float32
            )
            return xr.DataArray(
                data,
                coords={
                    "time": np.array([np.datetime64(t, "ns") for t in timestamps]),
                    "variable": variables_param,
                    "lat": lat,
                    "lon": lon,
                },
                dims=["time", "variable", "lat", "lon"],
            )

    class FakeCBottle3D:
        @staticmethod
        def load_default_package():
            return object()

        @staticmethod
        def load_model(_package, seed=None, lat_lon=True):
            return FakeCBottleModel()

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.CBottle3D",
        FakeCBottle3D,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.setup_device",
        lambda device, logger: "cpu",
        raising=True,
    )
    from nv_dfm_lib_weather.cbottle import CbottleDataGen

    adapter = CbottleDataGen(DummySite(), None)
    ds_iso_z = await adapter.body(
        variables=["msl"],
        time="2022-09-05T00:00:00Z",
        n_samples=2,
        seed=None,
        device=None,
        lat_lon=True,
    )
    ds_dt = await adapter.body(
        variables=["msl"],
        time=datetime.fromisoformat("2022-09-05T00:00:00"),
        n_samples=2,
        seed=None,
        device=None,
        lat_lon=True,
    )
    assert ds_iso_z.sizes["time"] == 2 and ds_dt.sizes["time"] == 2


@pytest.mark.asyncio
async def test_cbottle_datagen_hpx_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_fake_earth2studio_modules()
    variables = ["msl", "tcwv"]
    time_str = "2022-09-05T00:00:00"
    timestamp = datetime.fromisoformat(time_str)
    n_samples = 1

    class FakeCBottleModel:
        def __init__(self) -> None:
            self._device = "cpu"

        def to(self, device: str) -> "FakeCBottleModel":
            self._device = device
            return self

        def __call__(self, timestamps, variables_param):
            assert variables_param == variables
            assert len(timestamps) == n_samples
            hpx = np.arange(8)
            time_vals = np.array(
                [np.datetime64(timestamp, "ns") for _ in range(n_samples)]
            )
            data = np.random.rand(n_samples, len(variables), len(hpx)).astype(
                np.float32
            )
            return xr.DataArray(
                data,
                coords={
                    "time": time_vals,
                    "variable": variables,
                    "hpx": hpx,
                },
                dims=["time", "variable", "hpx"],
            )

    class FakeCBottle3D:
        @staticmethod
        def load_default_package():  # pragma: no cover - trivial
            return object()

        @staticmethod
        def load_model(_package, seed=None, lat_lon=True):  # pragma: no cover - trivial
            assert lat_lon is False
            return FakeCBottleModel()

    def fake_setup_device(device: str | None, _logger) -> str:
        return device or "cpu"

    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.CBottle3D",
        FakeCBottle3D,
        raising=True,
    )
    monkeypatch.setattr(
        "nv_dfm_lib_weather.cbottle._cbottle_data_gen.setup_device",
        fake_setup_device,
        raising=True,
    )

    from nv_dfm_lib_weather.cbottle import CbottleDataGen

    site = DummySite()
    adapter = CbottleDataGen(site=site, provider=None)

    ds = await adapter.body(
        variables=variables,
        time=time_str,
        n_samples=n_samples,
        seed=123,
        device="cpu",
        lat_lon=False,
    )

    assert isinstance(ds, xr.Dataset)
    for v in variables:
        assert v in ds.data_vars
        assert (
            "hpx" in ds[v].coords
            and "lat" not in ds[v].coords
            and "lon" not in ds[v].coords
        )
