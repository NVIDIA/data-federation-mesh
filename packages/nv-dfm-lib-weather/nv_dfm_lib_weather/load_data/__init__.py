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

# Always available - no earth2studio dependency
from ._load_ecmwf_era5_data import LoadEcmwfEra5Data  # noqa: F401
from ._load_hrrr_era5_data import LoadHrrrEra5Data  # noqa: F401

# Conditional imports - require earth2studio extra
try:
    from ._load_cmip6_data import LoadCmip6Data  # noqa: F401
    from ._load_gfs_era5_data import LoadGfsEra5Data  # noqa: F401
except ImportError as e:
    # Create placeholder classes that raise helpful errors

    class _MissingDependency:
        def __init__(self, name, extra):
            self.name = name
            self.extra = extra

        def __init_subclass__(cls, **kwargs):
            raise ImportError(
                f"{cls.__name__} requires earth2studio. "
                f"Install with: uv sync --package nv-dfm-lib-weather --extra {cls.extra}"
            )

    # Only create placeholders if the import actually failed due to missing earth2studio
    if "earth2studio" in str(e):
        LoadCmip6Data = type(
            "LoadCmip6Data",
            (),
            {
                "__init__": lambda self, *args, **kwargs: (
                    (_ for _ in ()).throw(
                        ImportError(
                            "LoadCmip6Data requires earth2studio. "
                            "Install with: uv sync --package nv-dfm-lib-weather --extra earth2studio"
                        )
                    )
                )
            },
        )
        LoadGfsEra5Data = type(
            "LoadGfsEra5Data",
            (),
            {
                "__init__": lambda self, *args, **kwargs: (
                    (_ for _ in ()).throw(
                        ImportError(
                            "LoadGfsEra5Data requires earth2studio. "
                            "Install with: uv sync --package nv-dfm-lib-weather --extra earth2studio"
                        )
                    )
                )
            },
        )
    else:
        # If it's a different import error, re-raise it
        raise

__all__ = [
    "LoadEcmwfEra5Data",
    "LoadHrrrEra5Data",
    "LoadCmip6Data",
    "LoadGfsEra5Data",
]
