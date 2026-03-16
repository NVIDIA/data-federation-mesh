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

import importlib
from pathlib import Path
import sys
from types import ModuleType


class StartupKitLoader:
    """
    Helper to dynamically load modules from a startup kit
    """

    def __init__(self, path: str, *modules: ModuleType | str):
        self._path = path
        self._modules_requested = modules
        self._modules_that_existed: list[ModuleType] | None = None
        self._modules_loaded_new: list[str] | None = None

    def __enter__(self):
        # prepend folder to sys.path
        assert Path(self._path).exists(), (
            f"Startup kit path {self._path} does not exist"
        )
        assert not sys.path[0].startswith(self._path)
        sys.path.insert(0, self._path)

        # check for each requested module if it existed before or was loaded new.
        # If it existed before, we will reload them on exit; if they were loaded new,
        # we will remove them on exit
        self._modules_that_existed = []
        self._modules_loaded_new = []
        for m in self._modules_requested:
            if isinstance(m, str):
                if m in sys.modules:
                    m = importlib.import_module(m)
                    self._modules_that_existed.append(m)
                else:
                    self._modules_loaded_new.append(m)
                    m = importlib.import_module(m)
                importlib.reload(m)
            else:
                # since m is a module already it must have been loaded before
                self._modules_that_existed.append(m)
                importlib.reload(m)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # remove path from sys path
        assert sys.path[0] == self._path
        path = sys.path.pop(0)
        assert path.startswith(self._path)
        assert not sys.path[0].startswith(self._path)
        assert self._modules_that_existed is not None
        assert self._modules_loaded_new is not None

        for module in self._modules_that_existed:
            importlib.reload(module)
        for mname in self._modules_loaded_new:
            del sys.modules[mname]
