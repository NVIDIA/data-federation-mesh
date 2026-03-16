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

import inspect
from pydoc import locate
from types import ModuleType


def load_fed_runtime_module(federation_module_name: str) -> ModuleType:
    module_name = f"{federation_module_name}.fed.runtime"
    mod = locate(module_name)
    if not mod or not inspect.ismodule(mod):
        raise ValueError(f"Could not locate a python module for name {module_name}")
    return mod


def load_site_runtime_module(federation_module_name: str, site_name: str) -> ModuleType:
    # make sure we escape any @ and . in the site name
    site_id = site_name_to_identifier(site_name)
    module_name = f"{federation_module_name}.fed.runtime.{site_id}"
    mod = locate(module_name)
    if not mod or not inspect.ismodule(mod):
        raise ValueError(f"Could not locate a python module for name {module_name}")
    assert_valid_runtime_module(mod)
    return mod


def assert_valid_runtime_module(runtime_module: ModuleType):
    if not hasattr(runtime_module, "API_VERSION") or not hasattr(
        runtime_module, "ThisSite"
    ):
        raise ValueError(
            f"Object '{runtime_module}' is not a valid federation runtime module."
        )


def site_name_to_identifier(site_name: str) -> str:
    # Flare admin sites have an email address as the enforced name. Only use the part before the @
    ident = site_name.split("@")[0]
    if not ident.isidentifier():
        raise ValueError(f"Site name {ident} is not a valid identifier")
    return ident
