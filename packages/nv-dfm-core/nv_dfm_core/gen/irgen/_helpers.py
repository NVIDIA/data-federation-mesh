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

import importlib.resources as pkg_resources
from types import ModuleType

from ._fed_info import FedInfo


def load_fed_info_json(fed_runtime_module: ModuleType) -> FedInfo:
    if not pkg_resources.is_resource(fed_runtime_module, "fed_info.json"):
        raise ValueError(
            f"No fed_info.json found in package {fed_runtime_module.__name__}. Did you pass the federation runtime module? e.g. atlantis.fed.runtime"
        )

    resource = pkg_resources.files(fed_runtime_module).joinpath("fed_info.json")
    with resource.open("r", encoding="utf-8") as f:
        json_data = f.read()
        return FedInfo.model_validate_json(json_data=json_data)
