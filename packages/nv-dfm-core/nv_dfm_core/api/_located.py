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

from abc import ABC

from pydantic import BaseModel

from ._advise import Advise
from ._best_of import BestOf


class Located(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """Located is a mixin class for statements that are assigned to
    a specific site."""

    # site, if specified, otherwise the optimizer will choose one
    # BestOf can be used to restrict this node to only specific sites.
    site: str | BestOf | Advise = BestOf()
    # If none, site interface; if specified, specific provider interface
    provider: None | str = None

    __api_name__: str
