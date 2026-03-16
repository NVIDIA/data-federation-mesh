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

from typing import Literal

from pydantic import BaseModel


class PlaceParam(BaseModel, frozen=True):
    """This marker is attached to Statements to indicate
    that the value is passed later as a pipeline a parameter.
    By default, all places in a pipeline must be unique (this includes Yield places),
    to avoid accidental aliasing. If multiple PlaceParams should reference the
    same place, multiuse must be set to True on ALL PlaceParams for this place.
    The place name must still be different from all Yield places, however."""

    tag: Literal["@dfm-place-param"] = "@dfm-place-param"
    place: str
    multiuse: bool = False
