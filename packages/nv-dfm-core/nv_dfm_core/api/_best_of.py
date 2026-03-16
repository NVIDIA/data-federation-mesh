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


class BestOf(BaseModel):
    """This marker is attached to site and provider to indicate that the user wants
    IRGen to deduce the optimal location."""

    tag: Literal["@dfm-best-of"] = "@dfm-best-of"
    # which sites are candidates?
    # None means, any site is okay. Otherwise, restrict to the given list
    sites: list[str] | None = None
