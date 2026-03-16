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

"""All the adapter implementations"""

from ._cbottle_data_gen import CbottleDataGen
from ._cbottle_infilling import CbottleInfilling
from ._cbottle_super_resolution import CbottleSuperResolution
from ._cbottle_tc_guidance import CBottleTropicalCycloneGuidance
from ._cbottle_video import CbottleVideo

__all__ = [
    "CbottleDataGen",
    "CbottleInfilling",
    "CbottleSuperResolution",
    "CBottleTropicalCycloneGuidance",
    "CbottleVideo",
]
