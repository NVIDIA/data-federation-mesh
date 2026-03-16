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

import os
from dataclasses import dataclass, field

from ._defs import Constant


@dataclass
class FlareOptions:
    """
    Flare options for the DFM Flare target.
    """

    # Flare task timeout in seconds. Default is Constant.TASK_TIMEOUT and can be overridden by environment variable DFM_FLARE_TASK_TIMEOUT
    task_timeout_s: int = field(
        default_factory=lambda: int(
            os.getenv("DFM_FLARE_TASK_TIMEOUT", Constant.TASK_TIMEOUT)
        )
    )
