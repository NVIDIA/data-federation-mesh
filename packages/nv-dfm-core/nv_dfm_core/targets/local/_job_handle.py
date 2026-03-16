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

# pyright: reportImportCycles=false

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nv_dfm_core.api import PreparedPipeline
from nv_dfm_core.exec import Frame

if TYPE_CHECKING:
    from ._job_execution import JobExecution


@dataclass
class JobHandle:
    """Lightweight tracking of running job execution.

    This is what FederationRunner keeps alive, NOT the full LocalJob.
    Contains only essential job metadata, no callbacks or threads.

    This separation allows LocalJob objects to be garbage collected
    after detach() while the job execution continues running.
    """

    job_id: str
    homesite: str
    pipeline: PreparedPipeline
    next_frame: Frame
    federation_name: str
    pipeline_api_version: str
    force_modgen: bool

    # Reference to the execution (NOT the LocalJob)
    execution: "JobExecution | None" = None
