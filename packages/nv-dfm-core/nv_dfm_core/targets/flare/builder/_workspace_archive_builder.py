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
import subprocess
from pathlib import Path

from nvflare.lighter.entity import Participant
from nvflare.lighter.spec import Builder, Project, ProvisionContext


class WorkspaceArchiveBuilder(Builder):
    def __init__(self):
        """Build the workspace archive for the project that is then used
        to initialize the workspace during deployment.
        """
        super().__init__()

    def initialize(self, project: Project, ctx: ProvisionContext):
        print("DFM Workspace Archive Builder initializing...")
        pass

    def _build_participant(self, participant: Participant, ctx: ProvisionContext):
        # Build a tarball of the participant's workspace. It will be used to initialize the workspace during deployment.
        print("Building workspace archive for participant:", participant.name)
        name = participant.name
        wip_dir = Path(ctx.get_wip_dir())
        run_args = ["tar", "-czf", wip_dir / f"{name}.tar.gz", "-C", wip_dir, name]
        try:
            subprocess.run(run_args, env=os.environ)
        except FileNotFoundError:
            raise RuntimeError("Unable to build tar archive.")

    def build(self, project: Project, ctx: ProvisionContext):
        """Create workspace archives for the project.
        Args:
            project (Project): project instance
            ctx (dict): the provision context
        """
        participants = project.get_all_participants(["client", "server"])
        print(
            f"DFM Workspace Archive Builder starting for {len(participants)} participants"
        )
        for participant in participants:
            self._build_participant(participant, ctx)
        print("DFM Workspace Archive Builder finished.")
