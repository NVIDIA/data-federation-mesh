#!/usr/bin/env python3
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

import uuid
from pathlib import Path
from typing import Any

from nvflare.fuel.f3.cellnet.fqcn import FQCN  # pyright: ignore[reportMissingImports]
from nvflare.job_config.api import FedJob  # pyright: ignore[reportMissingImports]

from nv_dfm_core.api import PreparedPipeline
from nv_dfm_core.exec import Frame

from ._controller import Controller
from ._executor import Executor
from ._flare_options import FlareOptions


class FlareApp:
    def __init__(
        self,
        pipeline: PreparedPipeline,
        input_params: list[tuple[Frame, dict[str, Any]]],
        options: FlareOptions,
        force_modgen: bool = False,
    ):
        self._pipeline = pipeline
        self._job: FedJob | None = None
        self._input_params = input_params
        self._force_modgen = force_modgen
        self._options = options

    def _prepare(self):
        # Create the Flare job configuration object
        rnd_id = str(uuid.uuid4())
        pipe_name = (
            f"{self._pipeline.pipeline_name}"
            if self._pipeline.pipeline_name
            else "pipeline"
        )
        job_name = f"{self._pipeline.federation_module}-{self._pipeline.homesite}-{pipe_name}-{rnd_id}"

        clients = self._pipeline.get_participating_sites()
        clients.remove(FQCN.ROOT_SERVER)

        job = FedJob(
            name=job_name,
            min_clients=len(clients),
            mandatory_clients=clients,
        )

        # Send DfmExecutor to each client and Controller to root server
        bound_net_irs = self._pipeline.bind_net_irs(input_params=self._input_params)

        # handle the server
        # NOTE: the server_netir should not be False ever, but some tests currently rely on this so I keep it in.
        server_netir = (
            bound_net_irs.pop(FQCN.ROOT_SERVER).model_dump()
            if FQCN.ROOT_SERVER in bound_net_irs
            else False
        )

        controller = Controller(
            submitted_api_version=self._pipeline.api_version,
            federation_name=self._pipeline.federation_module,
            homesite=self._pipeline.homesite,
            bound_net_ir=server_netir,
            options=self._options,
            force_modgen=self._force_modgen,
        )
        job.to_server(controller)

        # and the clients
        for bound_net_ir in bound_net_irs.values():
            assert bound_net_ir.site != FQCN.ROOT_SERVER, (
                "Root server net IR should have been handled above"
            )

            executor = Executor(
                submitted_api_version=self._pipeline.api_version,
                federation_name=self._pipeline.federation_module,
                homesite=self._pipeline.homesite,
                bound_net_ir=bound_net_ir.model_dump(),
                force_modgen=self._force_modgen,
            )
            job.to(obj=executor, target=bound_net_ir.site)

        self._job = job

    def simulate(self, workspace: str | Path | None = None) -> str:
        if not workspace:
            workspace = "/tmp/dfm_workspace"
        workspace = str(workspace)
        if not self._job:
            self._prepare()
        assert self._job
        self._job.simulator_run(workspace, n_clients=1)
        return workspace

    def export(self, workspace: Path) -> Path:
        """
        Exports the job to the given workspace.
        Returns the path to the job directory inside the workspace.
        """
        if not self._job:
            self._prepare()
        assert self._job
        self._job.export_job(str(workspace))
        return workspace.joinpath(self._job.name)
