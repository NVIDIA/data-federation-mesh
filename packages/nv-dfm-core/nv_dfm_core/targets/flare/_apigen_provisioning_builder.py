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

from pathlib import Path
from typing import List, Literal, Tuple

from nvflare.lighter.spec import Builder, Participant

from nv_dfm_core.provisioning.apigen import ApiGenerator
from nv_dfm_core.provisioning.apigen.ir import FederationInfo


class DfmApigenBuilder(Builder):
    def __init__(self, config_folder="../../configs"):
        self._config_folder = Path(config_folder)

    def _prepare_apigen(self, project_name: str) -> ApiGenerator:
        if not self._config_folder.exists():
            raise ValueError(
                "No config folder with *.api.yaml and/or *.site.yaml configs"
                f" found at location {self._config_folder.absolute()}"
            )
        apigen = ApiGenerator(project_name.lower(), warnings_will_raise=False)
        for conffile in self._config_folder.glob("*.*.yaml"):
            apigen.add_config_from_yaml_files(conffile)
        apigen.finalize()
        return apigen

    def _ensure_expected_type(
        self,
        participant: Participant,
        fed_info: FederationInfo,
        expected: Literal["client", "homesite", "worker"],
    ):
        site_info = fed_info.get_site_info(participant.name)
        if not site_info:
            raise ValueError(
                f"Project config mentions participant {participant.name} of type"
                f" {participant.type} but no DFM site configuration found for this name."
            )
        if not site_info.type == expected:
            raise ValueError(
                f"Project config mentions participant {participant.name} of type"
                f" {participant.type} but the DFM site configuration"
                f" has conflicting type {site_info.type} (expected DFM config type: {expected})."
            )

    def _sanity_checked(
        self, fed_info: FederationInfo, participants: List[Participant]
    ) -> Tuple[
        List[Participant], List[Participant], List[Participant], List[Participant]
    ]:
        clients = []
        homesites = []
        controls = []
        workers = []
        for participant in participants:
            if participant.type == "dfm-client":
                self._ensure_expected_type(participant, fed_info, "client")
                clients.append(participant)
            elif participant.type == "homesite":
                self._ensure_expected_type(participant, fed_info, "homesite")
                homesites.append(participant)
            elif participant.type == "server":
                # Note: control nodes aren't really dfm sites and aren't part of the
                # normal site configs, therefore there's no SiteInfo for them.
                # We still want to generate runtime code though.
                controls.append(participant)
            elif participant.type == "client":
                self._ensure_expected_type(participant, fed_info, "worker")
                workers.append(participant)

        if len(clients) == 0:
            raise ValueError(
                "Project configuration does not contain a single"
                " Participant of type dfm-client (dfm: client)."
            )
        if len(homesites) == 0:
            print(
                "INFO: Project configuration does not contain a single"
                " Participant of type homesite (dfm: homesite)."
            )
        if len(controls) == 0:
            print(
                "INFO: Project configuration does not contain a single"
                " Participant of type server (dfm: control)."
            )
        if len(workers) == 0:
            print(
                "INFO: Project configuration does not contain a single"
                " Participant of type client (dfm: worker)."
            )
        return (clients, homesites, controls, workers)

    def build(self, project, ctx):
        apigen = self._prepare_apigen(project.name)
        fed_info = apigen.context.fed_info

        clients, homesites, controls, workers = self._sanity_checked(
            fed_info, project.participants
        )

        for participant in clients:
            language = participant.props.get("language", "python")
            dest_dir = Path(self.get_kit_dir(participant, ctx))
            apigen.generate_client_startup_kit(
                sitename=participant.name,
                language=language,
                outpath=dest_dir,
            )
        for participant in homesites:
            dest_dir = Path(self.get_kit_dir(participant, ctx))
            apigen.generate_homesite_startup_kit(
                sitename=participant.name,
                outpath=dest_dir,
            )
        for participant in controls:
            dest_dir = Path(self.get_kit_dir(participant, ctx))
            apigen.generate_control_site_startup_kit(
                sitename=participant.name,
                outpath=dest_dir,
            )
        for participant in workers:
            dest_dir = Path(self.get_kit_dir(participant, ctx))
            apigen.generate_worker_site_startup_kit(
                sitename=participant.name,
                outpath=dest_dir,
            )
