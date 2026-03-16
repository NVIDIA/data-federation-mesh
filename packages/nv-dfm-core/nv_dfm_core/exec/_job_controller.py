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

from logging import Logger
from typing import TYPE_CHECKING, Any

from ._dfm_context import DfmContext
from ._net_runner import NetRunner
from ._panic_error import PanicError
from ._router import Router

if TYPE_CHECKING:
    from nv_dfm_core.gen.modgen.ir import BoundNetIR
else:
    BoundNetIR = object


class JobController:
    def __init__(
        self,
        router: Router,
        pipeline_api_version: str,
        federation_name: str,
        homesite: str,
        this_site: str,
        job_id: str,
        netir: BoundNetIR,
        logger: Logger,
        force_modgen: bool = False,
    ):
        self._router: Router = router
        self._pipeline_api_version: str = pipeline_api_version
        self._federation_name: str = federation_name
        self._homesite: str = homesite
        self._this_site: str = this_site
        self._job_id: str = job_id
        self._netir: BoundNetIR = netir
        self._logger: Logger = logger

        from nv_dfm_core.gen.modgen import ModGen

        dfm_context = DfmContext(
            pipeline_api_version=self._pipeline_api_version,
            federation_name=self._federation_name,
            homesite=self._homesite,
            this_site=self._this_site,
            job_id=self._job_id,
            router=self._router,
            logger=self._logger,
        )
        # trace_id is now derived from job_id inside telemetry collector

        modgen = ModGen()
        net_mod = modgen.generate_net_module(
            federation_name=self._federation_name,
            this_site_name=self._this_site,
            this_site_runtime_module=dfm_context.this_site_runtime_module,
            netir=netir.ir,
            logger=self._logger,
            force_modgen=force_modgen,
        )
        self._netrunner: NetRunner = NetRunner(
            dfm_context=dfm_context,
            net_module=net_mod,
            logger=self._logger,
        )

    @property
    def job_id(self) -> str:
        return self._job_id

    def start(self):
        self._netrunner.start()

    def shutdown(self):
        self._netrunner.shutdown()

    def submit_initial_tokens(self):
        from nv_dfm_core.gen.modgen.ir._in_place import START_PLACE_NAME

        # send initial input params to the net runner
        with self._netrunner.receive_token_transaction() as transaction:
            input_params = self._netir.deserialized_input_params()
            # all places that expect external input, including the start place
            all_external_places = self._netir.ir.find_places(
                kind=None, origin="external"
            )
            # each paramset that is not a stop frame must have
            # values for all external places. In particular, it must
            # have a value for the start place.
            for frame, paramset in input_params:
                places_to_send: set[str] = all_external_places
                if frame.is_stop_frame():
                    # stop frames are special because they don't have values for data places
                    places_to_send = {START_PLACE_NAME}

                for key in places_to_send:
                    if key not in paramset:
                        self._logger.warning(
                            f"Paramset is missing value for place param '{key}'. Forcing it to None: {paramset}"
                        )
                    if transaction.has_place(key):
                        value = paramset.get(key, None)
                        transaction.receive_token(place=key, frame=frame, data=value)
                    else:
                        self._logger.warning(f"No place {key} found")

    def wait_for_done(self, abort_signal: Any | None):
        self._netrunner.wait_for_done(abort_signal=abort_signal)

    def error_occurred(self) -> bool:
        return self._netrunner.error_occurred()

    def get_panic_error(self) -> PanicError:
        return self._netrunner.get_panic_error()
