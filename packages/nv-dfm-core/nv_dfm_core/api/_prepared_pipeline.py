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

# -*- coding: utf-8 -*-

# Just a mock for now
from logging import Logger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nv_dfm_core.exec import Frame
    from nv_dfm_core.gen.modgen.ir import BoundNetIR, NetIR, YieldPlace
else:
    BoundNetIR = Any
    NetIR = Any
    Frame = Any
    YieldPlace = Any


class PreparedPipeline:
    """A pipeline that has been prepared and optimized for execution in a federation.

    PreparedPipeline contains the intermediate representation (IR) for each
    participating site, along with metadata needed for execution.
    """

    def __init__(
        self,
        api_version: str,
        federation_module_name: str,
        homesite: str,
        pipeline_name: str | None,
        net_irs: list[NetIR],
        yield_places: list[YieldPlace],
    ):
        self._api_version: str = api_version
        self._federation_module_name: str = federation_module_name
        self._homesite: str = homesite
        self._name: str | None = pipeline_name
        self._net_irs: dict[str, NetIR] = {net_ir.site: net_ir for net_ir in net_irs}
        self._yield_places: list[YieldPlace] = yield_places

    @property
    def api_version(self) -> str:
        return self._api_version

    @api_version.setter
    def api_version(self, value: str) -> None:
        self._api_version = value

    @property
    def federation_module(self) -> str:
        return self._federation_module_name

    @property
    def homesite(self) -> str:
        return self._homesite

    @property
    def pipeline_name(self) -> str | None:
        return self._name

    def net_irs(self) -> dict[str, NetIR]:
        return self._net_irs

    def has_param_places(self) -> bool:
        return any(
            net_ir.find_places(kind="data", origin="external")
            for net_ir in self._net_irs.values()
        )

    def check_input_params(
        self, input_params: list[tuple[Frame, dict[str, Any]]]
    ) -> None:
        """Validate that all required input parameters are provided and no extra parameters are given."""
        # collect all required input parameters from all net IRs
        required_external_places: set[str] = set()

        for net_ir in self._net_irs.values():
            required_external_places.update(
                net_ir.find_places(kind=None, origin="external")
            )

        for i, (frame, paramset) in enumerate(input_params):
            provided = set(paramset.keys())

            if frame.is_stop_frame():
                continue

            if not required_external_places.issubset(provided):
                raise ValueError(
                    f"Parameter set {i} is missing input parameters that are required PlaceParams in the pipeline: {required_external_places - provided}"
                )

            if not provided.issubset(required_external_places):
                raise ValueError(
                    f"Parameter set {i} provided parameters that are not PlaceParams in the pipeline: {provided - required_external_places}"
                )

    def check_callbacks(self, callback_places: list[str], logger: Logger):
        """Check and warn if callback registration doesn't match yield places in the pipeline."""
        yield_places = set([yield_place.place for yield_place in self._yield_places])
        cb_places = set(callback_places)
        if not yield_places.issubset(cb_places):
            logger.warning(
                f"No callbacks were registered for yield places {yield_places - cb_places}."
            )

        if not cb_places.issubset(yield_places):
            logger.warning(
                f"Some callbacks were registered that don't have corresponding yields: {cb_places - yield_places}."
            )

    def get_participating_sites(self) -> list[str]:
        return list(self._net_irs.keys())

    def bind_net_ir(
        self,
        site: str,
        input_params: list[tuple[Frame, dict[str, Any]]],
    ) -> BoundNetIR:
        """Bind a specific site's NetIR with input parameters to create a BoundNetIR ready for execution."""
        from nv_dfm_core.gen.modgen.ir import BoundNetIR

        netir = self._net_irs[site]
        return BoundNetIR.bind_netir(netir, input_params)

    def bind_net_irs(
        self, input_params: list[tuple[Frame, dict[str, Any]]]
    ) -> dict[str, BoundNetIR]:
        """Bind all site NetIRs with input parameters to create BoundNetIRs for each participating site."""
        from nv_dfm_core.gen.modgen.ir import BoundNetIR

        return {
            site: BoundNetIR.bind_netir(net_ir, input_params)
            for site, net_ir in self._net_irs.items()
        }
