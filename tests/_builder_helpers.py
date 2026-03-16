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

from nv_dfm_core.gen.irgen import (
    ComputeCostInfo,
    FedInfo,
    OperationInfo,
    ProviderInfo,
    SendCostInfo,
    SiteInfo,
)


class ProviderInfoBuilder:
    def __init__(self, parent: "SiteInfoBuilder", name: str):
        self.parent: SiteInfoBuilder = parent
        self.name: str = name
        self.interface: dict[str, OperationInfo] = {}

    def site(self, name: str) -> "SiteInfoBuilder":
        return self.parent.site(name)

    def op(
        self,
        name: str,
        fixed_sec: float = 0,
        comp: int | None = None,
        fixed_size: int = 0,
        size_factor: float = 0.0,
    ) -> "ProviderInfoBuilder":
        self.interface[name] = OperationInfo(
            operation=name,
            compute_cost=ComputeCostInfo(
                fixed_time=fixed_sec,
                compute_throughput=comp,
                fixed_size=fixed_size,
                output_factor=size_factor,
            ),
        )
        return self

    def prov(self, name: str) -> "ProviderInfoBuilder":
        return self.parent.prov(name)

    def build(self) -> ProviderInfo:
        return ProviderInfo(interface=self.interface)


class SiteInfoBuilder:
    def __init__(self, parent: "FedInfoBuilder", name: str, is_homesite: bool):
        self.parent: FedInfoBuilder = parent
        self.name: str = name
        self.interface: dict[str, OperationInfo] = {}
        self.providers: dict[str, ProviderInfoBuilder] = {}
        self.send_cost: dict[str, SendCostInfo] = {}
        self.is_homesite: bool = is_homesite

    def site(self, name: str) -> "SiteInfoBuilder":
        return self.parent.site(name)

    fixed_time: float = 1.0
    # Throughput in bytes per second of the input data
    # by default, only fixed time is used
    compute_throughput: int | None = None
    # Fixed part of the result size in bytes
    fixed_size: int = 0
    # Factor of the result in terms of the input size
    output_factor: float | None = None

    def op(
        self,
        name: str,
        fixed_sec: float = 0,
        comp: int | None = None,
        fixed_size: int = 0,
        size_factor: float = 0.0,
    ) -> "SiteInfoBuilder":
        self.interface[name] = OperationInfo(
            operation=name,
            compute_cost=ComputeCostInfo(
                fixed_time=fixed_sec,
                compute_throughput=comp,
                fixed_size=fixed_size,
                output_factor=size_factor,
            ),
        )
        return self

    def prov(self, name: str) -> "ProviderInfoBuilder":
        self.providers[name] = ProviderInfoBuilder(parent=self, name=name)
        return self.providers[name]

    def build(self) -> SiteInfo:
        return SiteInfo(
            interface=self.interface,
            providers={
                name: provider.build() for name, provider in self.providers.items()
            },
            send_cost=self.send_cost,
        )


class FedInfoBuilder:
    def __init__(self, federation_module_name: str):
        self.federation_module_name: str = federation_module_name
        self.sites: dict[str, SiteInfoBuilder] = {}

    def site(self, name: str, is_homesite: bool = False) -> "SiteInfoBuilder":
        if name not in self.sites:
            self.sites[name] = SiteInfoBuilder(
                parent=self, name=name, is_homesite=is_homesite
            )
        return self.sites[name]

    def comm(self, src: str, dst: str, fixed_sec: float = 0, bw: int | None = None):
        self.sites[src].send_cost[dst] = SendCostInfo(
            fixed_time=fixed_sec, bandwidth=bw
        )

    def homesite(self, name: str) -> "SiteInfoBuilder":
        return self.site(name, is_homesite=True)

    def build(self) -> tuple[str, str, FedInfo]:
        """Returns: (module_name, homesite_name, FedInfo)"""
        homesite: str | None = None
        for site in self.sites.values():
            if site.is_homesite:
                homesite = site.name
        assert homesite is not None

        sites = {name: site.build() for name, site in self.sites.items()}
        return self.federation_module_name, homesite, FedInfo(sites=sites)
