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

from pydantic import BaseModel


class ComputeCostInfo(BaseModel):
    """Information about the performance of a node that does computation. Default parameters
    result in zero cost.

    Params:
    compute_throughput: in bytes per second of the incoming data size.
                       Default is None (compute time is independent of input size).
                       Used to compute resulting in the time it takes to compute the amount of incoming data.
    output_factor: in terms of % of the input size.
                       Default is 0.0 (output independent of input size)
                       Used to compute the size of the output data relative to the input size.
    fixed_time: in seconds. Default is 0. A fixed time added to the compute time.
    fixed_size: in bytes. Default is 0. A fixed size added to the compute size.
    """

    fixed_time: float = 0.0
    # Throughput in bytes per second of the input data
    # by default, only fixed time is used
    compute_throughput: int | None = None
    # Fixed part of the result size in bytes
    fixed_size: int = 0
    # Factor of the result in terms of the input size
    output_factor: float = 0.0

    def compute_time(self, input_size: int) -> float:
        compute_time = (
            input_size / self.compute_throughput if self.compute_throughput else 0
        )
        return self.fixed_time + compute_time

    def compute_size(self, input_size: int) -> int:
        res_size = int(self.output_factor * input_size) if self.output_factor else 0
        return self.fixed_size + res_size


class SendCostInfo(BaseModel):
    """Information about the cost of sending data between sites."""

    fixed_time: float = 0
    # Transfer/send bandwidth in bytes per second
    bandwidth: int | None = None

    def compute_time(self, input_size: int) -> float:
        return (
            float(input_size) / float(self.bandwidth) if self.bandwidth else 0
        ) + self.fixed_time


class OperationInfo(BaseModel):
    """Information about an operation including its cost and execution mode."""

    operation: str
    compute_cost: ComputeCostInfo
    is_async: bool = True


class ProviderInfo(BaseModel):
    """Information about a provider including its available operations."""

    interface: dict[str, OperationInfo]


class SiteInfo(BaseModel):
    """Information about a site including its interface, providers, and send costs."""

    interface: dict[str, OperationInfo]
    providers: dict[str, ProviderInfo]
    # cost to send data from this site to all other sites
    send_cost: dict[str, SendCostInfo]

    def get_send_cost(self, fromsite: str, tosite: str, logger: Logger) -> SendCostInfo:
        if tosite in self.send_cost:
            return self.send_cost[tosite]
        elif fromsite == tosite:
            # no cost to send data within the same site
            return SendCostInfo(fixed_time=0)
        else:
            logger.warning(
                f"Site info for {fromsite} doesn't have send cost info to site '{tosite}'. Using suboptimal default info."
            )
            # use something large to discourage using this
            return SendCostInfo(fixed_time=100_000_000, bandwidth=1)


class FedInfo(BaseModel):
    """Federation information containing details about all sites and their capabilities."""

    sites: dict[str, SiteInfo]

    def find_send_cost(
        self, fromsite: str, tosite: str, logger: Logger
    ) -> SendCostInfo:
        if fromsite not in self.sites:
            raise ValueError(f"Site {fromsite} not in fed info: {self.sites}")
        send_cost = self.sites[fromsite].get_send_cost(
            fromsite=fromsite, tosite=tosite, logger=logger
        )
        return send_cost

    def find_location_info(
        self, site: str, provider: str | None = None
    ) -> SiteInfo | ProviderInfo:
        if site not in self.sites:
            raise ValueError(f"Site {site} not in fed info: {self.sites}")
        info_to_use = self.sites[site]

        if provider:
            if provider not in info_to_use.providers:
                raise ValueError(
                    f"Provider {provider} not in site info: {info_to_use.providers}"
                )
            info_to_use = info_to_use.providers[provider]
        return info_to_use
