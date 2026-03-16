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

# pyright: reportPrivateUsage=false
from unittest.mock import MagicMock

import pytest

from nv_dfm_core.api import BestOf
from nv_dfm_core.gen.irgen._fed_info import (
    ComputeCostInfo,
    FedInfo,
    OperationInfo,
    ProviderInfo,
    SendCostInfo,
    SiteInfo,
)
from nv_dfm_core.gen.irgen.graph import (
    Graph,
)


@pytest.fixture
def fed_info() -> FedInfo:
    """Create a simple FedInfo with two sites for testing."""
    return FedInfo(
        sites={
            "SiteA": SiteInfo(
                interface={
                    "A": OperationInfo(
                        operation="A",
                        compute_cost=ComputeCostInfo(fixed_time=1.0, fixed_size=10),
                        is_async=True,
                    ),
                    "B": OperationInfo(
                        operation="B",
                        compute_cost=ComputeCostInfo(fixed_time=5.0, fixed_size=20),
                        is_async=True,
                    ),
                    "C": OperationInfo(
                        operation="C",
                        compute_cost=ComputeCostInfo(fixed_time=5.0, fixed_size=20),
                        is_async=True,
                    ),
                },
                providers={
                    "p1": ProviderInfo(
                        interface={
                            "A": OperationInfo(
                                operation="A",
                                compute_cost=ComputeCostInfo(
                                    fixed_time=1.0, fixed_size=10
                                ),
                                is_async=True,
                            ),
                        }
                    ),
                    "p2": ProviderInfo(
                        interface={
                            "A": OperationInfo(
                                operation="A",
                                compute_cost=ComputeCostInfo(
                                    fixed_time=1.0, fixed_size=10
                                ),
                                is_async=True,
                            ),
                        }
                    ),
                },
                send_cost={
                    "SiteB": SendCostInfo(fixed_time=0.1, bandwidth=1000),
                },
            ),
            "SiteB": SiteInfo(
                interface={
                    "A": OperationInfo(
                        operation="A",
                        compute_cost=ComputeCostInfo(fixed_time=10.0, fixed_size=10),
                        is_async=True,
                    ),
                    "B": OperationInfo(
                        operation="B",
                        compute_cost=ComputeCostInfo(fixed_time=2.0, fixed_size=20),
                        is_async=True,
                    ),
                },
                providers={
                    "p1": ProviderInfo(
                        interface={
                            "A": OperationInfo(
                                operation="A",
                                compute_cost=ComputeCostInfo(
                                    fixed_time=1.0, fixed_size=10
                                ),
                                is_async=True,
                            ),
                        }
                    ),
                    "p2": ProviderInfo(
                        interface={
                            "B": OperationInfo(
                                operation="B",
                                compute_cost=ComputeCostInfo(
                                    fixed_time=1.0, fixed_size=10
                                ),
                                is_async=True,
                            ),
                        }
                    ),
                },
                send_cost={
                    "SiteA": SendCostInfo(fixed_time=0.1, bandwidth=1000),
                },
            ),
        }
    )


def test_fixed_site_no_provider(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = "SiteB"
    op.provider = None
    op.__api_name__ = "B"

    locations = graph._locations_to_try_for_operation(op)
    assert locations == {"SiteB": fed_info.sites["SiteB"].interface["B"]}


def test_fixed_site_with_provider(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = "SiteB"
    op.provider = "p1"
    op.__api_name__ = "A"

    locations = graph._locations_to_try_for_operation(op)
    assert locations == {
        "SiteB": fed_info.sites["SiteB"].providers["p1"].interface["A"]
    }


def test_bestof_no_provider_one_match(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = BestOf(sites=["SiteA", "SiteB"])
    op.provider = None
    op.__api_name__ = "C"

    locations = graph._locations_to_try_for_operation(op)
    assert locations == {"SiteA": fed_info.sites["SiteA"].interface["C"]}


def test_bestof_no_provider_multiple_matches(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = BestOf(sites=["SiteA", "SiteB"])
    op.provider = None
    op.__api_name__ = "B"

    locations = graph._locations_to_try_for_operation(op)
    assert locations == {
        "SiteA": fed_info.sites["SiteA"].interface["B"],
        "SiteB": fed_info.sites["SiteB"].interface["B"],
    }


def test_bestof_default_no_provider(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = BestOf()
    op.provider = None
    op.__api_name__ = "B"

    locations = graph._locations_to_try_for_operation(op)
    assert locations == {
        "SiteA": fed_info.sites["SiteA"].interface["B"],
        "SiteB": fed_info.sites["SiteB"].interface["B"],
    }


def test_bestof_with_provider_one_match(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = BestOf(sites=["SiteA", "SiteB"])
    op.provider = "p2"
    op.__api_name__ = "B"

    locations = graph._locations_to_try_for_operation(op)
    assert locations == {
        "SiteB": fed_info.sites["SiteB"].providers["p2"].interface["B"]
    }


def test_bestof_with_provider_multiple_matches(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = BestOf(sites=["SiteA", "SiteB"])
    op.provider = "p1"
    op.__api_name__ = "A"

    locations = graph._locations_to_try_for_operation(op)
    assert locations == {
        "SiteA": fed_info.sites["SiteA"].providers["p1"].interface["A"],
        "SiteB": fed_info.sites["SiteB"].providers["p1"].interface["A"],
    }


def test_restricted_candidate_sites(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = BestOf()
    op.provider = "p1"
    op.__api_name__ = "A"

    locations = graph._locations_to_try_for_operation(op)
    assert locations == {
        "SiteB": fed_info.sites["SiteB"].providers["p1"].interface["A"]
    }


def test_using_excluded_site_with_restricted_candidate_sites_causes_error(
    fed_info: FedInfo,
):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = BestOf(sites=["SiteA", "SiteB"])
    op.provider = "p1"
    op.__api_name__ = "A"

    with pytest.raises(
        ValueError,
        match="Operation .* is associated with sites .*, some of which are not in site candidates: .*",
    ):
        _ = graph._locations_to_try_for_operation(op)


def test_default_bestof_with_provider(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = BestOf()
    op.provider = "p1"
    op.__api_name__ = "A"

    locations = graph._locations_to_try_for_operation(op)
    assert locations == {
        "SiteA": fed_info.sites["SiteA"].providers["p1"].interface["A"],
        "SiteB": fed_info.sites["SiteB"].providers["p1"].interface["A"],
    }


def test_fixed_site_with_unknown_site(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = "DoesntExist"
    op.provider = "p1"
    op.__api_name__ = "A"

    with pytest.raises(
        ValueError,
        match="Operation .* is associated with sites .*, some of which are not in site candidates: .*",
    ):
        _ = graph._locations_to_try_for_operation(op)


def test_bestof_with_unknown_site(fed_info: FedInfo):
    graph = Graph(
        pipeline=MagicMock(),
        homesite="SiteA",
        fed_info=fed_info,
        candidate_sites=["SiteA", "SiteB"],
        logger=None,
    )
    op = MagicMock()
    op.site = BestOf(sites=["DoesntExist", "SiteB"])
    op.provider = "p1"
    op.__api_name__ = "A"

    with pytest.raises(
        ValueError,
        match="Operation .* is associated with sites .*, some of which are not in site candidates: .*",
    ):
        _ = graph._locations_to_try_for_operation(op)
