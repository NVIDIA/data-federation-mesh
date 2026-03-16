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

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

# Import modules directly to avoid circular imports
from nv_dfm_core.exec._router import Router
from nv_dfm_core.exec._token_package import TokenPackage
from nv_dfm_core.exec._frame import Frame


class MockRouter(Router):
    """Concrete implementation of Router for testing purposes."""

    def __init__(self):
        super().__init__()
        self.yield_calls = []
        self.remote_calls = []
        self.other_job_calls = []

    def _send_this_job_yield_token_package_sync(self, token_package: TokenPackage):
        self.yield_calls.append(token_package)

    def _send_this_job_remote_token_package_sync(self, token_package: TokenPackage):
        self.remote_calls.append(token_package)

    def _send_other_job_remote_token_package_sync(self, token_package: TokenPackage):
        self.other_job_calls.append(token_package)


class AsyncMockRouter(MockRouter):
    """Async version of MockRouter that overrides async methods."""

    async def _send_this_job_yield_token_package_async(
        self, token_package: TokenPackage
    ):
        self.yield_calls.append(token_package)
        await asyncio.sleep(0.001)  # Simulate async work

    async def _send_this_job_remote_token_package_async(
        self, token_package: TokenPackage
    ):
        self.remote_calls.append(token_package)
        await asyncio.sleep(0.001)  # Simulate async work

    async def _send_other_job_remote_token_package_async(
        self, token_package: TokenPackage
    ):
        self.other_job_calls.append(token_package)
        await asyncio.sleep(0.001)  # Simulate async work


@pytest.fixture
def mock_netrunner():
    """Create a mock NetRunner with required attributes."""
    netrunner = MagicMock()
    netrunner.dfm_context.job_id = "test_job"
    netrunner.dfm_context.this_site = "test_site"
    netrunner.dfm_context.homesite = "home_site"
    netrunner.receive_token_transaction.return_value.__enter__.return_value.receive_token = MagicMock()
    return netrunner


@pytest.fixture
def router(mock_netrunner):
    """Create a test router instance."""
    router = MockRouter()
    router.set_netrunner(mock_netrunner)
    return router


@pytest.fixture
def async_router(mock_netrunner):
    """Create an async test router instance."""
    router = AsyncMockRouter()
    router.set_netrunner(mock_netrunner)
    return router


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return Frame.start_frame(0)


@pytest.fixture
def sample_token_package(sample_frame):
    """Create a sample token package for testing."""
    return TokenPackage.wrap_data(
        source_site="source_site",
        source_node=1,
        source_job="source_job",
        target_site="target_site",
        target_place="target_place",
        target_job="target_job",
        is_yield=False,
        frame=sample_frame,
        data={"test": "data"},
    )


class TestRouterInitialization:
    """Test Router initialization and setup."""

    def test_router_initialization(self):
        """Test that Router can be initialized."""
        router = MockRouter()
        with pytest.raises(AssertionError):
            router.netrunner
        with pytest.raises(AssertionError):
            router.job_id
        with pytest.raises(AssertionError):
            router.this_site
        with pytest.raises(AssertionError):
            router.homesite

    def test_set_netrunner(self, router, mock_netrunner):
        """Test that set_netrunner properly sets up the router."""
        router.set_netrunner(mock_netrunner)
        assert router.netrunner == mock_netrunner
        assert router.job_id == "test_job"
        assert router.this_site == "test_site"
        assert router.homesite == "home_site"


class TestRouteTokenAsync:
    """Test the route_token_async method."""

    @pytest.mark.asyncio
    async def test_local_token_routing(self, router, sample_frame):
        """Test routing a local token (no serialization needed)."""
        data = {"key": "value"}

        await router.route_token_async(
            to_job=None,
            to_site="test_site",
            to_place="test_place",
            is_yield=False,
            frame=sample_frame,
            data=data,
            node_id=1,
        )

        # Should call receive_token directly, not create a TokenPackage
        router._netrunner.receive_token_transaction.assert_called_once()
        transaction = router._netrunner.receive_token_transaction.return_value.__enter__.return_value
        transaction.receive_token.assert_called_once_with(
            place="test_place", frame=sample_frame, data=data
        )

        # Should not create any token packages
        assert len(router.yield_calls) == 0
        assert len(router.remote_calls) == 0
        assert len(router.other_job_calls) == 0

    @pytest.mark.asyncio
    async def test_remote_token_routing_same_job(self, router, sample_frame):
        """Test routing a remote token to the same job but different site."""
        data = {"key": "value"}

        await router.route_token_async(
            to_job="test_job",
            to_site="different_site",
            to_place="test_place",
            is_yield=False,
            frame=sample_frame,
            data=data,
            node_id=1,
        )

        # Should create a TokenPackage and call remote method
        assert len(router.remote_calls) == 1
        token_package = router.remote_calls[0]
        assert token_package.source_site == "test_site"
        assert token_package.source_node == 1
        assert token_package.source_job == "test_job"
        assert token_package.target_site == "different_site"
        assert token_package.target_place == "test_place"
        assert token_package.target_job == "test_job"
        assert token_package.is_yield is False
        assert token_package.frame == sample_frame
        assert token_package.unwrap_data() == data

    @pytest.mark.asyncio
    async def test_yield_token_routing(self, router, sample_frame):
        """Test routing a yield token."""
        data = {"key": "value"}

        await router.route_token_async(
            to_job="test_job",
            to_site="test_site",
            to_place="test_place",
            is_yield=True,
            frame=sample_frame,
            data=data,
            node_id=1,
        )

        # Should create a TokenPackage and call yield method
        assert len(router.yield_calls) == 1
        token_package = router.yield_calls[0]
        assert token_package.is_yield is True
        assert token_package.unwrap_data() == data

    @pytest.mark.asyncio
    async def test_other_job_token_routing(self, router, sample_frame):
        """Test routing a token to a different job."""
        data = {"key": "value"}

        await router.route_token_async(
            to_job="other_job",
            to_site="test_site",
            to_place="test_place",
            is_yield=False,
            frame=sample_frame,
            data=data,
            node_id=1,
        )

        # Should create a TokenPackage and call other job method
        assert len(router.other_job_calls) == 1
        token_package = router.other_job_calls[0]
        assert token_package.target_job == "other_job"
        assert token_package.unwrap_data() == data

    @pytest.mark.asyncio
    async def test_async_router_methods(self, async_router, sample_frame):
        """Test that async router properly calls async methods."""
        data = {"key": "value"}

        await async_router.route_token_async(
            to_job="other_job",
            to_site="test_site",
            to_place="test_place",
            is_yield=False,
            frame=sample_frame,
            data=data,
            node_id=1,
        )

        # Should call async method
        assert len(async_router.other_job_calls) == 1


class TestRouteTokenPackageAsync:
    """Test the route_token_package_async method."""

    @pytest.mark.asyncio
    async def test_local_token_package_routing(self, router, sample_frame):
        """Test routing a local token package."""
        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="test_site",
            target_place="test_place",
            target_job="test_job",
            is_yield=False,
            frame=sample_frame,
            data={"test": "data"},
        )

        await router.route_token_package_async(token_package)

        # Should call receive_token directly
        router._netrunner.receive_token_transaction.assert_called_once()
        transaction = router._netrunner.receive_token_transaction.return_value.__enter__.return_value
        transaction.receive_token.assert_called_once_with(
            place="test_place", frame=sample_frame, data={"test": "data"}
        )

        # Should not create any new token packages
        assert len(router.yield_calls) == 0
        assert len(router.remote_calls) == 0
        assert len(router.other_job_calls) == 0

    @pytest.mark.asyncio
    async def test_remote_token_package_routing(self, router, sample_frame):
        """Test routing a remote token package to the same job."""
        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="different_site",
            target_place="test_place",
            target_job="test_job",
            is_yield=False,
            frame=sample_frame,
            data={"test": "data"},
        )

        await router.route_token_package_async(token_package)

        # Should call remote method with the same token package
        assert len(router.remote_calls) == 1
        assert router.remote_calls[0] == token_package

    @pytest.mark.asyncio
    async def test_yield_token_package_routing(self, router, sample_frame):
        """Test routing a yield token package."""
        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="test_site",
            target_place="test_place",
            target_job="test_job",
            is_yield=True,
            frame=sample_frame,
            data={"test": "data"},
        )

        await router.route_token_package_async(token_package)

        # Should call yield method with the same token package
        assert len(router.yield_calls) == 1
        assert router.yield_calls[0] == token_package

    @pytest.mark.asyncio
    async def test_other_job_token_package_routing(self, router, sample_frame):
        """Test routing a token package to a different job."""
        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="test_site",
            target_place="test_place",
            target_job="other_job",
            is_yield=False,
            frame=sample_frame,
            data={"test": "data"},
        )

        await router.route_token_package_async(token_package)

        # Should call other job method with the same token package
        assert len(router.other_job_calls) == 1
        assert router.other_job_calls[0] == token_package


class TestRouteTokenPackageSync:
    """Test the route_token_package_sync method."""

    def test_local_token_package_routing_sync(self, router, sample_frame):
        """Test routing a local token package synchronously."""
        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="test_site",
            target_place="test_place",
            target_job="test_job",
            is_yield=False,
            frame=sample_frame,
            data={"test": "data"},
        )

        router.route_token_package_sync(token_package)

        # Should call receive_token directly
        router._netrunner.receive_token_transaction.assert_called_once()
        transaction = router._netrunner.receive_token_transaction.return_value.__enter__.return_value
        transaction.receive_token.assert_called_once_with(
            place="test_place", frame=sample_frame, data={"test": "data"}
        )

        # Should not create any new token packages
        assert len(router.yield_calls) == 0
        assert len(router.remote_calls) == 0
        assert len(router.other_job_calls) == 0

    def test_remote_token_package_routing_sync(self, router, sample_frame):
        """Test routing a remote token package synchronously."""
        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="different_site",
            target_place="test_place",
            target_job="test_job",
            is_yield=False,
            frame=sample_frame,
            data={"test": "data"},
        )

        router.route_token_package_sync(token_package)

        # Should call remote method with the same token package
        assert len(router.remote_calls) == 1
        assert router.remote_calls[0] == token_package

    def test_yield_token_package_routing_sync(self, router, sample_frame):
        """Test routing a yield token package synchronously."""
        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="test_site",
            target_place="test_place",
            target_job="test_job",
            is_yield=True,
            frame=sample_frame,
            data={"test": "data"},
        )

        router.route_token_package_sync(token_package)

        # Should call yield method with the same token package
        assert len(router.yield_calls) == 1
        assert router.yield_calls[0] == token_package

    def test_other_job_token_package_routing_sync(self, router, sample_frame):
        """Test routing a token package to a different job synchronously."""
        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="test_site",
            target_place="test_place",
            target_job="other_job",
            is_yield=False,
            frame=sample_frame,
            data={"test": "data"},
        )

        router.route_token_package_sync(token_package)

        # Should call other job method with the same token package
        assert len(router.other_job_calls) == 1
        assert router.other_job_calls[0] == token_package


class TestAsyncMethodDelegation:
    """Test that async methods properly delegate to sync methods when not overridden."""

    @pytest.mark.asyncio
    async def test_async_yield_delegation(self, router, sample_token_package):
        """Test that async yield method delegates to sync method."""
        await router._send_this_job_yield_token_package_async(sample_token_package)

        assert len(router.yield_calls) == 1
        assert router.yield_calls[0] == sample_token_package

    @pytest.mark.asyncio
    async def test_async_remote_delegation(self, router, sample_token_package):
        """Test that async remote method delegates to sync method."""
        await router._send_this_job_remote_token_package_async(sample_token_package)

        assert len(router.remote_calls) == 1
        assert router.remote_calls[0] == sample_token_package

    @pytest.mark.asyncio
    async def test_async_other_job_delegation(self, router, sample_token_package):
        """Test that async other job method delegates to sync method."""
        await router._send_other_job_remote_token_package_async(sample_token_package)

        assert len(router.other_job_calls) == 1
        assert router.other_job_calls[0] == sample_token_package


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_route_token_with_none_to_job(self, router, sample_frame):
        """Test routing with None to_job (should default to current job)."""
        data = {"key": "value"}

        await router.route_token_async(
            to_job=None,
            to_site="test_site",
            to_place="test_place",
            is_yield=False,
            frame=sample_frame,
            data=data,
            node_id=1,
        )

        # Should treat as local token
        router._netrunner.receive_token_transaction.assert_called_once()
        transaction = router._netrunner.receive_token_transaction.return_value.__enter__.return_value
        transaction.receive_token.assert_called_once_with(
            place="test_place", frame=sample_frame, data=data
        )

    @pytest.mark.asyncio
    async def test_route_token_with_none_node_id(self, router, sample_frame):
        """Test routing with None node_id."""
        data = {"key": "value"}

        await router.route_token_async(
            to_job="other_job",
            to_site="test_site",
            to_place="test_place",
            is_yield=False,
            frame=sample_frame,
            data=data,
            node_id=None,
        )

        # Should create token package with None node_id
        assert len(router.other_job_calls) == 1
        token_package = router.other_job_calls[0]
        assert token_package.source_node is None

    def test_route_token_package_with_complex_data(self, router, sample_frame):
        """Test routing with complex data structures."""
        complex_data = {
            "nested": {"list": [1, 2, 3], "dict": {"a": "b"}},
            "simple": "value",
            "numbers": [1.5, 2.7, 3.14],
        }

        token_package = TokenPackage.wrap_data(
            source_site="source_site",
            source_node=1,
            source_job="source_job",
            target_site="test_site",
            target_place="test_place",
            target_job="test_job",
            is_yield=False,
            frame=sample_frame,
            data=complex_data,
        )

        router.route_token_package_sync(token_package)

        # Should handle complex data correctly
        router._netrunner.receive_token_transaction.assert_called_once()
        transaction = router._netrunner.receive_token_transaction.return_value.__enter__.return_value
        transaction.receive_token.assert_called_once_with(
            place="test_place", frame=sample_frame, data=complex_data
        )


class TestRouterAbstractMethods:
    """Test that abstract methods are properly defined."""

    def test_router_is_abstract(self):
        """Test that Router cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Router()  # pyright: ignore[reportAbstractUsage]

    def test_concrete_router_can_be_instantiated(self):
        """Test that concrete implementation can be instantiated."""
        router = MockRouter()
        assert isinstance(router, Router)

    def test_abstract_methods_are_defined(self):
        """Test that all abstract methods are properly defined in concrete class."""
        router = MockRouter()

        # These should not raise AttributeError
        assert hasattr(router, "_send_this_job_yield_token_package_sync")
        assert hasattr(router, "_send_this_job_remote_token_package_sync")
        assert hasattr(router, "_send_other_job_remote_token_package_sync")

        # These should be callable
        assert callable(router._send_this_job_yield_token_package_sync)
        assert callable(router._send_this_job_remote_token_package_sync)
        assert callable(router._send_other_job_remote_token_package_sync)
