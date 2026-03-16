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
import logging
from typing import Any

from typing_extensions import override
from collections.abc import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from nv_dfm_core.api._yield import DISCOVERY_PLACE_NAME
from nv_dfm_core.api.discovery import BranchFieldAdvice, SingleFieldAdvice
from nv_dfm_core.exec._dfm_context import DfmContext
from nv_dfm_core.exec._frame import Frame


class MockDfmContext(DfmContext):
    """Concrete implementation of DfmContext for testing."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._send_to_place_internal_calls: list[dict[str, Any]] = []

    @override
    async def _send_to_place_internal(
        self,
        to_job: str | None,
        to_site: str,
        to_place: str,
        is_yield: bool,
        frame: Frame,
        data: Any,
        node_id: int | str | None = None,
    ):
        self._send_to_place_internal_calls.append(
            {
                "to_job": to_job,
                "to_site": to_site,
                "to_place": to_place,
                "is_yield": is_yield,
                "frame": frame,
                "data": data,
                "node_id": node_id,
            }
        )


class IterableButNotIterator(BaseModel):
    """A Pydantic model that is iterable but not an iterator."""

    items: list[str]

    @override
    def __iter__(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return iter(self.items)


class SyncIterator:
    """A custom sync iterator."""

    def __init__(self, items: list[Any]):
        self.items: list[Any] = items
        self.index: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.items):
            raise StopIteration
        item = self.items[self.index]
        self.index += 1
        return item


class AsyncIterator:
    """A custom async iterator."""

    def __init__(self, items: list[Any]):
        self.items: list[Any] = items
        self.index: int = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


def sync_generator(items: Any) -> Generator[Any, Any, None]:
    """A sync generator function."""
    for item in items:
        yield item


async def async_generator(items: Any) -> AsyncGenerator[Any, Any]:
    """An async generator function."""
    for item in items:
        yield item


@pytest.fixture
def mock_logger() -> MagicMock:
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def mock_runtime_module() -> MagicMock:
    module = MagicMock()
    module.API_VERSION = "1.0.0"
    return module


@pytest.fixture
def dfm_context(mock_logger: MagicMock, mock_runtime_module: MagicMock):
    # Mock the load_site_runtime_module function
    with pytest.MonkeyPatch().context() as m:
        m.setattr(
            "nv_dfm_core.exec._dfm_context.load_site_runtime_module",
            lambda fed, site: mock_runtime_module,  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
        )

        context = MockDfmContext(
            pipeline_api_version="1.0.0",
            federation_name="test_fed",
            homesite="homesite",
            this_site="this_site",
            job_id="test_job",
            router=MagicMock(),
            logger=mock_logger,
        )
        return context


class TestDfmContextProperties:
    """Test the properties of DfmContext."""

    def test_properties(self, dfm_context: MockDfmContext):
        assert dfm_context.api_version == "1.0.0"
        assert dfm_context.federation_name == "test_fed"
        assert dfm_context.this_site == "this_site"
        assert dfm_context.homesite == "homesite"
        assert dfm_context.job_id == "test_job"
        assert dfm_context.logger is not None


class TestSendToPlace:
    """Test the send_to_place method with different data types."""

    @pytest.mark.asyncio
    async def test_send_to_place_with_regular_data(self, dfm_context: MockDfmContext):
        """Test that regular data is sent directly."""
        data = {"key": "value"}

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=data,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 1
        call = dfm_context._send_to_place_internal_calls[0]
        assert call["to_job"] == "job1"
        assert call["to_site"] == "remote_site"
        assert call["to_place"] == "output"
        assert call["data"] == data
        assert call["node_id"] == 42

    @pytest.mark.asyncio
    async def test_send_to_place_with_sync_iterator(self, dfm_context: MockDfmContext):
        """Test that sync iterators are iterated and each item is sent."""
        items = ["item1", "item2", "item3"]
        iterator = iter(items)

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            data=iterator,
            frame=Frame.start_frame(0),
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 3
        for i, call in enumerate(dfm_context._send_to_place_internal_calls):
            assert call["to_job"] == "job1"
            assert call["to_site"] == "remote_site"
            assert call["to_place"] == "output"
            assert call["data"] == f"item{i + 1}"
            assert call["node_id"] == 42

    @pytest.mark.asyncio
    async def test_send_to_place_with_async_iterator(self, dfm_context: MockDfmContext):
        """Test that async iterators are iterated and each item is sent."""
        items = ["item1", "item2", "item3"]
        async_iterator = AsyncIterator(items)

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=async_iterator,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 3
        for i, call in enumerate(dfm_context._send_to_place_internal_calls):
            assert call["to_job"] == "job1"
            assert call["to_site"] == "remote_site"
            assert call["to_place"] == "output"
            assert call["frame"] == Frame(frame=[0, i])
            assert call["data"] == f"item{i + 1}"
            assert call["node_id"] == 42

    @pytest.mark.asyncio
    async def test_send_to_place_with_sync_generator(self, dfm_context: MockDfmContext):
        """Test that sync generators are iterated and each item is sent."""
        items = ["item1", "item2", "item3"]
        generator = sync_generator(items)

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=generator,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 3
        for i, call in enumerate(dfm_context._send_to_place_internal_calls):
            assert call["to_job"] == "job1"
            assert call["to_site"] == "remote_site"
            assert call["to_place"] == "output"
            assert call["frame"] == Frame(frame=[0, i])
            assert call["data"] == f"item{i + 1}"
            assert call["node_id"] == 42

    @pytest.mark.asyncio
    async def test_send_to_place_with_async_generator(
        self, dfm_context: MockDfmContext
    ):
        """Test that async generators are iterated and each item is sent."""

        # NOTE: sending iterators that are not Yields is technically not supported yet.
        # Don't do this in practice!
        items = ["item1", "item2", "item3"]
        async_gen = async_generator(items)

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=async_gen,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 3
        for i, call in enumerate(dfm_context._send_to_place_internal_calls):
            assert call["to_job"] == "job1"
            assert call["to_site"] == "remote_site"
            assert call["to_place"] == "output"
            assert call["frame"] == Frame(frame=[0, i])
            assert call["data"] == f"item{i + 1}"
            assert call["node_id"] == 42

    @pytest.mark.asyncio
    async def test_send_to_place_with_custom_sync_iterator(
        self, dfm_context: MockDfmContext
    ):
        """Test that custom sync iterators are handled correctly."""
        items = ["item1", "item2", "item3"]
        custom_iterator = SyncIterator(items)

        # NOTE: sending iterators that are not Yields is technically not supported yet.
        # Don't do this in practice!

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=custom_iterator,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 3
        for i, call in enumerate(dfm_context._send_to_place_internal_calls):
            assert call["data"] == f"item{i + 1}"

    @pytest.mark.asyncio
    async def test_send_to_place_with_custom_async_iterator(
        self, dfm_context: MockDfmContext
    ):
        """Test that custom async iterators are handled correctly."""
        items = ["item1", "item2", "item3"]
        custom_async_iterator = AsyncIterator(items)

        # NOTE: sending iterators that are not Yields is technically not supported yet.
        # Don't do this in practice!

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=custom_async_iterator,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 3
        for i, call in enumerate(dfm_context._send_to_place_internal_calls):
            assert call["data"] == f"item{i + 1}"

    @pytest.mark.asyncio
    async def test_send_to_place_with_iterable_but_not_iterator(
        self, dfm_context: MockDfmContext
    ):
        """Test that iterable objects that are not iterators are sent directly."""
        # Pydantic BaseModel is iterable but not an iterator
        iterable_data = IterableButNotIterator(items=["a", "b", "c"])

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=iterable_data,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 1
        call = dfm_context._send_to_place_internal_calls[0]
        assert call["data"] == iterable_data  # Should be sent as-is, not iterated

    @pytest.mark.asyncio
    async def test_send_to_place_with_list(self, dfm_context: MockDfmContext):
        """Test that lists (which are iterable but not iterators) are sent directly."""
        list_data = ["item1", "item2", "item3"]

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=list_data,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 1
        call = dfm_context._send_to_place_internal_calls[0]
        assert call["data"] == list_data  # Should be sent as-is, not iterated

    @pytest.mark.asyncio
    async def test_send_to_place_with_dict(self, dfm_context: MockDfmContext):
        """Test that dicts (which are iterable but not iterators) are sent directly."""
        dict_data = {"key1": "value1", "key2": "value2"}

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=dict_data,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 1
        call = dfm_context._send_to_place_internal_calls[0]
        assert call["data"] == dict_data  # Should be sent as-is, not iterated

    @pytest.mark.asyncio
    async def test_send_to_place_with_string(self, dfm_context: MockDfmContext):
        """Test that strings (which are iterable but not iterators) are sent directly."""
        string_data = "hello world"

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=string_data,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 1
        call = dfm_context._send_to_place_internal_calls[0]
        assert call["data"] == string_data  # Should be sent as-is, not iterated

    @pytest.mark.asyncio
    async def test_send_to_place_with_none(self, dfm_context: MockDfmContext):
        """Test that None is sent directly."""
        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=None,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 1
        call = dfm_context._send_to_place_internal_calls[0]
        assert call["data"] is None

    @pytest.mark.asyncio
    async def test_send_to_place_with_empty_iterator(self, dfm_context: MockDfmContext):
        """Test that empty iterators don't send anything."""
        empty_iterator = iter([])

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=empty_iterator,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 0

    @pytest.mark.asyncio
    async def test_send_to_place_with_empty_async_iterator(
        self, dfm_context: MockDfmContext
    ):
        """Test that empty async iterators don't send anything."""
        empty_async_iterator = AsyncIterator([])

        await dfm_context.send_to_place(
            to_job="job1",
            to_site="remote_site",
            to_place="output",
            frame=Frame.start_frame(0),
            data=empty_async_iterator,
            node_id=42,
            is_yield=False,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 0


class TestYieldData:
    """Test the yield_data method."""

    @pytest.mark.asyncio
    async def test_yield_data(self, dfm_context: MockDfmContext):
        """Test that yield_data calls send_to_place with correct parameters."""
        data = {"key": "value"}

        await dfm_context.send_to_place(
            to_job=None,
            to_site="homesite",
            to_place="output",
            frame=Frame.start_frame(0),
            data=data,
            node_id=42,
            is_yield=True,
        )

        assert len(dfm_context._send_to_place_internal_calls) == 1
        call = dfm_context._send_to_place_internal_calls[0]
        assert call["to_job"] is None
        assert call["to_site"] == "homesite"
        assert call["to_place"] == "output"
        assert call["frame"] == Frame.start_frame(0)
        assert call["data"] == data
        assert call["node_id"] == 42


class TestAddDiscovery:
    """Test the add_discovery method."""

    def test_add_discovery(self, dfm_context: MockDfmContext):
        """Test that add_discovery adds advice to the internal list."""
        node_id = 42
        advice = SingleFieldAdvice(field="test", value="value")

        dfm_context.add_discovery(node_id, advice)

        assert len(dfm_context._advice) == 1
        stored_node_id, stored_advice = dfm_context._advice[0]
        assert stored_node_id.ident == node_id
        assert stored_advice == advice

    def test_add_discovery_with_branch_advice(self, dfm_context: MockDfmContext):
        """Test that add_discovery works with BranchFieldAdvice."""
        node_id = 42
        advice = BranchFieldAdvice(field="test", branches=[("branch1", None)])

        dfm_context.add_discovery(node_id, advice)

        assert len(dfm_context._advice) == 1
        stored_node_id, stored_advice = dfm_context._advice[0]
        assert stored_node_id.ident == node_id
        assert stored_advice == advice

    def test_add_discovery_with_none_advice(self, dfm_context: MockDfmContext):
        """Test that add_discovery works with None advice."""
        node_id = 42

        dfm_context.add_discovery(node_id, None)

        assert len(dfm_context._advice) == 1
        stored_node_id, stored_advice = dfm_context._advice[0]
        assert stored_node_id.ident == node_id
        assert stored_advice is None


class TestSendDiscovery:
    """Test the send_discovery method."""

    @pytest.mark.asyncio
    async def test_send_discovery(self, dfm_context: MockDfmContext):
        """Test that send_discovery sends accumulated advice and clears the list."""
        # Add some advice
        advice1 = SingleFieldAdvice(field="test1", value="value1")
        advice2 = BranchFieldAdvice(field="test2", branches=[("branch1", None)])

        dfm_context.add_discovery(42, advice1)
        dfm_context.add_discovery(43, advice2)

        # Send discovery
        await dfm_context.send_discovery(frame=Frame.start_frame(0))

        # Check that advice was sent
        assert len(dfm_context._send_to_place_internal_calls) == 1
        call = dfm_context._send_to_place_internal_calls[0]
        assert call["to_job"] is None
        assert call["to_site"] == "homesite"
        assert call["to_place"] == DISCOVERY_PLACE_NAME

        # Check that the sent data contains the advice
        sent_data = call["data"]
        assert len(sent_data) == 2
        assert sent_data[0][0].ident == 42
        assert sent_data[0][1] == advice1
        assert sent_data[1][0].ident == 43
        assert sent_data[1][1] == advice2

        # Check that the internal list was cleared
        assert len(dfm_context._advice) == 0

    @pytest.mark.asyncio
    async def test_send_discovery_empty(self, dfm_context: MockDfmContext):
        """Test that send_discovery works when no advice has been added."""
        await dfm_context.send_discovery(frame=Frame.start_frame(0))

        assert len(dfm_context._send_to_place_internal_calls) == 1
        call = dfm_context._send_to_place_internal_calls[0]
        assert call["to_job"] is None
        assert call["to_site"] == "homesite"
        assert call["to_place"] == DISCOVERY_PLACE_NAME
        assert call["data"] == []
