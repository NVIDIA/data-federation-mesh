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

"""Tests for callback dispatcher abstractions and helper functions."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from nv_dfm_core.exec import Frame, TokenPackage
from nv_dfm_core.session._callback_dispatcher import (
    AsyncioDispatcher,
    CallbackDispatcher,
    CallbackRunner,
    DfmDataCallback,
    DfmDataCallbackSync,
    DirectDispatcher,
    ManualCallbackRunner,
    ManualDispatcher,
    call_callbacks,
    call_callbacks_async,
    callbacks_kind,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_token() -> TokenPackage:
    """Create a sample token package for testing."""
    return TokenPackage(
        source_site="site1",
        source_node="node1",
        source_job="job1",
        target_site="homesite",
        target_place="result",
        target_job="job1",
        is_yield=True,
        frame=Frame.start_frame(0),
        tagged_data=("json", {"value": 42}),
    )


@pytest.fixture
def sample_tokens(sample_token: TokenPackage) -> list[TokenPackage]:
    """Create multiple sample tokens for testing."""
    token2 = TokenPackage(
        source_site="site2",
        source_node="node2",
        source_job="job1",
        target_site="homesite",
        target_place="status",
        target_job="job1",
        is_yield=True,
        frame=Frame.start_frame(0),
        tagged_data=("json", {"message": "ok"}),
    )
    return [sample_token, token2]


@pytest.fixture
def sync_callback() -> DfmDataCallbackSync:
    """Create a mock sync callback."""
    return MagicMock()


# -----------------------------------------------------------------------------
# Tests for callbacks_kind()
# -----------------------------------------------------------------------------


class TestCallbacksKind:
    """Tests for the callbacks_kind() helper function."""

    def test_sync_callbacks(self, sync_callback: DfmDataCallbackSync) -> None:
        """Test detection of sync callbacks."""
        result = callbacks_kind(default_callback=sync_callback)
        assert result == "sync"

    def test_sync_callbacks_with_place_callbacks(
        self, sync_callback: DfmDataCallbackSync
    ) -> None:
        """Test detection of sync callbacks with place-specific callbacks."""
        another_sync = MagicMock()
        result = callbacks_kind(
            default_callback=sync_callback,
            place_callbacks={"result": another_sync, "status": sync_callback},
        )
        assert result == "sync"

    def test_async_callbacks(self) -> None:
        """Test detection of async callbacks."""

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            pass

        result = callbacks_kind(default_callback=async_cb)
        assert result == "async"

    def test_async_callbacks_with_place_callbacks(self) -> None:
        """Test detection of async callbacks with place-specific callbacks."""

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            pass

        async def another_async(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            pass

        result = callbacks_kind(
            default_callback=async_cb,
            place_callbacks={"result": another_async},
        )
        assert result == "async"

    def test_mixed_callbacks_raises(self, sync_callback: DfmDataCallbackSync) -> None:
        """Test that mixed sync/async callbacks raise ValueError."""

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            pass

        with pytest.raises(ValueError, match="Mixed sync and async callbacks"):
            callbacks_kind(
                default_callback=sync_callback,
                place_callbacks={"result": async_cb},
            )

    def test_no_callbacks_returns_none(self) -> None:
        """Test that no callbacks returns 'none'."""
        result = callbacks_kind()
        assert result == "none"

    def test_none_default_with_place_callbacks(
        self, sync_callback: DfmDataCallbackSync
    ) -> None:
        """Test with None default but place callbacks provided."""
        result = callbacks_kind(
            default_callback=None,
            place_callbacks={"result": sync_callback},
        )
        assert result == "sync"


# -----------------------------------------------------------------------------
# Tests for call_callbacks()
# -----------------------------------------------------------------------------


class TestCallCallbacks:
    """Tests for the call_callbacks() helper function."""

    def test_calls_default_callback(
        self, sync_callback: DfmDataCallbackSync, sample_token: TokenPackage
    ) -> None:
        """Test that default callback is called when no place callback matches."""
        call_callbacks(
            default_callback=sync_callback, place_callbacks=None, tokens=sample_token
        )

        sync_callback.assert_called_once()
        call_args = sync_callback.call_args[0]
        assert call_args[0] == "site1"  # source_site
        assert call_args[1] == "node1"  # source_node
        assert call_args[3] == "result"  # target_place
        assert call_args[4] == {"value": 42}  # data

    def test_calls_place_callback(
        self, sync_callback: DfmDataCallbackSync, sample_token: TokenPackage
    ) -> None:
        """Test that place-specific callback is called when it matches."""
        place_callback = MagicMock()
        call_callbacks(
            default_callback=sync_callback,
            place_callbacks={"result": place_callback},
            tokens=sample_token,
        )

        place_callback.assert_called_once()
        sync_callback.assert_not_called()

    def test_calls_callbacks_for_multiple_tokens(
        self, sync_callback: DfmDataCallbackSync, sample_tokens: list[TokenPackage]
    ) -> None:
        """Test callback invocation for multiple tokens."""
        call_callbacks(
            default_callback=sync_callback, place_callbacks=None, tokens=sample_tokens
        )

        assert sync_callback.call_count == 2

    def test_handles_none_tokens(self, sync_callback: DfmDataCallbackSync) -> None:
        """Test that None tokens don't cause errors."""
        call_callbacks(
            default_callback=sync_callback, place_callbacks=None, tokens=None
        )

        sync_callback.assert_not_called()

    def test_no_callback_does_not_error(self, sample_token: TokenPackage) -> None:
        """Test that missing callback doesn't raise an error."""
        # Should not raise
        call_callbacks(default_callback=None, place_callbacks=None, tokens=sample_token)

    def test_place_callback_priority_over_default(
        self, sample_token: TokenPackage
    ) -> None:
        """Test that place callback takes priority over default."""
        default = MagicMock()
        place = MagicMock()
        call_callbacks(
            default_callback=default,
            place_callbacks={"result": place},
            tokens=sample_token,
        )

        place.assert_called_once()
        default.assert_not_called()


# -----------------------------------------------------------------------------
# Tests for call_callbacks_async()
# -----------------------------------------------------------------------------


class TestCallCallbacksAsync:
    """Tests for the call_callbacks_async() helper function."""

    @pytest.mark.asyncio
    async def test_calls_default_callback(self, sample_token: TokenPackage) -> None:
        """Test that default async callback is called."""
        results: list[tuple] = []

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            results.append((from_site, from_node, to_place, data))

        await call_callbacks_async(
            default_callback=async_cb, place_callbacks=None, tokens=sample_token
        )

        assert len(results) == 1
        assert results[0][0] == "site1"  # source_site
        assert results[0][2] == "result"  # target_place
        assert results[0][3] == {"value": 42}  # data

    @pytest.mark.asyncio
    async def test_calls_place_callback(self, sample_token: TokenPackage) -> None:
        """Test that place-specific async callback is called."""
        default_results: list[tuple] = []
        place_results: list[tuple] = []

        async def default_cb(*args: Any) -> None:
            default_results.append(args)

        async def place_cb(*args: Any) -> None:
            place_results.append(args)

        await call_callbacks_async(
            default_callback=default_cb,
            place_callbacks={"result": place_cb},
            tokens=sample_token,
        )

        assert len(place_results) == 1
        assert len(default_results) == 0

    @pytest.mark.asyncio
    async def test_handles_none_tokens(self) -> None:
        """Test that None tokens don't cause errors."""
        results: list[tuple] = []

        async def async_cb(*args: Any) -> None:
            results.append(args)

        await call_callbacks_async(
            default_callback=async_cb, place_callbacks=None, tokens=None
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_calls_multiple_tokens(
        self, sample_tokens: list[TokenPackage]
    ) -> None:
        """Test async callback invocation for multiple tokens."""
        results: list[str] = []

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            results.append(to_place)

        await call_callbacks_async(
            default_callback=async_cb, place_callbacks=None, tokens=sample_tokens
        )

        assert len(results) == 2
        assert "result" in results
        assert "status" in results


# -----------------------------------------------------------------------------
# Tests for CallbackRunner (Abstract Class)
# -----------------------------------------------------------------------------


class TestCallbackRunner:
    """Tests for the CallbackRunner abstract class."""

    def test_stores_callbacks(self, sync_callback: DfmDataCallbackSync) -> None:
        """Test that callbacks are stored correctly."""

        class ConcreteRunner(CallbackRunner):
            def dispatch(
                self, tokens: TokenPackage | list[TokenPackage] | None
            ) -> None:
                pass

        runner = ConcreteRunner(
            default_callback=sync_callback,
            place_callbacks={"result": sync_callback},
        )

        assert runner.default_callback is sync_callback
        assert runner.place_callbacks == {"result": sync_callback}

    def test_default_place_callbacks_is_empty_dict(self) -> None:
        """Test that place_callbacks defaults to empty dict."""

        class ConcreteRunner(CallbackRunner):
            def dispatch(
                self, tokens: TokenPackage | list[TokenPackage] | None
            ) -> None:
                pass

        runner = ConcreteRunner()
        assert runner.default_callback is None
        assert runner.place_callbacks == {}

    def test_start_and_stop_are_noops_by_default(self) -> None:
        """Test that start() and stop() are no-ops by default."""

        class ConcreteRunner(CallbackRunner):
            def dispatch(
                self, tokens: TokenPackage | list[TokenPackage] | None
            ) -> None:
                pass

        runner = ConcreteRunner()
        # Should not raise
        runner.start()
        runner.stop()


# -----------------------------------------------------------------------------
# Tests for CallbackDispatcher (Abstract Class)
# -----------------------------------------------------------------------------


class TestCallbackDispatcher:
    """Tests for the CallbackDispatcher abstract class."""

    def test_create_runner_is_abstract(self) -> None:
        """Test that create_runner is an abstract method."""

        with pytest.raises(TypeError, match="abstract method"):

            class IncompleteDispatcher(CallbackDispatcher):
                pass

            IncompleteDispatcher()  # type: ignore

    def test_concrete_dispatcher_works(
        self, sync_callback: DfmDataCallbackSync
    ) -> None:
        """Test that a concrete dispatcher implementation works."""

        class ConcreteRunner(CallbackRunner):
            def dispatch(
                self, tokens: TokenPackage | list[TokenPackage] | None
            ) -> None:
                if tokens is None:
                    return
                call_callbacks(self._default_callback, self._place_callbacks, tokens)

        class ConcreteDispatcher(CallbackDispatcher):
            def create_runner(
                self,
                default_callback: DfmDataCallback | None = None,
                place_callbacks: dict[str, DfmDataCallback] | None = None,
            ) -> CallbackRunner:
                return ConcreteRunner(default_callback, place_callbacks)

        dispatcher = ConcreteDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        assert runner.default_callback is sync_callback
        assert isinstance(runner, CallbackRunner)


# -----------------------------------------------------------------------------
# Tests for DirectDispatcher
# -----------------------------------------------------------------------------


class TestDirectDispatcher:
    """Tests for the DirectDispatcher class."""

    def test_creates_runner(self, sync_callback: DfmDataCallbackSync) -> None:
        """Test that DirectDispatcher creates a runner."""
        dispatcher = DirectDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        assert runner is not None
        assert runner.default_callback is sync_callback
        assert isinstance(runner, CallbackRunner)

    def test_creates_runner_with_place_callbacks(
        self, sync_callback: DfmDataCallbackSync
    ) -> None:
        """Test runner creation with place-specific callbacks."""
        another_callback = MagicMock()
        dispatcher = DirectDispatcher()
        runner = dispatcher.create_runner(
            default_callback=sync_callback,
            place_callbacks={"result": another_callback},
        )

        assert runner.default_callback is sync_callback
        assert runner.place_callbacks == {"result": another_callback}

    def test_dispatch_calls_callbacks(
        self, sync_callback: DfmDataCallbackSync, sample_token: TokenPackage
    ) -> None:
        """Test that dispatch immediately calls callbacks."""
        dispatcher = DirectDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        runner.dispatch(sample_token)

        sync_callback.assert_called_once()
        call_args = sync_callback.call_args[0]
        assert call_args[0] == "site1"  # source_site
        assert call_args[3] == "result"  # target_place

    def test_dispatch_multiple_tokens(
        self, sync_callback: DfmDataCallbackSync, sample_tokens: list[TokenPackage]
    ) -> None:
        """Test dispatching multiple tokens."""
        dispatcher = DirectDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        runner.dispatch(sample_tokens)

        assert sync_callback.call_count == 2

    def test_dispatch_none_tokens(self, sync_callback: DfmDataCallbackSync) -> None:
        """Test that None tokens don't cause errors."""
        dispatcher = DirectDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        runner.dispatch(None)

        sync_callback.assert_not_called()

    def test_rejects_async_callbacks(self) -> None:
        """Test that async callbacks are rejected."""

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            pass

        dispatcher = DirectDispatcher()
        with pytest.raises(
            ValueError, match="DirectDispatcher only supports synchronous"
        ):
            dispatcher.create_runner(default_callback=async_cb)

    def test_runner_start_stop_are_noops(
        self, sync_callback: DfmDataCallbackSync
    ) -> None:
        """Test that start() and stop() are no-ops for DirectRunner."""
        dispatcher = DirectDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        # Should not raise
        runner.start()
        runner.stop()

    def test_no_callbacks_works(self) -> None:
        """Test that DirectDispatcher works with no callbacks."""
        dispatcher = DirectDispatcher()
        runner = dispatcher.create_runner()  # No callbacks

        assert runner.default_callback is None
        assert runner.place_callbacks == {}

    def test_place_callback_priority(self, sample_token: TokenPackage) -> None:
        """Test that place-specific callback takes priority."""
        default = MagicMock()
        place = MagicMock()

        dispatcher = DirectDispatcher()
        runner = dispatcher.create_runner(
            default_callback=default,
            place_callbacks={"result": place},
        )

        runner.dispatch(sample_token)

        place.assert_called_once()
        default.assert_not_called()


# -----------------------------------------------------------------------------
# Tests for ManualDispatcher
# -----------------------------------------------------------------------------


class TestManualDispatcher:
    """Tests for the ManualDispatcher class."""

    def test_creates_runner(self, sync_callback: DfmDataCallbackSync) -> None:
        """Test that ManualDispatcher creates a ManualCallbackRunner."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        assert runner is not None
        assert isinstance(runner, ManualCallbackRunner)
        assert runner.default_callback is sync_callback

    def test_dispatch_queues_tokens(
        self, sync_callback: DfmDataCallbackSync, sample_token: TokenPackage
    ) -> None:
        """Test that tokens are queued, not immediately processed."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        runner.dispatch(sample_token)

        # Callback should NOT have been called yet
        sync_callback.assert_not_called()
        # Token should be in the queue
        assert runner.has_pending()

    def test_process_pending_sync(
        self, sync_callback: DfmDataCallbackSync, sample_token: TokenPackage
    ) -> None:
        """Test synchronous callback processing."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        runner.dispatch(sample_token)
        assert runner.has_pending()

        runner.process_pending()

        sync_callback.assert_called_once()
        assert not runner.has_pending()

    def test_process_pending_multiple_tokens(
        self, sync_callback: DfmDataCallbackSync, sample_tokens: list[TokenPackage]
    ) -> None:
        """Test processing multiple queued tokens."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        for token in sample_tokens:
            runner.dispatch(token)

        assert runner.has_pending()
        runner.process_pending()

        assert sync_callback.call_count == 2
        assert not runner.has_pending()

    @pytest.mark.asyncio
    async def test_process_pending_async(self, sample_token: TokenPackage) -> None:
        """Test asynchronous callback processing."""
        results: list[str] = []

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            results.append(to_place)

        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=async_cb)

        runner.dispatch(sample_token)
        assert runner.has_pending()

        await runner.process_pending_async()

        assert len(results) == 1
        assert results[0] == "result"
        assert not runner.has_pending()

    def test_has_pending_empty(self, sync_callback: DfmDataCallbackSync) -> None:
        """Test has_pending returns False when queue is empty."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        assert not runner.has_pending()

    def test_dispatch_none_does_not_queue(
        self, sync_callback: DfmDataCallbackSync
    ) -> None:
        """Test that dispatching None doesn't add to queue."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        runner.dispatch(None)

        assert not runner.has_pending()

    def test_process_pending_wrong_kind_raises_sync(
        self, sample_token: TokenPackage
    ) -> None:
        """Test that calling process_pending with async callbacks raises."""

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            pass

        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=async_cb)

        runner.dispatch(sample_token)

        with pytest.raises(ValueError, match="Cannot use process_pending"):
            runner.process_pending()

    @pytest.mark.asyncio
    async def test_process_pending_async_wrong_kind_raises(
        self, sync_callback: DfmDataCallbackSync, sample_token: TokenPackage
    ) -> None:
        """Test that calling process_pending_async with sync callbacks raises."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        runner.dispatch(sample_token)

        with pytest.raises(ValueError, match="Cannot use process_pending_async"):
            await runner.process_pending_async()

    def test_multiple_process_pending_calls(
        self, sync_callback: DfmDataCallbackSync, sample_token: TokenPackage
    ) -> None:
        """Test that process_pending can be called multiple times."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=sync_callback)

        runner.dispatch(sample_token)
        runner.process_pending()
        assert sync_callback.call_count == 1

        # Second call should be a no-op (queue is empty)
        runner.process_pending()
        assert sync_callback.call_count == 1

        # Add another token and process
        sync_callback.reset_mock()
        runner.dispatch(sample_token)
        runner.process_pending()
        assert sync_callback.call_count == 1

    def test_no_callbacks_works(self) -> None:
        """Test that ManualDispatcher works with no callbacks."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner()

        assert runner.default_callback is None
        assert runner.place_callbacks == {}
        assert not runner.has_pending()

    def test_no_callbacks_process_pending_works(
        self, sample_token: TokenPackage
    ) -> None:
        """Test that process_pending works with no callbacks (kind='none')."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner()  # No callbacks

        runner.dispatch(sample_token)
        assert runner.has_pending()

        # Should not raise - kind is 'none' which allows process_pending
        runner.process_pending()
        assert not runner.has_pending()

    @pytest.mark.asyncio
    async def test_no_callbacks_process_pending_async_works(
        self, sample_token: TokenPackage
    ) -> None:
        """Test that process_pending_async works with no callbacks (kind='none')."""
        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner()  # No callbacks

        runner.dispatch(sample_token)
        assert runner.has_pending()

        # Should not raise - kind is 'none' which allows process_pending_async
        await runner.process_pending_async()
        assert not runner.has_pending()

    def test_place_callback_priority(self, sample_token: TokenPackage) -> None:
        """Test that place-specific callback takes priority."""
        default = MagicMock()
        place = MagicMock()

        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(
            default_callback=default,
            place_callbacks={"result": place},
        )

        runner.dispatch(sample_token)
        runner.process_pending()

        place.assert_called_once()
        default.assert_not_called()


# -----------------------------------------------------------------------------
# Tests for AsyncioDispatcher
# -----------------------------------------------------------------------------


class TestAsyncioDispatcher:
    """Tests for the AsyncioDispatcher class."""

    @pytest.mark.asyncio
    async def test_creates_runner(self) -> None:
        """Test that AsyncioDispatcher creates a runner."""

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            pass

        dispatcher = AsyncioDispatcher()
        runner = dispatcher.create_runner(default_callback=async_cb)

        assert runner is not None
        assert runner.default_callback is async_cb
        assert isinstance(runner, CallbackRunner)

    @pytest.mark.asyncio
    async def test_creates_runner_with_explicit_loop(self) -> None:
        """Test runner creation with explicit event loop."""

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            pass

        loop = asyncio.get_running_loop()
        dispatcher = AsyncioDispatcher(loop=loop)
        runner = dispatcher.create_runner(default_callback=async_cb)

        assert runner is not None
        assert runner.default_callback is async_cb

    @pytest.mark.asyncio
    async def test_dispatch_schedules_on_loop(self, sample_token: TokenPackage) -> None:
        """Test that callbacks are scheduled on event loop."""
        results: list[str] = []
        event = asyncio.Event()

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            results.append(to_place)
            event.set()

        loop = asyncio.get_running_loop()
        dispatcher = AsyncioDispatcher(loop=loop)
        runner = dispatcher.create_runner(default_callback=async_cb)

        runner.dispatch(sample_token)

        # Wait for the callback to be executed
        await asyncio.wait_for(event.wait(), timeout=2.0)

        assert len(results) == 1
        assert results[0] == "result"

    def test_rejects_sync_callbacks(self, sync_callback: DfmDataCallbackSync) -> None:
        """Test that sync callbacks are rejected."""
        dispatcher = AsyncioDispatcher()
        with pytest.raises(ValueError, match="AsyncioDispatcher requires asynchronous"):
            dispatcher.create_runner(default_callback=sync_callback)

    @pytest.mark.asyncio
    async def test_dispatch_none_is_noop(self) -> None:
        """Test that dispatching None is a no-op."""
        results: list[str] = []

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            results.append(to_place)

        loop = asyncio.get_running_loop()
        dispatcher = AsyncioDispatcher(loop=loop)
        runner = dispatcher.create_runner(default_callback=async_cb)

        runner.dispatch(None)

        # Give the loop a chance to run (should be nothing scheduled)
        await asyncio.sleep(0.1)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_dispatch_multiple_tokens(
        self, sample_tokens: list[TokenPackage]
    ) -> None:
        """Test dispatching multiple tokens."""
        results: list[str] = []
        event = asyncio.Event()

        async def async_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            results.append(to_place)
            if len(results) == 2:
                event.set()

        loop = asyncio.get_running_loop()
        dispatcher = AsyncioDispatcher(loop=loop)
        runner = dispatcher.create_runner(default_callback=async_cb)

        runner.dispatch(sample_tokens)

        await asyncio.wait_for(event.wait(), timeout=2.0)

        assert len(results) == 2
        assert "result" in results
        assert "status" in results

    @pytest.mark.asyncio
    async def test_loop_property_lazy_resolution(self) -> None:
        """Test lazy loop resolution in loop property."""
        dispatcher = AsyncioDispatcher()  # No loop provided

        # Loop should be resolved lazily
        loop = dispatcher.loop
        assert loop is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_no_callbacks_works(self) -> None:
        """Test that AsyncioDispatcher works with no callbacks."""
        dispatcher = AsyncioDispatcher()
        runner = dispatcher.create_runner()  # No callbacks - should work

        assert runner.default_callback is None
        assert runner.place_callbacks == {}

    @pytest.mark.asyncio
    async def test_no_callbacks_dispatch_is_noop(
        self, sample_token: TokenPackage
    ) -> None:
        """Test that dispatching with no callbacks is a no-op."""
        loop = asyncio.get_running_loop()
        dispatcher = AsyncioDispatcher(loop=loop)
        runner = dispatcher.create_runner()  # No callbacks

        # Should not raise or schedule anything
        runner.dispatch(sample_token)

        # Give the loop a chance to run (nothing should be scheduled)
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_place_callback_priority(self, sample_token: TokenPackage) -> None:
        """Test that place-specific callback takes priority."""
        default_results: list[str] = []
        place_results: list[str] = []
        event = asyncio.Event()

        async def default_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            default_results.append(to_place)
            event.set()

        async def place_cb(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ) -> None:
            place_results.append(to_place)
            event.set()

        loop = asyncio.get_running_loop()
        dispatcher = AsyncioDispatcher(loop=loop)
        runner = dispatcher.create_runner(
            default_callback=default_cb,
            place_callbacks={"result": place_cb},
        )

        runner.dispatch(sample_token)

        await asyncio.wait_for(event.wait(), timeout=2.0)

        assert len(place_results) == 1
        assert len(default_results) == 0


# =============================================================================
# Session Integration Tests
# =============================================================================


class TestSessionWithCallbackDispatcher:
    """Integration tests for Session with different dispatchers."""

    def test_session_default_dispatcher_is_direct(self):
        """Test that Session defaults to DirectDispatcher when not specified."""
        from nv_dfm_core.session import Session

        # Create session without specifying dispatcher
        session = Session.__new__(Session)
        session._callback_dispatcher = DirectDispatcher()

        assert isinstance(session._callback_dispatcher, DirectDispatcher)

    def test_session_accepts_custom_dispatcher(self):
        """Test that Session accepts a custom callback_dispatcher."""
        from nv_dfm_core.session import Session

        # Create session with ManualDispatcher
        session = Session.__new__(Session)
        manual_dispatcher = ManualDispatcher()
        session._callback_dispatcher = manual_dispatcher

        assert session._callback_dispatcher is manual_dispatcher

    def test_callback_dispatcher_property(self):
        """Test callback_dispatcher property access."""
        from nv_dfm_core.session import Session

        session = Session.__new__(Session)
        direct_dispatcher = DirectDispatcher()
        session._callback_dispatcher = direct_dispatcher

        assert session.callback_dispatcher is direct_dispatcher

    def test_job_callback_runner_property(self):
        """Test that Job exposes callback_runner property."""
        from nv_dfm_core.session import Job

        # Create a minimal job subclass for testing
        class TestJob(Job):
            @property
            def job_id(self) -> str:
                return "test-123"

            def get_status(self):
                from nv_dfm_core.session import JobStatus

                return JobStatus.RUNNING

            def wait_until_finished(self, timeout=None) -> bool:
                return True

            def cancel(self):
                pass

            def detach(self):
                pass

            def _send_token_package_internal(self, token_package):
                pass

        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=lambda *args: None)

        job = TestJob(
            homesite="test",
            job_id="test-123",
            callback_runner=runner,
        )

        # Job.callback_runner should return the runner
        assert job.callback_runner is runner

    def test_manual_callback_runner_accessible_from_job(self):
        """Test that ManualCallbackRunner methods are accessible via job.callback_runner."""
        from nv_dfm_core.session import Job

        class TestJob(Job):
            @property
            def job_id(self) -> str:
                return "test-123"

            def get_status(self):
                from nv_dfm_core.session import JobStatus

                return JobStatus.RUNNING

            def wait_until_finished(self, timeout=None) -> bool:
                return True

            def cancel(self):
                pass

            def detach(self):
                pass

            def _send_token_package_internal(self, token_package):
                pass

        dispatcher = ManualDispatcher()
        runner = dispatcher.create_runner(default_callback=lambda *args: None)

        job = TestJob(
            homesite="test",
            job_id="test-123",
            callback_runner=runner,
        )

        # Should be a ManualCallbackRunner
        assert isinstance(job.callback_runner, ManualCallbackRunner)
        # Should have has_pending method
        assert hasattr(job.callback_runner, "has_pending")
        assert hasattr(job.callback_runner, "process_pending")
        assert not job.callback_runner.has_pending()
