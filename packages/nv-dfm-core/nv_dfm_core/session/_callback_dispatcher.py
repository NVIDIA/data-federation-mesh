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

"""Callback dispatching framework for DFM pipelines.

This module provides abstractions for controlling where and when pipeline result
callbacks are executed. This is critical for applications like Kit that require
callbacks to run on the main thread (e.g., for USD updates).

Key concepts:
- CallbackDispatcher: Factory that creates CallbackRunner instances (set on Session)
- CallbackRunner: Executes callbacks for a specific job according to a dispatch strategy
"""

from __future__ import annotations

import asyncio
import queue
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal, TypeAlias

if TYPE_CHECKING:
    from nv_dfm_core.exec import Frame, TokenPackage

# -----------------------------------------------------------------------------
# Callback Type Definitions
# -----------------------------------------------------------------------------

# Sync callback signature
DfmDataCallbackSync: TypeAlias = Callable[
    [str, int | str | None, "Frame", str, Any], None
]

# Async callback signature
DfmDataCallbackAsync: TypeAlias = Callable[
    [str, int | str | None, "Frame", str, Any], Coroutine[Any, Any, None]
]

# Union type for both (the primary type to use in APIs)
DfmDataCallback: TypeAlias = DfmDataCallbackSync | DfmDataCallbackAsync


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def callbacks_kind(
    default_callback: DfmDataCallback | None = None,
    place_callbacks: dict[str, DfmDataCallback] | None = None,
) -> Literal["sync", "async", "none"]:
    """Determine whether all callbacks are sync, all are async, or none provided.

    Args:
        default_callback: The default callback for unspecified places.
        place_callbacks: Dictionary mapping place names to callbacks.

    Returns:
        'sync' if all callbacks are synchronous.
        'async' if all callbacks are asynchronous.
        'none' if no callbacks are provided.

    Raises:
        ValueError: If callbacks are mixed (some sync, some async).
    """
    callbacks: list[DfmDataCallback] = []
    if default_callback is not None:
        callbacks.append(default_callback)
    if place_callbacks:
        callbacks.extend(place_callbacks.values())

    if not callbacks:
        return "none"  # No callbacks provided

    async_count = sum(1 for cb in callbacks if asyncio.iscoroutinefunction(cb))
    sync_count = len(callbacks) - async_count

    if async_count > 0 and sync_count > 0:
        raise ValueError(
            "Mixed sync and async callbacks are not supported. "
            f"Found {sync_count} sync and {async_count} async callbacks."
        )

    return "async" if async_count > 0 else "sync"


def call_callbacks(
    default_callback: DfmDataCallback | None,
    place_callbacks: dict[str, DfmDataCallback] | None,
    tokens: TokenPackage | list[TokenPackage] | None,
) -> None:
    """Call sync callbacks for the given tokens.

    This unifies the callback invocation logic from FlareJob._distribute_tokens()
    and LocalJob._invoke_token_callback().

    Args:
        default_callback: Fallback callback for tokens without a place-specific callback.
        place_callbacks: Dictionary mapping place names to callbacks.
        tokens: Single token or list of tokens to process.
    """
    if tokens is None:
        return

    if not isinstance(tokens, list):
        tokens = [tokens]

    for token in tokens:
        callback = None
        if place_callbacks and token.target_place in place_callbacks:
            callback = place_callbacks[token.target_place]
        elif default_callback:
            callback = default_callback

        if callback is not None:
            callback(
                token.source_site,
                token.source_node,
                token.frame,
                token.target_place,
                token.unwrap_data(),
            )


async def call_callbacks_async(
    default_callback: DfmDataCallback | None,
    place_callbacks: dict[str, DfmDataCallback] | None,
    tokens: TokenPackage | list[TokenPackage] | None,
) -> None:
    """Call async callbacks for the given tokens.

    Args:
        default_callback: Fallback callback for tokens without a place-specific callback.
        place_callbacks: Dictionary mapping place names to callbacks.
        tokens: Single token or list of tokens to process.
    """
    if tokens is None:
        return

    if not isinstance(tokens, list):
        tokens = [tokens]

    for token in tokens:
        callback = None
        if place_callbacks and token.target_place in place_callbacks:
            callback = place_callbacks[token.target_place]
        elif default_callback:
            callback = default_callback

        if callback is not None:
            await callback(  # type: ignore[misc]
                token.source_site,
                token.source_node,
                token.frame,
                token.target_place,
                token.unwrap_data(),
            )


# -----------------------------------------------------------------------------
# Abstract Base Classes
# -----------------------------------------------------------------------------


class CallbackRunner(ABC):
    """Executes callbacks for a specific job.

    Created by CallbackDispatcher.create_runner() when a job starts.
    The runner is bound to the callbacks provided at creation time
    and dispatches tokens to them according to the dispatcher's strategy.

    Attributes:
        default_callback: The default callback for unspecified places.
        place_callbacks: Dictionary mapping place names to specific callbacks.
    """

    def __init__(
        self,
        default_callback: DfmDataCallback | None = None,
        place_callbacks: dict[str, DfmDataCallback] | None = None,
    ):
        self._default_callback = default_callback
        self._place_callbacks = place_callbacks or {}

    @property
    def default_callback(self) -> DfmDataCallback | None:
        """The default callback for places without specific callbacks."""
        return self._default_callback

    @property
    def place_callbacks(self) -> dict[str, DfmDataCallback]:
        """Dictionary mapping place names to specific callbacks."""
        return self._place_callbacks

    @abstractmethod
    def dispatch(self, tokens: TokenPackage | list[TokenPackage] | None) -> None:
        """Dispatch tokens to callbacks according to the dispatcher's strategy.

        Args:
            tokens: Single token, list of tokens, or None to process.
        """
        pass

    def start(self) -> None:
        """Called when the job's background thread starts.

        Override if the runner needs to perform setup when the job starts.
        """
        pass

    def stop(self) -> None:
        """Called when the job's background thread stops.

        Override if the runner needs to perform cleanup when the job ends.
        """
        pass


class CallbackDispatcher(ABC):
    """Factory for creating CallbackRunner instances.

    Set on Session to control how callbacks are invoked across all jobs.
    Each dispatcher implementation provides a different strategy for
    where and when callbacks are executed (e.g., directly, queued, on event loop).
    """

    @abstractmethod
    def create_runner(
        self,
        default_callback: DfmDataCallback | None = None,
        place_callbacks: dict[str, DfmDataCallback] | None = None,
    ) -> CallbackRunner:
        """Create a new CallbackRunner bound to the given callbacks.

        Args:
            default_callback: Fallback callback for tokens without a place-specific callback.
            place_callbacks: Dictionary mapping place names to callbacks.

        Returns:
            A new CallbackRunner instance configured for this dispatcher's strategy.
        """
        pass


# -----------------------------------------------------------------------------
# Dispatcher Implementations
# -----------------------------------------------------------------------------


class _DirectRunner(CallbackRunner):
    """Runner that calls callbacks immediately from the calling thread."""

    def dispatch(self, tokens: TokenPackage | list[TokenPackage] | None) -> None:
        """Call callbacks directly in the current thread."""
        call_callbacks(self._default_callback, self._place_callbacks, tokens)


class DirectDispatcher(CallbackDispatcher):
    """Default dispatcher: calls callbacks immediately from the background thread.

    This preserves the current behavior where callbacks are invoked directly
    in the thread that receives the tokens. This is the default dispatcher
    used by Session when no other dispatcher is specified.

    Note: DirectDispatcher only supports synchronous callbacks. If you need
    to use async callbacks, use AsyncioDispatcher instead.

    Example:
        # Explicit usage (same as default behavior)
        session = Session(
            federation_name="my_fed",
            homesite="site1",
            callback_dispatcher=DirectDispatcher()
        )
    """

    def create_runner(
        self,
        default_callback: DfmDataCallback | None = None,
        place_callbacks: dict[str, DfmDataCallback] | None = None,
    ) -> CallbackRunner:
        """Create a DirectRunner for synchronous callback execution.

        Args:
            default_callback: Fallback callback for tokens without a place-specific callback.
            place_callbacks: Dictionary mapping place names to callbacks.

        Returns:
            A _DirectRunner instance that calls callbacks immediately.

        Raises:
            ValueError: If any callback is asynchronous (use AsyncioDispatcher instead).
        """
        # DirectDispatcher only supports sync callbacks (or no callbacks)
        kind = callbacks_kind(default_callback, place_callbacks)
        if kind == "async":
            raise ValueError(
                "DirectDispatcher only supports synchronous callbacks. "
                "Use AsyncioDispatcher for async callbacks."
            )
        # kind is "sync" or "none" - both are valid

        return _DirectRunner(default_callback, place_callbacks)


class ManualCallbackRunner(CallbackRunner):
    """Runner that queues tokens for manual processing.

    Tokens are queued and the user must periodically call:
    - `process_pending()` for sync callbacks
    - `process_pending_async()` for async callbacks

    This enables callbacks to run on a specific thread (e.g., main thread)
    by having that thread call `process_pending()`.
    """

    def __init__(
        self,
        default_callback: DfmDataCallback | None = None,
        place_callbacks: dict[str, DfmDataCallback] | None = None,
    ):
        super().__init__(default_callback, place_callbacks)
        self._queue: queue.Queue[TokenPackage | list[TokenPackage]] = queue.Queue()
        self._kind: Literal["sync", "async", "none"] = callbacks_kind(
            default_callback, place_callbacks
        )

    def dispatch(self, tokens: TokenPackage | list[TokenPackage] | None) -> None:
        """Queue tokens for later processing.

        Args:
            tokens: Single token, list of tokens, or None to queue.
        """
        if tokens is not None:
            self._queue.put(tokens)

    def process_pending(self) -> None:
        """Process all pending tokens with sync callbacks.

        Call this from your desired thread (e.g., main thread).
        This is a non-blocking call that processes all currently queued tokens.

        Raises:
            ValueError: If callbacks are async (use process_pending_async instead).
        """
        if self._kind == "async":
            raise ValueError(
                "Cannot use process_pending() with async callbacks. "
                "Use process_pending_async() instead."
            )
        # kind is "sync" or "none" - both are valid
        while not self._queue.empty():
            try:
                tokens = self._queue.get_nowait()
                call_callbacks(self._default_callback, self._place_callbacks, tokens)
            except queue.Empty:
                break

    async def process_pending_async(self) -> None:
        """Process all pending tokens with async callbacks.

        Call this from your event loop.
        This is a non-blocking call that processes all currently queued tokens.

        Raises:
            ValueError: If callbacks are sync (use process_pending instead).
        """
        if self._kind == "sync":
            raise ValueError(
                "Cannot use process_pending_async() with sync callbacks. "
                "Use process_pending() instead."
            )
        # kind is "async" or "none" - both are valid
        while not self._queue.empty():
            try:
                tokens = self._queue.get_nowait()
                await call_callbacks_async(
                    self._default_callback, self._place_callbacks, tokens
                )
            except queue.Empty:
                break

    def has_pending(self) -> bool:
        """Check if there are pending tokens in the queue.

        Returns:
            True if there are tokens waiting to be processed.
        """
        return not self._queue.empty()


class ManualDispatcher(CallbackDispatcher):
    """Dispatcher that queues tokens for manual processing.

    Tokens are queued and the user must periodically call:
    - `job.callback_runner.process_pending()` for sync callbacks
    - `await job.callback_runner.process_pending_async()` for async callbacks

    This is useful for applications like Kit that need callbacks to run
    on the main thread.

    Example:
        # Setup
        session = Session(
            federation_name="my_fed",
            homesite="site1",
            callback_dispatcher=ManualDispatcher()
        )
        job = session.execute(pipeline, default_callback=my_callback)

        # In your main thread's update loop:
        def on_update():
            runner = job.callback_runner
            if runner and runner.has_pending():
                runner.process_pending()
    """

    def create_runner(
        self,
        default_callback: DfmDataCallback | None = None,
        place_callbacks: dict[str, DfmDataCallback] | None = None,
    ) -> ManualCallbackRunner:
        """Create a ManualCallbackRunner for queued callback execution.

        Args:
            default_callback: Fallback callback for tokens without a place-specific callback.
            place_callbacks: Dictionary mapping place names to callbacks.

        Returns:
            A ManualCallbackRunner instance with a token queue.
        """
        return ManualCallbackRunner(default_callback, place_callbacks)


class _AsyncioRunner(CallbackRunner):
    """Runner that schedules callbacks on an asyncio event loop."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        default_callback: DfmDataCallback | None = None,
        place_callbacks: dict[str, DfmDataCallback] | None = None,
    ):
        super().__init__(default_callback, place_callbacks)
        self._loop = loop

    def dispatch(self, tokens: TokenPackage | list[TokenPackage] | None) -> None:
        """Schedule callbacks on the event loop from any thread.

        Args:
            tokens: Single token, list of tokens, or None to schedule.
        """
        if tokens is None:
            return
        # Skip scheduling if there are no callbacks to call
        if self._default_callback is None and not self._place_callbacks:
            return
        asyncio.run_coroutine_threadsafe(
            call_callbacks_async(self._default_callback, self._place_callbacks, tokens),
            self._loop,
        )


class AsyncioDispatcher(CallbackDispatcher):
    """Dispatcher that schedules callbacks on an asyncio event loop.

    Expects async callbacks. The callbacks will be scheduled to run
    on the specified event loop (or the current event loop if not specified).
    This enables callbacks to be executed in an async context even when
    dispatched from a background thread.

    Note: AsyncioDispatcher only supports asynchronous callbacks. If you need
    to use sync callbacks, use DirectDispatcher instead.

    Example:
        # Use current event loop (resolved lazily)
        session = Session(
            federation_name="my_fed",
            homesite="site1",
            callback_dispatcher=AsyncioDispatcher()
        )

        # Use specific event loop
        loop = asyncio.get_event_loop()
        session = Session(
            federation_name="my_fed",
            homesite="site1",
            callback_dispatcher=AsyncioDispatcher(loop=loop)
        )

        # With async callback
        async def my_async_callback(site, node, frame, place, data):
            await process_data_async(data)

        job = session.execute(pipeline, default_callback=my_async_callback)
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        """Initialize the AsyncioDispatcher.

        Args:
            loop: The event loop to schedule callbacks on. If not specified,
                  the loop is resolved lazily when create_runner() is called.
        """
        self._loop = loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop (lazily resolved if not explicitly set).

        Returns:
            The event loop to use for scheduling callbacks.
        """
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.get_event_loop()
        return self._loop

    def create_runner(
        self,
        default_callback: DfmDataCallback | None = None,
        place_callbacks: dict[str, DfmDataCallback] | None = None,
    ) -> CallbackRunner:
        """Create an _AsyncioRunner for event loop callback scheduling.

        Args:
            default_callback: Fallback callback for tokens without a place-specific callback.
            place_callbacks: Dictionary mapping place names to callbacks.

        Returns:
            An _AsyncioRunner instance that schedules callbacks on the event loop.

        Raises:
            ValueError: If any callback is synchronous (use DirectDispatcher instead).
        """
        # AsyncioDispatcher requires async callbacks (or no callbacks)
        kind = callbacks_kind(default_callback, place_callbacks)
        if kind == "sync":
            raise ValueError(
                "AsyncioDispatcher requires asynchronous callbacks. "
                "Use DirectDispatcher for sync callbacks."
            )
        # kind is "async" or "none" - both are valid

        return _AsyncioRunner(self.loop, default_callback, place_callbacks)
