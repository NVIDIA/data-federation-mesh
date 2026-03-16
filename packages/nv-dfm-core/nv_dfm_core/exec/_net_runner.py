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

# pyright: reportImportCycles=false
import asyncio
import dataclasses
import inspect
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from logging import Logger
from types import ModuleType
from typing import TYPE_CHECKING, Any

from nv_dfm_core.api._error_token import ErrorToken
from nv_dfm_core.api._yield import STATUS_PLACE_NAME
from nv_dfm_core.telemetry import (
    TELEMETRY_PLACE_NAME,
    TelemetryBatch,
    create_collector,
    telemetry_enabled,
)

from ._activation_functions import Activation, TransitionTryActivateFunc
from ._dfm_context import DfmContext
from ._frame import Frame
from ._net import Net
from ._panic_error import PanicError
from ._places import Place
from ._site import Site

if TYPE_CHECKING:
    from nv_dfm_core.telemetry import SiteTelemetryCollector

    from ._router import Router
else:
    Router = object


class TokenTransaction:
    """Context manager for receiving tokens in a transaction-safe way."""

    def __init__(self, net_runner: "NetRunner"):
        self._net_runner: NetRunner = net_runner
        self._places: dict[str, Place] = net_runner._places  # pyright: ignore[reportPrivateUsage]
        self._places_lock: threading.Lock = net_runner.places_lock

    def __enter__(self):
        # Use a loop to ensure we actually acquire the lock (in case of interruptions)
        acquired = False
        while not acquired:
            acquired = self._places_lock.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ):
        self._places_lock.release()
        # Notify the activation thread that new work might be available
        with self._net_runner.work_available:
            self._net_runner.work_available.notify()

    def receive_token(self, place: str, frame: Frame, data: Any):
        if place not in self._places:
            self._net_runner._logger.error(f"Place {place} not found in {self._places}")  # pyright: ignore[reportPrivateUsage]
            raise ValueError(f"Place {place} not found")
        self._places[place].put(frame, data)

    def has_place(self, place: str) -> bool:
        return place in self._places


def _get_telemetry_collector(
    dfm_context: DfmContext,
) -> "SiteTelemetryCollector | None":
    """Get or create the telemetry collector for this site.

    Returns None if telemetry is disabled.
    The trace_id is derived from job_id automatically.
    """
    if not telemetry_enabled():
        return None

    # Check if we already have a collector attached to the context
    if dfm_context.telemetry_collector is None:
        # trace_id is derived from job_id inside create_collector
        collector = create_collector(
            site=dfm_context.this_site,
            job_id=dfm_context.job_id,
        )

        # Set up periodic flush callback for long-running jobs
        # This sends telemetry batches to the homesite periodically
        def _periodic_flush_callback(batch: TelemetryBatch) -> None:
            """Send telemetry batch to homesite during long-running jobs.

            If this raises an exception, the collector will restore the
            spans to the buffer for the final flush.
            """
            dfm_context.send_to_place_sync(
                to_job=None,
                to_site=dfm_context.homesite,
                to_place=TELEMETRY_PLACE_NAME,
                is_yield=True,
                frame=Frame.start_frame(num=0),
                data=batch,
                node_id=None,
            )

        collector.set_flush_callback(_periodic_flush_callback)

        # Attach to context for reuse
        dfm_context.telemetry_collector = collector
    return dfm_context.telemetry_collector


def transition_async_wrapper(
    activation: Activation,
    this_site: Site,
    dfm_context: DfmContext,
    parent_logger: Logger,
    net_runner: "NetRunner",
) -> None:
    """A task submitted to the thread pool. Starts a new event loop so that
    the transition and adapter code can be async."""
    func = activation.scheduled_func
    logger = parent_logger.getChild(f"RunningActivation.{func.__name__}")

    logger.info(f"running transition func {func.__name__}")
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    # Track this loop so we can stop it if the NetRunner shuts down
    with net_runner._loops_lock:
        net_runner._active_loops.add(event_loop)

    # Get telemetry collector (may be None if disabled)
    collector = _get_telemetry_collector(dfm_context)

    try:
        if collector is not None:
            # Record transition execution as a span
            with collector.span(
                f"transition.{func.__name__}",
                attributes={
                    "transition.name": func.__name__,
                    "site": dfm_context.this_site,
                    "job_id": dfm_context.job_id,
                    "frame": str(activation.frame.frame),
                    "is_stop_frame": activation.frame.is_stop_frame(),
                },
            ) as span:
                event_loop.run_until_complete(
                    func(this_site, dfm_context, activation.frame, activation.data)
                )
                span.set_ok()
        else:
            # No telemetry, just run normally
            event_loop.run_until_complete(
                func(this_site, dfm_context, activation.frame, activation.data)
            )
        logger.info(f"DONE running transition func {func.__name__}")
    except Exception as e:
        # NOTE: we log the error here, because the more the merrier.
        import traceback

        logger.error(
            f"Transition {func.__name__} failed with exception: {e}."
            + " The future will be marked as failed and the netrunner will handle the exception later."
            + f" Stack trace: {traceback.format_exc()}"
        )
        raise e
    finally:
        with net_runner._loops_lock:
            net_runner._active_loops.remove(event_loop)
        event_loop.close()
        asyncio.set_event_loop(None)


class NetRunner:
    """
    The NetRunner implements the subset of the overall pipeline that is
    local to this site. The modgen tool is creating a concrete NetRunner
    class given the IR that got sent from the applicatin (and the
    application used the irgen tool to create the IR).

    The semantics behind the Net implemented by a NetRunner is that of a
    petri net. Places are data buffers that get filled by sending tokens
    (possibly from other sites). When tokens get added or removed from
    places, then connected transitions may fire. The "transitions" are
    where the actual logic of the pipeline is implemented.

    The modgen generator will generate subclasses of NetRunner that
    define the places of this net (in create_places()). And for each
    transition "transition_name" in the net, modgen will generate two methods:
    - <transition_name>_try_activate(): async method that returns activation data or None
    - <transition_name>_fire(): async method that implements the transition logic
    """

    def __init__(
        self,
        dfm_context: DfmContext,
        net_module: ModuleType,
        logger: Logger,
    ):
        self._dfm_context: DfmContext = dfm_context

        # hook up the router to the netrunner
        self._router: Router = dfm_context.router
        logger.info(f"Setting netrunner for router {self._router}")
        self._router.set_netrunner(self)

        self._logger: Logger = logger.getChild(dfm_context.this_site)

        # computed properties
        self._this_site_instance: Site = (
            self._dfm_context.this_site_runtime_module.ThisSite(self._dfm_context)
        )
        # instantiate the generated Net class from the module
        self._net: Net = self._instantiate_this_net(net_module)

        # individual places are not thread safe, because they need to be changed
        # all together. Whenever accessing a place, the lock needs to be held.
        self.places_lock: threading.Lock = threading.Lock()
        # extract the places as a dict for easier handling
        # NOTE: dataclasses.asdict appears to create copies of the Place objects, which doesn't
        # work. Therefore we extract the places manually.
        self._places: dict[str, Place] = {
            field.name: getattr(self._net, field.name)
            for field in dataclasses.fields(self._net)
        }

        # extract the activation functions from a list; we keep track of all active activation
        # functions until a transition is done, then we remove its activation function
        self._transitions: list[TransitionTryActivateFunc] = (
            self._net.get_activation_functions()
        )

        # currently running futures, grouped by their activation function
        self._futures: dict[TransitionTryActivateFunc, set[Future[Any]]] = {
            func: set() for func in self._transitions
        }
        # transitions that want to be stopped, waiting for possibly outstanding futures to finish
        self._wind_down: dict[TransitionTryActivateFunc, Activation] = {}
        self._stopping: dict[TransitionTryActivateFunc, Future[Any]] = {}
        self._scheduling_lock: threading.Lock = threading.Lock()
        self._panic_error: PanicError | None = None

        self._check_activations_thread: threading.Thread | None = None
        self._shutdown_event: threading.Event = threading.Event()
        self.work_available: threading.Condition = threading.Condition()

        self._active_loops: set[asyncio.AbstractEventLoop] = set()
        self._loops_lock: threading.Lock = threading.Lock()

    def error_occurred(self) -> bool:
        return self._panic_error is not None

    def get_panic_error(self) -> PanicError:
        assert self._panic_error is not None
        return self._panic_error

    @property
    def dfm_context(self) -> DfmContext:
        return self._dfm_context

    def start(self):
        """Start the background thread that checks for transitions."""
        if self._check_activations_thread is not None:
            raise RuntimeError("NetRunner is already running")
        self._logger.info("Starting NetRunner")
        self._shutdown_event.clear()
        self._check_activations_thread = threading.Thread(
            target=self._run_activation_thread, daemon=True
        )
        self._check_activations_thread.start()

    def is_active(self, func: TransitionTryActivateFunc) -> bool:
        """Only call with the lock held"""
        return (
            func in self._transitions
            and func not in self._wind_down
            and func not in self._stopping
        )

    # done callbacks for futures
    def _cleanup_when_stopping_future_is_done(self, func: TransitionTryActivateFunc):
        def handler(f: Future[Any]):
            try:
                f.result()  # This will raise the exception if any
            except Exception as e:
                self._logger.critical(
                    "Signal stop function %s failed with exception: %s. This is a real issue, aborting.",
                    func.__name__,
                    str(e),
                )
                with self._scheduling_lock:
                    self._panic_error = PanicError(error=e)
                    self._shutdown_event.set()
                # Notify the activation thread to wake up immediately
                with self.work_available:
                    self.work_available.notify()
            with self._scheduling_lock:
                if func in self._stopping:
                    del self._stopping[func]
                # and remove the transition, we are done
                if func in self._transitions:
                    self._transitions.remove(func)
                if len(self._transitions) == 0:
                    self._logger.info("All transitions are done, exiting")
                    self._shutdown_event.set()

        return handler

    def _schedule_wind_down_if_needed(
        self, executor: ThreadPoolExecutor, func: TransitionTryActivateFunc
    ):
        try:
            if func in self._wind_down and (
                func not in self._futures or len(self._futures[func]) == 0
            ):
                self._logger.info(
                    f"Transition {func.__name__} wants to be stopped, no outstanding futures, submitting signal_stop activation to thread pool"
                )
                with self._scheduling_lock:
                    activation = self._wind_down.pop(func)
                # an error may have caused the pool to shut down already
                if not self._shutdown_event.is_set():
                    # Submit to thread pool
                    f = executor.submit(
                        transition_async_wrapper,
                        activation,
                        self._this_site_instance,
                        self._dfm_context,
                        self._logger,
                        self,
                    )
                    # Update tracking after submission
                    with self._scheduling_lock:
                        if func in self._futures:
                            del self._futures[func]
                        self._stopping[func] = f
                    # NOTE: don't hold the lock here. If f finished quickly, add_done_callback will run
                    # immediately in this thread and it will also try to acquire the lock.
                    f.add_done_callback(
                        self._cleanup_when_stopping_future_is_done(func)
                    )

                else:
                    self._logger.info(
                        f"Transition {func.__name__} wants to be stopped, but the thread pool is shutting down, skipping"
                    )
        except Exception as e:
            self._logger.error(
                f"Error scheduling wind down for transition {func.__name__}: {e}"
            )
            raise e

    def _cleanup_when_normal_future_is_done(
        self, executor: ThreadPoolExecutor, func: TransitionTryActivateFunc
    ):
        def handler(f: Future[Any]):
            # check if there was an exception; if so, it's bad
            try:
                f.result()  # This will raise the exception if any
            except Exception as e:
                self._logger.critical(
                    "Activation function %s failed with exception: %s. This is a real issue, aborting.",
                    func.__name__,
                    str(e),
                )
                with self._scheduling_lock:
                    self._panic_error = PanicError(error=e)
                    self._shutdown_event.set()
                # Notify the activation thread to wake up immediately
                with self.work_available:
                    self.work_available.notify()
            # remove the future from the dict
            with self._scheduling_lock:
                if func in self._futures:
                    self._futures[func].remove(f)
                else:
                    self._logger.warning(
                        "Transition %s not found in _futures",
                        func.__name__,
                    )
            # And check if the func is winding down
            self._schedule_wind_down_if_needed(executor, func)

        return handler

    def _run_activation_thread(self):
        """Background thread that continuously checks for transitions that
        can be activated and submits them to the thread pool."""
        with self._scheduling_lock:
            self._logger.info(
                "Starting activation thread, %d transitions", len(self._transitions)
            )
            if len(self._transitions) == 0:
                self._logger.info("No transitions, activation_thread exiting")
                return

        try:
            # Note: len(self._transitions) is not a good indicator of the amount of parallel work and
            # should not be used to set the max_workers.
            executor = ThreadPoolExecutor(
                thread_name_prefix="netrunner_activation_thread"
            )
            try:
                self._logger.info(
                    "Started activation thread and ThreadPoolExecutor with max %d workers",
                    executor._max_workers,
                )

                logging_health_counter = 0
                while not self._shutdown_event.is_set():
                    # first, collect all possible activations. Then release the lock again
                    # This is to avoid possible deadlocks when the thread pool is full and blocks
                    activations_by_func: dict[
                        TransitionTryActivateFunc, list[Activation]
                    ] = {}
                    with self.places_lock:
                        # Check for new transitions that can be activated
                        transitions = self._transitions.copy()
                        logging_health_counter += 1
                        if logging_health_counter % 50 == 0:
                            self._logger.info(
                                "Health check: NetRunner is still alive. Checking %d transitions",
                                len(transitions),
                            )
                            logging_health_counter = 0
                        for try_activate_func in transitions:
                            with self._scheduling_lock:
                                if not self.is_active(try_activate_func):
                                    self._logger.info(
                                        "Transition %s is winding down. Not calling the activation function anymore",
                                        try_activate_func.__name__,
                                    )
                                    continue

                            activations_by_func[try_activate_func] = try_activate_func()

                    # now we can submit the activations to the thread pool
                    future: Future[Any]
                    for try_activate_func, activations in activations_by_func.items():
                        # Check for panic error before processing each function
                        with self._scheduling_lock:
                            if self._panic_error is not None:
                                self._logger.error(
                                    "Panic error detected, aborting",
                                )
                                self._shutdown_event.set()
                                # Notify any waiting threads
                                with self.work_available:
                                    self.work_available.notify()
                                return

                        for activation in activations:
                            if activation.frame.is_stop_frame():
                                if self.is_active(try_activate_func):
                                    self._logger.info(
                                        "Transition %s wants to wind down. Got signal_stop func %s. wind down is %s, stopping is %s",
                                        try_activate_func.__name__,
                                        activation.scheduled_func.__name__,
                                        self._wind_down,
                                        self._stopping,
                                    )
                                    with self._scheduling_lock:
                                        self._wind_down[try_activate_func] = activation
                                    # possible that there is no running transition for this function
                                    self._schedule_wind_down_if_needed(
                                        executor, try_activate_func
                                    )
                                else:
                                    self._logger.info(
                                        "Transition %s activation returned stop frame but transition is already winding down",
                                        try_activate_func.__name__,
                                    )
                                continue

                            # try activate wants us to schedule an activation
                            self._logger.info(
                                "Transition %s is ready to activate, submitting %s to thread pool. Got activation data (keys: %s)",
                                try_activate_func.__name__,
                                activation.scheduled_func.__name__,
                                activation.data.keys()
                                if isinstance(activation.data, dict)
                                else "exception",
                            )
                            future = executor.submit(
                                transition_async_wrapper,
                                activation,
                                self._this_site_instance,
                                self._dfm_context,
                                self._logger,
                                self,
                            )
                            # Add future to tracking without holding the lock during thread pool submission
                            with self._scheduling_lock:
                                if try_activate_func not in self._futures:
                                    self._futures[try_activate_func] = set()
                                self._futures[try_activate_func].add(future)
                            future.add_done_callback(
                                self._cleanup_when_normal_future_is_done(
                                    executor, try_activate_func
                                )
                            )

                    # end of the busy loop,
                    # Wait for new work to become available
                    if not self._shutdown_event.is_set():
                        with self.work_available:
                            _ = self.work_available.wait(
                                timeout=0.1
                            )  # Still have a timeout for safety in case the work available signal gets swallowed
            finally:
                self._logger.info("Shutting down ThreadPoolExecutor (wait=False)")
                executor.shutdown(wait=False)
        except Exception as e:
            self._logger.critical(
                "Activation thread failed uncontrolled with exception: %s",
                str(e),
            )
            # raising e doesn't do much because this is a daemon thread
            raise e

    def is_done(self) -> bool:
        # we didn't start the thread yet
        if self._check_activations_thread is None:
            return False

        return not self._check_activations_thread.is_alive()

    def wait_for_done(self, abort_signal: Any | None):
        """Wait for the NetRunner to finish."""
        if self._check_activations_thread is None:
            return

        # Wait for either the thread to finish or an abort signal
        while (
            self._check_activations_thread.is_alive()
            and not self._shutdown_event.is_set()
        ):
            # Check abort signal first
            if abort_signal:
                # support Event objects as well as flare signals
                triggered = (
                    abort_signal.triggered
                    if hasattr(abort_signal, "triggered")
                    else abort_signal.is_set()
                )
                if triggered:
                    self.shutdown()
                    return

            # Wait a bit before checking again
            _ = self._shutdown_event.wait(timeout=0.1)

        # Give the thread a bit more time to finish naturally
        if self._check_activations_thread.is_alive():
            self._check_activations_thread.join(timeout=1.0)

        # if this netrunner stopped with a PanicError, try to yield it to the status place
        if self.error_occurred():
            try:
                self._logger.error(
                    f"NetRunner error occurred: {self.get_panic_error()}"
                )
                err = self.get_panic_error()
                assert err is not None
                self.dfm_context.send_to_place_sync(
                    to_job=None,
                    to_site=self.dfm_context.homesite,
                    to_place=STATUS_PLACE_NAME,
                    is_yield=True,
                    frame=Frame.stop_frame(),
                    data=ErrorToken.from_exception(err),
                    node_id=None,
                )
            except Exception as e:
                self._logger.error(f"Error sending error token: {e}")
            # Fall through to send telemetry even on error

        # Flush and send telemetry to homesite
        self._flush_telemetry_to_homesite()

    def _flush_telemetry_to_homesite(self) -> None:
        """Flush collected telemetry and send to homesite for aggregation."""
        if not telemetry_enabled():
            return

        # Get the collector from DfmContext (if it exists)
        collector = self._dfm_context.telemetry_collector
        if collector is None:
            return

        # Flush the collector
        batch = collector.flush()
        if batch.is_empty:
            return

        self._logger.info(
            f"Sending telemetry batch to homesite: {batch.span_count} spans, {batch.metric_count} metrics"
        )

        try:
            # Send to homesite via the telemetry place
            self.dfm_context.send_to_place_sync(
                to_job=None,
                to_site=self.dfm_context.homesite,
                to_place=TELEMETRY_PLACE_NAME,
                is_yield=True,
                frame=Frame.stop_frame(),
                data=batch,
                node_id=None,
            )
        except Exception as e:
            self._logger.warning(f"Failed to send telemetry: {e}")

        # Shutdown the collector (close streaming file)
        collector.shutdown()

    def shutdown(self):
        """Gracefully shutdown the NetRunner."""
        if self._check_activations_thread is None:
            return
        self._logger.info("Shutting down NetRunner")
        self._shutdown_event.set()

        # Stop all active event loops to wake up threads stuck in run_until_complete
        with self._loops_lock:
            for loop in self._active_loops:
                try:
                    if loop.is_running():
                        loop.call_soon_threadsafe(loop.stop)
                except Exception as e:
                    self._logger.warning(
                        f"Error stopping event loop during shutdown: {e}"
                    )

        # Notify the activation thread to wake up immediately
        with self.work_available:
            self.work_available.notify()
        # give some time for transitions to be done
        self._check_activations_thread.join(timeout=5.0)
        with self._scheduling_lock:
            for futures in self._futures.values():
                for future in futures:
                    _ = future.cancel()
            for future in self._stopping.values():
                _ = future.cancel()
            self._futures.clear()
            self._wind_down.clear()
            self._stopping.clear()
            self._transitions.clear()
            self._check_activations_thread = None

    def receive_token_transaction(self) -> TokenTransaction:
        """Returns a context manager for receiving tokens in a transaction-safe way.

        Usage:
            with net_runner.receive_token_transaction() as transaction:
                transaction.receive_token("place1", data1)
                transaction.receive_token("place2", data2)
        """
        return TokenTransaction(self)

    def _instantiate_this_net(self, net_module: ModuleType) -> Net:
        """Instantiate the ThisNet class from the module."""
        classes = inspect.getmembers(net_module, predicate=inspect.isclass)
        candidates = [c[1] for c in classes if c[0] == "ThisNet"]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one ThisNet class, got {len(candidates)}"
            )
        net_class = candidates[0]
        # instantiate it to create the places
        instance: Net = net_class()
        instance.assert_places_empty()
        return instance
