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

"""
*******************************************************************************
*               Welcome to the examplefed's Concierge app!                   *
*******************************************************************************

The Concierge app demonstrates executing greeting pipelines at different
locations in the examplefed federation.

This tool allows you to request greetings from different sites in the
federation:
- local: Execute greeting locally, optionally with a custom message
- server: Execute greeting on the server. The federation configuration does
  not allow custom messages for the server.
- reception: Execute greeting at reception. The federation configuration
  specifies a different adapter implementation for the reception site,
  which is more formal in its tone.

For local greetings, you can provide a custom message format like:
"Hello {name}! It's a nice day today to try the dfm."
If no message is provided, the federation's default greeting will be used.

"""

import argparse
from pathlib import Path
from typing import Any, Literal

import examplefed.fed.runtime.concierge
from examplefed.fed.api.users import GreetMe
from examplefed.fed.site.concierge.users import (
    GreetMe as ConciergeGreetMe,
)
from nv_dfm_core.api import Pipeline, PlaceParam, Yield
from nv_dfm_core.exec._frame import Frame
from nv_dfm_core.session import JobStatus
from nv_dfm_core.session._session import Session, configure_session_logging


class ConciergeApp:
    def __init__(self):
        # Create default session
        self._session: Session | None = None

    def disconnect(self):
        """Properly close the session to flush telemetry and release resources."""
        if self._session:
            self._session.close()
            self._session = None

    def connect(self, federation_path: str, target: Literal["flare", "local"]):
        # We need to create a new session with the given federation path to work with a remote federation
        if target == "flare":
            user = "concierge@examplefed.com"
            flare_workspace = Path(federation_path)
            admin_package = flare_workspace / "prod_00" / user
            if not admin_package.exists():
                raise RuntimeError(f"Admin package not found at {admin_package}")
            kwargs = {
                "user": user,
                "flare_workspace": flare_workspace,
                "admin_package": admin_package,
            }
        else:
            assert target == "local"
            kwargs = {}

        self._session = examplefed.fed.runtime.concierge.get_session(
            target=target,
            logger=configure_session_logging(force_enable=False),
            **kwargs,
        )
        self._session.connect()

    def example1_execute_specialized_greetme_on_homesite_with_custom_message(
        self,
        name: str,
        greeting: str | None,
        debug: bool = False,
    ):
        """
        The configuration specifies that the GreetMe operation on the homesite exposes
        an additional parameter "greeting" that lets the app submit a custom format.
        This allows us to directly use the GreetMe operation that is specific to the
        'concierge' site, which exposes this parameter, as opposed to the base GreetMe, which
        doesn't. This specialization ties our example pipeline to the concierge site.
        """
        if not self._session:
            raise RuntimeError("Session not connected")

        print(
            "Executing the GreetMe that's specialized on the concierge homesite with custom message"
        )

        with Pipeline() as p:
            greet = ConciergeGreetMe(
                name=PlaceParam(place="name"), greeting=PlaceParam(place="greeting")
            )
            _ = Yield(value=greet)

        prepared_pipeline = self._session.prepare(
            pipeline=p, restrict_to_sites="homesite", debug=debug
        )

        def _receive_greeting_callback(
            from_site: str,
            _node: int | str | None,
            _frame: Frame,
            target_place: str,
            data: Any,
        ):
            # can do more complex things of course.
            print(
                f"Received greeting from {from_site} for place '{target_place}', frame {_frame}:"
            )
            print(f"  {data}")

        def _receive_other_callback(
            from_site: str,
            _node: int | str | None,
            frame: Frame,
            target_place: str,
            data: Any,
        ):
            if frame.is_stop_frame():
                print("Received overall pipeline stop frame")
            else:
                print(f"Received stop of call frame {frame.frame}")

        job = self._session.execute(
            pipeline=prepared_pipeline,
            input_params={"name": name, "greeting": greeting},
            default_callback=_receive_other_callback,
            place_callbacks={"yield": _receive_greeting_callback},
        )
        _ = job.wait_until_finished(timeout=25.0)
        print(f"Job (hopefully) finished with status FINISHED: {job.get_status()}")
        assert job.get_status() == JobStatus.FINISHED, (
            f"Job finished with status {job.get_status()}"
        )

    def example2_execute_general_greetme_on_any_site(
        self,
        site: Literal["concierge", "server", "reception"],
        name: str,
        debug: bool = False,
    ):
        """This example shows how to use the 'generic' federation api.
        Note that the site (and possibly provider, but they are not used here)
        parameters can vary. Here, we also opted to prepare the pipeline
        every time again, as an example that pipelines can be built
        dynamically, and because there are several
        target/site combinations and it's easier this way.

        This example registers the callback as a place callback.
        "yield" is the default place the Yield statement sends its data to.
        However, it is possible to change the place name to something else
        as in `Yield(value=greet, place='greetings')`."""

        if not self._session:
            raise RuntimeError("Session not connected")

        print(f"Executing the general GreetMe on site {site}")

        with Pipeline() as p:
            greet = GreetMe(site=site, name=name)
            _ = Yield(value=greet)

        def _receive_greeting_callback(
            from_site: str,
            _node: int | str | None,
            _frame: Frame,
            target_place: str,
            data: Any,
        ):
            # can do more complex things of course.
            print(f"Received greeting from {from_site} for place '{target_place}':")
            print(f"  {data}")

        def _receive_other_callback(
            from_site: str,
            _node: int | str | None,
            frame: Frame,
            target_place: str,
            data: Any,
        ):
            if frame.is_stop_frame():
                print("Received overall pipeline stop frame")
            else:
                print(f"Received stop of call frame {frame.frame}")

        prepared = self._session.prepare(p, debug=debug)
        job = self._session.execute(
            prepared,
            input_params={},  # in this example, we "compile" the name into the pipeline
            default_callback=_receive_other_callback,
            place_callbacks={"yield": _receive_greeting_callback},
        )
        # don't wait forever
        _ = job.wait_until_finished(timeout=25.0)

        print(f"Job (hopefully) finished with status FINISHED: {job.get_status()}")
        assert job.get_status() == JobStatus.FINISHED, (
            f"Job finished with status {job.get_status()}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    _ = parser.add_argument("--name", required=True, help="Your name")
    _ = parser.add_argument(
        "--greetme-from",
        required=True,
        choices=["app", "server", "reception"],
        help="The site that should do the greeting",
    )
    _ = parser.add_argument(
        "--message", help="Custom greeting message format (for app greetings only)"
    )
    _ = parser.add_argument(
        "--target",
        choices=["flare", "local"],
        default="flare",
        help="Target network to run the app on. NOTE: --greetme-from=app currently is only supported with the local target.",
    )
    _ = parser.add_argument(
        "--federation-path",
        help="Path to the federation directory",
    )
    _ = parser.add_argument(
        "--debug", action="store_true", help="Enable debug output from IRGen"
    )

    args = parser.parse_args()
    if args.message and args.greetme_from != "app":
        parser.error("--message can only be used with --greetme-from=app")

    name = args.name
    message = args.message
    debug = args.debug

    if args.target == "flare":
        if not args.federation_path:
            parser.error("--federation-path is required when using --target=flare")

    app = ConciergeApp()
    app.connect(federation_path=args.federation_path, target=args.target)

    try:
        if args.greetme_from == "app" and message:
            # The homesite GreetMe operation is configured to accept a custom greeting message,
            # which we use here. In this example, only the homesite GreetMe accepts a custom
            # greeting message because of the configuration.
            # A prepared pipeline can be executed multiple times with different input parameters
            app.example1_execute_specialized_greetme_on_homesite_with_custom_message(
                name=name, greeting=message, debug=debug
            )
        else:
            # If there's no custom message, we can execute the GreetMe operation on any site,
            # including the homesite
            exec_site = "concierge" if args.greetme_from == "app" else args.greetme_from
            app.example2_execute_general_greetme_on_any_site(
                site=exec_site, name=name, debug=debug
            )
    finally:
        # Always close the session to flush telemetry and release resources
        app.disconnect()

    print("The concierge app says goodbye!")


if __name__ == "__main__":
    main()
