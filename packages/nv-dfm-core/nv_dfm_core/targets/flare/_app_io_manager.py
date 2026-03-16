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

from multiprocessing import Lock, Queue
from pathlib import Path
from queue import Empty

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext

from nv_dfm_core.exec import TokenPackage


class AppIOManager:
    """
    Manages the IO of the app through flare app_commands.
    Essentially, it lets the app send tokens to server and client places and it
    lets the app receive tokens from the server and client net runners.

    It's not much more than a queue with some helper functions.
    """

    def __init__(self):
        # use a multiprocessing queue because different threads may need to access it
        self._queue = Queue()
        self._get_all_lock = Lock()
        self._frozen = False

    def receive_token_package(self, token: TokenPackage):
        if self._frozen:
            # this shouldn't happen because we wait for all clients and the server
            # netrunners to finish, so nobody should be sending tokens to the client
            # anymore.
            raise RuntimeError("AppIOManager is frozen and shutting down.")
        self._queue.put(token)

    def get_all(self) -> list[TokenPackage]:
        """Allow only one thread to get all the tokens at a time."""
        with self._get_all_lock:
            # in case there was an app_command underway while we were finishing up.
            if self._frozen:
                return []
            results = []
            try:
                while not self._queue.empty():
                    results.append(self._queue.get())
            except Empty:
                pass
            return results

    def save_all_remaining(self, fl_ctx: FLContext):
        if self._frozen:
            return
        remaining = self.get_all()
        self._frozen = True
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        if not app_root:
            raise RuntimeError("app_root is missing from FL context.")
        directory = Path(app_root) / "results"
        if directory.exists():
            # this shouldn't happen if flare's job IDs are random enough across runs
            raise RuntimeError(
                f"Results directory {directory} already exists. Cannot save results."
            )
        directory.mkdir(parents=True, exist_ok=True)

        for idx, token_package in enumerate(remaining):
            with open(directory / f"token_{idx}.json", "w") as f:
                # we store the tokens as json, just in case somebody wants to look at them
                # data in the token that is not json serializable will be pickled and encoded
                json_dict = token_package.model_dump_json(indent=2)
                _ = f.write(json_dict)
