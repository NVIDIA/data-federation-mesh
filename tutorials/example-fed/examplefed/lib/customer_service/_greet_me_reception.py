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

from nv_dfm_core.exec import Provider, Site


class GreetMeReception:
    def __init__(self, site: Site, provider: Provider | None):
        self._site = site
        self._provider = provider

    async def body(self, name: str, venue: str) -> str:
        """The reception desk has a more formal protocol for greeting customers."""

        from datetime import datetime

        # use a fixed date to simplify testing
        now = datetime.strptime("2025-04-17 10:12:00", "%Y-%m-%d %H:%M:%S")
        weekday = now.strftime("%A")
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%I")

        # For name="Sunlight" and venue="examplefed", this will return:
        # "Welcome to the reception desk at examplefed, Sunlight. Today is Thursday, 2025-04-17 and it is 10 o'clock. We hope you are having a good experience."
        return (
            f"Welcome to the reception desk at {venue}, {name}."
            f" Today is {weekday}, {date} and it is {time} o'clock."
            " We hope you are having a good experience."
        )
