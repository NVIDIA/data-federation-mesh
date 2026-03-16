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

from ._site import Site


class Provider:
    """Base class for all providers. Providers may get specialized
    to hold additional data or functionality, e.g. an FsspecProvider or a
    WebsocketProvider to manage state across multiple operation calls.
    Providers get instantiated when the site gets instantiated and are
    passed to the constructors of the dfm.contrib implementation classes."""

    def __init__(self, site: Site):
        self._site: Site = site

    @property
    def site(self) -> Site:
        return self._site
