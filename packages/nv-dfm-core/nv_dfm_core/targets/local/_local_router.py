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

# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownMemberType=false
from multiprocessing import Queue

from typing_extensions import override

from nv_dfm_core.exec import Router, TokenPackage


class LocalRouter(Router):
    def __init__(
        self,
        channels: dict[str, Queue],
        yield_queue: Queue,
        inter_job_queue: Queue,
    ):
        super().__init__()
        self._out_channels: dict[str, Queue] = channels
        self._yield_queue: Queue = yield_queue
        self._inter_job_queue: Queue = inter_job_queue

    @override
    def _send_this_job_remote_token_package_sync(self, token_package: TokenPackage):
        # the token package is either for a different site, or for a different job. But we
        # aren't responsible for either. Send the token to the outgoing channel to the receiving site
        # which will pick the correct local router to receive it
        assert token_package.target_job == self.job_id
        assert token_package.target_site != self.this_site

        target_site = token_package.target_site
        if target_site not in self._out_channels:
            raise ValueError(f"Target site {target_site} not found in channels")
        self._out_channels[target_site].put(token_package)

    @override
    def _send_this_job_yield_token_package_sync(self, token_package: TokenPackage):
        assert token_package.is_yield
        assert token_package.target_job == self.job_id
        self._yield_queue.put(token_package)

    @override
    def _send_other_job_remote_token_package_sync(self, token_package: TokenPackage):
        assert token_package.target_job != self.job_id
        self._inter_job_queue.put(token_package)
