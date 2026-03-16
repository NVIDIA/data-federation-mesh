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

# pyright: reportMissingTypeStubs=false

from logging import Logger

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import (
    Shareable,
)
from nvflare.apis.signal import Signal
from nvflare.apis.utils.reliable_message import ReliableMessage
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from typing_extensions import override

from nv_dfm_core.exec import Router, TokenPackage
from nv_dfm_core.targets.flare._app_io_manager import AppIOManager

from ._defs import Constant


class FlareRouter(Router):
    def __init__(
        self,
        fl_ctx: FLContext,
        abort_signal: Signal,
        logger: Logger,
        app_io_manager: AppIOManager | None = None,  # for the server only
        client_names: list[str]
        | None = None,  # for the server controller, just to check that recipients are valid
    ):
        super().__init__()
        self._fl_ctx: FLContext = fl_ctx
        self._abort_signal: Signal = abort_signal
        self._client_names: list[str] | None = client_names
        self._app_io_manager: AppIOManager | None = app_io_manager

    def _send_flare_message(self, recipient: str, token_package: TokenPackage):
        request = Shareable()
        request[Constant.MSG_KEY_TOKEN_PACKAGE_DICT] = token_package.model_dump()
        _ = ReliableMessage.send_request(
            target=recipient,
            topic=Constant.TOPIC_SEND_TO_PLACE,
            request=request,
            fl_ctx=self._fl_ctx,
            per_msg_timeout=10,
            tx_timeout=50,
            abort_signal=self._abort_signal,
        )

    @override
    def _send_other_job_remote_token_package_sync(self, token_package: TokenPackage):
        raise RuntimeError(
            f"Target job {token_package.target_job} is not the current job {self.job_id}. Cross-job communication is not yet supported in the Flare target."
        )

    @override
    def _send_this_job_remote_token_package_sync(self, token_package: TokenPackage):
        assert (
            token_package.target_site != self.this_site
        )  # this package is certainly not for us
        assert token_package.target_job == self.job_id

        # if we are not on the server, we need to send the message via the server
        if self.this_site != FQCN.ROOT_SERVER:
            self._send_flare_message(
                recipient=FQCN.ROOT_SERVER, token_package=token_package
            )

        # we are on the server, message is for the homesite
        elif token_package.target_site == self.homesite:
            assert self._app_io_manager is not None
            # target is the app
            self._app_io_manager.receive_token_package(token_package)

        # we are on the server, target is a client
        else:
            if (
                self._client_names is not None
                and token_package.target_site not in self._client_names
            ):
                raise RuntimeError(f"Invalid target site: {token_package.target_site}")
            self._send_flare_message(
                recipient=token_package.target_site, token_package=token_package
            )

    @override
    def _send_this_job_yield_token_package_sync(self, token_package: TokenPackage):
        if self.this_site != FQCN.ROOT_SERVER:
            self._send_flare_message(
                recipient=FQCN.ROOT_SERVER, token_package=token_package
            )
        else:
            assert self._app_io_manager is not None
            self._app_io_manager.receive_token_package(token_package)
