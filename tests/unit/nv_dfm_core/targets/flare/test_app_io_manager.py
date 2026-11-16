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

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nvflare.apis.fl_constant import FLContextKey

from nv_dfm_core.exec import Frame, TokenPackage
from nv_dfm_core.targets.flare._app_io_manager import AppIOManager


def _token(idx: int) -> TokenPackage:
    return TokenPackage.wrap_data(
        source_site="client1",
        source_node=None,
        source_job="job",
        target_site="homesite",
        target_place=f"place_{idx}",
        target_job="job",
        is_yield=True,
        frame=Frame.start_frame(num=0),
        data={"value": idx},
        trace_context=None,
    )


def _fl_ctx(app_root: Path) -> MagicMock:
    ctx = MagicMock()
    ctx.get_prop.side_effect = lambda key, default=None: (
        str(app_root) if key == FLContextKey.APP_ROOT else default
    )
    return ctx


def test_get_all_returns_tokens_put_immediately_before():
    manager = AppIOManager()
    for idx in range(50):
        manager.receive_token_package(_token(idx))

    tokens = manager.get_all()

    assert [t.target_place for t in tokens] == [f"place_{idx}" for idx in range(50)]
    assert manager.get_all() == []


@pytest.mark.parametrize("attempt", range(20))
def test_save_all_remaining_keeps_token_put_immediately_before(tmp_path, attempt):
    manager = AppIOManager()
    manager.receive_token_package(_token(attempt))

    manager.save_all_remaining(_fl_ctx(tmp_path))

    saved = sorted((tmp_path / "results").glob("token_*.json"))
    assert len(saved) == 1
    assert TokenPackage.model_validate_json(saved[0].read_text()).target_place == (
        f"place_{attempt}"
    )


def test_frozen_after_save(tmp_path):
    manager = AppIOManager()
    manager.save_all_remaining(_fl_ctx(tmp_path))

    with pytest.raises(RuntimeError, match="frozen"):
        manager.receive_token_package(_token(0))
    assert manager.get_all() == []
