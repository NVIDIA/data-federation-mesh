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

# pyright: reportPrivateUsage=false
import pytest

from nv_dfm_core.targets.flare._flare_options import FlareOptions
from nv_dfm_core.targets.flare._defs import Constant


def test_flare_options_uses_env_default(monkeypatch):
    monkeypatch.setenv("DFM_FLARE_TASK_TIMEOUT", "321")
    opts = FlareOptions()
    assert opts.task_timeout_s == 321


def test_flare_options_explicit_over_env(monkeypatch):
    monkeypatch.setenv("DFM_FLARE_TASK_TIMEOUT", "456")
    opts = FlareOptions(task_timeout_s=123)
    assert opts.task_timeout_s == 123


def test_flare_options_default_to_constant(monkeypatch):
    monkeypatch.delenv("DFM_FLARE_TASK_TIMEOUT", raising=False)
    opts = FlareOptions()
    assert opts.task_timeout_s == int(Constant.TASK_TIMEOUT)
