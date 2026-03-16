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

"""Tests for the DFM CLI main module."""

import pytest
from click.testing import CliRunner

from nv_dfm_core.cli.main import cli


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


def test_version(runner):
    """Test that --version flag displays the correct version."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    # Get the actual version from the nv_dfm_core module
    import nv_dfm_core

    expected_output = f"DFM, version {nv_dfm_core.__version__}\n"
    assert result.output == expected_output
