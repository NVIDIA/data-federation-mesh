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

"""Custom version schemes for setuptools-scm / hatch-vcs.

These schemes produce versions in the format:
- On tagged commit: "3.0.10"
- Between tags: "3.0.10+gabc1234"

This avoids .devN suffixes while still distinguishing non-release builds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from setuptools_scm.version import ScmVersion


def version_scheme(version: ScmVersion) -> str:
    """Return the tag version without any dev/post suffix."""
    # Always return the tag version; local_scheme adds +node for non-tagged
    return str(version.tag)


def local_scheme(version: ScmVersion) -> str:
    """Return just the git node (commit hash) without date."""
    if version.exact or version.node is None:
        # On a tagged commit or no node info, no local version
        return ""
    # Return just the node (e.g., "+gabc1234")
    return f"+{version.node}"
