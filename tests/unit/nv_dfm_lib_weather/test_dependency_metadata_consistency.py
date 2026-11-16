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

"""Guard: ``[tool.uv.dependency-metadata]`` must stay consistent with ``uv.lock``."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 (tomllib added in 3.11, PEP 680)
    import tomli as tomllib  # pyright: ignore[reportMissingImports]


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "uv.lock").is_file() and (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(
        "Could not locate repo root (dir containing uv.lock + pyproject.toml)"
    )


REPO_ROOT = _repo_root()

# Packages that MUST keep a dependency-metadata entry. Removing one would let uv build
# that no-build-isolation package during lock verification and re-break non-CUDA sync.
REQUIRED_METADATA_PACKAGES = {"earth2grid", "torch-harmonics"}


def _dependency_metadata() -> list[dict]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return data.get("tool", {}).get("uv", {}).get("dependency-metadata", [])


def _locked_versions() -> dict[str, set[str]]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    versions: dict[str, set[str]] = {}
    for pkg in lock.get("package", []):
        name, version = pkg.get("name"), pkg.get("version")
        if name and version:
            versions.setdefault(name, set()).add(version)
    return versions


def test_required_dependency_metadata_entries_present() -> None:
    declared = {entry["name"] for entry in _dependency_metadata() if "name" in entry}
    missing = REQUIRED_METADATA_PACKAGES - declared
    assert not missing, (
        f"Missing [[tool.uv.dependency-metadata]] entries for {sorted(missing)} in the root "
        "pyproject.toml. Without them, uv builds these no-build-isolation packages during "
        "lock verification, which breaks `uv sync` on any machine without torch/CUDA."
    )


@pytest.mark.parametrize(
    "entry",
    _dependency_metadata(),
    ids=lambda entry: entry.get("name", "<unnamed>"),
)
def test_versioned_dependency_metadata_matches_lock(entry: dict) -> None:
    name = entry.get("name")
    version = entry.get("version")
    if version is None:
        # Version-less entries (registry deps) intentionally apply to all versions.
        pytest.skip(f"{name!r} dependency-metadata is version-less; nothing to pin")

    locked = _locked_versions().get(name, set())
    assert version in locked, (
        f"[[tool.uv.dependency-metadata]] for {name!r} declares version {version!r}, but "
        f"uv.lock resolves it to {sorted(locked) or ['<absent>']}. When this drifts, uv "
        f"silently falls back to BUILDING {name!r} during lock verification and every "
        f"non-CUDA `uv sync` breaks with 'ModuleNotFoundError: No module named torch'. "
        f"Update the pyproject dependency-metadata version (and requires-dist) atomically "
        f"with uv.lock."
    )
