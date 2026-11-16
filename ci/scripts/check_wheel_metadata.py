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

"""Check published DFM distributions for dependency, license and version errors.

Usage:
    python3 ci/scripts/check_wheel_metadata.py [DIST_DIR] [--release-version VERSION]

    DIST_DIR defaults to ``artifacts/dist``. Every ``*.whl`` and ``*.tar.gz`` in
    that directory is inspected. DFM packages must carry the approved Apache
    license metadata and payload. Supplying VERSION also requires exactly one
    wheel and one sdist per DFM package, all matching that version. Any violation
    (or an empty directory) produces a non-zero exit.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

DFM_PACKAGES = {"nv-dfm-core", "nv-dfm-lib-common", "nv-dfm-lib-weather"}

# Internal / workspace-only distribution names that must NEVER be published as a
# dependency or extra of a shipped DFM wheel. Names are compared using PEP 503
# normalization, so casing and ``-``/``_``/``.`` differences do not matter.
FORBIDDEN_NAMES = {
    "examplefed",
    "weather-fed",
    "weatherfed",
    "roguefed",
    "nv-dfm-lib-all",
}


def _normalize(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


_FORBIDDEN_NORMALIZED = {_normalize(n) for n in FORBIDDEN_NAMES}

# A Requires-Dist value looks like: "name[extra] (>=1.0) ; marker". The
# distribution name is the leading run of name characters before any spec.
_DIST_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _requires_dist_name(value: str) -> str | None:
    match = _DIST_NAME_RE.match(value)
    return match.group(1) if match else None


def _read_wheel_metadata(path: Path) -> str | None:
    """Return the METADATA text from a wheel, or None if not found."""
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".dist-info/METADATA"):
                return zf.read(name).decode("utf-8", errors="replace")
    return None


def _read_sdist_metadata(path: Path) -> str | None:
    """Return the PKG-INFO text from a source distribution, or None."""
    with tarfile.open(path, "r:*") as tf:
        # PKG-INFO lives at <name>-<version>/PKG-INFO (shallowest match wins).
        candidates = [
            m for m in tf.getmembers() if m.isfile() and m.name.endswith("/PKG-INFO")
        ]
        candidates.sort(key=lambda m: m.name.count("/"))
        for member in candidates:
            fileobj = tf.extractfile(member)
            if fileobj is not None:
                return fileobj.read().decode("utf-8", errors="replace")
    return None


def _check_metadata(dist: Path, metadata_text: str) -> list[str]:
    """Return a list of violation strings for a single distribution."""
    message = Parser().parsestr(metadata_text)
    violations: list[str] = []

    for raw in message.get_all("Requires-Dist", []):
        dep_name = _requires_dist_name(raw)
        if dep_name and _normalize(dep_name) in _FORBIDDEN_NORMALIZED:
            violations.append(f"Requires-Dist: {raw.strip()}")

    for raw in message.get_all("Provides-Extra", []):
        if _normalize(raw) in _FORBIDDEN_NORMALIZED:
            violations.append(f"Provides-Extra: {raw.strip()}")

    return violations


def _check_license(dist: Path, message) -> list[str]:
    """Require the approved project license in each standalone DFM artifact."""
    if _normalize(message.get("Name", "")) not in DFM_PACKAGES:
        return []
    violations = []
    if message.get("License-Expression") != "Apache-2.0":
        violations.append("License-Expression must be Apache-2.0")
    if message.get_all("License-File", []) != ["LICENSE"]:
        violations.append("License-File must identify LICENSE")
    expected = Path(__file__).resolve().parents[2] / "LICENSE"
    try:
        if dist.suffix == ".whl":
            with zipfile.ZipFile(dist) as archive:
                metadata = [
                    n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
                ]
                if len(metadata) != 1:
                    raise ValueError("ambiguous wheel metadata")
                data = archive.read(
                    metadata[0].removesuffix("METADATA") + "licenses/LICENSE"
                )
        else:
            with tarfile.open(dist, "r:*") as archive:
                metadata = sorted(
                    (
                        m
                        for m in archive.getmembers()
                        if m.isfile() and m.name.endswith("/PKG-INFO")
                    ),
                    key=lambda m: m.name.count("/"),
                )[0]
                member = archive.getmember(
                    metadata.name.removesuffix("PKG-INFO") + "LICENSE"
                )
                if not member.isfile():
                    raise ValueError("license is not a regular file")
                data = archive.extractfile(member).read()
        if data != expected.read_bytes():
            violations.append(
                "LICENSE payload does not match the approved repository license"
            )
    except (KeyError, IndexError, ValueError):
        violations.append("missing or invalid LICENSE payload")
    return violations


def check_dist(dist: Path, expected_version: str | None = None) -> list[str]:
    """Inspect one distribution file. Returns a list of violations (possibly empty)."""
    if dist.suffix == ".whl":
        metadata_text = _read_wheel_metadata(dist)
    elif dist.name.endswith(".tar.gz"):
        metadata_text = _read_sdist_metadata(dist)
    else:
        return []

    if metadata_text is None:
        return [f"could not locate metadata inside {dist.name}"]

    message = Parser().parsestr(metadata_text)
    violations = _check_metadata(dist, metadata_text) + _check_license(dist, message)
    if expected_version and message.get("Version") != expected_version:
        violations.append(
            f"Version must be {expected_version}, found {message.get('Version')}"
        )
    return violations


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_dir", nargs="?", type=Path, default=Path("artifacts/dist")
    )
    parser.add_argument("--release-version")
    args = parser.parse_args(argv[1:])
    if args.release_version is not None and not args.release_version.strip():
        parser.error("--release-version must not be empty")
    dist_dir = args.dist_dir

    if not dist_dir.is_dir():
        print(f"ERROR: distribution directory not found: {dist_dir}", file=sys.stderr)
        return 2

    dists = sorted(
        p
        for p in dist_dir.iterdir()
        if p.suffix == ".whl" or p.name.endswith(".tar.gz")
    )

    if not dists:
        print(f"ERROR: no wheels or sdists found in {dist_dir}", file=sys.stderr)
        return 2

    print(
        f"Checking {len(dists)} distribution(s) in {dist_dir} for internal names and license payloads:"
    )
    for name in sorted(FORBIDDEN_NAMES):
        print(f"  - {name}")
    print()

    failed = False
    inventory = []
    for dist in dists:
        violations = check_dist(dist, args.release_version)
        if violations:
            failed = True
            print(f"FAIL: {dist.name} has invalid distribution metadata or licensing:")
            for v in violations:
                print(f"    {v}")
        else:
            print(f"OK:   {dist.name}")
        if args.release_version:
            text = (
                _read_wheel_metadata(dist)
                if dist.suffix == ".whl"
                else _read_sdist_metadata(dist)
            )
            name = _normalize(Parser().parsestr(text or "").get("Name", ""))
            inventory.append((name, "wheel" if dist.suffix == ".whl" else "sdist"))

    if args.release_version:
        expected = {
            (name, kind) for name in DFM_PACKAGES for kind in ("wheel", "sdist")
        }
        if set(inventory) != expected or len(inventory) != len(expected):
            failed = True
            print(
                "FAIL: release requires exactly one wheel and one sdist for each DFM package"
            )

    print()
    if failed:
        print(
            "ERROR: distribution metadata/license checks failed. See the violations above.",
            file=sys.stderr,
        )
        return 1

    print("PASS: distribution metadata and license payloads checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
