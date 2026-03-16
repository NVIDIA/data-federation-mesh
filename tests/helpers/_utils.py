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

from filecmp import dircmp
from pathlib import Path
import shutil


def prep_testfolder(suffix: str) -> Path:
    target = Path(f"tests/assets/outputs/{suffix}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(exist_ok=False)
    return target


def assert_files_and_folders_equal(left: str | Path, right: str | Path):
    def assert_equality_recursively(dcmp: dircmp):
        for name in dcmp.left_only:
            assert False, (
                f"dircmp: File or folder '{name}' is present in {dcmp.left} but is missing in {dcmp.right}"
            )
        for name in dcmp.right_only:
            assert False, (
                f"dircmp: File or folder '{name}' is missing in {dcmp.left} but is present in {dcmp.right}"
            )
        for name in dcmp.diff_files:
            # files that exist in left and in right but are different
            assert False, (
                f"dircmp: File or folder '{name}' differs between {dcmp.left} and {dcmp.right}"
            )
        for sub_dcmp in dcmp.subdirs.values():
            assert_equality_recursively(sub_dcmp)

    dcmp = dircmp(left, right)
    assert_equality_recursively(dcmp)
