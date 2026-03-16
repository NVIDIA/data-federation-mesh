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

import pytest

from nv_dfm_core.gen.apigen import ApiGen
from tests.helpers import assert_files_and_folders_equal


def test_apigen_generate_api():
    apigen = ApiGen.from_yaml_file(
        "tutorials/example-fed/examplefed/configs/federation.dfm.yaml"
    )
    _ = apigen.generate_api(
        language="python",
        outpath=Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/"),
    )
    assert_files_and_folders_equal(
        Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/examplefed/fed/api"),
        Path("tutorials/example-fed/examplefed/fed/api"),
    )
    assert_files_and_folders_equal(
        Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/examplefed/fed/site"),
        Path("tutorials/example-fed/examplefed/fed/site"),
    )


def test_apigen_generate_runtime():
    apigen = ApiGen.from_yaml_file(
        "tutorials/example-fed/examplefed/configs/federation.dfm.yaml"
    )
    _ = apigen.generate_runtime(
        language="python",
        outpath=Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/"),
    )
    assert_files_and_folders_equal(
        Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/examplefed/fed/runtime"),
        Path("tutorials/example-fed/examplefed/fed/runtime"),
    )


def test_apigen_generate_api_testpkg_config():
    apigen = ApiGen.from_yaml_file(
        "tests/assets/inputs/apigen_testfiles/testpkg.dfm.yaml"
    )
    _ = apigen.generate_api(
        language="python",
        outpath=Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/"),
    )
    assert_files_and_folders_equal(
        Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/testpkg/fed/schema"),
        Path("tests/assets/golden/dfm_gen_apigen_test_apigen/testpkg/fed/schema"),
    )
    assert_files_and_folders_equal(
        Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/testpkg/fed/api"),
        Path("tests/assets/golden/dfm_gen_apigen_test_apigen/testpkg/fed/api"),
    )
    assert_files_and_folders_equal(
        Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/testpkg/fed/site"),
        Path("tests/assets/golden/dfm_gen_apigen_test_apigen/testpkg/fed/site"),
    )


def test_apigen_generate_runtime_testpkg_config():
    apigen = ApiGen.from_yaml_file(
        "tests/assets/inputs/apigen_testfiles/testpkg.dfm.yaml"
    )
    _ = apigen.generate_runtime(
        language="python",
        outpath=Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/"),
    )
    assert_files_and_folders_equal(
        Path("tests/assets/outputs/dfm_gen_apigen_test_apigen/testpkg/fed/runtime"),
        Path("tests/assets/golden/dfm_gen_apigen_test_apigen/testpkg/fed/runtime"),
    )


@pytest.mark.skip(
    reason="Use this (for now) to generate the actual examplefed code. It's not a test!"
)
def test_apigen_actually_generate_the_code():
    apigen = ApiGen.from_yaml_file(
        "tutorials/example-fed/examplefed/configs/federation.dfm.yaml"
    )
    _ = apigen.generate_api(language="python", outpath=Path(""))
    _ = apigen.generate_runtime(language="python", outpath=Path(""))
