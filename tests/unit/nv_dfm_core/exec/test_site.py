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

# pyright: reportArgumentType=false
import os
import shutil
from typing import Any

import appdirs

from nv_dfm_core.exec import FsspecConfig, SecretsVault, SecretsVaultConfig, Site


def test_site_cache_storage_is_created():
    test_cache_folder = "tests/assets/outputs/dfm_exec_test_site/cache"
    if os.path.exists(test_cache_folder):
        shutil.rmtree(test_cache_folder)

    site = Site(
        dfm_context=type("MockDfmContext", (), {"federation_name": "atlantis"})(),  # type: ignore
        cache_config=FsspecConfig(
            protocol="file", storage_options={}, base_path=test_cache_folder
        ),
        secrets_vault_config=SecretsVaultConfig(path=None, args={}),
    )

    # check cache storage
    s1 = site.cache_storage()
    assert s1.exists()
    assert s1.is_dir()
    assert s1.protocol == "file"
    assert s1.storage_options == {}
    assert s1.path.endswith(test_cache_folder)

    s2 = site.cache_storage("my_subpath")
    assert s2.exists()
    assert s2.is_dir()
    assert s2.protocol == "file"
    assert s2.storage_options == {}
    assert s2.path.endswith(f"{test_cache_folder}/my_subpath")


def test_site_cache_storage_defaults_to_user_cache_dir():
    site = Site(
        dfm_context=type("MockDfmContext", (), {"federation_name": "dfm_testsuite"})(),  # type: ignore
        cache_config=FsspecConfig(protocol="file", storage_options={}, base_path=None),
        secrets_vault_config=SecretsVaultConfig(path=None, args={}),
    )

    # check cache storage
    s1 = site.cache_storage()
    assert s1.exists()
    assert s1.is_dir()
    assert s1.protocol == "file"
    assert s1.storage_options == {}
    assert s1.path.endswith(appdirs.user_cache_dir("dfm_testsuite"))


def test_site_default_vault():
    site = Site(
        dfm_context=type("MockDfmContext", (), {"federation_name": "atlantis"})(),  # type: ignore
        cache_config=FsspecConfig(protocol="file", storage_options={}, base_path=None),
        secrets_vault_config=SecretsVaultConfig(
            path=None, args={"prefix": "TOP_SECRET_", "additional_stuff": "ignored"}
        ),
    )

    assert site._secrets_vault
    assert isinstance(site._secrets_vault, SecretsVault)


class CustomVault(SecretsVault):
    def __init__(self, my_arg: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._my_arg = my_arg


def test_site_custom_vault_is_instantiated():
    site = Site(
        dfm_context=type("MockDfmContext", (), {"federation_name": "atlantis"})(),  # type: ignore
        cache_config=FsspecConfig(protocol="file", storage_options={}, base_path=None),
        secrets_vault_config=SecretsVaultConfig(
            path="test_site.CustomVault", args={"my_arg": "my_value"}
        ),
    )

    assert site._secrets_vault
    assert isinstance(site._secrets_vault, CustomVault)
    assert site._secrets_vault._my_arg == "my_value"


def test_illegal_custom_vault_falls_back_to_default():
    site = Site(
        dfm_context=type("MockDfmContext", (), {"federation_name": "atlantis"})(),  # type: ignore
        cache_config=FsspecConfig(protocol="file", storage_options={}, base_path=None),
        secrets_vault_config=SecretsVaultConfig(
            path="test_site.IllegalCustomVault", args={"my_arg": "my_value"}
        ),
    )

    assert site._secrets_vault
    assert isinstance(site._secrets_vault, SecretsVault)
