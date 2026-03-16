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

from nv_dfm_core.exec import SecretsVault, SecretsVaultConfig


def test_secrets_vault_default():
    vault = SecretsVault.from_config(SecretsVaultConfig(path=None, args={}))
    assert isinstance(vault, SecretsVault)
    assert vault._prefix == "DFM_SECRET_"


def test_secrets_vault_prefix():
    vault = SecretsVault.from_config(
        SecretsVaultConfig(path=None, args={"prefix": "TOP_SECRET_"})
    )
    assert isinstance(vault, SecretsVault)
    assert vault._prefix == "TOP_SECRET_"


def test_secrets_vault_env_vars():
    vault = SecretsVault.from_config(SecretsVaultConfig(path=None, args={}))
    import os

    # Set a test environment variable
    os.environ["DFM_SECRET_TEST_KEY"] = "test_secret_value"

    # Test that the vault can retrieve the secret
    secret = vault.secret_for_key("TEST_KEY")
    assert secret == "test_secret_value"

    # Test vault converts key to upper case
    secret = vault.secret_for_key("test_key")
    assert secret == "test_secret_value"

    # Test that missing keys raise ValueError
    try:
        vault.secret_for_key("MISSING_KEY")
        assert False, "Expected ValueError for missing key"
    except ValueError:
        pass

    # Clean up

    del os.environ["DFM_SECRET_TEST_KEY"]
