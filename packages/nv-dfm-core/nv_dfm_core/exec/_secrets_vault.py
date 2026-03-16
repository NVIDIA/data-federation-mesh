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

import os
from typing import Any

from ._secrets_vault_config import SecretsVaultConfig


class SecretsVault:
    def __init__(self, prefix: str = "DFM_SECRET_", **kwargs: Any):
        self._prefix: str = prefix

    def secret_for_key(self, key: str) -> str:
        """Return the secret for the given key from environment variables.

        Args:
            key: The environment variable name to retrieve

        Returns:
            The value of the environment variable

        Raises:
            ValueError: If the environment variable is not set
        """
        env_var_name = (self._prefix + key).upper()
        value = os.getenv(env_var_name)
        if value is None:
            raise ValueError(f"Environment variable '{env_var_name}' is not set")
        return value

    @classmethod
    def from_config(cls, config: SecretsVaultConfig) -> "SecretsVault":
        """Create a SecretsVault instance from configuration.

        Args:
            config: Configuration containing the class path and constructor arguments

        Returns:
            An instance of the specified SecretsVault subclass
        """
        import importlib

        # Split the fully qualified class path into module and class name
        if config.path:
            module_path, class_name = config.path.rsplit(".", 1)

            # Import the module
            module = importlib.import_module(module_path)

            # Get the class
            secrets_vault_class = getattr(module, class_name)
        else:
            secrets_vault_class = cls

        # Create and return an instance with the provided arguments
        return secrets_vault_class(**config.args)
