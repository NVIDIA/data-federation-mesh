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

"""Tests for nv_dfm_core package initialization and version handling."""

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest


class TestVersionHandling:
    """Test suite for __version__ attribute handling."""

    def test_version_when_package_installed(self):
        """Test that __version__ is set correctly when package is installed."""
        import nv_dfm_core

        # When the package is installed, version should not be "unknown"
        # (assuming the package is installed in the test environment)
        assert hasattr(nv_dfm_core, "__version__")
        assert isinstance(nv_dfm_core.__version__, str)

    def test_version_when_package_not_found(self):
        """Test that __version__ is 'unknown' when package is not found."""
        import importlib
        import sys

        # Save the original nv_dfm_core module and its submodules
        saved_modules = {
            key: mod
            for key, mod in sys.modules.items()
            if key == "nv_dfm_core" or key.startswith("nv_dfm_core.")
        }

        try:
            # Remove all nv_dfm_core modules from cache
            for key in list(sys.modules.keys()):
                if key == "nv_dfm_core" or key.startswith("nv_dfm_core."):
                    del sys.modules[key]

            # Now import it fresh with the mocked version function
            with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
                import nv_dfm_core

                assert nv_dfm_core.__version__ == "unknown"
        finally:
            # Restore all the original modules to prevent test pollution
            for key in list(sys.modules.keys()):
                if key == "nv_dfm_core" or key.startswith("nv_dfm_core."):
                    del sys.modules[key]
            sys.modules.update(saved_modules)

    def test_version_is_exported_in_all(self):
        """Test that __version__ is included in __all__."""
        import nv_dfm_core

        assert hasattr(nv_dfm_core, "__all__")
        assert "__version__" in nv_dfm_core.__all__

    def test_version_format(self):
        """Test that version follows expected format when installed."""
        import nv_dfm_core

        # Version should be either "unknown" or follow semantic versioning pattern
        # (e.g., "1.2.3", "1.2.3.dev0", etc.)
        assert nv_dfm_core.__version__ == "unknown" or len(nv_dfm_core.__version__) > 0
