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

"""
Tests for the bump.py module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nv_dfm_core.cli.core._version import (
    BumpType,
    Config,
    JsonUpdater,
    ModuleUpdater,
    UvUpdater,
    YamlUpdater,
    bump,
    get,
    get_updater,
    get_version_from_module,
    get_version_from_uv,
    increment_version,
    ok_to_update,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


def test_bump_version_increment():
    """Test version increment functionality."""
    # Test patch bump
    assert increment_version("1.0.0", BumpType.PATCH) == "1.0.1"
    assert increment_version("1.0.1", BumpType.PATCH) == "1.0.2"

    # Test minor bump
    assert increment_version("1.0.0", BumpType.MINOR) == "1.1.0"
    assert increment_version("1.1.0", BumpType.MINOR) == "1.2.0"

    # Test major bump
    assert increment_version("1.0.0", BumpType.MAJOR) == "2.0.0"
    assert increment_version("2.0.0", BumpType.MAJOR) == "3.0.0"


def test_bump_type_from_name():
    """Test BumpType.from_name functionality."""
    assert BumpType.from_name("patch") == BumpType.PATCH
    assert BumpType.from_name("minor") == BumpType.MINOR
    assert BumpType.from_name("major") == BumpType.MAJOR
    assert BumpType.from_name("1.0.0") == BumpType.LITERAL


def test_bump_updaters(temp_dir):
    """Test different updater types."""
    # Test ModuleUpdater
    module_file = temp_dir / "test_bump_module.py"
    module_file.write_text('__version__ = "1.0.0"')

    # Mock both find_spec and file operations
    with (
        patch(
            "nv_dfm_core.cli.core._version.importlib.util.find_spec"
        ) as mock_find_spec,
        patch("builtins.open", create=True) as mock_open,
    ):
        # Configure mocks
        mock_find_spec.return_value = type("Spec", (), {"origin": str(module_file)})
        mock_open.return_value.__enter__.return_value.read.return_value = (
            '__version__ = "1.0.0"'
        )

        updater = ModuleUpdater()
        updater.update("test_bump_module", "1.0.1")

        # Verify the correct content was written
        mock_open.return_value.__enter__.return_value.write.assert_called_once_with(
            '__version__ = "1.0.1"'
        )

    # Test YamlUpdater
    yaml_file = temp_dir / "test.yaml"
    yaml_file.write_text("""
version:
  number: "1.0.0"
""")
    updater = YamlUpdater()
    updater.update(str(yaml_file), "1.0.1", "version.number")
    print("Result from YamlUpdater:")
    print(yaml_file.read_text())
    assert "number: 1.0.1" in yaml_file.read_text()

    # Test JsonUpdater
    json_file = temp_dir / "test.json"
    json_file.write_text('{"version": {"number": "1.0.0"}}')
    updater = JsonUpdater()
    updater.update(str(json_file), "1.0.1", "version.number")
    assert '"number": "1.0.1"' in json_file.read_text()


def test_bump_dry_run(temp_dir):
    """Test dry run functionality."""
    # Test ModuleUpdater dry run
    module_file = temp_dir / "test_bump_module.py"
    module_file.write_text('__version__ = "1.0.0"')

    # Mock both find_spec and file operations
    with (
        patch(
            "nv_dfm_core.cli.core._version.importlib.util.find_spec"
        ) as mock_find_spec,
        patch("builtins.open", create=True) as mock_open,
    ):
        # Configure mocks
        mock_find_spec.return_value = type("Spec", (), {"origin": str(module_file)})
        mock_open.return_value.__enter__.return_value.read.return_value = (
            '__version__ = "1.0.0"'
        )

        updater = ModuleUpdater()
        updater.update("test_bump_module", "1.0.1", dry_run=True)

        # Verify no write was called due to dry_run
        mock_open.return_value.__enter__.return_value.write.assert_not_called()

    # Test YamlUpdater dry run
    yaml_file = temp_dir / "test.yaml"
    yaml_file.write_text("""
version:
  number: 1.0.0
""")
    updater = YamlUpdater()
    updater.update(str(yaml_file), "1.0.1", "version.number", dry_run=True)
    print("Result from YamlUpdater:")
    print(yaml_file.read_text())
    assert "number: 1.0.0" in yaml_file.read_text()

    # Test JsonUpdater dry run
    json_file = temp_dir / "test.json"
    json_file.write_text('{"version": {"number": "1.0.0"}}')
    updater = JsonUpdater()
    updater.update(str(json_file), "1.0.1", "version.number", dry_run=True)
    assert '"number": "1.0.0"' in json_file.read_text()


def test_bump_config_validation():
    """Test configuration validation."""
    # Test valid config
    valid_config = {
        "version_source": {"type": "module", "path": "test.module"},
        "files_to_update": [
            {"type": "init", "file": "test.py"},
            {"type": "yaml", "file": "test.yaml", "key": "version.number"},
        ],
    }
    config = Config(**valid_config)
    assert config.version_source.type == "module"
    assert config.version_source.path == "test.module"
    assert len(config.files_to_update) == 2

    # Test invalid config (missing required field)
    invalid_config = {"version_source": {"type": "module"}, "files_to_update": []}
    with pytest.raises(ValueError):
        Config(**invalid_config)


def test_bump_get_updater():
    """Test get_updater function."""
    assert isinstance(get_updater("init"), ModuleUpdater)
    assert isinstance(get_updater("yaml"), YamlUpdater)
    assert isinstance(get_updater("json"), JsonUpdater)

    with pytest.raises(ValueError, match="Unsupported update type"):
        get_updater("invalid_type")


def test_bump_get_version_from_module(temp_dir):
    """Test getting version from a module."""
    with patch("nv_dfm_core.cli.core._version.import_module") as mock_import_module:
        mock_import_module.return_value = type("Module", (), {"__version__": "1.0.0"})

        # Test getting version
        version = get_version_from_module("test_bump_module")
        assert version == "1.0.0"


def test_bump_get_version_from_module_not_found():
    """Test getting version from a non-existent module."""
    with pytest.raises(RuntimeError):
        get_version_from_module("nonexistent_module")


def test_bump_get_version_from_uv():
    """Test getting version from uv."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"1.0.0\n"
        version = get_version_from_uv()
        assert version == "1.0.0"
        mock_run.assert_called_once_with(
            ["uv", "version", "--short"], capture_output=True, shell=(os.name == "nt")
        )


def test_bump_uv_updater():
    """Test UvUpdater functionality."""
    with patch("subprocess.run") as mock_run:
        updater = UvUpdater()
        updater.update(None, "1.0.1")
        mock_run.assert_called_once_with(
            ["uv", "version", "1.0.1"], shell=(os.name == "nt")
        )


def test_bump_uv_updater_dry_run():
    """Test UvUpdater dry run functionality."""
    with patch("subprocess.run") as mock_run:
        updater = UvUpdater()
        updater.update(None, "1.0.1", dry_run=True)
        mock_run.assert_called_once_with(
            ["uv", "version", "1.0.1", "--dry-run"], shell=(os.name == "nt")
        )


def test_bump_yaml_updater_missing_key():
    """Test YamlUpdater with missing key."""
    with pytest.raises(ValueError, match="Key must be provided for YAML files"):
        updater = YamlUpdater()
        updater.update("test.yaml", "1.0.1")


def test_bump_json_updater_missing_key():
    """Test JsonUpdater with missing key."""
    with pytest.raises(ValueError, match="Key must be provided for JSON files"):
        updater = JsonUpdater()
        updater.update("test.json", "1.0.1")


def test_bump_increment_version_literal():
    """Test increment_version with literal version."""
    with pytest.raises(ValueError, match="Invalid bump type LITERAL"):
        increment_version("1.0.0", BumpType.LITERAL)


def test_bump_ok_to_update():
    """Test ok_to_update functionality."""
    with patch("git.Repo") as mock_repo:
        # Mock repo with no existing tags
        mock_repo.return_value.tags = []
        assert ok_to_update("1.0.0") is True

        # Mock repo with existing tag
        mock_repo.return_value.tags = ["nv-dfm-v1.0.0"]
        assert ok_to_update("1.0.0") is False


def test_bump_get():
    """Test get() function."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = b"1.0.0\n"
        version = get()
        assert version == "1.0.0"
        mock_run.assert_called_once_with(
            ["uv", "version", "--short"], capture_output=True, shell=(os.name == "nt")
        )


def test_bump_main_function():
    """Test the main bump function."""
    # Define a mock configuration that's independent of the real config file
    mock_config_dict = {
        "version_source": {"type": "uv", "path": ""},
        "files_to_update": [
            {"type": "init", "file": "test_module"},
            {"type": "yaml", "file": "test.yaml", "key": "appVersion"},
            {"type": "yaml", "file": "test.yaml", "key": "version"},
            {"type": "json", "file": "test.json", "key": "version"},
        ],
    }

    with (
        patch("nv_dfm_core.cli.core._version.get_version_from_uv") as mock_get_version,
        patch("nv_dfm_core.cli.core._version.ok_to_update") as mock_ok_to_update,
        patch("nv_dfm_core.cli.core._version.UvUpdater") as mock_uv_updater,
        patch("nv_dfm_core.cli.core._version.ModuleUpdater") as mock_module_updater,
        patch("nv_dfm_core.cli.core._version.YamlUpdater") as mock_yaml_updater,
        patch("nv_dfm_core.cli.core._version.JsonUpdater") as mock_json_updater,
        patch("builtins.open", create=True) as mock_open,
        patch("nv_dfm_core.cli.core._version.YAML") as mock_yaml_class,
    ):
        # Setup mocks
        mock_get_version.return_value = "1.0.0"
        mock_ok_to_update.return_value = True

        # Mock YAML loading
        mock_yaml_instance = mock_yaml_class.return_value
        mock_yaml_instance.load.return_value = mock_config_dict

        # Test bump with patch
        bump(BumpType.PATCH)
        mock_get_version.assert_called_once()
        mock_ok_to_update.assert_called_once_with("1.0.1")
        mock_uv_updater.return_value.update.assert_called_once_with(
            None, "1.0.1", dry_run=False
        )
        mock_module_updater.return_value.update.assert_called_once_with(
            "test_module", "1.0.1", None, False
        )

        # Check YAML updates
        assert mock_yaml_updater.return_value.update.call_count == 2
        mock_yaml_updater.return_value.update.assert_any_call(
            "test.yaml", "1.0.1", "appVersion", False
        )
        mock_yaml_updater.return_value.update.assert_any_call(
            "test.yaml", "1.0.1", "version", False
        )

        # Check JSON updates
        assert mock_json_updater.return_value.update.call_count == 1
        mock_json_updater.return_value.update.assert_called_with(
            "test.json", "1.0.1", "version", False
        )

        # Reset mocks for next test
        mock_get_version.reset_mock()
        mock_ok_to_update.reset_mock()
        mock_uv_updater.reset_mock()
        mock_module_updater.reset_mock()
        mock_yaml_updater.reset_mock()
        mock_json_updater.reset_mock()

        # Test bump with force
        bump(BumpType.PATCH, force=True)
        mock_get_version.assert_called_once()
        mock_ok_to_update.assert_called_once_with("1.0.1")
        mock_uv_updater.return_value.update.assert_called_once_with(
            None, "1.0.1", dry_run=False
        )
        mock_module_updater.return_value.update.assert_called_once_with(
            "test_module", "1.0.1", None, False
        )

        # Check YAML updates
        assert mock_yaml_updater.return_value.update.call_count == 2
        mock_yaml_updater.return_value.update.assert_any_call(
            "test.yaml", "1.0.1", "appVersion", False
        )
        mock_yaml_updater.return_value.update.assert_any_call(
            "test.yaml", "1.0.1", "version", False
        )

        # Check JSON updates
        assert mock_json_updater.return_value.update.call_count == 1
        mock_json_updater.return_value.update.assert_called_with(
            "test.json", "1.0.1", "version", False
        )

        # Reset mocks for next test
        mock_get_version.reset_mock()
        mock_ok_to_update.reset_mock()
        mock_uv_updater.reset_mock()
        mock_module_updater.reset_mock()
        mock_yaml_updater.reset_mock()
        mock_json_updater.reset_mock()

        # Test bump with literal version
        bump(BumpType.LITERAL, new_version="2.0.0")
        mock_get_version.assert_called_once()
        mock_ok_to_update.assert_called_once_with("2.0.0")
        mock_uv_updater.return_value.update.assert_called_once_with(
            None, "2.0.0", dry_run=False
        )
        mock_module_updater.return_value.update.assert_called_once_with(
            "test_module", "2.0.0", None, False
        )

        # Check YAML updates
        assert mock_yaml_updater.return_value.update.call_count == 2
        mock_yaml_updater.return_value.update.assert_any_call(
            "test.yaml", "2.0.0", "appVersion", False
        )
        mock_yaml_updater.return_value.update.assert_any_call(
            "test.yaml", "2.0.0", "version", False
        )

        # Check JSON updates
        assert mock_json_updater.return_value.update.call_count == 1
        mock_json_updater.return_value.update.assert_called_with(
            "test.json", "2.0.0", "version", False
        )
