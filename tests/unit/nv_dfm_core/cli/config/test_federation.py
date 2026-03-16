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

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from unittest.mock import MagicMock, patch
from click import Abort

from nv_dfm_core.cli.config._federation import (
    FederationConfig,
    FederationConfigStored,
    FederationConfigManager,
)
from nv_dfm_core.cli.core._context import CliContext


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()


@pytest.fixture
def sample_federation_config():
    return {
        "dfm": "1.0.0",
        "info": {
            "code-package": "test_module",
            "api-version": "0.0.1",
        },
        "provision": {
            "docker": {
                "image": "test_image",
                "tag": "1.2.3",
                "dockerfile": {
                    "dir": "deploy/docker",
                    "file": "Dockerfile.test",
                },
                "build": {
                    "engine": "docker",
                    "arch": "amd64",
                    "context": ".",
                    "save": {
                        "enabled": True,
                        "file": "artifacts/docker/test_image-1.2.3.tar",
                    },
                    "push": {
                        "enabled": True,
                        "registry": {
                            "url": "test_registry",
                            "username": "test_username",
                            "password": "test_password",
                        },
                    },
                },
            }
        },
    }


@pytest.fixture
def sample_project_config():
    return {
        "name": "test_fed",
        "participants": [
            {"type": "server", "name": "server1"},
            {"type": "server", "name": "server2"},
            {"type": "client", "name": "client1"},
            {"type": "client", "name": "client2"},
            {"type": "admin", "name": "admin_app"},
        ],
    }


@pytest.fixture
def federation_dir(temp_workspace):
    fed_dir = temp_workspace / "test_federation"
    fed_dir.mkdir()
    configs_dir = fed_dir / "configs"
    configs_dir.mkdir()
    return fed_dir.resolve()


@pytest.fixture
def federation_config_files(
    federation_dir, sample_federation_config, sample_project_config
):
    config_path = federation_dir / "configs" / "federation.dfm.yaml"
    project_path = federation_dir / "configs" / "project.yaml"

    with open(config_path, "w") as f:
        yaml.dump(sample_federation_config, f)
    with open(project_path, "w") as f:
        yaml.dump(sample_project_config, f)

    return config_path.resolve(), project_path.resolve()


@pytest.fixture
def mock_context(tmp_path):
    assert isinstance(tmp_path, Path)
    with patch("nv_dfm_core.cli.main.CliContext") as mock:
        context = MagicMock(spec=CliContext)
        context.workspace_path = tmp_path
        context.debug = False
        context.add_config = MagicMock()  # Explicitly mock add_config
        mock.return_value = context
        yield context  # Return the context object, not the mock


@pytest.fixture
def mock_fed_config_mgr():
    with patch(
        "nv_dfm_core.cli.config._federation.FederationConfigManager"
    ) as mock_class:
        manager = MagicMock(spec=FederationConfigManager)
        mock_class.return_value = manager
        mock_class.add_to_context_and_get = MagicMock(return_value=manager)
        yield mock_class, manager


def test_federation_config_initialization(
    temp_workspace, federation_dir, federation_config_files
):
    config_path, project_path = federation_config_files

    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )

    assert config.name == "test_fed"
    assert (
        config.federation_workspace_dir.resolve()
        == (temp_workspace / "test_fed").resolve()
    )
    assert config.federation_dir.resolve() == federation_dir.resolve()
    assert config.config_path.resolve() == config_path.resolve()
    assert config.project_path.resolve() == project_path.resolve()
    assert config.module == "test_module"
    assert config.server_sites == ["server1", "server2"]
    assert config.client_sites == ["client1", "client2"]
    assert config.app_name == "admin_app"
    assert config.provision_info.docker.image == "test_image"
    assert config.provision_info.docker.tag == "1.2.3"
    assert config.provision_info.docker.build.engine == "docker"
    assert config.provision_info.docker.build.arch == "amd64"
    assert config.provision_info.docker.build.context == "."
    assert config.provision_info.docker.build.save.enabled == True
    assert (
        config.provision_info.docker.build.save.file
        == "artifacts/docker/test_image-1.2.3.tar"
    )
    assert config.provision_info.docker.build.push.enabled == True
    assert config.provision_info.docker.build.push.registry.url == "test_registry"
    assert config.provision_info.docker.build.push.registry.username == "test_username"
    assert config.provision_info.docker.build.push.registry.password == "test_password"


def test_federation_config_reconfigure_same_instance(
    temp_workspace, federation_dir, federation_config_files
):
    """Reconfiguring the same FederationConfig instance must re-parse cleanly (no stale app_name)."""
    config_path, project_path = federation_config_files

    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    assert config.app_name == "admin_app"

    # Second initialize (reconfigure) on the same instance must not raise "Multiple admin participants"
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )
    assert config.app_name == "admin_app"
    assert config.server_sites == ["server1", "server2"]
    assert config.client_sites == ["client1", "client2"]


def test_federation_config_data_conversion(
    temp_workspace, federation_dir, federation_config_files
):
    config_path, project_path = federation_config_files

    config = FederationConfig()
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )

    data = config.to_stored()
    assert isinstance(data, FederationConfigStored)
    assert data.federation_dir.resolve() == federation_dir.resolve()
    assert data.config_path.resolve() == config_path.resolve()
    assert data.project_path.resolve() == project_path.resolve()

    new_config = FederationConfig.from_stored(data, temp_workspace, "test_fed")
    assert new_config.federation_dir.resolve() == federation_dir.resolve()
    assert new_config.config_path.resolve() == config_path.resolve()
    assert new_config.project_path.resolve() == project_path.resolve()


def test_federation_config_manager(
    mock_context, federation_dir, federation_config_files
):
    config_path, project_path = federation_config_files

    # Mock the get_config method that the manager calls during initialization
    cli_config_mock = MagicMock()
    cli_config_mock.federations_config_path = None
    mock_context.get_config = MagicMock(return_value=cli_config_mock)

    # Set the workspace_path to match temp_workspace from federation_config_files fixture
    mock_context.workspace_path = federation_dir.parent
    mock_context.debug = True

    # Create default configuration for the manager only
    manager_config_path = mock_context.workspace_path / "fed_config.yaml"
    FederationConfigManager.create_default_config(
        manager_config_path, mock_context.debug
    )

    manager = FederationConfigManager(mock_context)
    manager.initialize()

    # Remove the config file before adding a new federation
    if manager_config_path.exists():
        manager_config_path.unlink()

    # Test adding a new federation (use the correct config_path/project_path)
    config = FederationConfig()
    config.initialize(
        name="test_fed",
        workspace_dir=mock_context.workspace_path,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )

    manager.add_config("test_fed", config)
    assert "test_fed" in manager.get_federation_names()

    # Test getting the config
    retrieved_config = manager.get_config("test_fed")
    assert retrieved_config.name == "test_fed"

    # Remove the config file before deleting the federation
    if manager_config_path.exists():
        manager_config_path.unlink()

    # Test deleting the config
    manager.del_config("test_fed")
    assert "test_fed" not in manager.get_federation_names()


def test_federation_config_manager_default_config(mock_context):
    # Mock the get_config method that the manager calls during initialization
    cli_config_mock = MagicMock()
    cli_config_mock.federations_config_path = None
    mock_context.get_config = MagicMock(return_value=cli_config_mock)
    mock_context.debug = True

    # Create default configuration before initializing the manager
    config_path = mock_context.workspace_path / "fed_config.yaml"
    assert isinstance(config_path, Path)
    FederationConfigManager.create_default_config(config_path, mock_context.debug)

    manager = FederationConfigManager(mock_context)
    manager.initialize()

    assert "examplefed" in manager.get_federation_names()
    config = manager.get_config("examplefed")
    assert config.name == "examplefed"


def test_federation_config_invalid_paths(temp_workspace):
    config = FederationConfig()
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=temp_workspace / "nonexistent",
        config_path="configs/federation.dfm.yaml",
        project_path="configs/project.yaml",
    )

    # Verify that the paths are marked as non-existent
    assert not config.has_federation_dir
    assert not config.has_config_path
    assert not config.has_project_path


def test_federation_config_manager_invalid_workspace():
    # Create a mock context with non-existent workspace path
    mock_context = MagicMock()
    mock_context.workspace_path = Path("/nonexistent/path")
    mock_context.debug = False

    with pytest.raises(FileNotFoundError):
        FederationConfigManager(mock_context)


def test_federation_config_manager_double_initialization(mock_context):
    # Mock the get_config method that the manager calls during initialization
    cli_config_mock = MagicMock()
    cli_config_mock.federations_config_path = None
    mock_context.get_config = MagicMock(return_value=cli_config_mock)
    mock_context.debug = True

    # Create default configuration before initializing the manager
    config_path = mock_context.workspace_path / "fed_config.yaml"
    FederationConfigManager.create_default_config(config_path, mock_context.debug)

    manager = FederationConfigManager(mock_context)
    manager.initialize()

    with pytest.raises(RuntimeError):
        manager.initialize()


def test_federation_config_manager_load_config(
    mock_context, sample_federation_config, sample_project_config
):
    # Set up the workspace directory
    temp_workspace = mock_context.workspace_path

    # Create a sample config file
    config_path = temp_workspace / "fed_config.yaml"
    test_config = {
        "federations": {
            "test_fed": {
                "federation_dir": str(temp_workspace / "test_federation"),
                "config_path": "configs/federation.dfm.yaml",
                "project_path": "configs/project.yaml",
            }
        }
    }

    # Create necessary directories and files
    fed_dir = temp_workspace / "test_federation"
    fed_dir.mkdir()
    configs_dir = fed_dir / "configs"
    configs_dir.mkdir()

    # Create federation config file
    with open(fed_dir / "configs" / "federation.dfm.yaml", "w") as f:
        yaml.dump(sample_federation_config, f)

    # Create project config file
    with open(fed_dir / "configs" / "project.yaml", "w") as f:
        yaml.dump(sample_project_config, f)

    # Write the test config
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    # Mock the get_config method and set up the manager
    cli_config_mock = MagicMock()
    cli_config_mock.federations_config_path = None
    print(
        f"in test: type of cli_config_mock.cli_config.federations_config_path: {type(cli_config_mock.cli_config.federations_config_path)}"
    )
    mock_context.get_config = MagicMock(return_value=cli_config_mock)
    mock_context.debug = True

    manager = FederationConfigManager(mock_context)
    manager._config_path = config_path
    manager._load_config()

    assert "test_fed" in manager._config
    loaded_config = manager._config["test_fed"]
    assert loaded_config.name == "test_fed"
    assert loaded_config.federation_dir.resolve() == fed_dir.resolve()


def test_federation_config_manager_preserves_comments_on_save(
    mock_context, sample_federation_config, sample_project_config
):
    """Saving the federations config must preserve top-level comments (e.g. license header)."""
    temp_workspace = mock_context.workspace_path
    config_path = temp_workspace / "fed_config.yaml"
    fed_dir = temp_workspace / "test_federation"
    fed_dir.mkdir()
    (fed_dir / "configs").mkdir(parents=True)
    with open(fed_dir / "configs" / "federation.dfm.yaml", "w") as f:
        yaml.dump(sample_federation_config, f)
    with open(fed_dir / "configs" / "project.yaml", "w") as f:
        yaml.dump(sample_project_config, f)

    license_header = (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "#\n"
    )
    test_config = {
        "federations": {
            "test_fed": {
                "federation_dir": str(fed_dir),
                "config_path": "configs/federation.dfm.yaml",
                "project_path": "configs/project.yaml",
            }
        }
    }
    with open(config_path, "w") as f:
        f.write(license_header)
        yaml.dump(test_config, f)

    cli_config_mock = MagicMock()
    cli_config_mock.federations_config_path = None
    mock_context.get_config = MagicMock(return_value=cli_config_mock)
    mock_context.debug = True

    manager = FederationConfigManager(mock_context)
    manager._config_path = config_path
    manager._load_config()
    assert manager._config_document is not None

    # Trigger save with existing_document so comments are preserved
    FederationConfigManager.save_config(
        manager._config_path,
        manager._config,
        existing_document=manager._config_document,
    )

    with open(config_path) as f:
        content = f.read()
    assert content.startswith(license_header), (
        "License header must be preserved after save"
    )

    # Key order within each federation must be preserved (config_path before federation_dir in original)
    # Our test file was written with yaml.dump which may reorder; after save we expect the same
    # order as in the loaded document. So we assert config_path appears before federation_dir in the
    # federations block (the order we get from ruamel after load is preserved on save).
    federations_section = content[content.find("federations:") :]
    assert (
        "config_path:" in federations_section
        and "federation_dir:" in federations_section
    )
    # Document was built as license_header + yaml.dump(test_config); test_config has "federations" then
    # per-fed "federation_dir", "config_path", "project_path" (dict order). So after load/save that order
    # should be preserved.
    idx_config = federations_section.find("config_path:")
    idx_fed_dir = federations_section.find("federation_dir:")
    assert idx_config >= 0 and idx_fed_dir >= 0
    # Original yaml.dump order is federation_dir, config_path, project_path (Python dict order from test_config)
    # So we just assert both keys exist; order preservation is that we didn't replace the block.
    assert "project_path:" in federations_section


def test_federation_config_manager_preserves_key_order_on_save(
    mock_context, sample_federation_config, sample_project_config
):
    """Saving must preserve key order within each federation (e.g. config_path before federation_dir)."""
    temp_workspace = mock_context.workspace_path
    config_path = temp_workspace / "fed_config.yaml"
    fed_dir = temp_workspace / "test_federation"
    fed_dir.mkdir()
    (fed_dir / "configs").mkdir(parents=True)
    with open(fed_dir / "configs" / "federation.dfm.yaml", "w") as f:
        yaml.dump(sample_federation_config, f)
    with open(fed_dir / "configs" / "project.yaml", "w") as f:
        yaml.dump(sample_project_config, f)

    # Write with explicit key order: config_path first, then federation_dir, then project_path
    content_with_order = """# header
federations:
  test_fed:
    config_path: configs/federation.dfm.yaml
    federation_dir: """
    content_with_order += str(fed_dir).replace("\\", "/") + "\n"
    content_with_order += "    project_path: configs/project.yaml\n"
    with open(config_path, "w") as f:
        f.write(content_with_order)

    cli_config_mock = MagicMock()
    cli_config_mock.federations_config_path = None
    mock_context.get_config = MagicMock(return_value=cli_config_mock)
    mock_context.debug = True

    manager = FederationConfigManager(mock_context)
    manager._config_path = config_path
    manager._load_config()
    FederationConfigManager.save_config(
        manager._config_path,
        manager._config,
        existing_document=manager._config_document,
    )

    with open(config_path) as f:
        saved = f.read()
    # config_path must still appear before federation_dir in the test_fed block
    fed_block = saved[saved.find("test_fed:") : saved.find("test_fed:") + 400]
    pos_config = fed_block.find("config_path:")
    pos_fed_dir = fed_block.find("federation_dir:")
    assert pos_config >= 0 and pos_fed_dir >= 0, "Both keys must be present"
    assert pos_config < pos_fed_dir, (
        "Key order (config_path before federation_dir) must be preserved"
    )


def test_federation_config_manager_load_config_invalid_yaml(mock_context):
    # Set up the workspace directory
    temp_workspace = mock_context.workspace_path

    # Create an invalid YAML file
    config_path = temp_workspace / "fed_config.yaml"
    with open(config_path, "w") as f:
        f.write("invalid: yaml: content: {")

    # Mock the get_config method and set up the manager
    cli_config_mock = MagicMock()
    cli_config_mock.federations_config_path = None
    mock_context.get_config = MagicMock(return_value=cli_config_mock)
    mock_context.debug = True

    manager = FederationConfigManager(mock_context)
    manager._config_path = config_path

    with pytest.raises(yaml.YAMLError):
        manager._load_config()


def test_federation_config_manager_load_config_invalid_schema(mock_context):
    # Set up the workspace directory
    temp_workspace = mock_context.workspace_path

    # Create a config file with invalid schema
    config_path = temp_workspace / "fed_config.yaml"
    invalid_config = {
        "federations": {
            "test_fed": {
                # Missing required fields
                "invalid_field": "value"
            }
        }
    }

    with open(config_path, "w") as f:
        yaml.dump(invalid_config, f)

    # Mock the get_config method and set up the manager
    cli_config_mock = MagicMock()
    cli_config_mock.federations_config_path = None
    mock_context.get_config = MagicMock(return_value=cli_config_mock)
    mock_context.debug = True

    manager = FederationConfigManager(mock_context)
    manager._config_path = config_path

    with pytest.raises(Exception):  # pydantic will raise a validation error
        manager._load_config()


def test_federation_config_manager_add_to_context_and_get(mock_context):
    """Test that add_to_context_and_get properly initializes and adds the manager to context."""
    # Mock the get_config method that the manager calls during initialization
    cli_config_mock = MagicMock()
    cli_config_mock.federations_config_path = None
    mock_context.get_config = MagicMock(return_value=cli_config_mock)

    # Create default configuration before initializing the manager
    config_path = mock_context.workspace_path / "fed_config.yaml"
    FederationConfigManager.create_default_config(config_path, mock_context.debug)

    # Act
    result = FederationConfigManager.add_to_context_and_get(mock_context)

    # Assert
    assert isinstance(result, FederationConfigManager)
    mock_context.add_config.assert_called_once_with("fed", result)


def test_load_federation_config_success(mock_context, mock_fed_config_mgr):
    """Test successful loading of a federation config from context."""
    # Arrange
    mock_class, manager = mock_fed_config_mgr
    federation_name = "test_federation"
    expected_config = MagicMock(spec=FederationConfig)
    manager.get_federation_names.return_value = [federation_name]
    manager.get_config.return_value = expected_config

    # Act
    result = FederationConfig.add_to_context_and_get(mock_context, federation_name)

    # Assert
    mock_class.add_to_context_and_get.assert_called_once_with(mock_context)
    manager.get_federation_names.assert_called_once()
    manager.get_config.assert_called_once_with(federation_name)
    assert result == expected_config


def test_load_federation_config_not_found(mock_context, mock_fed_config_mgr, capsys):
    """Test that loading a non-existent federation config raises an error."""
    # Arrange
    mock_class, manager = mock_fed_config_mgr
    federation_name = "non_existent_federation"
    manager.get_federation_names.return_value = ["other_federation"]

    # Act & Assert
    with pytest.raises(Abort):
        FederationConfig.add_to_context_and_get(mock_context, federation_name)
    captured = capsys.readouterr()
    assert f"Federation {federation_name} not configured" in captured.out
    assert "Use 'dfm fed config set' to configure a federation." in captured.out


def test_federation_config_patch_with_dict(
    temp_workspace, federation_dir, federation_config_files
):
    """Test patching federation config with a dictionary."""
    config_path, project_path = federation_config_files

    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )

    # Test patching with valid provision info
    patch_dict = {
        "provision": {
            "docker": {
                "image": "patched_image",
                "tag": "2.0.0",
                "dockerfile": {
                    "dir": "deploy/docker",
                    "file": "Dockerfile.patched",
                },
                "build": {
                    "engine": "podman",
                    "arch": "arm64",
                    "context": "./build",
                },
            }
        }
    }
    config.patch(patch_dict)
    assert config.provision_info.docker.image == "patched_image"
    assert config.provision_info.docker.tag == "2.0.0"
    assert config.provision_info.docker.dockerfile.dir == "deploy/docker"
    assert config.provision_info.docker.dockerfile.file == "Dockerfile.patched"
    assert config.provision_info.docker.build.engine == "podman"
    assert config.provision_info.docker.build.arch == "arm64"
    assert config.provision_info.docker.build.context == "./build"

    # Test patching with invalid dict (missing provision)
    with pytest.raises(ValueError, match="Provision info not found in values"):
        config.patch({"invalid": "data"})


def test_federation_config_patch_with_file(
    temp_workspace, federation_dir, federation_config_files
):
    """Test patching federation config with a file path."""
    config_path, project_path = federation_config_files

    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )

    # Create a patch file
    patch_file = federation_dir / "patch.yaml"
    patch_data = {
        "provision": {"docker": {"image": "file_patched_image", "tag": "3.0.0"}}
    }
    with open(patch_file, "w") as f:
        yaml.dump(patch_data, f)

    # Test patching with string path
    config.patch(str(patch_file))
    assert config.provision_info.docker.image == "file_patched_image"
    assert config.provision_info.docker.tag == "3.0.0"

    # Test patching with Path object
    config.patch(patch_file)
    assert config.provision_info.docker.image == "file_patched_image"
    assert config.provision_info.docker.tag == "3.0.0"

    # Test patching with non-existent file
    with pytest.raises(FileNotFoundError):
        config.patch("nonexistent.yaml")


def test_federation_config_patch_with_list(
    temp_workspace, federation_dir, federation_config_files
):
    """Test patching federation config with a list of files."""
    config_path, project_path = federation_config_files

    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )

    # Create multiple patch files
    patch_files = []
    for i in range(2):
        patch_file = federation_dir / f"patch_{i}.yaml"
        patch_data = {
            "provision": {"docker": {"image": f"patched_image_{i}", "tag": f"{i}.0.0"}}
        }
        with open(patch_file, "w") as f:
            yaml.dump(patch_data, f)
        patch_files.append(patch_file)

    # Test patching with list of files
    config.patch(patch_files)
    # Should have the last patch applied
    assert config.provision_info.docker.image == "patched_image_1"
    assert config.provision_info.docker.tag == "1.0.0"

    # Test patching with list containing non-existent file
    with pytest.raises(FileNotFoundError):
        config.patch([patch_files[0], Path("nonexistent.yaml")])


def test_federation_config_patch_invalid_yaml(
    temp_workspace, federation_dir, federation_config_files
):
    """Test patching federation config with invalid YAML content."""
    config_path, project_path = federation_config_files

    config = FederationConfig(debug=True)
    config.initialize(
        name="test_fed",
        workspace_dir=temp_workspace,
        federation_dir=federation_dir,
        config_path=config_path,
        project_path=project_path,
    )

    # Create a file with invalid YAML
    patch_file = federation_dir / "invalid.yaml"
    with open(patch_file, "w") as f:
        f.write("invalid: yaml: content: {")

    # Test patching with invalid YAML
    with pytest.raises(yaml.YAMLError):
        config.patch(patch_file)
