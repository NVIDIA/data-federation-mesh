#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
Version management utilities for DFM CLI.

This module provides functionality for managing version numbers across the DFM project.
It includes utilities for:
- Reading version numbers from various sources (modules, uv)
- Updating version numbers in different file types (Python, YAML, JSON)
- Version bumping with semantic versioning support using uv
- Version validation and conflict checking
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
from enum import Enum, auto
from importlib import import_module
from pathlib import Path
from typing import Literal

import git
from packaging.version import Version
from pydantic import BaseModel, Field, ValidationError
from ruamel.yaml import YAML
from typing_extensions import override


class Updater:
    """Base class for version updaters.

    This abstract class defines the interface for updating version numbers
    in different types of files.
    """

    def update(
        self,
        file_path: str,
        new_version: str,
        key: str | None = None,
        dry_run: bool | None = False,
    ) -> None:
        """Update version number in a file.

        Args:
            file_path: Path to the file to update.
            new_version: New version number to set.
            key: Optional key for structured files (YAML, JSON).
            dry_run: If True, don't actually write changes.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class ModuleUpdater(Updater):
    """Updater for Python module files.

    Updates version numbers in Python module files by modifying the __version__ variable.
    """

    @override
    def update(
        self,
        file_path: str,
        new_version: str,
        key: str | None = None,
        dry_run: bool | None = False,
    ) -> None:
        """Update version number in a Python module.

        Args:
            module_path: Path to the Python module.
            new_version: New version number to set.
            key: Not used for module updates.
            dry_run: If True, don't actually write changes.

        Raises:
            ModuleNotFoundError: If the module cannot be found.
        """
        # We get module path, so we need to create a file path from that
        spec = importlib.util.find_spec(file_path)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(f"Module {file_path} not found")
        file_path = spec.origin

        with open(file_path, "r") as file:
            content = file.read()
        content = re.sub(
            r'(__version__\s*=\s*[\'"])([^\'"]+)([\'"])',
            r"\g<1>" + new_version + r"\3",
            content,
        )
        if dry_run:
            return
        with open(file_path, "w") as file:
            file.write(content)


class YamlUpdater(Updater):
    """Updater for YAML files.

    Updates version numbers in YAML files using a dot-notation key path.
    """

    @override
    def update(
        self,
        file_path: str,
        new_version: str,
        key: str | None = None,
        dry_run: bool | None = False,
    ) -> None:
        """Update version number in a YAML file.

        Args:
            file_path: Path to the YAML file.
            new_version: New version number to set.
            key: Dot-notation path to the version field.
            dry_run: If True, don't actually write changes.

        Raises:
            ValueError: If key is not provided.
        """
        if key is None:
            raise ValueError("Key must be provided for YAML files")
        with open(file_path, "r") as file:
            yaml = YAML()
            content = yaml.load(file)
        keys = key.split(".")
        target = content
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = new_version
        if dry_run:
            return
        with open(file_path, "w") as file:
            yaml.dump(content, file)


class JsonUpdater(Updater):
    """Updater for JSON files.

    Updates version numbers in JSON files using a dot-notation key path.
    """

    @override
    def update(
        self,
        file_path: str,
        new_version: str,
        key: str | None = None,
        dry_run: bool | None = False,
    ) -> None:
        """Update version number in a JSON file.

        Args:
            file_path: Path to the JSON file.
            new_version: New version number to set.
            key: Dot-notation path to the version field.
            dry_run: If True, don't actually write changes.

        Raises:
            ValueError: If key is not provided.
        """
        if key is None:
            raise ValueError("Key must be provided for JSON files")
        with open(file_path, "r") as file:
            content = json.load(file)
        keys = key.split(".")
        target = content
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = new_version
        if dry_run:
            return
        with open(file_path, "w") as file:
            json.dump(content, file, indent=4)


class UvUpdater(Updater):
    """Updater for uv project files.

    Updates version numbers using uv's version command.
    """

    @override
    def update(
        self,
        file_path: str | None,
        new_version: str,
        key: str | None = None,
        dry_run: bool | None = False,
    ) -> None:
        """Update version number using uv.

        Args:
            file_path: Not used for uv updates.
            new_version: New version number to set.
            key: Not used for uv updates.
            dry_run: If True, don't actually write changes.
        """
        cmd = ["uv", "version", new_version]
        if dry_run:
            cmd.append("--dry-run")
        subprocess.run(cmd, shell=(os.name == "nt"))


# Pydantic models for configuration validation
class VersionSource(BaseModel):
    """Configuration for version source.

    Attributes:
        type: Type of version source ("module" or "uv").
        path: Path to the version source (module path for "module" type).
    """

    type: Literal["module", "uv"]
    path: str  # For module this is something like "dfm.service.execute"


class FileToUpdate(BaseModel):
    """Configuration for a file to update.

    Attributes:
        type: Type of file ("init", "yaml", "json", "toml").
        file: Path to the file.
        key: Optional key for structured files.
    """

    type: Literal["init", "yaml", "json", "toml"]
    file: str
    key: str | None = None


class Config(BaseModel):
    """Configuration for version management.

    Attributes:
        version_source: Source of the version number.
        files_to_update: List of files to update with the new version.
    """

    version_source: VersionSource
    files_to_update: list[FileToUpdate] = Field(default_factory=list)


class BumpType(Enum):
    """Types of version bumps supported.

    Attributes:
        MAJOR: Major version bump (1.0.0 -> 2.0.0)
        MINOR: Minor version bump (1.0.0 -> 1.1.0)
        PATCH: Patch version bump (1.0.0 -> 1.0.1)
        LITERAL: Set to a specific version
    """

    MAJOR = auto()
    MINOR = auto()
    PATCH = auto()
    LITERAL = auto()

    @classmethod
    def from_name(cls: type["BumpType"], name: str) -> "BumpType":
        """Get bump type from a name.

        Args:
            name: Name of the bump type or a version number.

        Returns:
            The corresponding BumpType.

        Raises:
            KeyError: If the name is not a valid bump type.
        """
        if name[0].isdigit():
            return BumpType.LITERAL
        return cls[name.upper()]


def get_version_from_module(module_path: str) -> str:
    """Get version number from a Python module.

    Args:
        module_path: Path to the module.

    Returns:
        The version number.

    Raises:
        RuntimeError: If the module cannot be imported.
    """
    try:
        m = import_module(module_path)
    except ModuleNotFoundError as e:
        print(f"Failed to import {module_path}: {e}")
        raise RuntimeError from e
    return m.__version__


def get_version_from_uv() -> str:
    """Get version number from uv.

    Returns:
        The version number from uv.
    """
    rr = subprocess.run(
        ["uv", "version", "--short"], capture_output=True, shell=(os.name == "nt")
    )
    return str(rr.stdout.decode(encoding="utf-8")).rstrip()


def increment_version(initial_version: str, type: BumpType = BumpType.PATCH) -> str:
    """Increment a version number according to semantic versioning.

    Args:
        version: Current version number.
        type: Type of version bump to perform.

    Returns:
        The incremented version number.

    Raises:
        ValueError: If the bump type is invalid.
    """
    version = Version(
        initial_version
    )  # Convert to a convenient object for easier parsing
    major, minor, patch = version.release

    if type == BumpType.MAJOR:
        major += 1
        minor = 0
        patch = 0
    elif type == BumpType.MINOR:
        minor += 1
        patch = 0
    elif type == BumpType.PATCH:
        patch += 1
    else:
        raise ValueError(f"Invalid bump type {type.name}")

    return f"{major}.{minor}.{patch}"


def get_updater(update_type: str) -> Updater:
    """Get an updater instance for the specified file type.

    Args:
        update_type: Type of file to update.

    Returns:
        An appropriate updater instance.

    Raises:
        ValueError: If the update type is not supported.
    """
    if update_type == "init":
        return ModuleUpdater()
    elif update_type == "yaml":
        return YamlUpdater()
    elif update_type == "json":
        return JsonUpdater()
    else:
        raise ValueError(f"Unsupported update type: {update_type}")


def ok_to_update(new_version: str) -> bool:
    """Check if it's safe to update to the new version.

    Args:
        new_version: The new version number to check.

    Returns:
        True if the version can be updated, False if a tag already exists.
    """
    # Open the repository
    repo = git.Repo(os.getcwd())

    # Get all tags
    tags = repo.tags

    # Check if tag corresponding to our new version exists
    new_tag = f"nv-dfm-v{new_version}"
    return new_tag not in tags


def get():
    """Get the current version number from uv.

    Returns:
        The current version number.
    """
    return get_version_from_uv()


def bump(
    type: BumpType = BumpType.PATCH,
    new_version: str = "",
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Bump the version number according to the specified type.

    Args:
        type: Type of version bump to perform.
        new_version: Specific version to set (for LITERAL type).
        force: If True, allow updating to a version that already has a tag.
        dry_run: If True, don't actually write changes.

    Note:
        This function reads configuration from bump.yaml and updates all
        specified files with the new version number.
    """
    config_path = Path(__file__).parent.joinpath("bump.yaml")
    with open(config_path, "r") as file:
        yaml = YAML(typ="safe")
        config_dict = yaml.load(file)

    try:
        config = Config(**config_dict)
    except ValidationError as e:
        print("Configuration validation error:", e)
        return

    if config.version_source.type == "module":
        version_source_module = config.version_source.path
        initial_version = get_version_from_module(version_source_module)
        print(
            f"Initial version read from {version_source_module} is: {initial_version}"
        )
    elif config.version_source.type == "uv":
        initial_version = get_version_from_uv()
        print(f"Initial version read from uv is: {initial_version}")
    else:
        raise ValueError(
            f"Unsupported version source type: {config.version_source.type}"
        )
    if type != BumpType.LITERAL:
        new_version = increment_version(initial_version, type)
    print(f"Setting version to {new_version}")

    if ok_to_update(new_version) or force:
        # Update all configured files
        for file_info in config.files_to_update:
            updater = get_updater(file_info.type)
            updater.update(file_info.file, new_version, file_info.key, dry_run)

        # Update the source of truth file
        UvUpdater().update(None, new_version, dry_run=dry_run)
        print(f"Version updated to {new_version}")
    else:
        print(
            f"Tag for {new_version} already exists. If you are sure you want to use that version - use --force."
        )


def main() -> None:
    """Main entry point for the version management script.

    Parses command line arguments and performs the requested version bump.
    """
    parser = argparse.ArgumentParser(description="Increment version number.")
    parser.add_argument(
        "bump_type",
        type=str,
        nargs="?",
        default="patch",
        help="Type of version bump: major, minor, patch or a specific version number. Default is patch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry run - don't write changes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force using a version for which tag already exists.",
    )
    new_version = ""
    args = parser.parse_args()
    bump_type = BumpType.from_name(args.bump_type)
    if bump_type == BumpType.LITERAL:
        new_version = str(args.bump_type)
    print(f"Bump type: {bump_type.name.lower()}")

    if args.dry_run:
        print("This is a dry run! Don't expect any changes on disk.")

    bump(
        type=bump_type, new_version=new_version, force=args.force, dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
