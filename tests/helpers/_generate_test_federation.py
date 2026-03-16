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
import shutil
import sys
import tempfile
from typing import Any
import yaml

from nv_dfm_core.gen.apigen import ApiGen


class GenerateTestFederation:
    """
    Context manager for generating and managing test federation environments.

    This class provides a convenient way to generate a federation from
    configuration, temporarily add it to the Python path, and clean up
    afterwards. It can be used as a context manager to ensure proper setup
    and teardown of the test environment.

    Attributes:
        config_path (str, optional): Path to a YAML configuration file.
        config_dict (dict, optional): Configuration dictionary (alternative
            to config_path).
        output_path (str, optional): Directory where the federation will be
            generated. If None, a temporary directory will be created.
        delete_output_at_exit (bool): Whether to delete the output directory
            on exit. Defaults to True.
        restore_globals_and_modules (bool): If True, the globals and modules
            will be restored when exiting the context manager. Defaults to True.
        modules_snapshot (dict): Snapshot of sys.modules for cleanup.
        globals_snapshot (dict): Snapshot of globals() for cleanup.

    Example:
        ```python
        # Using with config file
        with GenerateTestFederation(
            config_path="config.yaml",
            output_path="/tmp/test_federation",
            delete_output_at_exit=True
        ) as federation:
            # Federation is now available in sys.path
            import generated_module
        # Federation is automatically cleaned up

        # Using with config dict
        config = {"federation": {"name": "test"}}
        with GenerateTestFederation(config_dict=config) as federation:
            # Use the generated federation
            pass
        ```
    """

    def __init__(
        self,
        config_path: str | None = None,
        config_dict: dict[str, Any] | None = None,
        output_path: str | None = None,
        delete_output_at_exit: bool = True,
        restore_globals_and_modules: bool = True,
    ):
        """
        Initialize the GenerateTestFederation context manager.

        Args:
            config_path (str, optional): Path to YAML configuration file.
                Cannot be used together with config_dict.
            config_dict (dict, optional): Configuration dictionary.
                Cannot be used together with config_path.
            output_path (str, optional): Directory where federation will be
                generated. If None, a temporary directory will be created.
            delete_output_at_exit (bool): If True, the output directory will be
                deleted when exiting the context manager. Defaults to True.
            restore_globals_and_modules (bool): If True, the globals and modules
                will be restored when exiting the context manager. Defaults to True.

        Raises:
            ValueError: If neither config_path nor config_dict is provided,
                or if both are provided.
        """
        self.config_path = config_path
        self.config_dict = config_dict
        self.output_path = output_path
        self.delete_output_at_exit = delete_output_at_exit
        self.b_restore_globals_and_modules = restore_globals_and_modules
        self.modules_snapshot = None
        self.globals_snapshot = None

    def __enter__(self):
        """
        Enter the context manager by generating the federation and adding it
        to sys.path.

        This method:
        1. Generates the federation from configuration
        2. Adds the output path to sys.path
        3. Backs up the current state of modules and globals

        Returns:
            GenerateTestFederation: The instance itself for use in 'with'
                statements.
        """
        self.generate_federation()
        assert self.output_path is not None
        self.add_to_sys_path()
        self.backup_globals_and_modules()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager by cleaning up the federation environment.

        This method:
        1. Optionally deletes the output directory if delete_output_at_exit is True
        2. Removes the output path from sys.path
        3. Restores the previous state of modules and globals

        Args:
            exc_type: Exception type (if an exception occurred)
            exc_value: Exception value (if an exception occurred)
            traceback: Exception traceback (if an exception occurred)
        """
        assert self.output_path is not None
        if self.delete_output_at_exit:
            shutil.rmtree(self.output_path)
        self.remove_from_sys_path()
        if self.b_restore_globals_and_modules:
            self.restore_globals_and_modules()

    def generate_federation(self):
        """
        Generate the federation from the provided configuration.

        This method:
        1. Creates a temporary directory if output_path is None
        2. Reads configuration from either config_path or config_dict
        3. Creates an ApiGen instance
        4. Generates both the API and runtime components of the federation

        Raises:
            ValueError: If no input dictionary is provided (neither config_path
                nor config_dict was specified).
            FileNotFoundError: If config_path is specified but the file
                doesn't exist.
        """
        if self.output_path is None:
            self.output_path = tempfile.mkdtemp()
        input_dict = None
        if self.config_path:
            with open(self.config_path, "r") as f:
                input_dict = yaml.safe_load(f)
        elif self.config_dict:
            input_dict = self.config_dict
        if input_dict:
            api_gen = ApiGen.from_dict(input_dict)
            api_gen.generate_api(language="python", outpath=Path(self.output_path))
            api_gen.generate_runtime(language="python", outpath=Path(self.output_path))
        else:
            raise ValueError("No input dictionary provided")

    def add_to_sys_path(self):
        """
        Add the output path to Python's sys.path.

        This method appends the output path to sys.path, allowing direct
        importing of the generated federation modules. The path is added to
        the end of sys.path to ensure proper module resolution.
        """
        assert self.output_path is not None
        sys.path.append(self.output_path)

    def remove_from_sys_path(self):
        """
        Remove the output path from Python's sys.path.

        This method removes the output path from sys.path that was added by
        add_to_sys_path(). It should be called during cleanup to avoid
        polluting the Python path.

        Raises:
            ValueError: If the output_path is not found in sys.path.
        """
        assert self.output_path is not None
        sys.path.remove(self.output_path)

    def backup_globals_and_modules(self):
        """
        Create snapshots of the current sys.modules and globals() state.

        This method is called during context entry to preserve the state
        of the Python environment before federation generation. The snapshots
        are used later to restore the environment during cleanup.
        """
        self.modules_snapshot = sys.modules.copy()
        self.globals_snapshot = globals().copy()

    def restore_globals_and_modules(self):
        """
        Restore the previous state of sys.modules and globals().

        This method:
        1. Removes any newly added modules that originate from the generated
           output path to avoid polluting the environment across tests
        2. Removes any new globals that were added
        3. Restores the previous values of existing globals

        Note: The globals restoration may not cover all edge cases due to
        Python's scoping rules and module system.
        """
        # Restore modules (safely):
        # Instead of clearing sys.modules (which can break stdlib internals
        # like multiprocessing's registered reducers), we remove only modules
        # that were introduced during the context and that come from the
        # generated federation's output path.
        snapshot_names = (
            set(self.modules_snapshot.keys()) if self.modules_snapshot else set()
        )
        current_names = list(sys.modules.keys())

        for name in current_names:
            if name in snapshot_names:
                continue
            mod = sys.modules.get(name)
            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                continue
            # Remove modules whose files live under the generated output path
            # (these were created by ApiGen and should be cleaned up)
            try:
                if str(mod_file).startswith(str(self.output_path)):
                    del sys.modules[name]
            except Exception:
                # Best-effort cleanup; ignore issues for robustness
                pass

        # Restore globals (careful: may not cover all edge cases)
        current_globals = set(globals())
        globals_snapshot_names = (
            set(self.globals_snapshot) if self.globals_snapshot else set()
        )
        for name in current_globals - globals_snapshot_names:
            del globals()[name]
        if self.globals_snapshot:
            for name, value in self.globals_snapshot.items():
                globals()[name] = value
