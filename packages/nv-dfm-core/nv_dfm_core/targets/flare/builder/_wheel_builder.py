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
import shutil
import subprocess
from pathlib import Path

from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.spec import Builder


class WheelBuilder(Builder):
    DFM_INCLUDE = [
        "dfm/api",
        "dfm/session",
        "dfm/__init__.py",
    ]

    def __init__(self, py_project: str | None = None, include: list[str] | None = None):
        """Build the wheel file for the project.
        Creates the wheel file for the project.
        """
        super().__init__()
        self._py_project = py_project
        self._include = include or []
        self._my_dir = Path(__file__).absolute().parent
        self._dfm_dir = self._my_dir.parent.parent.parent
        # There is no way to know inside the builder what is the project root.
        # So we need to pass it via environment variable.
        # Assume current working directory is the project root by default.
        self._project_root = Path(
            os.environ.get("DFM_WHEEL_BUILDER_PROJECT_ROOT", os.getcwd())
        )

    def initialize(self, ctx: dict):
        if not self._py_project:
            raise RuntimeError("py_project is required")
        self._py_project = Path(self._py_project).absolute()
        if not self._py_project.exists():
            raise RuntimeError(f"Project file {self._py_project} does not exist")
        print(f"Using Python project file: {self._py_project}")
        if not self._dfm_dir.exists():
            raise RuntimeError(f"DFM directory {self._dfm_dir} does not exist")
        print(f"Using DFM directory: {self._dfm_dir}")

    def _copy_includes(self, base_dir: Path, dst_dir: Path, includes: list[str]):
        # Copy the include files/directories
        for include in includes:
            src_path = base_dir / include
            if src_path.exists():
                dst_path = dst_dir / include
                print(f"Adding: {src_path}")
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy(src_path, dst_path)
            else:
                raise RuntimeError(f"Include file {src_path} does not exist")

    def _prepare_participant(
        self, participant: Participant, ctx: dict
    ) -> tuple[Path, Path] | None:
        print(f"Checking participant: {participant.name}")
        if participant.type in ["client", "server"]:
            return None
        ws_dir = Path(self.get_ws_dir(participant, ctx))
        print(f"Preparing build files for participant: {participant.name} in {ws_dir}")
        if not ws_dir.exists():
            raise RuntimeError(f"Workspace directory {ws_dir} does not exist")
        dst_dir = ws_dir / "wheel"
        build_dir = dst_dir / "build"
        dst_dir.mkdir(parents=True, exist_ok=True)
        # Copy DFM files
        self._copy_includes(self._dfm_dir, build_dir, self.DFM_INCLUDE)
        # Copy project-specific files and pyproject.toml
        self._copy_includes(
            self._project_root, build_dir, self._include + ["pyproject.toml"]
        )
        return build_dir, dst_dir

    def _build_participant(self, build_dir: Path, dst_dir: Path, ctx: dict):
        # Call setuptools to build the wheel file, using the current script as the build script
        cwd = os.getcwd()
        os.chdir(build_dir)
        run_args = ["python3", "-m", "build", "--outdir", build_dir / "dist"]
        try:
            subprocess.run(run_args, env=os.environ)
        except FileNotFoundError:
            raise RuntimeError("Unable to build wheel file. Is Python 3 installed?")
        finally:
            os.chdir(cwd)
        wheel_file = next(build_dir.glob("dist/*.whl"))  # Find the first .whl file
        shutil.copy(wheel_file, dst_dir)
        tar_file = next(build_dir.glob("dist/*.tar.gz"))  # Find the first .tar.gz file
        shutil.copy(tar_file, dst_dir)

    def _cleanup(self, build_dir: Path):
        shutil.rmtree(build_dir)

    def build(self, project: Project, ctx: dict):
        """Create a wheel file for the project.
        Args:
            project (Project): project instance
            ctx (dict): the provision context
        """
        print("DFM Wheel Builder starting...")
        for p in project.participants:
            paths = self._prepare_participant(p, ctx)
            if paths:
                build_dir, dst_dir = paths
                self._build_participant(build_dir, dst_dir, ctx)
                self._cleanup(build_dir)
        print("DFM Wheel Builder finished.")
