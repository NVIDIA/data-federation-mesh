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
import os


def get_repo_root() -> str:
    # Prefer kernel CWD (often set by VS Code to the notebook folder)
    nb_dir = Path.cwd()

    # Optional: override if VS Code exposes a working dir env (org-specific setups)
    if not nb_dir.exists():
        nb_dir = Path(os.environ.get("VSCODE_CWD"))

    repo_root = next(
        (p for p in [nb_dir, *nb_dir.parents] if (p / ".git").exists()), nb_dir
    )
    return repo_root.resolve()


def change_directory_to_repo_root():
    repo_root = get_repo_root()
    os.chdir(repo_root)
