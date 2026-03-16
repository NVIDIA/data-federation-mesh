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
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict
from typing_extensions import override

from ._block import Block
from ._pipeline_build_helper import PipelineBuildHelper

if TYPE_CHECKING:
    from ._api_visitor import ApiVisitor
else:
    ApiVisitor = Any


class Pipeline(Block):
    """Represents a DFM pipeline containing a sequence of operations and control flow.

    A Pipeline is the top-level construct that defines the workflow to be executed
    across a federation. It contains statements, operations, and control structures.
    """

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True)

    api_version: str = ""  # filled by Session
    mode: Literal["execute", "discovery"] = "execute"
    name: str | None = None  # can be used to name the pipeline

    @override
    def __enter__(self):  # pyright: ignore[reportMissingSuperCall]
        """Overrides Block/s enter/exit methods"""
        PipelineBuildHelper.enter_pipeline(self)
        return self

    @override
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:  # pyright: ignore[reportMissingSuperCall]
        PipelineBuildHelper.exit_pipeline(self)
        if exc_type is not None:
            return False
        return True

    @override
    def accept(self, visitor: ApiVisitor) -> None:
        visitor.visit_pipeline(self)

    @classmethod
    def load_from_file(cls, file: str | Path, **fsspec_kwargs: Any) -> "Pipeline":
        """
        Load a stored pipeline from a .json file.

        The file argument can be:
          - a local filename (str or Path)
          - a remote URI (e.g., s3://bucket/file.json, gs://bucket/file.json, etc.)
          - any fsspec-compatible URL

        If fsspec is installed, it will be used for all paths (including local).
        If fsspec is not installed, only local files are supported.

        Additional keyword arguments are passed to fsspec.filesystem().
        """
        try:
            import fsspec  # pyright: ignore[reportMissingTypeStubs]
        except ImportError:
            fsspec = None

        file_url = str(file)

        if fsspec is not None:
            fs, path_in_fs = fsspec.core.url_to_fs(file_url, **fsspec_kwargs)  # pyright: ignore[reportUnknownMemberType]
            if not fs.exists(path_in_fs):
                raise FileNotFoundError(f"Pipeline file not found: {file}")
            with fs.open(path_in_fs, "r") as f:
                pipeline = Pipeline.model_validate_json(f.read())
        else:
            file = Path(file)
            if not file.exists():
                raise FileNotFoundError(f"Pipeline file not found: {file}")
            with open(file, "r") as f:
                pipeline = Pipeline.model_validate_json(f.read())
        return pipeline

    @classmethod
    def save_to_file(
        cls, pipeline: "Pipeline", file: str | Path, **fsspec_kwargs: Any
    ) -> None:
        """
        Store a pipeline to a .json file.

        The file argument can be:
          - a local filename (str or Path)
          - a remote URI (e.g., s3://bucket/file.json, gs://bucket/file.json, etc.)
          - any fsspec-compatible URL

        If fsspec is installed, it will be used for all paths (including local).
        If fsspec is not installed, only local files are supported.

        Additional keyword arguments are passed to fsspec.filesystem().
        """
        try:
            import fsspec  # pyright: ignore[reportMissingTypeStubs]
        except ImportError:
            fsspec = None

        file_url = str(file)

        if fsspec is not None:
            fs, path_in_fs = fsspec.core.url_to_fs(file_url, **fsspec_kwargs)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            # Try to ensure parent directory exists (if supported)
            try:
                parent = fs._parent(path_in_fs)
                if parent and not fs.exists(parent):
                    fs.mkdirs(parent, exist_ok=True)
            except Exception:
                pass  # Some filesystems may not support mkdirs or _parent
            with fs.open(path_in_fs, "w") as f:
                _ = f.write(pipeline.model_dump_json(indent=4))
        else:
            # Fallback: only support local files
            file = Path(file)
            file.parent.mkdir(parents=True, exist_ok=True)
            with open(file, "w") as f:
                _ = f.write(pipeline.model_dump_json(indent=4))
