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

import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import pytest

from nv_dfm_core.api import Pipeline, PipelineBuildHelper


def test_finds_api_metadata():
    import examplefed.fed.runtime.reception

    assert examplefed.fed.runtime.reception.API_VERSION == "0.0.1"

    assert "atlantis" not in sys.modules


def test_Param_type_annotations():
    import examplefed.fed.site.reception.users as reception_users  # type: ignore

    orig_Operation = reception_users.GreetMe.__bases__[0]
    field_annotations = orig_Operation.__annotations__["name"]
    assert field_annotations == "Union[str, NodeParam, PlaceParam, Advise]"


def test_pipeline_to_json_and_back():
    import examplefed.fed.site.reception.users as reception_users  # type: ignore

    with Pipeline() as p:
        assert PipelineBuildHelper.build_helper_active()
        reception_users.GreetMe(name="John")

    json = p.model_dump_json(exclude_defaults=False)

    # now pretend to be server-side
    pafter = Pipeline.model_validate_json(json)
    assert p == pafter
    assert type(pafter) is Pipeline


def test_save_to_file_local():
    """Test saving a pipeline to a local file"""
    with Pipeline() as p:
        # Create a simple pipeline
        pass

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Save pipeline to file
        Pipeline.save_to_file(p, tmp_path)

        # Verify file was created and contains valid JSON
        assert Path(tmp_path).exists()
        with open(tmp_path, "r") as f:
            content = f.read()
            loaded_data = json.loads(content)
            assert "api_version" in loaded_data
            assert "mode" in loaded_data
    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def test_load_from_file_local():
    """Test loading a pipeline from a local file"""
    # Create a simple pipeline
    with Pipeline() as p:
        pass

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(p.model_dump_json(indent=4))

    try:
        # Load pipeline from file
        loaded_pipeline = Pipeline.load_from_file(tmp_path)

        # Verify loaded pipeline matches original
        assert loaded_pipeline == p
        assert isinstance(loaded_pipeline, Pipeline)
    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def test_save_and_load_roundtrip():
    """Test saving and loading a pipeline maintains equality"""
    with Pipeline() as p:
        # Create a pipeline with some content
        pass

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Save pipeline
        Pipeline.save_to_file(p, tmp_path)

        # Load pipeline back
        loaded_pipeline = Pipeline.load_from_file(tmp_path)

        # Verify roundtrip preserves equality
        assert loaded_pipeline == p
        assert loaded_pipeline.model_dump_json() == p.model_dump_json()
    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def test_load_from_file_not_found():
    """Test loading from non-existent file raises FileNotFoundError"""
    non_existent_path = "/path/that/does/not/exist/pipeline.json"

    with pytest.raises(FileNotFoundError, match="Pipeline file not found"):
        Pipeline.load_from_file(non_existent_path)


def test_save_to_file_creates_parent_directories():
    """Test that save_to_file creates parent directories if they don't exist"""
    with Pipeline() as p:
        pass

    # Create a path with non-existent parent directories
    temp_dir = tempfile.mkdtemp()
    nested_path = Path(temp_dir) / "nested" / "deep" / "pipeline.json"

    try:
        # Save pipeline to nested path
        Pipeline.save_to_file(p, nested_path)

        # Verify file was created
        assert nested_path.exists()

        # Verify parent directories were created
        assert nested_path.parent.exists()
        assert (nested_path.parent / "..").resolve().exists()
    finally:
        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def _mock_url_to_fs_no_s3(url, *args, **kwargs):
    """Mock fsspec.core.url_to_fs that raises ImportError for S3 URLs."""
    if url.startswith("s3://"):
        raise ImportError("No module named 's3fs' (mocked)")
    # For other URLs, use the real implementation
    import fsspec

    return fsspec.core.url_to_fs(url, *args, **kwargs)


def test_save_to_file_with_fsspec():
    """Test saving to file with fsspec when S3 backend is unavailable"""
    try:
        import fsspec
    except ImportError:
        pytest.skip("fsspec not available")

    with Pipeline() as p:
        pass

    # Mock fsspec.core.url_to_fs to simulate S3 backend not being available
    with patch("fsspec.core.url_to_fs", side_effect=_mock_url_to_fs_no_s3):
        with pytest.raises((FileNotFoundError, OSError, ImportError)):
            Pipeline.save_to_file(
                p,
                "s3://tests/assets/outputs/dfm_api_test_pipeline/bucket/pipeline.json",
            )


def test_load_from_file_with_fsspec():
    """Test loading from file with fsspec when S3 backend is unavailable"""
    try:
        import fsspec
    except ImportError:
        pytest.skip("fsspec not available")

    # Mock fsspec.core.url_to_fs to simulate S3 backend not being available
    with patch("fsspec.core.url_to_fs", side_effect=_mock_url_to_fs_no_s3):
        with pytest.raises((FileNotFoundError, OSError, ImportError)):
            Pipeline.load_from_file("s3://bucket/pipeline.json")


def test_load_from_file_fsspec_not_found():
    """Test loading from non-existent remote file when S3 backend is unavailable"""
    try:
        import fsspec
    except ImportError:
        pytest.skip("fsspec not available")

    # Mock fsspec.core.url_to_fs to simulate S3 backend not being available
    with patch("fsspec.core.url_to_fs", side_effect=_mock_url_to_fs_no_s3):
        with pytest.raises((FileNotFoundError, OSError, ImportError)):
            Pipeline.load_from_file("s3://bucket/nonexistent/pipeline.json")


def test_save_to_file_without_fsspec():
    """Test saving to file when fsspec is not available (local only)"""
    with Pipeline() as p:
        pass

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Save pipeline to local file (should work without fsspec)
        Pipeline.save_to_file(p, tmp_path)

        # Verify file was created
        assert Path(tmp_path).exists()
    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def test_load_from_file_without_fsspec():
    """Test loading from file when fsspec is not available (local only)"""
    with Pipeline() as p:
        pass

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(p.model_dump_json(indent=4))

    try:
        # Load pipeline from local file (should work without fsspec)
        loaded_pipeline = Pipeline.load_from_file(tmp_path)

        # Verify pipeline was loaded correctly
        assert loaded_pipeline == p
    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def test_save_to_file_without_fsspec_remote_uri():
    """Test that remote URIs fail when fsspec is not available"""
    # This test requires fsspec to NOT be available
    try:
        import fsspec

        # If fsspec is available, we can't test this scenario
        pytest.skip("fsspec is available, cannot test without fsspec scenario")
    except ImportError:
        # fsspec is not available, test that remote URIs fail
        with Pipeline() as p:
            pass

        with pytest.raises(FileNotFoundError):
            # This should fail because fsspec is not available for remote URIs
            Pipeline.save_to_file(p, "s3://bucket/pipeline.json")
