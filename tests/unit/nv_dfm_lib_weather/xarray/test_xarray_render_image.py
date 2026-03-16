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

import json
import pytest
import xarray
import numpy as np
from unittest.mock import Mock
from pathlib import Path

from nv_dfm_lib_weather.xarray import RenderUint8ToImages
from nv_dfm_lib_common.schemas import TextureFile, TextureFileList
from nv_dfm_lib_weather.xarray._render_uint8_to_images import _sanitize_for_json


def create_test_dataset(num_timesteps=3, has_time_attr=True):
    """Create a test xarray dataset with proper structure."""
    time_values = np.array(
        [f"2024-01-01T{i:02d}:00:00" for i in range(num_timesteps)], dtype="datetime64"
    )

    # Create coordinate arrays with proper ranges
    lat_values = np.linspace(-90, 90, 10)
    lon_values = np.linspace(-180, 180, 10)

    data = xarray.Dataset(
        data_vars={
            "t2m": xarray.DataArray(
                # Schema expects dims in order: [time, lon, lat] (time_dimension + xydims)
                np.random.randint(50, 200, (num_timesteps, 10, 10), dtype=np.uint8),
                dims=["time", "lon", "lat"],  # Fixed order to match schema expectation
                attrs={"long_name": "2m temperature", "units": "K"},
            ),
        },
        coords={
            "time": xarray.DataArray(
                time_values,
                dims=["time"],
            ),
            "lat": xarray.DataArray(
                lat_values,
                dims=["lat"],
            ),
            "lon": xarray.DataArray(
                lon_values,
                dims=["lon"],
            ),
        },
    )

    # Add required attributes for schema validation
    data.attrs["data_min"] = np.array([0])
    data.attrs["data_max"] = np.array([255])

    return data


def create_mock_site():
    """Create a mock Site object with all necessary methods."""
    mock_site = Mock()
    mock_dfm_context = Mock()
    mock_logger = Mock()

    mock_dfm_context.logger = mock_logger
    mock_site.dfm_context = mock_dfm_context

    # Mock cache storage and path operations
    mock_path = Mock(spec=Path)
    mock_path.joinpath.return_value = mock_path
    mock_path.mkdir = Mock()
    mock_path.write_text = Mock()
    mock_path.write_bytes = Mock()
    mock_path.as_posix.return_value = "/mock/path/file.ext"

    mock_site.cache_storage.return_value = mock_path

    return mock_site, mock_logger, mock_path


@pytest.mark.asyncio
async def test_render_uint8_to_images_no_site():
    """Test rendering with no site (no file saving)."""
    data = create_test_dataset()
    adapter = RenderUint8ToImages(site=None, provider=None)

    outputs = await adapter.body(
        data=data,
        variable="t2m",
        xydims=["lon", "lat"],
        time_dimension="time",
        additional_meta_data={"test": "value"},
        return_meta_data=True,
        return_image_data=True,
    )

    assert outputs is not None
    assert isinstance(outputs, TextureFileList)
    assert len(outputs.texture_files) == 3  # 3 timesteps

    for texture_file in outputs.texture_files:
        assert isinstance(texture_file, TextureFile)
        assert texture_file.format == "PNG"
        assert texture_file.timestamp is not None
        assert texture_file.metadata is not None
        assert texture_file.metadata["test"] == "value"
        assert texture_file.base64_image_data is not None
        assert texture_file.url is None  # No saving when site is None
        assert texture_file.metadata_url is None


@pytest.mark.asyncio
async def test_render_uint8_to_images_with_site():
    """Test rendering with site (with file saving)."""
    data = create_test_dataset()
    mock_site, mock_logger, mock_path = create_mock_site()

    adapter = RenderUint8ToImages(site=mock_site, provider=None)

    outputs = await adapter.body(
        data=data,
        variable="t2m",
        xydims=["lon", "lat"],
        time_dimension="time",
        additional_meta_data={"project": "nv_dfm_lib_all"},
        return_meta_data=True,
        return_image_data=False,
    )

    assert outputs is not None
    assert isinstance(outputs, TextureFileList)
    assert len(outputs.texture_files) == 3

    # Verify site interactions
    mock_site.cache_storage.assert_called_once()
    mock_path.mkdir.assert_called_once_with(parents=True)
    mock_path.write_text.assert_called_once()  # metadata.json
    assert mock_path.write_bytes.call_count == 3  # 3 images

    for texture_file in outputs.texture_files:
        assert texture_file.url is not None
        assert texture_file.metadata_url is not None
        assert texture_file.base64_image_data is None  # return_image_data=False


@pytest.mark.asyncio
async def test_render_uint8_to_images_auto_variable_selection():
    """Test automatic variable selection when only one variable exists."""
    data = create_test_dataset()
    adapter = RenderUint8ToImages(site=None, provider=None)

    outputs = await adapter.body(
        data=data,
        variable="",  # Empty string should auto-select
        xydims=["lon", "lat"],
        time_dimension="time",
    )

    assert len(outputs.texture_files) == 3


@pytest.mark.asyncio
async def test_render_uint8_to_images_error_missing_variable():
    """Test error when specified variable doesn't exist."""
    data = create_test_dataset()
    adapter = RenderUint8ToImages(site=None, provider=None)

    with pytest.raises(
        ValueError, match="Variable nonexistent selected for rendering does not exist"
    ):
        await adapter.body(
            data=data,
            variable="nonexistent",
            xydims=["lon", "lat"],
            time_dimension="time",
        )


@pytest.mark.asyncio
async def test_render_uint8_to_images_error_multiple_variables():
    """Test error when multiple variables exist but none specified."""
    data = create_test_dataset()
    # Add another variable
    data["wind_speed"] = xarray.DataArray(
        np.random.randint(0, 255, (3, 10, 10), dtype=np.uint8),
        dims=["time", "lon", "lat"],  # Fixed order to match schema expectation
    )

    adapter = RenderUint8ToImages(site=None, provider=None)

    with pytest.raises(
        ValueError,
        match="Data has multiple variables.*Need an explicit 'variable' parameter",
    ):
        await adapter.body(
            data=data,
            variable="",  # No variable specified
            xydims=["lon", "lat"],
            time_dimension="time",
        )


@pytest.mark.asyncio
async def test_render_uint8_to_images_different_formats():
    """Test different image formats."""
    data = create_test_dataset(num_timesteps=1)
    adapter = RenderUint8ToImages(site=None, provider=None)

    # Test PNG format
    outputs_png = await adapter.body(
        data=data,
        variable="t2m",
        xydims=["lon", "lat"],
        time_dimension="time",
        format="png",
    )
    assert outputs_png.texture_files[0].format == "PNG"

    # Test JPEG format
    outputs_jpeg = await adapter.body(
        data=data,
        variable="t2m",
        xydims=["lon", "lat"],
        time_dimension="time",
        format="jpeg",
    )
    assert outputs_jpeg.texture_files[0].format == "JPEG"


@pytest.mark.asyncio
async def test_render_uint8_to_images_quality_parameter():
    """Test different quality settings."""
    data = create_test_dataset(num_timesteps=1)
    adapter = RenderUint8ToImages(site=None, provider=None)

    # Test with different quality settings
    for quality in [50, 90, 99]:
        outputs = await adapter.body(
            data=data,
            variable="t2m",
            xydims=["lon", "lat"],
            time_dimension="time",
            quality=quality,
            format="jpeg",
        )
        assert len(outputs.texture_files) == 1
        assert outputs.texture_files[0].base64_image_data is not None


@pytest.mark.asyncio
async def test_render_uint8_to_images_return_flags():
    """Test different combinations of return flags."""
    data = create_test_dataset(num_timesteps=1)
    adapter = RenderUint8ToImages(site=None, provider=None)

    # Test return_meta_data=False, return_image_data=False
    outputs = await adapter.body(
        data=data,
        variable="t2m",
        xydims=["lon", "lat"],
        time_dimension="time",
        return_meta_data=False,
        return_image_data=False,
    )

    texture_file = outputs.texture_files[0]
    assert texture_file.metadata is None
    assert texture_file.base64_image_data is None

    # Test return_meta_data=True, return_image_data=True
    outputs = await adapter.body(
        data=data,
        variable="t2m",
        xydims=["lon", "lat"],
        time_dimension="time",
        return_meta_data=True,
        return_image_data=True,
    )

    texture_file = outputs.texture_files[0]
    assert texture_file.metadata is not None
    assert texture_file.base64_image_data is not None


@pytest.mark.asyncio
async def test_render_uint8_to_images_metadata_structure():
    """Test that metadata contains expected fields."""
    data = create_test_dataset(num_timesteps=1)
    adapter = RenderUint8ToImages(site=None, provider=None)

    additional_meta = {"custom_field": "custom_value", "number": 42}
    outputs = await adapter.body(
        data=data,
        variable="t2m",
        xydims=["lon", "lat"],
        time_dimension="time",
        additional_meta_data=additional_meta,
        return_meta_data=True,
    )

    metadata = outputs.texture_files[0].metadata
    assert metadata is not None

    # Check for coordinate bounds
    assert "lon_minmax" in metadata
    assert "lat_minmax" in metadata
    assert len(metadata["lon_minmax"]) == 2
    assert len(metadata["lat_minmax"]) == 2

    # Check for variable attributes
    assert metadata["long_name"] == "2m temperature"
    assert metadata["units"] == "K"

    # Check for additional metadata
    assert metadata["custom_field"] == "custom_value"
    assert metadata["number"] == 42


@pytest.mark.asyncio
async def test_render_uint8_to_images_empty_additional_metadata():
    """Test with None additional metadata."""
    data = create_test_dataset(num_timesteps=1)
    adapter = RenderUint8ToImages(site=None, provider=None)

    outputs = await adapter.body(
        data=data,
        variable="t2m",
        xydims=["lon", "lat"],
        time_dimension="time",
        additional_meta_data=None,
    )

    assert len(outputs.texture_files) == 1
    metadata = outputs.texture_files[0].metadata
    assert metadata is not None
    # Should still have coordinate bounds and variable attributes
    assert "lon_minmax" in metadata
    assert "lat_minmax" in metadata


@pytest.mark.asyncio
async def test_render_uint8_to_images_coordinate_bounds():
    """Test that coordinate bounds are calculated correctly."""
    # Create dataset with known coordinate values
    time_values = np.array(["2024-01-01T00:00:00"], dtype="datetime64")
    lat_values = np.array([45.0, 46.0, 47.0])  # Known values
    lon_values = np.array([10.0, 11.0, 12.0])  # Known values

    data = xarray.Dataset(
        data_vars={
            "temp": xarray.DataArray(
                np.random.randint(0, 255, (1, 3, 3), dtype=np.uint8),
                dims=["time", "lon", "lat"],  # Fixed order to match schema expectation
            ),
        },
        coords={
            "time": xarray.DataArray(time_values, dims=["time"]),
            "lat": xarray.DataArray(lat_values, dims=["lat"]),
            "lon": xarray.DataArray(lon_values, dims=["lon"]),
        },
    )
    data.attrs["data_min"] = np.array([0])
    data.attrs["data_max"] = np.array([255])

    adapter = RenderUint8ToImages(site=None, provider=None)
    outputs = await adapter.body(
        data=data, variable="temp", xydims=["lon", "lat"], time_dimension="time"
    )

    metadata = outputs.texture_files[0].metadata

    # Check longitude bounds (should include step size correction)
    lon_step = 1.0  # difference between consecutive lon values
    expected_lon_max = 12.0 + lon_step  # 13.0
    assert metadata["lon_minmax"][0] == 10.0
    assert metadata["lon_minmax"][1] == expected_lon_max

    # Check latitude bounds
    assert metadata["lat_minmax"][0] == 45.0
    assert metadata["lat_minmax"][1] == 47.0


@pytest.mark.asyncio
async def test_render_uint8_to_images_all_zero_warning(caplog):
    """Test warning when image data is all zeros."""
    time_values = np.array(["2024-01-01T00:00:00"], dtype="datetime64")

    data = xarray.Dataset(
        data_vars={
            "zeros": xarray.DataArray(
                np.zeros((1, 5, 5), dtype=np.uint8),  # All zeros
                dims=["time", "lon", "lat"],  # Fixed order to match schema expectation
            ),
        },
        coords={
            "time": time_values,
            "lat": np.linspace(0, 1, 5),
            "lon": np.linspace(0, 1, 5),
        },
    )
    data.attrs["data_min"] = np.array([0])
    data.attrs["data_max"] = np.array([255])

    adapter = RenderUint8ToImages(site=None, provider=None)

    with caplog.at_level("WARNING"):
        outputs = await adapter.body(
            data=data, variable="zeros", xydims=["lon", "lat"], time_dimension="time"
        )

    assert len(outputs.texture_files) == 1
    assert "Image was all 0, possibly bad normalization?" in caplog.text


@pytest.mark.asyncio
async def test_render_uint8_to_images_timestamp_format():
    """Test that timestamps are formatted correctly."""
    time_values = np.array(
        ["2024-01-15T14:30:00", "2024-12-31T23:59:00"], dtype="datetime64"
    )

    data = xarray.Dataset(
        data_vars={
            "data": xarray.DataArray(
                np.random.randint(0, 255, (2, 3, 3), dtype=np.uint8),
                dims=["time", "lon", "lat"],  # Fixed order to match schema expectation
            ),
        },
        coords={
            "time": time_values,
            "lat": np.linspace(0, 1, 3),
            "lon": np.linspace(0, 1, 3),
        },
    )
    data.attrs["data_min"] = np.array([0])
    data.attrs["data_max"] = np.array([255])

    adapter = RenderUint8ToImages(site=None, provider=None)
    outputs = await adapter.body(
        data=data, variable="data", xydims=["lon", "lat"], time_dimension="time"
    )

    # Check timestamp formatting
    assert outputs.texture_files[0].timestamp == "2024-01-15T14:30"
    assert outputs.texture_files[1].timestamp == "2024-12-31T23:59"


# Tests for _sanitize_for_json helper function
class TestSanitizeForJson:
    """Tests for the _sanitize_for_json helper function."""

    def test_sanitize_numpy_array(self):
        """Test that numpy arrays are converted to lists."""
        input_dict = {"data": np.array([1, 2, 3])}
        result = _sanitize_for_json(input_dict)
        assert result["data"] == [1, 2, 3]
        assert isinstance(result["data"], list)

    def test_sanitize_numpy_2d_array(self):
        """Test that 2D numpy arrays are converted to nested lists."""
        input_dict = {"matrix": np.array([[1, 2], [3, 4]])}
        result = _sanitize_for_json(input_dict)
        assert result["matrix"] == [[1, 2], [3, 4]]

    def test_sanitize_numpy_integer(self):
        """Test that numpy integers are converted to Python ints."""
        input_dict = {"count": np.int64(42)}
        result = _sanitize_for_json(input_dict)
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_sanitize_numpy_float(self):
        """Test that numpy floats are converted to Python floats."""
        input_dict = {"value": np.float64(3.14159)}
        result = _sanitize_for_json(input_dict)
        assert abs(result["value"] - 3.14159) < 1e-5
        assert isinstance(result["value"], float)

    def test_sanitize_numpy_bool(self):
        """Test that numpy booleans are converted to Python bools."""
        input_dict = {"flag_true": np.bool_(True), "flag_false": np.bool_(False)}
        result = _sanitize_for_json(input_dict)
        assert result["flag_true"] is True
        assert result["flag_false"] is False
        assert isinstance(result["flag_true"], bool)

    def test_sanitize_nested_dict(self):
        """Test that nested dicts are recursively sanitized."""
        input_dict = {
            "outer": {
                "inner_array": np.array([1, 2, 3]),
                "inner_int": np.int32(10),
            }
        }
        result = _sanitize_for_json(input_dict)
        assert result["outer"]["inner_array"] == [1, 2, 3]
        assert result["outer"]["inner_int"] == 10

    def test_sanitize_list_with_numpy(self):
        """Test that lists containing numpy types are sanitized."""
        input_dict = {"items": [np.int64(1), np.float32(2.5), np.array([3, 4])]}
        result = _sanitize_for_json(input_dict)
        assert result["items"][0] == 1
        assert abs(result["items"][1] - 2.5) < 1e-5
        assert result["items"][2] == [3, 4]

    def test_sanitize_preserves_native_types(self):
        """Test that native Python types are preserved."""
        input_dict = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "none": None,
        }
        result = _sanitize_for_json(input_dict)
        assert result == input_dict

    def test_sanitize_mixed_types(self):
        """Test mixed numpy and native types."""
        input_dict = {
            "native_int": 10,
            "numpy_int": np.int64(20),
            "native_list": [1, 2],
            "numpy_array": np.array([3, 4]),
            "native_str": "test",
        }
        result = _sanitize_for_json(input_dict)
        assert result["native_int"] == 10
        assert result["numpy_int"] == 20
        assert result["native_list"] == [1, 2]
        assert result["numpy_array"] == [3, 4]
        assert result["native_str"] == "test"

    def test_sanitize_result_is_json_serializable(self):
        """Test that the sanitized output can be JSON serialized."""
        input_dict = {
            "array": np.array([1.5, 2.5, 3.5]),
            "int": np.int32(100),
            "nested": {"arr": np.array([[1, 2], [3, 4]])},
        }
        result = _sanitize_for_json(input_dict)
        # This should not raise
        json_str = json.dumps(result)
        assert json_str is not None
        # Verify round-trip
        parsed = json.loads(json_str)
        assert parsed["array"] == [1.5, 2.5, 3.5]
        assert parsed["int"] == 100

    def test_sanitize_empty_dict(self):
        """Test that empty dict returns empty dict."""
        result = _sanitize_for_json({})
        assert result == {}


@pytest.mark.asyncio
async def test_render_uint8_to_images_with_numpy_attrs():
    """Test that xarray attrs with numpy types are properly sanitized."""
    time_values = np.array(["2024-01-01T00:00:00"], dtype="datetime64")

    data = xarray.Dataset(
        data_vars={
            "temp": xarray.DataArray(
                np.random.randint(0, 255, (1, 3, 3), dtype=np.uint8),
                dims=["time", "lon", "lat"],
                # Add attrs with various numpy types
                attrs={
                    "scale_factor": np.float64(0.1),
                    "add_offset": np.float32(273.15),
                    "valid_range": np.array([0, 255]),
                    "missing_value": np.int32(-9999),
                    "flag": np.bool_(True),
                },
            ),
        },
        coords={
            "time": time_values,
            "lat": np.array([45.0, 46.0, 47.0]),
            "lon": np.array([10.0, 11.0, 12.0]),
        },
    )
    data.attrs["data_min"] = np.array([0])
    data.attrs["data_max"] = np.array([255])

    adapter = RenderUint8ToImages(site=None, provider=None)
    outputs = await adapter.body(
        data=data, variable="temp", xydims=["lon", "lat"], time_dimension="time"
    )

    metadata = outputs.texture_files[0].metadata
    assert metadata is not None

    # Verify numpy types were converted
    assert isinstance(metadata["scale_factor"], float)
    assert isinstance(metadata["add_offset"], float)
    assert isinstance(metadata["valid_range"], list)
    assert isinstance(metadata["missing_value"], int)
    assert isinstance(metadata["flag"], bool)

    # Verify values are correct
    assert abs(metadata["scale_factor"] - 0.1) < 1e-5
    assert metadata["valid_range"] == [0, 255]
    assert metadata["missing_value"] == -9999
    assert metadata["flag"] is True

    # Verify the metadata is JSON serializable
    json_str = json.dumps(metadata)
    assert json_str is not None
