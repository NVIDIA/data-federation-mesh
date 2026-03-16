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

from typing import List, Tuple
from PIL import Image
from io import BytesIO
from base64 import b64decode
import numpy as np
from nv_dfm_lib_common.schemas import TextureFile, TextureFileList


def texturefile_to_image_bytes(texture_file: TextureFile) -> bytes:
    return b64decode(texture_file.base64_image_data)


def texturefile_to_image_bytes_list(texture_file_list: TextureFileList) -> List[bytes]:
    return [
        b64decode(texture_file.base64_image_data)
        for texture_file in texture_file_list.texture_files
    ]


def texturefile_to_image(texture_file: TextureFile) -> Image.Image:
    image_data = b64decode(texture_file.base64_image_data)
    return Image.open(BytesIO(image_data))


def texturefilelist_to_images(texture_file_list: TextureFileList) -> List[Image.Image]:
    return [
        texturefile_to_image(texture_file)
        for texture_file in texture_file_list.texture_files
    ]


def texturefile_to_image_url(texture_file: TextureFile) -> str:
    return f"data:image/{texture_file.format.lower()};base64,{texture_file.base64_image_data}"


def texturefilelist_to_image_urls(texture_file_list: TextureFileList) -> List[str]:
    return [
        texturefile_to_image_url(texture_file)
        for texture_file in texture_file_list.texture_files
    ]


def roll_image_by_pixels(img: Image.Image, shift_px: int) -> Image.Image:
    """
    Circularly shift a PIL image horizontally by shift_px pixels.
    Positive shift moves content to the right; pixels wrap around.
    Works for any mode (L, RGB, RGBA, etc.).
    """
    arr = np.asarray(img)
    rolled = np.roll(arr, shift=shift_px, axis=1)
    return Image.fromarray(rolled)


def roll_image_by_degrees(
    img: Image.Image,
    offset_deg: float,
    long_min: float = -180.0,
    long_max: float = 180.0,
) -> Tuple[Image.Image, int]:
    """
    Roll an equirectangular image by offset_deg in longitude.
    Assumes columns map linearly to longitudes in [long_min, long_max].
    Returns (rolled_image, shift_px_applied).
    Positive offset shifts content to the right.
    """
    width = img.width
    span = float(long_max - long_min)
    if span <= 0:
        raise ValueError("Invalid longitude span: long_max must be > long_min")

    px_per_deg = width / span
    shift_px = int(round(offset_deg * px_per_deg))
    return roll_image_by_pixels(img, shift_px), shift_px


def image_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
