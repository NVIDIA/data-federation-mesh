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

from ._convert_to_uint8 import ConvertToUint8OutputSchema


class RenderUint8ToImagesInputSchema(ConvertToUint8OutputSchema):
    """
    Schema for the input data to the RenderUint8ToImages function.
    """

    def __init__(self, time_dimension: str, xydims: list[str], data_vars):
        """
        Initialize the RenderUint8ToImagesInputSchema.

        This schema inherits from ConvertToUint8OutputSchema since the input
        to the render function should be uint8 data.

        Args:
            time_dimension: Name of the time dimension
            xydims: List of x and y dimension names
            data_vars: List of data variable names to validate
        """
        super().__init__(time_dimension, xydims, data_vars)
