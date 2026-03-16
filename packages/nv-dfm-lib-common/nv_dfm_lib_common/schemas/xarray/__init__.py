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

"""xarray-schema: Schema validation for xarray Datasets."""

from .checks import (
    check_attrs,
    check_cf_standard_name,
    check_cf_units,
    check_coordinate_system,
    check_dims,
    check_dtype,
    check_grid_mapping,
    check_grid_mapping_reference,
    check_max_missing,
    check_range,
    check_time_coordinate,
)
from .schema import (
    Attribute,
    Coordinate,
    DataVariable,
    XArraySchema,
)

__version__ = "0.1.0"

__all__ = [
    "XArraySchema",
    "DataVariable",
    "Coordinate",
    "Attribute",
    "check_dims",
    "check_dtype",
    "check_max_missing",
    "check_attrs",
    "check_range",
    "check_cf_standard_name",
    "check_cf_units",
    "check_coordinate_system",
    "check_grid_mapping",
    "check_grid_mapping_reference",
    "check_time_coordinate",
]
