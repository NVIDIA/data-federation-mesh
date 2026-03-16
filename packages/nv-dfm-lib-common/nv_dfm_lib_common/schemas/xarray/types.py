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

"""Type definitions and utility functions for xarray validation."""

from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import xarray as xr
from pandera.api.checks import Check

# Type aliases
XArrayDtypeInputTypes = Union[
    np.dtype, Type[np.number], Type[np.bool_], Type[np.datetime64]
]
CheckList = Union[Check, List[Check]]
CheckArg = Optional[
    Union[
        Check,
        List[Check],
        Callable[[xr.DataArray], bool],
        List[Callable[[xr.DataArray], bool]],
    ]
]


def to_checklist(checks: CheckArg) -> List[Check]:
    """Convert checks to a list of Check objects.

    Args:
        checks: Checks to convert. Can be:
            - None
            - A single Check object
            - A single callable function
            - A list of Check objects
            - A list of callable functions
            - A list mixing Check objects and callable functions

    Returns:
        List of Check objects
    """
    if checks is None:
        return []

    if isinstance(checks, Check):
        return [checks]

    if isinstance(checks, Callable):
        return [Check(checks)]

    if isinstance(checks, list):
        return [
            Check(check) if isinstance(check, Callable) else check for check in checks
        ]

    # This should never happen due to type checking, but just in case
    raise TypeError(f"Invalid check type: {type(checks)}")


# Utility functions
def is_dataset(obj: Any) -> bool:
    """Check if object is an xarray Dataset.

    Args:
        obj: object to check

    Returns:
        True if object is a Dataset, False otherwise
    """
    return isinstance(obj, xr.Dataset)


def is_dataarray(obj: Any) -> bool:
    """Check if object is an xarray DataArray.

    Args:
        obj: object to check

    Returns:
        True if object is a DataArray, False otherwise
    """
    return isinstance(obj, xr.DataArray)


def is_numeric(dtype: np.dtype) -> bool:
    """Check if dtype is numeric.

    Args:
        dtype: dtype to check

    Returns:
        True if dtype is numeric, False otherwise
    """
    return np.issubdtype(dtype, np.number)


def is_bool(dtype: np.dtype) -> bool:
    """Check if dtype is boolean.

    Args:
        dtype: dtype to check

    Returns:
        True if dtype is boolean, False otherwise
    """
    return np.issubdtype(dtype, np.bool_)


def is_datetime(dtype: np.dtype) -> bool:
    """Check if dtype is datetime.

    Args:
        dtype: dtype to check

    Returns:
        True if dtype is datetime, False otherwise
    """
    return np.issubdtype(dtype, np.datetime64)
