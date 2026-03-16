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

"""Container classes for xarray schema validation."""

from typing import Any, Dict, Optional, cast

import numpy as np
from pandera.api.base.schema import BaseSchema

from .components import Attribute, Coordinate, DataArraySchema, DataVariable
from .types import (
    CheckList,
    XArrayDtypeInputTypes,
    is_dataarray,
    is_dataset,
    to_checklist,
)


class DataArrayContainer(DataArraySchema):
    """Container for xarray DataArray validation."""

    def __init__(
        self,
        dtype: Optional[XArrayDtypeInputTypes] = None,
        checks: Optional[CheckList] = None,
        nullable: bool = False,
        coerce: bool = False,
        required: bool = True,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create DataArrayContainer validator object.

        Args:
            dtype: datatype of the data array
            checks: checks to verify validity of the data array
            nullable: Whether or not the data array can contain null values
            coerce: If True, coerce the data array to the specified dtype
            required: Whether or not the data array is required
            name: name of the data array
            title: A human-readable label for the data array
            description: An arbitrary textual description
            metadata: An optional key-value data
        """
        super().__init__(
            dtype=dtype,
            checks=checks,
            nullable=nullable,
            coerce=coerce,
            required=required,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )

    def validate(self, data: Any) -> bool:
        """Validate a DataArray against the schema.

        Args:
            data: The DataArray to validate.

        Returns:
            True if validation passes, raises an exception otherwise.
        """
        if not is_dataarray(data):
            raise ValueError(f"Expected DataArray, got {type(data)}")

        if data is None:
            if not self.nullable:
                raise ValueError(f"DataArray {self.name} cannot be null")
            # Null value is allowed, continue to return True at end
        else:
            if self.dtype is not None and not np.issubdtype(
                data.dtype, cast(np.dtype, self.dtype)
            ):
                raise ValueError(f"DataArray has incorrect dtype: {data.dtype}")

            if self.checks:
                for check in self.checks:
                    if not check(data):
                        raise ValueError(
                            f"DataArray {data.name} failed check: {check.__name__}"
                        )

        return True


class DatasetContainer(BaseSchema):
    """Container for xarray Dataset validation."""

    def __init__(
        self,
        data_vars: Optional[Dict[str, DataVariable]] = None,
        coords: Optional[Dict[str, Coordinate]] = None,
        attrs: Optional[Dict[str, Attribute]] = None,
        checks: Optional[CheckList] = None,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create DatasetContainer validator object.

        Args:
            data_vars: Dictionary of data variable schemas
            coords: Dictionary of coordinate schemas
            attrs: Dictionary of attribute schemas
            checks: checks to verify validity of the dataset
            name: name of the dataset
            title: A human-readable label for the dataset
            description: An arbitrary textual description
            metadata: An optional key-value data
        """
        super().__init__(
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )
        self.data_vars = data_vars or {}
        self.coords = coords or {}
        self.attrs = attrs or {}
        self.checks = to_checklist(checks)

    def validate(self, data: Any) -> bool:
        """Validate a Dataset against the schema.

        Args:
            data: The Dataset to validate.

        Returns:
            True if validation passes, raises an exception otherwise.
        """
        if not is_dataset(data):
            raise ValueError(f"Expected Dataset, got {type(data)}")

        # Validate data variables
        for name, schema in self.data_vars.items():
            if name not in data.data_vars:
                if schema.required:
                    raise ValueError(f"Required data variable {name} not found")
                continue
            schema.validate(data.data_vars[name])

        # Validate coordinates
        for name, schema in self.coords.items():
            if name not in data.coords:
                if schema.required:
                    raise ValueError(f"Required coordinate {name} not found")
                continue
            schema.validate(data.coords[name])

        # Validate attributes
        for name, schema in self.attrs.items():
            if name not in data.attrs:
                if schema.required:
                    raise ValueError(f"Required attribute {name} not found")
                continue
            schema.validate(data.attrs[name])

        # Validate dataset-level checks
        if self.checks:
            for check in self.checks:
                if not check(data):
                    raise ValueError(f"Dataset failed check: {check.__name__}")

        return True
