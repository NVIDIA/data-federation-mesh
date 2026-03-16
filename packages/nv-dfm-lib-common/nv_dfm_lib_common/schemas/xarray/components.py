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

"""Core xarray schema component specifications."""

from typing import Any, Dict, Optional, cast

import numpy as np
import xarray as xr
from pandera.api.base.schema import BaseSchema

from .types import CheckList, XArrayDtypeInputTypes


class DataArraySchema(BaseSchema):
    """Validate types and properties of xarray DataArrays."""

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
        """Create DataArray validator object.

        Args:
            dtype: datatype of the DataArray
            checks: checks to verify validity of the DataArray
            nullable: Whether or not the DataArray can contain null values
            coerce: If True, coerce the DataArray to the specified dtype
            required: Whether or not the DataArray is required
            name: name of the DataArray
            title: A human-readable label for the DataArray
            description: An arbitrary textual description
            metadata: An optional key-value data
        """
        super().__init__(
            dtype=dtype,
            checks=checks,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )
        self.required = required
        self.name = name
        self.metadata = metadata or {}
        self._nullable = nullable
        self._coerce = coerce
        self._dtype = dtype

    def validate(self, data: xr.DataArray) -> bool:
        """Validate a DataArray against the schema.

        Args:
            data: The DataArray to validate.

        Returns:
            True if validation passes, raises an exception otherwise.
        """
        if not isinstance(data, xr.DataArray):
            raise ValueError(f"Expected DataArray, got {type(data)}")

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

    @property
    def dtype(self) -> Optional[XArrayDtypeInputTypes]:
        """Get the dtype of the DataArray."""
        return self._dtype

    @dtype.setter
    def dtype(self, value: Optional[XArrayDtypeInputTypes]) -> None:
        """Set the dtype of the DataArray."""
        self._dtype = value

    @property
    def coerce(self) -> bool:
        """Whether to coerce the DataArray to the specified dtype."""
        return self._coerce

    @coerce.setter
    def coerce(self, value: bool) -> None:
        """Set whether to coerce the DataArray to the specified dtype."""
        self._coerce = value

    @property
    def nullable(self) -> bool:
        """Whether the DataArray can contain null values."""
        return self._nullable

    @nullable.setter
    def nullable(self, value: bool) -> None:
        """Set whether the DataArray can contain null values."""
        self._nullable = value


class DataVariable(DataArraySchema):
    """Schema for xarray data variables."""

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
        """Create DataVariable validator object.

        Args:
            dtype: datatype of the data variable
            checks: checks to verify validity of the data variable
            nullable: Whether or not the data variable can contain null values
            coerce: If True, coerce the data variable to the specified dtype
            required: Whether or not the data variable is required
            name: name of the data variable
            title: A human-readable label for the data variable
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


class Coordinate(DataArraySchema):
    """Schema for xarray coordinates."""

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
        """Create Coordinate validator object.

        Args:
            dtype: datatype of the coordinate
            checks: checks to verify validity of the coordinate
            nullable: Whether or not the coordinate can contain null values
            coerce: If True, coerce the coordinate to the specified dtype
            required: Whether or not the coordinate is required
            name: name of the coordinate
            title: A human-readable label for the coordinate
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


class Attribute(BaseSchema):
    """Schema for xarray attributes."""

    def __init__(
        self,
        dtype: Optional[XArrayDtypeInputTypes] = None,
        checks: Optional[CheckList] = None,
        nullable: bool = False,
        required: bool = True,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create Attribute validator object.

        Args:
            dtype: datatype of the attribute
            checks: checks to verify validity of the attribute
            nullable: Whether or not the attribute can contain null values
            required: Whether or not the attribute is required
            name: name of the attribute
            title: A human-readable label for the attribute
            description: An arbitrary textual description
            metadata: An optional key-value data
        """
        super().__init__(
            dtype=dtype,
            checks=checks,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )
        self.required = required
        self.name = name
        self.metadata = metadata or {}
        self._nullable = nullable
        self._dtype = dtype

    def validate(self, data: Any) -> bool:
        """Validate an attribute against the schema.

        Args:
            data: The attribute to validate.

        Returns:
            True if validation passes, raises an exception otherwise.
        """
        if data is None:
            if not self.nullable:
                raise ValueError(f"Attribute {self.name} cannot be null")
            # Null value is allowed, continue to return True at end
        else:
            if self.dtype is not None and not np.issubdtype(
                type(data), cast(np.dtype, self.dtype)
            ):
                raise ValueError(f"Attribute has incorrect type: {type(data)}")

            if self.checks:
                for check in self.checks:
                    if not check(data):
                        raise ValueError(f"Attribute failed check: {check.__name__}")

        return True

    @property
    def dtype(self) -> Optional[XArrayDtypeInputTypes]:
        """Get the dtype of the attribute."""
        return self._dtype

    @dtype.setter
    def dtype(self, value: Optional[XArrayDtypeInputTypes]) -> None:
        """Set the dtype of the attribute."""
        self._dtype = value

    @property
    def nullable(self) -> bool:
        """Whether the attribute can contain null values."""
        return self._nullable

    @nullable.setter
    def nullable(self, value: bool) -> None:
        """Set whether the attribute can contain null values."""
        self._nullable = value
