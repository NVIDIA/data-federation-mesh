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

"""Model components for xarray schema validation."""

from typing import Any, Dict, Optional, Sequence, cast

import numpy as np
from pandera.api.base.schema import BaseSchema
from pandera.api.checks import Check

from .components import DataArraySchema
from .types import CheckList, XArrayDtypeInputTypes, to_checklist


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


class FieldInfo:
    """Captures extra information about a field."""

    def __init__(
        self,
        checks: Optional[Sequence[Check]] = None,
        nullable: bool = False,
        coerce: bool = False,
        required: bool = True,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize field info.

        Args:
            checks: List of validation checks
            nullable: Whether the field can be null
            coerce: Whether to coerce the field to the specified dtype
            required: Whether the field is required
            name: Name of the field
            title: A human-readable label for the field
            description: An arbitrary textual description
            metadata: An optional key-value data
        """
        self.checks = list(checks) if checks else []
        self.nullable = nullable
        self.coerce = coerce
        self.required = required
        self.name = name
        self.title = title
        self.description = description
        self.metadata = metadata or {}

    def to_data_variable(
        self,
        dtype: Optional[XArrayDtypeInputTypes] = None,
        checks: Optional[CheckList] = None,
        required: bool = True,
        name: Optional[str] = None,
    ) -> DataVariable:
        """Create a DataVariable from a field."""
        combined_checks = to_checklist(self.checks + to_checklist(checks))
        return DataVariable(
            dtype=dtype,
            checks=combined_checks,
            nullable=self.nullable,
            coerce=self.coerce,
            required=required,
            name=name or self.name,
            title=self.title,
            description=self.description,
            metadata=self.metadata,
        )

    def to_coordinate(
        self,
        dtype: Optional[XArrayDtypeInputTypes] = None,
        checks: Optional[CheckList] = None,
        required: bool = True,
        name: Optional[str] = None,
    ) -> Coordinate:
        """Create a Coordinate from a field."""
        combined_checks = to_checklist(self.checks + to_checklist(checks))
        return Coordinate(
            dtype=dtype,
            checks=combined_checks,
            nullable=self.nullable,
            coerce=self.coerce,
            required=required,
            name=name or self.name,
            title=self.title,
            description=self.description,
            metadata=self.metadata,
        )

    def to_attribute(
        self,
        dtype: Optional[XArrayDtypeInputTypes] = None,
        checks: Optional[CheckList] = None,
        required: bool = True,
        name: Optional[str] = None,
    ) -> Attribute:
        """Create an Attribute from a field."""
        combined_checks = to_checklist(self.checks + to_checklist(checks))
        return Attribute(
            dtype=dtype,
            checks=combined_checks,
            nullable=self.nullable,
            required=required,
            name=name or self.name,
            title=self.title,
            description=self.description,
            metadata=self.metadata,
        )
