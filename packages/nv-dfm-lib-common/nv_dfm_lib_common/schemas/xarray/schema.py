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

"""Core schema validation for xarray Datasets."""

import copy
from typing import (
    Any,
    Callable,
    Hashable,
    Literal,
    Protocol,
)

import numpy as np
import xarray as xr
from pandera.api.base.schema import BaseSchema
from xarray.core.types import InterpOptions

from .checks import (
    check_cf_standard_name,
    check_cf_units,
    check_dims,
    check_dtype,
    check_range,
)
from .types import CheckList, to_checklist

# Define valid interpolation methods
InterpMethod = Literal[
    "linear",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "polynomial",
    "barycentric",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
]


class CheckFunction(Protocol):
    """Protocol for check functions."""

    def __call__(self, dataarray: xr.DataArray) -> bool: ...


class SchemaComponent:
    """Base class for schema components."""

    def __init__(
        self,
        dtype: np.dtype | type,
        checks: list[Callable],
        nullable: bool = False,
        required: bool = True,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a schema component.

        Args:
            dtype: The expected data type.
            checks: List of validation check functions.
            nullable: Whether the component can be null.
            required: Whether the component is required.
            name: The name of the component.
            title: A human-readable title.
            description: A description of the component.
            metadata: Additional metadata.
        """
        self.dtype: np.dtype | type = dtype
        self.checks: list[Callable] = checks
        self.nullable: bool = nullable
        self.required: bool = required
        self.name: str | None = name
        self.title: str | None = title
        self.description: str | None = description
        self.metadata: dict[str, Any] = metadata or {}

    def __eq__(self, other: Any) -> bool:
        """Compare two schema components for equality."""
        if not isinstance(other, type(self)):
            return False
        return (
            self.dtype == other.dtype
            and self.checks == other.checks
            and self.nullable == other.nullable
            and self.required == other.required
            and self.name == other.name
            and self.title == other.title
            and self.description == other.description
            and self.metadata == other.metadata
        )

    def validate(self, data: Any) -> bool:
        """Validate the data against the schema.

        Args:
            data: The data to validate.

        Returns:
            True if validation passes, raises an exception otherwise.
        """
        if data is None:
            if not self.nullable:
                raise ValueError(f"Component {self.name} cannot be null")
            # Null value is allowed, continue to return True at end
        else:
            if not np.issubdtype(type(data), self.dtype):
                raise ValueError(
                    f"Component {self.name} has incorrect type: {type(data)}"
                )

            for check in self.checks:
                if not check(data):
                    raise ValueError(
                        f"Component {self.name} failed check: {getattr(check, '__name__', type(check).__name__)}"
                    )

        return True


class DataVariable(SchemaComponent):
    """Schema for xarray DataVariables."""

    def __init__(
        self,
        dtype: np.dtype | type,
        checks: list[Callable],
        nullable: bool = False,
        required: bool = True,
        coerce: bool = False,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        required_attrs: list[str] | None = None,
    ):
        """Initialize a DataVariable schema.

        Args:
            dtype: The expected data type.
            checks: List of validation check functions.
            nullable: Whether the component can be null.
            required: Whether the component is required.
            coerce: Whether to coerce the data to the specified dtype.
            name: The name of the component.
            title: A human-readable title.
            description: A description of the component.
            metadata: Additional metadata.
            required_attrs: List of required attribute names.
        """
        super().__init__(
            dtype=dtype,
            checks=checks,
            nullable=nullable,
            required=required,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )
        self.coerce = coerce
        self.required_attrs = required_attrs or []

    def __eq__(self, other: Any) -> bool:
        """Compare two DataVariable schemas for equality."""
        if not isinstance(other, DataVariable):
            return False
        return (
            super().__eq__(other)
            and self.coerce == other.coerce
            and self.required_attrs == other.required_attrs
        )

    def validate(self, data: xr.DataArray) -> bool:
        """Validate a DataArray against the schema.

        Args:
            data: The DataArray to validate.

        Returns:
            True if validation passes, raises an exception otherwise.
        """
        if not isinstance(data, xr.DataArray):
            raise ValueError(f"Expected DataArray, got {type(data)}")

        # Run all checks including dtype check
        for check in self.checks:
            if not check(data):
                raise ValueError(
                    f"DataArray {data.name} failed check: {getattr(check, '__name__', type(check).__name__)}"
                )

        # Check for required attributes
        if hasattr(self, "required_attrs"):
            for attr_name in self.required_attrs:
                if attr_name not in data.attrs:
                    raise ValueError(
                        f"DataArray missing required attribute: {attr_name}"
                    )

        return True


class Coordinate(SchemaComponent):
    """Schema for xarray Coordinates."""

    def __init__(
        self,
        dtype: np.dtype | type,
        checks: list[Callable],
        nullable: bool = False,
        required: bool = True,
        coerce: bool = False,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a Coordinate schema.

        Args:
            dtype: The expected data type.
            checks: List of validation check functions.
            nullable: Whether the component can be null.
            required: Whether the component is required.
            coerce: Whether to coerce the data to the specified dtype.
            name: The name of the component.
            title: A human-readable title.
            description: A description of the component.
            metadata: Additional metadata.
        """
        super().__init__(
            dtype=dtype,
            checks=checks,
            nullable=nullable,
            required=required,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )
        self.coerce = coerce

    def __eq__(self, other: Any) -> bool:
        """Compare two Coordinate schemas for equality."""
        if not isinstance(other, Coordinate):
            return False
        return super().__eq__(other) and self.coerce == other.coerce

    def validate(self, data: xr.DataArray) -> bool:
        """Validate a Coordinate against the schema.

        Args:
            data: The Coordinate to validate.

        Returns:
            True if validation passes, raises an exception otherwise.
        """
        if not isinstance(data, xr.DataArray):
            raise ValueError(f"Expected DataArray, got {type(data)}")

        # Run all checks including dtype check
        for check in self.checks:
            if not check(data):
                raise ValueError(
                    f"Coordinate failed check: {getattr(check, '__name__', type(check).__name__)}"
                )

        return True


class Attribute(SchemaComponent):
    """Schema for xarray Attributes."""

    def __eq__(self, other: Any) -> bool:
        """Compare two Attribute schemas for equality."""
        if not isinstance(other, Attribute):
            return False
        return super().__eq__(other)

    def validate(self, data: Any) -> bool:
        """Validate an Attribute against the schema.

        Args:
            data: The Attribute to validate.

        Returns:
            True if validation passes, raises an exception otherwise.
        """
        if not np.issubdtype(type(data), self.dtype):
            raise ValueError(f"Attribute has incorrect type: {type(data)}")

        for check in self.checks:
            if not check(data):
                raise ValueError(
                    f"Attribute failed check: {getattr(check, '__name__', type(check).__name__)}"
                )

        return True


class XArraySchema(BaseSchema):
    """Schema for xarray Datasets."""

    def __init__(
        self,
        data_vars: dict[str, DataVariable] | None = None,
        coords: dict[str, Coordinate] | None = None,
        attrs: dict[str, Attribute] | None = None,
        checks: CheckList | None = None,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        frozen: bool = True,
    ):
        """Initialize an XArraySchema.

        Args:
            data_vars: Dictionary of data variable schemas.
            coords: Dictionary of coordinate schemas.
            attrs: Dictionary of attribute schemas.
            checks: List of validation check functions.
            name: The name of the schema.
            title: A human-readable title.
            description: A description of the schema.
            metadata: Additional metadata.
            frozen: If True, the schema is immutable.
        """
        self.data_vars = data_vars or {}
        self.coords = coords or {}
        self.attrs = attrs or {}
        self.checks = checks or []
        self.name = name
        self.title = title
        self.description = description
        self.metadata = metadata or {}
        self.frozen = frozen

    def mutable_clone(self) -> "XArraySchema":
        """Create a deep copy of the schema with frozen=False.

        Returns:
            A new XArraySchema instance with frozen=False.
        """
        # Create new instances of DataVariable, Coordinate, and Attribute
        new_data_vars = {}
        for name, var in self.data_vars.items():
            new_data_vars[name] = DataVariable(
                dtype=var.dtype,
                checks=copy.deepcopy(var.checks),
                nullable=var.nullable,
                required=var.required,
                coerce=var.coerce,
                name=var.name,
                title=var.title,
                description=var.description,
                metadata=copy.deepcopy(var.metadata),
                required_attrs=copy.deepcopy(var.required_attrs)
                if hasattr(var, "required_attrs")
                else None,
            )

        new_coords = {}
        for name, coord in self.coords.items():
            new_coords[name] = Coordinate(
                dtype=coord.dtype,
                checks=copy.deepcopy(coord.checks),
                nullable=coord.nullable,
                required=coord.required,
                coerce=coord.coerce,
                name=coord.name,
                title=coord.title,
                description=coord.description,
                metadata=copy.deepcopy(coord.metadata),
            )

        new_attrs = {}
        for name, attr in self.attrs.items():
            new_attrs[name] = Attribute(
                dtype=attr.dtype,
                checks=copy.deepcopy(attr.checks),
                nullable=attr.nullable,
                required=attr.required,
                name=attr.name,
                title=attr.title,
                description=attr.description,
                metadata=copy.deepcopy(attr.metadata),
            )

        return XArraySchema(
            data_vars=new_data_vars,
            coords=new_coords,
            attrs=new_attrs,
            checks=copy.deepcopy(self.checks),
            name=self.name,
            title=self.title,
            description=self.description,
            metadata=copy.deepcopy(self.metadata),
            frozen=False,
        )

    def freeze(self) -> None:
        """Freeze the schema, making it immutable."""
        self.frozen = True

    def add_data_var(self, name: str, data_var: DataVariable) -> None:
        """Add a data variable to the schema.

        Args:
            name: The name of the data variable.
            data_var: The DataVariable schema to add.

        Raises:
            ValueError: If the schema is frozen.
        """
        if self.frozen:
            raise ValueError("Cannot modify a frozen schema.")
        self.data_vars[name] = data_var

    def remove_data_var(self, name: str) -> None:
        """Remove a data variable from the schema.

        Args:
            name: The name of the data variable to remove.

        Raises:
            ValueError: If the schema is frozen.
        """
        if self.frozen:
            raise ValueError("Cannot modify a frozen schema.")
        if name in self.data_vars:
            del self.data_vars[name]

    def add_coord(self, name: str, coord: Coordinate) -> None:
        """Add a coordinate to the schema.

        Args:
            name: The name of the coordinate.
            coord: The Coordinate schema to add.

        Raises:
            ValueError: If the schema is frozen.
        """
        if self.frozen:
            raise ValueError("Cannot modify a frozen schema.")
        self.coords[name] = coord

    def remove_coord(self, name: str) -> None:
        """Remove a coordinate from the schema.

        Args:
            name: The name of the coordinate to remove.

        Raises:
            ValueError: If the schema is frozen.
        """
        if self.frozen:
            raise ValueError("Cannot modify a frozen schema.")
        if name in self.coords:
            del self.coords[name]

    def add_attr(self, name: str, attr: Attribute) -> None:
        """Add an attribute to the schema.

        Args:
            name: The name of the attribute.
            attr: The Attribute schema to add.

        Raises:
            ValueError: If the schema is frozen.
        """
        if self.frozen:
            raise ValueError("Cannot modify a frozen schema.")
        self.attrs[name] = attr

    def remove_attr(self, name: str) -> None:
        """Remove an attribute from the schema.

        Args:
            name: The name of the attribute to remove.

        Raises:
            ValueError: If the schema is frozen.
        """
        if self.frozen:
            raise ValueError("Cannot modify a frozen schema.")
        if name in self.attrs:
            del self.attrs[name]

    def get_variable(self, name: str) -> DataVariable | None:
        """Get a data variable schema by name.

        Args:
            name: The name of the data variable.

        Returns:
            The DataVariable schema if found, None otherwise.
        """
        return self.data_vars.get(name)

    def get_coord(self, name: str) -> Coordinate | None:
        """Get a coordinate schema by name.

        Args:
            name: The name of the coordinate.

        Returns:
            The Coordinate schema if found, None otherwise.
        """
        return self.coords.get(name)

    def get_attr(self, name: str) -> Attribute | None:
        """Get an attribute schema by name.

        Args:
            name: The name of the attribute.

        Returns:
            The Attribute schema if found, None otherwise.
        """
        return self.attrs.get(name)

    def copy(self) -> "XArraySchema":
        """Create a deep copy of the schema.

        Returns:
            A new XArraySchema instance with copied components.
        """
        return XArraySchema(
            data_vars={k: copy.deepcopy(v) for k, v in self.data_vars.items()},
            coords={k: copy.deepcopy(v) for k, v in self.coords.items()},
            attrs={k: copy.deepcopy(v) for k, v in self.attrs.items()},
            checks=copy.deepcopy(self.checks),
            name=self.name,
            title=self.title,
            description=self.description,
            metadata=copy.deepcopy(self.metadata),
            frozen=self.frozen,
        )

    def validate(self, data: xr.Dataset, strict: bool = True) -> bool:
        """Validate a Dataset against the schema.

        Args:
            data: The Dataset to validate.
            strict: Whether to enforce strict validation (no extra components allowed).

        Returns:
            True if validation passes, raises an exception otherwise.
        """
        if not isinstance(data, xr.Dataset):
            raise ValueError(f"Expected Dataset, got {type(data)}")

        # Validate required data variables
        for name, schema in self.data_vars.items():
            if schema.required and name not in data.data_vars:
                raise ValueError(f"missing required data variable: {name}")
            if name in data.data_vars:
                schema.validate(data[name])

        # Validate required coordinates
        for name, schema in self.coords.items():
            if schema.required and name not in data.coords:
                raise ValueError(f"missing required coordinate: {name}")
            if name in data.coords:
                schema.validate(data[name])

        # Validate required attributes
        for name, schema in self.attrs.items():
            if schema.required and name not in data.attrs:
                raise ValueError(f"missing required attribute: {name}")
            if name in data.attrs:
                schema.validate(data.attrs[name])

        # Check for extra components in strict mode
        if strict:
            for name in data.data_vars:
                if name not in self.data_vars:
                    raise ValueError(f"unexpected data variable: {name}")
            for name in data.coords:
                if name not in self.coords:
                    raise ValueError(f"unexpected coordinate: {name}")
            for name in data.attrs:
                if name not in self.attrs:
                    raise ValueError(f"unexpected attribute: {name}")

        return True

    def clean(self, data: xr.Dataset) -> xr.Dataset:
        """Remove extra variables, coordinates, dimensions, and attributes not present in the schema.

        Args:
            data: The dataset to clean.

        Returns:
            A new dataset containing only the components specified in the schema.
        """
        # Create a copy to avoid modifying the original
        cleaned = data.copy()

        # Remove extra data variables
        extra_vars = set(cleaned.data_vars) - set(self.data_vars)
        if extra_vars:
            cleaned = cleaned.drop_vars(extra_vars)

        # Remove extra coordinates
        extra_coords = set(cleaned.coords) - set(self.coords)
        if extra_coords:
            cleaned = cleaned.drop_vars(extra_coords)

        # Remove extra attributes
        if self.attrs:
            extra_attrs = set(cleaned.attrs) - set(self.attrs)
            for attr in extra_attrs:
                del cleaned.attrs[attr]

        return cleaned

    def _get_dims_from_checks(self, var_name: str) -> list[Hashable] | None:
        """Get dimensions from a variable's check_dims check.

        Args:
            var_name: Name of the variable to get dimensions for.

        Returns:
            List of dimensions if found, None otherwise.
        """
        for check in self.data_vars[var_name].checks:
            if hasattr(check, "__name__") and check.__name__.startswith("check_dims("):
                if check.__closure__ is not None and len(check.__closure__) > 0:
                    return check.__closure__[0].cell_contents  # type: ignore
        return None

    def stack_to_ndarray(
        self,
        data: xr.Dataset,
        var_names: list[str] | None = None,
        dim_order: list[Hashable] | None = None,
    ) -> np.ndarray:
        """Stack all data variables into a single n-dimensional array.

        The resulting array will have dimensions ordered as:
        1. All dimensions from the schema (in order of appearance)
        2. A final dimension for the variables

        Args:
            data: The dataset to stack.
            var_names: Optional list of variable names to control stacking order.
                      If not provided, uses the order from the schema definition.
            dim_order: Optional list of dimension names specifying the desired order.
                      If not provided, uses the order from the first variable's check_dims.
                      If no check_dims is found, raises ValueError.

        Returns:
            A numpy array containing all data variables stacked along a new dimension.

        Raises:
            ValueError: If the dataset doesn't match the schema.
            ValueError: If var_names contains invalid variable names.
            ValueError: If variables have different dimensions.
            ValueError: If no dimension order can be determined.
        """
        # First validate the dataset
        self.validate(data)

        # Determine variable order
        if var_names is None:
            # Use schema definition order
            var_names = list(self.data_vars.keys())
        else:
            # Validate var_names
            invalid_vars = set(var_names) - set(self.data_vars.keys())
            if invalid_vars:
                raise ValueError(f"Invalid variable names: {invalid_vars}")

        # If no dimension order provided, try to get it from the first variable's check_dims
        if dim_order is None:
            dim_order = self._get_dims_from_checks(var_names[0])
            if dim_order is None:
                raise ValueError(
                    "No dimension order specified and no check_dims found in schema"
                )

        # Stack all variables in the specified order
        stacked = []
        for var_name in var_names:
            var = data[var_name]
            # Ensure variables have all required dimensions
            missing_dims = set(dim_order) - set(var.dims)
            if missing_dims:
                raise ValueError(
                    f"Variable {var_name} is missing dimensions: {missing_dims}"
                )
            # Reorder dimensions to match the specified order
            var = var.transpose(*dim_order)
            stacked.append(var)

        # Stack along a new dimension
        return np.stack([var.values for var in stacked], axis=-1)

    def unstack_from_ndarray(
        self,
        array: np.ndarray,
        var_names: list[str] | None = None,
        original_dataset: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Convert a stacked array back to a dataset.

        Args:
            array: Stacked array with shape (..., n_vars)
            var_names: Optional list of variable names to use
            original_dataset: Optional dataset to copy attributes from

        Returns:
            Dataset with variables unstacked from array

        Raises:
            ValueError: If array shape doesn't match schema
        """
        if var_names is None:
            var_names = list(self.data_vars.keys())
        else:
            # Validate that all variable names are valid Python identifiers
            if not all(isinstance(vn, str) and vn.isidentifier() for vn in var_names):
                raise ValueError("All variable names must be valid Python identifiers")

            # Validate that all variable names exist in the schema
            invalid_vars = set(var_names) - set(self.data_vars.keys())
            if invalid_vars:
                raise ValueError(f"Variables not found in schema: {invalid_vars}")

        # Get dimensions from the first variable's check_dims
        dims = self._get_dims_from_checks(var_names[0])
        if dims is None:
            raise ValueError("No dimension order found in schema")

        # Verify all variables have the same dimensions
        for var_name in var_names[1:]:
            var_dims = self._get_dims_from_checks(var_name)
            if var_dims != dims:
                raise ValueError(
                    f"Variable {var_name} has different dimensions than {var_names[0]}"
                )

        # Verify array shape matches dimensions
        if array.shape[-1] != len(var_names):
            raise ValueError(
                f"Last dimension of array ({array.shape[-1]}) doesn't match number of variables ({len(var_names)})"
            )

        # Create a dataset from the unstacked array
        data_vars = {}
        for i, var_name in enumerate(var_names):
            # Get attributes from original dataset if available
            attrs = {}
            if original_dataset is not None and var_name in original_dataset:
                attrs = original_dataset[var_name].attrs.copy()
            else:
                # Use schema metadata
                attrs = self.data_vars[var_name].metadata.copy()

            data_vars[var_name] = xr.DataArray(
                array[..., i], dims=dims, name=var_name, attrs=attrs
            )

        return xr.Dataset(data_vars)

    def coerce_dtypes(self, dataset: xr.Dataset) -> xr.Dataset:
        """Coerce dataset variables to their schema-specified dtypes.

        Args:
            dataset: Dataset to coerce

        Returns:
            Dataset with coerced dtypes

        Raises:
            ValueError: If coercion fails
        """
        result = dataset.copy(deep=True)

        # Coerce data variables
        for name, var_schema in self.data_vars.items():
            if name in result.data_vars:
                try:
                    result[name] = result[name].astype(var_schema.dtype)
                except Exception as e:
                    raise ValueError(
                        f"Failed to coerce {name} to {var_schema.dtype}: {e}"
                    )

        # Coerce coordinates
        for name, coord_schema in self.coords.items():
            if name in result.coords:
                try:
                    result[name] = result[name].astype(coord_schema.dtype)
                except Exception as e:
                    raise ValueError(
                        f"Failed to coerce {name} to {coord_schema.dtype}: {e}"
                    )

        return result

    def _get_range_from_checks(self, var_name: str) -> tuple[float, float] | None:
        """Get min/max range from a variable's check_range check.

        Args:
            var_name: Name of the variable to get range for.

        Returns:
            Tuple of (min_val, max_val) if found, None otherwise.
        """
        for check in self.data_vars[var_name].checks:
            if hasattr(check, "__name__") and check.__name__.startswith("check_range"):
                if check.__closure__ is not None and len(check.__closure__) >= 2:
                    # Get both values from closure
                    val1 = check.__closure__[0].cell_contents
                    val2 = check.__closure__[1].cell_contents
                    # Return them in the correct order (min, max)
                    return (min(val1, val2), max(val1, val2))
        return None

    def clip_values(self, dataset: xr.Dataset) -> xr.Dataset:
        """Clip dataset values to their schema-specified ranges.

        Args:
            dataset: Dataset to clip

        Returns:
            Dataset with clipped values
        """
        result = dataset.copy(deep=True)

        # Find range checks for each variable
        for name, var_schema in self.data_vars.items():
            if name not in result.data_vars:
                continue

            # Get range from checks
            range_check = self._get_range_from_checks(name)
            if range_check is not None:
                min_val, max_val = range_check
                assert min_val <= max_val, f"Invalid range: {min_val} <= {max_val}"
                # Create a copy of the values to avoid modifying the original
                values = result[name].values.copy()
                # Clip only the out-of-range values
                values[values < min_val] = min_val
                values[values > max_val] = max_val
                result[name].values = values

        return result

    def _get_max_missing_from_checks(self, var_name: str) -> float | None:
        """Get max_missing threshold from a variable's check_max_missing check.

        Args:
            var_name: Name of the variable to get max_missing for.

        Returns:
            Max missing threshold if found, None otherwise.
        """
        for check in self.data_vars[var_name].checks:
            if check.__name__ == "check_max_missing":
                if check.__closure__ is not None and len(check.__closure__) > 0:
                    return check.__closure__[0].cell_contents
        return None

    def fill_missing(
        self,
        dataset: xr.Dataset,
        method: str = "interpolate",
        dim: str = "time",
        interpolate_method: InterpOptions = "linear",
    ) -> xr.Dataset:
        """Fill missing values in dataset according to schema constraints.

        Args:
            dataset: Dataset to fill
            method: Method to use for filling ('ffill', 'bfill', or 'interpolate')
            dim: Dimension along which to fill missing values (default: "time")
            interpolate_method: Method to use for interpolation when method='interpolate'.
                              Options: 'linear', 'nearest', 'zero', 'slinear', 'quadratic',
                              'cubic', 'polynomial', 'barycentric', 'krogh', 'piecewise_polynomial',
                              'spline', 'pchip', 'akima'. Default is 'linear'.

        Returns:
            Dataset with filled missing values

        Raises:
            ValueError: If method is invalid
            ImportError: If bottleneck is not installed (required for ffill/bfill)
        """
        if method in ["ffill", "bfill"]:
            try:
                import bottleneck  # noqa: F401
            except ImportError:
                raise ImportError(
                    "bottleneck is required for forward/backward fill operations. "
                    "Please install it with: pip install bottleneck"
                )

        if method not in ["ffill", "bfill", "interpolate"]:
            raise ValueError("method must be one of: ffill, bfill, interpolate")

        result = (
            dataset.copy()
        )  # Shallow copy is sufficient since fill operations return new DataArrays

        # Fill missing values for all variables
        for name, var_schema in self.data_vars.items():
            if name not in result.data_vars:
                continue
            # Fill missing values
            if method == "ffill":
                result[name] = result[name].ffill(dim=dim)
            elif method == "bfill":
                result[name] = result[name].bfill(dim=dim)
            else:  # interpolate
                result[name] = result[name].interpolate_na(
                    dim=dim, method=interpolate_method
                )

        return result

    @classmethod
    def infer_schema(
        cls,
        dataset: xr.Dataset,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> "XArraySchema":
        """Infer a schema from a dataset.

        Args:
            dataset: The dataset to infer the schema from
            name: Optional name for the schema
            title: Optional title for the schema
            description: Optional description for the schema

        Returns:
            An XArraySchema instance inferred from the dataset
        """
        if not isinstance(dataset, xr.Dataset):
            raise ValueError(f"Expected Dataset, got {type(dataset)}")

        # Infer data variables
        data_vars = {}
        for vname, var in dataset.data_vars.items():
            dtype = var.dtype
            dims = list(var.dims)
            checks: list[Callable[[xr.DataArray], bool]] = []
            if np.issubdtype(dtype, np.number):
                min_val = float(var.min().values)
                max_val = float(var.max().values)
                checks.append(check_range(min_val, max_val))
            checks.append(check_dims(dims))
            checks.append(check_dtype(dtype))
            if "standard_name" in var.attrs:
                checks.append(check_cf_standard_name(str(var.attrs["standard_name"])))
            if "units" in var.attrs:
                checks.append(check_cf_units(str(var.attrs["units"])))
            data_vars[str(vname)] = DataVariable(
                dtype=dtype,
                checks=to_checklist(checks),
                required=True,
                name=str(vname),
                metadata=var.attrs,
            )

        # Infer coordinates
        coords = {}
        for cname, coord in dataset.coords.items():
            dtype = coord.dtype
            dims = list(coord.dims)
            checks: list[Callable[[xr.DataArray], bool]] = []
            if np.issubdtype(dtype, np.number):
                min_val = float(coord.min().values)
                max_val = float(coord.max().values)
                checks.append(check_range(min_val, max_val))
            checks.append(check_dims(dims))
            checks.append(check_dtype(dtype))
            if "standard_name" in coord.attrs:
                checks.append(check_cf_standard_name(str(coord.attrs["standard_name"])))
            coords[str(cname)] = Coordinate(
                dtype=dtype,
                checks=to_checklist(checks),
                required=True,
                name=str(cname),
                metadata=coord.attrs,
            )

        # Infer attributes
        attrs = {}
        for aname, value in dataset.attrs.items():
            dtype = type(value)
            attrs[str(aname)] = Attribute(
                dtype=dtype,
                checks=[],  # Always pass a list
                required=True,
                name=str(aname),
            )

        return cls(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs,
            name=name,
            title=title,
            description=description,
        )

    def to_code(self) -> str:
        """Generate Python code to recreate this schema.

        Returns:
            A string containing Python code that recreates this schema
        """
        code = [
            "import numpy as np",
            "from nv_dfm_lib_common.schemas.xarray import XArraySchema, DataVariable, Coordinate, Attribute",
            "from nv_dfm_lib_common.schemas.xarray.checks import (",
            "    check_dims,",
            "    check_dtype,",
            "    check_range,",
            "    check_cf_standard_name,",
            "    check_cf_units,",
            "    check_max_missing,",
            ")",
            "",
            "schema = XArraySchema(",
        ]

        # Add data variables
        if self.data_vars:
            code.append("    data_vars={")
            for name, var in self.data_vars.items():
                code.append(f"        '{name}': DataVariable(")
                code.append(f"            dtype={self._dtype_to_code(var.dtype)},")
                code.append("            checks=[")
                for check in var.checks:
                    if hasattr(check, "dims"):
                        code.append(f"                check_dims({check.dims}),")
                    elif hasattr(check, "min_value") and hasattr(check, "max_value"):
                        code.append(
                            f"                check_range({check.min_value}, {check.max_value}),"
                        )
                    elif hasattr(check, "standard_name"):
                        code.append(
                            f"                check_cf_standard_name('{check.standard_name}'),"
                        )
                    elif hasattr(check, "units"):
                        code.append(f"                check_cf_units('{check.units}'),")
                code.append("            ],")
                code.append("            required=True,")
                code.append(f"            name='{name}',")
                if var.metadata:
                    code.append("            metadata={")
                    for k, v in var.metadata.items():
                        code.append(f"                '{k}': {repr(v)},")
                    code.append("            },")
                code.append("        ),")
            code.append("    },")

        # Add coordinates
        if self.coords:
            code.append("    coords={")
            for name, coord in self.coords.items():
                code.append(f"        '{name}': Coordinate(")
                code.append(f"            dtype={self._dtype_to_code(coord.dtype)},")
                code.append("            checks=[")
                for check in coord.checks:
                    if hasattr(check, "dims"):
                        code.append(f"                check_dims({check.dims}),")
                    elif hasattr(check, "min_value") and hasattr(check, "max_value"):
                        code.append(
                            f"                check_range({check.min_value}, {check.max_value}),"
                        )
                    elif hasattr(check, "standard_name"):
                        code.append(
                            f"                check_cf_standard_name('{check.standard_name}'),"
                        )
                    elif hasattr(check, "units"):
                        code.append(f"                check_cf_units('{check.units}'),")
                code.append("            ],")
                code.append("            required=True,")
                code.append(f"            name='{name}',")
                if coord.metadata:
                    code.append("            metadata={")
                    for k, v in coord.metadata.items():
                        code.append(f"                '{k}': {repr(v)},")
                    code.append("            },")
                code.append("        ),")
            code.append("    },")

        # Add attributes
        if self.attrs:
            code.append("    attrs={")
            for name, attr in self.attrs.items():
                code.append(f"        '{name}': Attribute(")
                code.append(f"            dtype={attr.dtype.__name__},")
                code.append("            checks=[],")
                code.append("            required=True,")
                code.append(f"            name='{name}',")
                code.append("        ),")
            code.append("    },")

        # Add schema metadata
        if self.name:
            code.append(f"    name='{self.name}',")
        if self.title:
            code.append(f"    title='{self.title}',")
        if self.description:
            code.append(f"    description='{self.description}',")
        if self.metadata:
            code.append("    metadata={")
            for k, v in self.metadata.items():
                code.append(f"        '{k}': {repr(v)},")
            code.append("    },")

        code.append(")")

        return "\n".join(code)

    def _dtype_to_code(self, dtype) -> str:
        """Helper to convert dtype to code string for to_code method."""
        if np.issubdtype(dtype, np.floating):
            return f"np.{np.dtype(dtype).name}"
        elif np.issubdtype(dtype, np.integer):
            return f"np.{np.dtype(dtype).name}"
        elif np.issubdtype(dtype, np.bool_):
            return "np.bool_"
        elif np.issubdtype(dtype, np.str_):
            return "np.str_"
        elif np.issubdtype(dtype, np.datetime64):
            return "np.datetime64"
        else:
            # fallback to generic dtype string
            return f"np.dtype('{str(dtype)}')"
