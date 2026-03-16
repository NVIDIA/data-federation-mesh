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

"""A BaseModel extension adding automatic polymorphism to Pydantic."""

from pydoc import locate
from typing import Any, Callable

from pydantic import BaseModel, model_serializer, model_validator
from pydantic_core import PydanticCustomError


class PolymorphicBaseModel(BaseModel):
    """
    Pydantic does not support polymorphism out of the box, which makes
    this approach necessary. That is, if a subclass of Bar, Foo(Bar), is
    deserialized json = Foo().model_dump_json() and then reserialized
    as the base class: model = Bar().model_validate_json(json) you actually
    get a Bar object back, not the original Foo.

    The PolymorphicBaseModel will check if the data contains a 'dfm_class_name' field during
    model validation. If yes, the dfm_class_name value will be interpreted as the actual class name.
    Pydantic doesn't support this, and when de-serializing a field declared as List[Function]
    Pydantic will instantiate actual Function objects, instead of the correct subclass."""

    @classmethod
    def _discriminator_name(cls) -> str:
        return "dfm_class_name"

    @classmethod
    def _rewrite_discriminator_value_to_model_class(
        cls, module_path: str, dfm_class_name: str
    ) -> tuple[str, str]:
        # Backwards compatibility: rewrite old 'dfm.' paths to 'nv_dfm_core.'
        if module_path.startswith("dfm."):
            module_path = "nv_dfm_core." + module_path[4:]
        elif module_path == "dfm":
            module_path = "nv_dfm_core"
        return (module_path, dfm_class_name)

    @model_validator(mode="wrap")
    @classmethod
    def _replace_with_tagged_class(
        cls, values: dict[str, Any] | Any, handler: Callable[[Any], Any], _info: Any
    ) -> Any:
        """When using polymorphic base models with a discriminator tag, a user model will
        often not specify the concrete leaf class as the type, but use an abstract base model.
        For example, a "Script" BaseModel may have a field funcs: List[Function] and would expect
        that this array contains not actual Function objects, but the correct subclasses
        of Function. However, pydantic would instantiate actual Function objects in this list,
        instead of the correct subclasses.
        This model_validator runs in this case on the Function class that Pydantic would try
        to instantiate and checks if there is a type tag in the data. If yes, it will get the
        Python BaseModel class from this tag and run model_validate on there.
        """
        if not isinstance(values, dict):
            return handler(values)

        # this happens when a user doesn't explicitly specify the api key (which they shouldn't)
        # We simply let Pydantic take care of this; we only do polymorphic stuff when the api
        # key is provided explicitly
        if cls._discriminator_name() not in values:
            return handler(values)

        try:
            klass_value: Any = values[cls._discriminator_name()]  # pyright: ignore[reportUnknownVariableType]
            assert isinstance(klass_value, str), "Discriminator value must be a string"
            module_path, dfm_class_name = klass_value.rsplit(".", 1)
            # give the class a chance to rewrite the classname written in the PydanticModel
            module_path, dfm_class_name = (
                cls._rewrite_discriminator_value_to_model_class(
                    module_path, dfm_class_name
                )
            )
        except ValueError as ex:
            raise PydanticCustomError(
                "PolymorphicBaseModel",
                "PolymorphicBaseModel: The discriminator literal doesn't look like a module path",
            ) from ex

        try:
            the_classname = f"{module_path}.{dfm_class_name}"
            the_class = locate(the_classname)
            if the_class == cls:
                return handler(values)
            else:
                if not isinstance(the_class, type):
                    raise ValueError(
                        f"PolymorphicBaseModel trying to instantiate {the_classname}."
                        + f" Found object {the_class} of type {type(the_class)} but expected a type"
                    )
                if not issubclass(the_class, BaseModel):
                    raise ValueError(
                        f"PolymorphicBaseModel trying to instantiate {the_classname}."
                        + f" Found object {the_class} of type {type(the_class).__name__} but expected a"  # pyright: ignore[reportUnknownArgumentType]
                        + " subclass of BaseModel."
                    )
                return the_class.model_validate(values)
        except AttributeError as ex:
            raise PydanticCustomError(
                "PolymorphicBaseModel-object",
                "PolymorphicBaseModel: Could not instantiate class",
            ) from ex

    @model_serializer(mode="wrap", when_used="always")
    def serialize_model(
        self, _handler: Callable[[Any], Any], _info: Any
    ) -> dict[str, Any]:
        """Pydantic serializes a field like List[FunctionParam] as actual
        FunctionParam objects, not as the concrete subclass, missing out fields from the
        subclass. This model serializer adds back those missing fields"""
        # the pydantic handler will only serialize the base class fields
        d = {field: getattr(self, field) for field in type(self).model_fields}
        return d
