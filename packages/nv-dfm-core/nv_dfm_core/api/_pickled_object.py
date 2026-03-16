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

import base64
import json
import pickle
from binascii import Error as BinAsciiError
from typing import Any, Literal

from pydantic import BaseModel, model_validator
from typing_extensions import Self, override


class PickledObject(BaseModel):
    """
    A class that can serialize and deserialize a Python object to a JSON string by pickling
    and base64 encoding it.
    """

    tag: Literal["__dfm__pickled__"] = "__dfm__pickled__"
    value: Any

    @model_validator(mode="before")
    @classmethod
    def validate_value(cls, data: dict[str, Any] | Any) -> Any:
        """Validate that the value can be pickled before accepting it."""
        if isinstance(data, dict) and "value" in data:
            v: Any = data["value"]  # pyright: ignore[reportUnknownVariableType]
            if v is not None:
                try:
                    _ = pickle.dumps(v)
                except (pickle.PickleError, TypeError, AttributeError) as e:
                    raise ValueError(f"Value cannot be pickled: {str(e)}")
        return data  # pyright: ignore[reportUnknownVariableType]

    @override
    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialize the object by pickling and base64 encoding it."""
        pickled = pickle.dumps(self.value)
        encoded = base64.b64encode(pickled).decode("utf-8")
        return {"tag": self.tag, "value": encoded}

    @override
    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        return json.dumps(self.model_dump(*args, **kwargs))

    @classmethod
    @override
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Deserialize the object by base64 decoding and unpickling it."""
        if isinstance(obj, dict) and "value" in obj:
            try:
                assert isinstance(obj["value"], str), "Value must be a string"
                decoded = base64.b64decode(obj["value"].encode("utf-8"))
                value = pickle.loads(decoded)
                return cls(value=value)
            except (BinAsciiError, pickle.UnpicklingError) as e:
                raise ValueError(f"Failed to decode/unpickle value: {str(e)}")
        return cls(value=obj)

    @classmethod
    @override
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        return cls.model_validate(
            obj=json.loads(json_data),
            strict=strict,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )
