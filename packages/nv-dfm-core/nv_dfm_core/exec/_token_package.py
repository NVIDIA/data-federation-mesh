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

import math
from typing import Any

from pydantic import BaseModel, JsonValue

from ._frame import Frame


class TokenPackage(BaseModel):
    """This is an envelope to send data between the sites.

    The trace_context field enables distributed tracing across sites.
    When a token is sent, the current trace context is captured and
    propagated to the receiving site, allowing spans to be correlated
    across the entire federation.
    """

    source_site: str
    source_node: int | str | None
    source_job: str
    target_site: str
    target_place: str
    target_job: str
    is_yield: bool
    frame: Frame
    tagged_data: tuple[str, JsonValue]

    # Telemetry: trace context for distributed tracing (optional, backward compatible)
    # Contains trace_id, span_id, and trace_flags for W3C Trace Context compatibility
    trace_context: dict[str, str] | None = None

    def unwrap_data(self) -> Any:
        return tagged_json_value_to_object(self.tagged_data)

    @staticmethod
    def wrap_data(
        source_site: str,
        source_node: int | str | None,
        source_job: str,
        target_site: str,
        target_place: str,
        target_job: str,
        is_yield: bool,
        frame: Frame,
        data: Any,
        trace_context: dict[str, str] | None = None,
    ) -> "TokenPackage":
        return TokenPackage(
            source_site=source_site,
            source_node=source_node,
            source_job=source_job,
            target_site=target_site,
            target_place=target_place,
            target_job=target_job,
            is_yield=is_yield,
            frame=frame,
            tagged_data=any_object_to_tagged_json_value(data),
            trace_context=trace_context,
        )


def is_json_serializable(obj: Any, _seen: set[int] | None = None) -> bool:
    """Check if an object is JSON serializable without actually serializing it.

    This performs a conservative check that guarantees json.dumps() will succeed
    with default settings if this function returns True.
    """
    if _seen is None:
        _seen = set()

    # Primitives
    if obj is None or obj is True or obj is False:
        return True
    if isinstance(obj, (str, int)):
        return True
    if isinstance(obj, float):
        # Reject NaN and Infinity (json.dumps() default behavior)
        return math.isfinite(obj)

    # Check for circular references
    obj_id = id(obj)
    if obj_id in _seen:
        return False

    # Containers
    if isinstance(obj, (list, tuple)):
        _seen.add(obj_id)
        result = all(is_json_serializable(item, _seen) for item in obj)  # type: ignore
        _seen.remove(obj_id)
        return result

    if isinstance(obj, dict):
        _seen.add(obj_id)
        # Check keys are valid types
        for key in obj.keys():  # type: ignore
            if not isinstance(key, (str, int, float, bool, type(None))):
                _seen.remove(obj_id)
                return False
        # Check all values are serializable
        result = all(is_json_serializable(value, _seen) for value in obj.values())  # type: ignore
        _seen.remove(obj_id)
        return result

    return False


def any_object_to_tagged_json_value(obj: Any) -> tuple[str, JsonValue]:
    # returns (tag, JsonValue) where tag tells us how it was serialized
    # JsonValue is union of int, str, float, dict, list
    # only pickle if really necessary, otherwise send data more directly
    if isinstance(obj, BaseModel):
        # try to avoid pickling BaseModel, that fails on some systems
        # We must remember the actual class for deserialization, Pydantic doesn't know
        payload = {
            "class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
            "data": obj.model_dump(mode="json"),
        }
        return ("BaseModel", payload)  # type: ignore
    elif is_json_serializable(obj):
        # JSON-serializable: return as-is to avoid string conversion
        # (which could collide with NVFlare's {var} substitution)
        if isinstance(obj, tuple):
            # Convert tuple to list since JsonValue doesn't include tuple
            return ("json", list(obj))  # type: ignore
        else:
            return ("json", obj)  # type: ignore
    else:
        # Not JSON-serializable, pickle it
        from ..api._pickled_object import PickledObject

        serialized = PickledObject(value=obj)
        json_dict = serialized.model_dump()
        return ("PickledObject", json_dict)


def tagged_json_value_to_object(val: tuple[str, JsonValue]) -> Any:
    tag, jv = val
    if tag == "BaseModel":
        assert isinstance(jv, dict)
        class_path = jv["class"]
        assert isinstance(class_path, str)
        try:
            module_name, class_name = class_path.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except (AttributeError, ImportError):
            raise ValueError(f"Could not locate class: {class_path}")
        # Validate the data using the class
        if not issubclass(cls, BaseModel):
            raise ValueError(f"Located class {class_path} is not a Pydantic model")
        return cls.model_validate(jv["data"])
    elif tag == "json":
        # Already a dict/list/primitive, return as-is
        return jv
    elif tag == "PickledObject":
        assert isinstance(jv, dict)
        # unpickle from base64
        from ..api._pickled_object import PickledObject

        return PickledObject.model_validate(jv).value
    else:
        raise ValueError(f"Unknown tag: {tag} in tagged_json_value_to_object")
