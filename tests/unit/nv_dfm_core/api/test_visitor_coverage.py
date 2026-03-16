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

import re

import pytest

from nv_dfm_core.api import ApiVisitor, Operation, Statement


def camel_to_snake(name):
    # Convert CamelCase to snake_case
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def all_subclasses(cls):
    """Recursively find all subclasses of a class."""
    subclasses = set(cls.__subclasses__())
    for subclass in cls.__subclasses__():
        subclasses.update(all_subclasses(subclass))
    return subclasses


def test_pipeline_visitor_coverage():
    visitor_methods = dir(ApiVisitor)
    missing = []
    for cls in all_subclasses(Statement):
        # Only check subclasses defined in the dfm.api package
        if cls.__module__.startswith("nv_dfm_core.api"):
            is_abstract = bool(getattr(cls, "__abstractmethods__", False))
            # Always require visit method for Operation, even if abstract
            if not is_abstract or cls is Operation:
                method_name = f"visit_{camel_to_snake(cls.__name__)}"
                if method_name not in visitor_methods:
                    missing.append((cls.__name__, method_name))
    if missing:
        missing_str = ", ".join(
            f"{name} (expected {method})" for name, method in missing
        )
        pytest.fail(f"PipelineVisitor is missing visit methods for: {missing_str}")
