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

from typing import TypeAlias

from ._expression import Expression
from ._node_id import NodeRef

NodeParam: TypeAlias = Expression | NodeRef
"""NodeParam is a marker for parameters that accept inputs from other nodes.
That's usually the case because the type is not serializable in
json and therefore must come from another node. 
If an Expression is passed as a parameter, the field_validator
in Statement will replace this Expression object with
its node_id:NodeRef, which is really stored in the pydantic
model. We add Expression here, however, to make pylint happy.
"""
