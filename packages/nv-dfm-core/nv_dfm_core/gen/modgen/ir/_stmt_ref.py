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

from pydantic import BaseModel


class StmtRef(BaseModel):
    """Using a ssa value as a parameter to another
    statement. Var can refer to a field inside var, such as var[0] or
    var.field or var["key"]."""

    stmt_id: str
    sel: str | int | None = None
    issubs: bool = True

    def to_python(self) -> str:
        if isinstance(self.sel, int):
            assert self.issubs, (
                f"Cannot use index selection on non-subscriptable variable {self.stmt_id}"
            )
            return f"{self.stmt_id}[{self.sel}]"
        elif isinstance(self.sel, str):
            if self.issubs:
                return f'{self.stmt_id}["{self.sel}"]'
            else:
                return f"{self.stmt_id}.{self.sel}"
        else:
            return self.stmt_id
