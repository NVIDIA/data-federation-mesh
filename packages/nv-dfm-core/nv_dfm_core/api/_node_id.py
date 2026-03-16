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

from typing import Any, Literal

from pydantic import BaseModel, field_validator
from typing_extensions import override


class NodeId(BaseModel, frozen=True):
    """
    NodeId uniquely identifies a node in a Pipeline graph. The NodeId is used at the
    node itself. Other nodes use NodeRef, not NodeId, to reference this node. NodeRef is different
    from the NodeId in that it adds a possible `selector` to reference individual
    fields of the node's result.
    """

    tag: Literal["@dfm-node"] = "@dfm-node"
    ident: int | str

    @field_validator("ident")
    @classmethod
    def validate_ident(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.isidentifier():
            raise ValueError(f"NodeId.ident '{v}' is not a valid identifier")
        return v

    @override
    def __str__(self) -> str:
        nid = f"%{self.ident}" if isinstance(self.ident, int) else f"#{self.ident}"
        return f"{nid}"

    def to_ref(self, sel: int | str | None = None, issubs: bool | None = None):
        """
        Create a NodeRef from a NodeId. If issubs is not provided, it is inferred from
        whether sel is an int or a str. For ints we assume a subscript like
        some_val[42] whereas for strings we assume a field selector like some_val.myname
        """
        if not issubs:
            # if sel is None it doesn't really matter what issubs is, but
            # we use True as the default (e.g. in testing), so let's use True
            issubs = True if sel is None else isinstance(sel, int)
        return NodeRef(id=self, sel=sel, issubs=issubs)

    def as_identifier(self) -> str:
        """Returns this node_id as a valid identifier"""
        if isinstance(self.ident, int):
            return f"node{self.ident}"
        else:
            assert self.ident.isidentifier(), (
                f"Node ID {self.ident} is not a valid identifier"
            )
            return self.ident


class NodeRef(BaseModel, frozen=True):
    """A NodeRef is used to identify (reference) a node in a pipeline when passing
    the output of a node as a parameter to another node.
    NodeRef is a tuple ['@dfmref', NodeId, Field]. The first element is a magic
    string that helps identifying a tuple as a node ref. The NodeID is the id
    of the referenced node. The optional Field is a field inside the node's result
    that should get passed as a parameter. If None, the whole result object is passed.
    But for example `op = OtherOp(name=['@dfmref', 42, 'filename'])` would only pass
    the 'filename' field of node 42's result."""

    id: NodeId
    sel: int | str | None = None
    # if sel is a string, issubs=True is ref[sel] whereas issubs=False is ref.sel
    issubs: bool = True

    @override
    def __str__(self) -> str:
        selector = f".{self.sel}" if self.sel else ""
        return f"{str(self.id)}{selector}"


def make_auto_id(ident: int) -> NodeId:
    """Auto IDs are prefixed with %. Auto NodeIds are managed by PipelineBuildHelper.
    This function should only be used by users if they know what they are doing,
    otherwise there's a risk of naming conflict. Users should use the
    well_known_id variant below"""
    return NodeId(ident=ident)


def well_known_id(ident: int | str) -> NodeId:
    """Well-known IDs (user-defined IDs) are prefixed with #. The user should make
    sure that there is really only one node in a pipeline with this ID"""
    return NodeId(ident=ident if isinstance(ident, str) else f"wkid{ident}")
