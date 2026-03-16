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

from abc import ABC, abstractmethod
from typing import Any

from pydantic import ConfigDict, Field, field_validator
from typing_extensions import override

from nv_dfm_core.api._place_param import PlaceParam
from nv_dfm_core.api.pydantic import PolymorphicBaseModel

from ._node_id import NodeId, NodeRef
from ._pipeline_build_helper import PipelineBuildHelper


class Statement(PolymorphicBaseModel, ABC):
    """A Statement is the base class for all syntactic constructs that can appear
    in a Pipeline body. A Statement does not have a node_id, therefore it cannot
    be referenced by other nodes (it's a syntactic construct without a value)"""

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True)  # pyright: ignore[reportIncompatibleVariableOverride]

    # Every statement node should have a node_id, we need that for IR generation.
    # But it should only be possible
    # with expression nodes to pass them as a parameter to another statement, because
    # only expressions have a value.
    # by default, get a fresh node ID. But the user can set a well_known_node_id if needed
    dfm_node_id: NodeId = Field(default_factory=PipelineBuildHelper.get_fresh_node_id)
    # By default, the dfm orders statements only by their data flow. If two statements
    # must be ordered but there is no explicit data flow between them, you can use the
    # after field to enforce this ordering.
    dfm_after: "Statement | NodeId | None" = None

    @field_validator("*", mode="before")
    @classmethod
    def rewrite_statement_object_to_noderef(cls, v: Any, ctx: Any) -> Any:
        """Replace a Statement object reference with its corresponding node id,
        which is what's actually stored in the object.
        Note that technically only Expressions have values. Our Pydantic models
        use NodeRef as the type, which enforces fields to be Expressions on the Pydantic level.
        But we rewrite all Statement fields here, because like the after field above,
        we may want the user to pass/reference nodes that are not Expressions."""
        if ctx.field_name == "dfm_body":
            # don't rewrite the list of statements in the block body, that's the only place where
            # we actually want to keep the Statement objects and don't replace them with NodeRefs
            return v

        if isinstance(v, Statement):
            return v.dfm_node_id.to_ref()
        elif isinstance(v, list):
            # rewrite all Expression objects to their node_id
            return [
                e.dfm_node_id.to_ref() if isinstance(e, Statement) else e for e in v
            ]  # pyright: ignore[reportUnknownVariableType]
        elif isinstance(v, tuple):
            # rewrite all Expression objects to their node_id
            return tuple(
                [
                    e.dfm_node_id.to_ref() if isinstance(e, Statement) else e
                    for e in v  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
                ]
            )
        elif isinstance(v, dict):
            # rewrite all Expression objects to their node_id
            return {
                k: (e.dfm_node_id.to_ref() if isinstance(e, Statement) else e)  # pyright: ignore[reportUnknownVariableType]
                for k, e in v.items()  # pyright: ignore[reportUnknownVariableType]
            }
        return v

    @override
    def model_post_init(self, __context: Any):
        # We want to prevent users to accidentally instantiate Operation
        # outside a pipeline to prevent errors. But on the server side
        # we need to load the model from json, in which case we don't
        # want to do this
        if PipelineBuildHelper.build_helper_active():
            PipelineBuildHelper.get_current_block().add_to_body(self)

    def get_noderef_and_placeparam_pydantic_fields(
        self,
    ) -> list[tuple[str, NodeRef | PlaceParam]]:
        """Returns all pydantic fields that have a NodeRef or PlaceParam value.
        Those are fields that get their values from references and not from literals."""
        result: list[tuple[str, NodeRef | PlaceParam]] = []
        for fieldname in self.model_fields_set:
            if hasattr(self, fieldname):
                value = getattr(self, fieldname)
                if isinstance(value, NodeRef) or isinstance(value, PlaceParam):
                    result.append((fieldname, value))
        return result

    @abstractmethod
    def accept(self, visitor: Any) -> None:
        pass
