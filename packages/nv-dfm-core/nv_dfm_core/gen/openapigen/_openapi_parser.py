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

import os
from typing import Any

from datamodel_code_generator import (
    DataModelType,
    Error,
    LiteralType,
    OpenAPIScope,
    PythonVersion,
)
from datamodel_code_generator.model import get_data_model_types
from datamodel_code_generator.parser.openapi import (
    OpenAPIParser,
    Operation,
    ReferenceObject,
    RequestBodyObject,
)
from datamodel_code_generator.types import DataType
from pydantic import BaseModel

data_model_types = get_data_model_types(
    DataModelType.PydanticV2BaseModel,  # or PydanticV1BaseModel for Pydantic v1
    target_python_version=PythonVersion.PY_312,
)


class ParsedOperation(BaseModel):
    operation_id: str
    parameters: DataType | None
    request_body: dict[str, DataType]
    responses: dict[str | int, dict[str, DataType]]
    tags: list[str]


class DFMOpenAPIParser(OpenAPIParser):
    def __init__(self, openapi_file_path: str):
        self.openapi_file_path = openapi_file_path
        if not os.path.exists(self.openapi_file_path):
            raise FileNotFoundError(f"OpenAPI file not found: {self.openapi_file_path}")

        super().__init__(
            open(openapi_file_path).read(),
            data_model_type=data_model_types.data_model,
            data_model_root_type=data_model_types.root_model,
            data_model_field_type=data_model_types.field_model,
            data_type_manager_type=data_model_types.data_type_manager,
            dump_resolve_reference_action=data_model_types.dump_resolve_reference_action,
            reuse_model=True,
            enum_field_as_literal=LiteralType.One,
            treat_dot_as_module=True,
            field_constraints=True,
            use_union_operator=True,
            collapse_root_models=True,
            parent_scoped_naming=True,
            openapi_scopes=[
                OpenAPIScope.Schemas,
                OpenAPIScope.Paths,
                OpenAPIScope.Parameters,
            ],
            include_path_parameters=True,
            use_operation_id_as_name=False,
        )

        self.parsed_operations: dict[str, ParsedOperation] = {}

    def parse_operation(
        self,
        raw_operation: dict[str, Any],
        path: list[str],
    ) -> None:
        """Extension of the parents method that returns parsed parameters, request body and responses.

        Args:
            raw_operation (dict[str, Any]): _description_
            path (list[str]): _description_

        Raises:
            Error: _description_
        """
        operation = Operation.model_validate(raw_operation)
        parsed_params: DataType | None = None
        parsed_request_body: dict[str, DataType] = {}
        parsed_response: dict[str | int, dict[str, DataType]] = {}
        parsed_tags: list[str] = []
        path_name, method = path[-2:]
        if self.use_operation_id_as_name:
            if not operation.operationId:
                msg = (
                    "All operations must have an operationId when --use_operation_id_as_name is set."
                    "The following path was missing an operationId: {path_name}"
                )  # type: ignore
                raise Error(msg)
            path_name = operation.operationId
            method = ""
        parsed_params = self.parse_all_parameters(
            self._get_model_name(
                path_name,
                method,
                suffix="Parameters"
                if self.include_path_parameters
                else "ParametersQuery",
            ),
            operation.parameters,
            [*path, "parameters"],
        )
        if operation.requestBody:
            if isinstance(operation.requestBody, ReferenceObject):
                ref_model = self.get_ref_model(operation.requestBody.ref)
                request_body = RequestBodyObject.model_validate(ref_model)
            else:
                request_body = operation.requestBody
            parsed_request_body = self.parse_request_body(
                name=self._get_model_name(path_name, method, suffix="Request"),
                request_body=request_body,
                path=[*path, "requestBody"],
            )
        parsed_response = self.parse_responses(
            name=self._get_model_name(path_name, method, suffix="Response"),
            responses=operation.responses,
            path=[*path, "responses"],
        )
        if OpenAPIScope.Tags in self.open_api_scopes:
            parsed_tags = self.parse_tags(
                name=self._get_model_name(path_name, method, suffix="Tags"),
                tags=operation.tags,
                path=[*path, "tags"],
            )

        if operation.operationId in self.parsed_operations:
            raise ValueError(f"Repeating operation ID {operation.operationId}")

        self.parsed_operations[operation.operationId] = ParsedOperation(
            operation_id=operation.operationId or "",
            parameters=parsed_params,
            request_body=parsed_request_body,
            responses=parsed_response,
            tags=parsed_tags,
        )
