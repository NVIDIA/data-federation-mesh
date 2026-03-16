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

# pyright: reportUnnecessaryComparison=false
"""
Parser for DFM federation configuration files.

This module takes a FederationObject configuration and generates Python code using
datamodel-code-generator. The parser:

1. Converts JSON Schema objects into Pydantic models
2. Generates operation classes with parameter type options
3. Creates site-specific adapter classes that inherit from operations
4. Handles references to external schemas and operations
5. Adds metadata fields like class names and API names

The main entry point is ConfigParser.parse_federation_object() which processes
the entire federation configuration and generates the corresponding Python code.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar

from datamodel_code_generator import LiteralType, snooper_to_methods
from datamodel_code_generator.model import pydantic as pydantic_model
from datamodel_code_generator.parser.jsonschema import (
    JsonSchemaObject,
    JsonSchemaParser,
    get_model_by_path,
)
from datamodel_code_generator.types import DataType
from pydantic import BaseModel
from typing_extensions import override

from nv_dfm_core.exec import site_name_to_identifier

from ._config_models import (
    AdapterObject,
    FederationObject,
    OperationObject,
    ProviderObject,
    ReferencesToDicts,
    ReferenceToObject,
    SiteObject,
)

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def determine_final_json_schema(
    param_name: str, user_schema: JsonSchemaObject, info_dict: dict[str, Any]
) -> JsonSchemaObject:
    """
    The user supplies the main data type of an operation parameter as a JsonSchemaObject.
    But we also support some additional flags that may change the type. This
    function resolves the user schema and creates a new schema object with a Union type
    if necessary.
    """

    # retrieve the flags from the info_dict
    accepts_literal_input = info_dict.get(
        "accepts-literal-input", info_dict.get("accepts_literal_input", True)
    )
    accepts_node_input = info_dict.get(
        "accepts-node-input", info_dict.get("accepts_node_input", True)
    )
    accepts_place_input = info_dict.get(
        "accepts-place-input", info_dict.get("accepts_place_input", True)
    )
    discoverable = info_dict.get("discoverable", info_dict.get("discover_able", True))

    # add options if necessary
    possible_options: list[JsonSchemaObject] = []
    if accepts_literal_input:
        # did the user specify a type?
        if user_schema.type:
            # yes. Type can be single string or a list, so make it a list
            types: list[str] = (
                [user_schema.type]
                if isinstance(user_schema.type, str)
                else user_schema.type
            )
            # The type list may contain a mix of JsonSchema builtin types and user-defined types.
            # For each type we create a new JsonSchema option by copying.
            allowed_builtin_types = [
                "string",
                "number",
                "integer",
                "boolean",
                "array",
                "object",
                "null",
            ]
            for type_name in types:
                if "." in type_name:
                    # this is a reference to a user-defined type. It's assumed to point to a Pydantic model.
                    # TODO: maybe we should check that it's really a Pydantic model?
                    option = JsonSchemaObject(customTypePath=type_name)
                    possible_options.append(option)
                elif type_name in allowed_builtin_types:
                    # we add each builtin type as a separate option to flatten the Union type
                    # otherwise you get something like Union[Union[bool, float], NodeParam, PlaceParam, Advise]
                    # in the pydantic model
                    option = user_schema.model_copy(update={"type": type_name})
                    possible_options.append(option)
                else:
                    raise ValueError(
                        f"Operation parameter {param_name} tries to use an unknown type: {type_name}."
                        + f" Allowed values for type are JsonSchema builtin types {allowed_builtin_types}"
                        + " as well as the fully-qualified class name of user-defined Pydantic models,"
                        + " which are type strings that contain a dot."
                        + " Note that the magic 'python-object' value was removed. Use the user-defined type"
                        + " 'dfm.api.PickledObject' instead."
                    )
        else:
            # no type given, which implies Any. Nothing to do for us, just append it
            possible_options.append(user_schema)
    if accepts_node_input:
        possible_options.append(
            JsonSchemaObject(customTypePath="nv_dfm_core.api.NodeParam")
        )
    if accepts_place_input:
        possible_options.append(
            JsonSchemaObject(customTypePath="nv_dfm_core.api.PlaceParam")
        )
    # check that we have at least some type
    if len(possible_options) == 0:
        raise ValueError(f"The parameter '{param_name}' does not accept anything.")
    # now add the discoverable option
    if discoverable:
        possible_options.append(
            JsonSchemaObject(customTypePath="nv_dfm_core.api.Advise")
        )

    if len(possible_options) > 1:
        # create a union type
        return JsonSchemaObject(oneOf=possible_options)
    else:
        # only a single option, return it
        return possible_options[0]


def create_param_description(
    param_name: str,
    param_desc: str | None,
    accepts_literal_input: bool,
    accepts_node_input: bool,
    accepts_place_input: bool,
    discoverable: bool,
) -> str:
    """Creates a formatted description for a parameter showing what types it accepts."""
    accepts: list[str] = []
    if accepts_literal_input:
        accepts.append("literal input")
    if accepts_node_input:
        accepts.append("node input")
    if accepts_place_input:
        accepts.append("place input")

    accepts_text = f"Accepts: {', '.join(accepts)}." if len(accepts) > 0 else ""

    discoverable_text = "Discoverable" if discoverable else ""
    marker_text = f" [{accepts_text} {discoverable_text}]"

    param_desc = f"\n        {param_desc}" if param_desc else ""
    return f"    {param_name}:{marker_text}{param_desc}"


def create_model_description(
    description: str | None,
    param_docs: dict[str, tuple[str, bool, bool, bool, bool]],
    returns: str | None,
) -> str:
    """Creates a comprehensive description for an operation model including parameters and return type."""
    desc = description if description else ""

    param_descs: list[str] = []
    for param_name, param_doc in param_docs.items():
        param_descs.append(
            create_param_description(
                param_name=param_name,
                param_desc=param_doc[0],
                accepts_literal_input=param_doc[1],
                accepts_node_input=param_doc[2],
                accepts_place_input=param_doc[3],
                discoverable=param_doc[4],
            )
        )

    # Python 3.10 doesn't allow \ inside f-strings, so pulling it out
    # for backwards compatibility
    newline = "\n"
    param_desc = (
        f"Parameters:{newline}{newline.join(param_descs)}"
        if len(param_descs) > 0
        else ""
    )

    ret = f"Returns: {returns}" if returns else ""

    newlines1 = "\n\n" if desc != "" and (param_desc != "" or ret != "") else ""
    newlines2 = "\n\n" if param_desc != "" and ret != "" else ""

    return f"{desc}{newlines1}{param_desc}{newlines2}{ret}"


def fully_qualified_schema_name(base_package: str, schema_name: str) -> str:
    return f"{base_package}.fed.schema.{schema_name}"


class MyFieldModel(pydantic_model.DataModelField):
    """Using a custom field model that does not fall back to nullable if it has a default value.
    Otherwise we get stuff like:

    site: Optional[Literal['central']] = 'central'

    instead of:

    site: Literal['central'] = 'central'
    """

    @property
    @override
    def fall_back_to_nullable(self) -> bool:
        return False


@snooper_to_methods()
class ConfigParser(JsonSchemaParser):
    """Parser that converts DFM federation configuration into Python code.

    Extends JsonSchemaParser to handle the specific needs of DFM configuration:
    - Operation parameter type options (literal, node, place, discoverable)
    - Site-specific operation adapters
    - Reference resolution for external schemas and operations
    - Generation of metadata fields like class names and API names
    """

    def __init__(
        self,
        base_path: Path | None = None,
    ):
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            source="",
            data_model_field_type=MyFieldModel,
            base_path=base_path,
            # make sure the "description" from the config is added as a comment
            use_schema_description=True,
            # make sure the "description" from the config is added as a comment
            use_field_description=True,
            use_exact_imports=True,
            # We want to generate things like "site: Literal['central'] = 'central'"
            # With this flag, json schema enums with a single element will be translated to this
            enum_field_as_literal=LiteralType.One,
            # and we want the default value to actually be outputted
            apply_default_values_for_required_fields=True,
            # just in case our custom template may need extra info
            extra_template_data=defaultdict(dict),
            # we want to generate fields like
            # __api_name__: Literal['utils.SomeOp'] = 'utils.SomeOp'
            # By default, the hidden __api_name__ field would get a 'field' prefix, we override that.
            special_field_name_prefix="",
            # we want, for example, operations with a name users.GreetMe to be placed
            # in the users module
            treat_dot_as_module=True,
            # we want the generated code to be frozen
            # enable_faux_immutability=True,
            additional_imports=[],
        )
        self.defined_schemas_fully_qualified: list[str] = []

    def resolve_raw_obj(self, ref: str) -> dict[str, Any]:
        """Resolves a JSON reference to its actual object."""
        if self.raw_obj is None:
            raise ValueError(
                "Raw object is not set. Please set the raw object before calling resolve_raw_obj."
            )
        ref_file, ref_path = self.model_resolver.resolve_ref(ref).split("#", 1)
        ref_body = self._get_ref_body(ref_file) if ref_file else self.raw_obj
        if ref_body is None:
            raise ValueError(
                "Could not resolve reference into a valid JSON object, couldn't find referenced file or URL."
                + f" Ref was {ref}, which was split into ref file {ref_file} and ref path {ref_path}, with base path {self.base_path}"
            )
        try:
            resolved = get_model_by_path(ref_body, ref_path.split("/")[1:])
        except Exception as e:
            raise ValueError(
                "Error while resolving"
                + f" ref {ref}, which was split into ref file {ref_file} and ref path {ref_path}, with base path {self.base_path},"
                + f" in the referenced body: {ref_body}."
            ) from e

        if (
            not resolved
        ):  # get_model_by_path returns an empty {} if the path is not found
            raise ValueError(
                f"Could not resolve JSON reference {ref} into a valid json object."
                + f" Local reference path {ref_path} pointed nowhere in the resolved body."
                + f" The resolved body was retrieved from ref file {ref_file} with base path {self.base_path}"
                + f" and has contents: {ref_body}."
            )
        return resolved

    def resolve_object(
        self, obj: str | ReferenceToObject | BaseModelT, object_type: type[BaseModelT]
    ) -> BaseModelT:
        """Resolves an object that might be a reference into the actual object."""
        if isinstance(obj, ReferenceToObject):
            ref_obj = self.resolve_raw_obj(obj.ref)
            return object_type.model_validate(ref_obj)
        elif isinstance(obj, str):
            ref_obj = self.resolve_raw_obj(obj)
            return object_type.model_validate(ref_obj)
        return obj

    def resolve_data_type_object(
        self, base_package: str, obj: str | ReferenceToObject | JsonSchemaObject
    ) -> JsonSchemaObject:
        """Resolves a Json schema object that is used as a data type
        (i.e. a schema parameter or an operation parameter)
        Data types can be defined in-place as a schema object or referenced from somewhere else.
        There's an ambiguity however in whether the referenced object is getting inlined in place
        or if it's referencing a common "central" data type from the schemas section. Example:
        my_op_param1: { type: {$ref: "some_file#/some_path/MyModel"} }
        my_op_param2: { type: {$ref: "some_file#/some_path/MyModel"} }
        here, it's not directly obvious from the $ref whether MyModel is supposed to be
        duplicated (inlined) or not. We resolve this ambiguity by storing all classes we create
        in the parse_schema method. Then we check for a reference whether we have a definition in
        the schema model. If yes, we use this class as a custom type. If no, we inline it.
        This method here is a wrapper around resolve_object to add this additional logic.
        """
        if isinstance(obj, ReferenceToObject):
            datatype: DataType = self.get_ref_data_type(obj.ref)
            if datatype.reference:
                relative_name = datatype.reference.name
                schemaname = fully_qualified_schema_name(
                    base_package=base_package, schema_name=relative_name
                )
                if schemaname in self.defined_schemas_fully_qualified:
                    return JsonSchemaObject(customTypePath=schemaname)
        # otherwise, treat normally
        return self.resolve_object(obj, JsonSchemaObject)

    def parse_schema(
        self,
        path_parts: list[str],
        base_package: str,
        schema_name: str,
        schema: JsonSchemaObject,
    ) -> None:
        """Parses a single JSON Schema object into a Pydantic model."""
        klassname = fully_qualified_schema_name(
            base_package=base_package, schema_name=schema_name
        )
        self.parse_obj(
            klassname,
            schema,
            path_parts,
        )
        self.defined_schemas_fully_qualified.append(klassname)

    def parse_schemas(
        self,
        path_parts: list[str],
        base_package: str,
        schemas: dict[str, ReferencesToDicts | ReferenceToObject | JsonSchemaObject],
    ) -> None:
        """Parses all schemas in the federation configuration.

        Handles three types of schema definitions:
        - Direct JsonSchemaObject instances
        - References to external schemas (ReferenceToObject)
        - Collections of references (ReferencesToDicts)
        """
        for schema_name, rhs in schemas.items():
            if isinstance(rhs, JsonSchemaObject):
                self.parse_schema(path_parts, base_package, schema_name, rhs)
            elif isinstance(rhs, ReferenceToObject):
                ref_model = self.resolve_data_type_object(base_package, rhs)
                self.parse_schema(path_parts, base_package, schema_name, ref_model)
            else:
                assert isinstance(rhs, ReferencesToDicts), (
                    f"Unexpected type: {type(rhs)}"
                )
                # for remote json schemas referenced in this way, we don't support
                # recursive refs. Those must be full JsonSchemaObjects.
                for ref in rhs.root:
                    remote_dict = self.resolve_raw_obj(ref)
                    for remote_name, remote_obj_dict in remote_dict.items():
                        ref_model = JsonSchemaObject.model_validate(remote_obj_dict)
                        self.parse_schema(
                            path_parts, base_package, remote_name, ref_model
                        )

    def _make_optional(self, schema: JsonSchemaObject) -> JsonSchemaObject:
        """Makes a schema optional by adding None to the union and setting default to None.

        This is used for operation parameters that are not in the required list.
        """
        # Create a null type option for None
        null_option = JsonSchemaObject(type="null")

        # If the schema already has a oneOf (union), add None to it
        if schema.oneOf:
            # Make a copy and add the null option
            new_one_of = list(schema.oneOf) + [null_option]
            return JsonSchemaObject(oneOf=new_one_of, default=None)
        else:
            # Create a new union with the existing schema and None
            return JsonSchemaObject(oneOf=[schema, null_option], default=None)

    def parse_operation_property(
        self,
        param_name: str,
        param_config: JsonSchemaObject,
    ) -> tuple[JsonSchemaObject, tuple[str, bool, bool, bool, bool]]:
        """Parses a single operation property. Returns the actual JsonSchemaObject that
        should be used plus the tuple for the parameter documentation."""
        actual_type = determine_final_json_schema(
            param_name, param_config, param_config.extras
        )
        return (
            actual_type,
            (
                param_config.extras.get("description", ""),
                param_config.extras.get("accepts-literal-input", True),
                param_config.extras.get("accepts-node-input", True),
                param_config.extras.get("accepts-place-input", True),
                param_config.extras.get("discoverable", True),
            ),
        )

    def parse_operation(
        self,
        path_parts: list[str],
        base_package: str,
        operation_name: str,
        operation: OperationObject,
    ) -> None:
        """Parses a single operation into a Pydantic model.

        Creates an operation class that:
        - Inherits from nv_dfm_core.api.Operation
        - Has metadata fields (dfm_class_name, __api_name__)
        - Has parameter type options based on configuration
        - Includes comprehensive documentation
        """
        klassname = f"{base_package}.fed.api.{operation_name}"

        # we add a few things to the properties that come from the config file
        new_properties = {}
        new_properties["dfm_class_name"] = {  ## this creates a literal constant
            "type": "string",
            "enum": [klassname],
            "default": klassname,
        }
        new_properties["__api_name__"] = {  ## this creates a literal constant
            "type": "string",
            "enum": [operation_name],
            "default": operation_name,
        }

        # Get the list of required parameters from the operation
        required_params = operation.required if operation.required else []

        param_docs: dict[str, tuple[str, bool, bool, bool, bool]] = {}
        if operation.properties:
            for param_name, rhs in operation.properties.items():
                if isinstance(rhs, JsonSchemaObject):
                    actual_type, param_doc = self.parse_operation_property(
                        param_name, rhs
                    )
                    # If the parameter is not required, add None to the Union and set default to None
                    if param_name not in required_params:
                        actual_type = self._make_optional(actual_type)
                    new_properties[param_name] = actual_type
                    param_docs[param_name] = param_doc
                elif isinstance(rhs, ReferenceToObject):
                    ref_model = self.resolve_data_type_object(base_package, rhs)
                    actual_type, param_doc = self.parse_operation_property(
                        param_name,
                        ref_model,
                    )
                    # If the parameter is not required, add None to the Union and set default to None
                    if param_name not in required_params:
                        actual_type = self._make_optional(actual_type)
                    new_properties[param_name] = actual_type
                    param_docs[param_name] = param_doc
                else:
                    assert isinstance(rhs, ReferencesToDicts), (
                        f"Unexpected type: {type(rhs)}"
                    )
                    for ref in rhs.root:
                        remote_dict = self.resolve_raw_obj(ref)
                        for remote_name, remote_obj in remote_dict.items():
                            ref_model = JsonSchemaObject.model_validate(remote_obj)
                            actual_type, param_doc = self.parse_operation_property(
                                param_name,
                                ref_model,
                            )
                            # If the parameter is not required, add None to the Union and set default to None
                            if remote_name not in required_params:
                                actual_type = self._make_optional(actual_type)
                            new_properties[remote_name] = actual_type
                            param_docs[remote_name] = param_doc

        # get the raw object and override the properties
        raw_obj = operation.model_dump()
        raw_obj["customBasePath"] = "nv_dfm_core.api.Operation"
        raw_obj["properties"] = new_properties

        if "required" not in raw_obj:
            raw_obj["required"] = []
        raw_obj["required"].append("dfm_class_name")  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        raw_obj["required"].append("__api_name__")  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]

        raw_obj["description"] = create_model_description(
            description=raw_obj.get("description", None),
            param_docs=param_docs,
            returns=raw_obj.get("returns", None),
        )

        self.parse_raw_obj(
            klassname,
            raw_obj,
            path_parts,
        )

    def parse_operations(
        self,
        path_parts: list[str],
        base_package: str,
        operations: dict[str, ReferencesToDicts | ReferenceToObject | OperationObject],
    ) -> None:
        """Parses all operations in the federation configuration.

        Handles the same three types as parse_schemas: direct objects, references,
        and collections of references.
        """
        for operation_name, rhs in operations.items():
            if isinstance(rhs, OperationObject):
                self.parse_operation(
                    [*path_parts, operation_name], base_package, operation_name, rhs
                )
            elif isinstance(rhs, ReferenceToObject):
                ref_model = self.resolve_object(rhs, OperationObject)
                self.parse_operation(
                    [*path_parts, operation_name],
                    base_package,
                    operation_name,
                    ref_model,
                )
            else:
                assert isinstance(rhs, ReferencesToDicts), (
                    f"Unexpected type: {type(rhs)}"
                )
                for ref in rhs.root:
                    remote_dict = self.resolve_raw_obj(ref)
                    for remote_name, remote_obj in remote_dict.items():
                        ref_model = OperationObject.model_validate(remote_obj)
                        self.parse_operation(
                            [*path_parts, remote_name],
                            base_package,
                            remote_name,
                            ref_model,
                        )

    def parse_adapter(
        self,
        path_parts: list[str],
        base_package: str,
        site_name: str,
        provider_name: str | None,
        operation_ref: str,
        adapter: AdapterObject,
    ) -> None:
        """Parses an adapter that specializes an operation for a specific site/provider.

        Creates a Pydantic model that:
        - Inherits from the original operation (via allOf)
        - Adds site and provider metadata
        - Exposes any configured adapter arguments as optional parameters
        """
        # for each operation in a site or provider interface we
        # generate a pydantic model that specializes the operation to
        # this site or provider

        # declare the new op as allOf the original op, which will make
        # the datamodel_code_generator inherit from the original op
        # operation_ref is like #/operations/utils.Greetme, we only want the last part
        operation_name = operation_ref.split("/")[-1]
        provider_part = f"{provider_name}." if provider_name else ""
        klassname = (
            f"{base_package}.fed.site.{site_name}.{provider_part}{operation_name}"
        )

        if provider_name:
            provider_json_schema = {
                "type": "string",
                "enum": [provider_name],
                "default": provider_name,
            }
        else:
            provider_json_schema = {"type": "null", "default": None}

        raw_obj = {
            "type": "object",
            "allOf": [{"$ref": operation_ref}],
            "properties": {
                "site": {"type": "string", "enum": [site_name], "default": site_name},
                "provider": provider_json_schema,
                "dfm_class_name": {
                    "type": "string",
                    "enum": [klassname],
                    "default": klassname,
                },
                "__api_name__": {
                    "type": "string",
                    "enum": [operation_name],
                    "default": operation_name,
                },
            },
            "required": ["site", "provider", "dfm_class_name", "__api_name__"],
        }

        # check if any arguments should be exposed
        # Exposed fields are always optional and get a default value of None
        # to the outside
        param_docs: dict[Any, Any] = {}
        if adapter.args != "from-operation":
            # args don't simply come from the operation
            for arg_value in adapter.args.values():
                # is this arg spec one that supports exposing?
                if hasattr(arg_value, "expose_as"):
                    exposed_field = getattr(arg_value, "expose_as")
                    # yes, and should it be exposed?
                    if exposed_field:
                        base_type = JsonSchemaObject(
                            type=exposed_field.type, default=None
                        )
                        props: Any = raw_obj["properties"]
                        assert isinstance(props, dict), (
                            f"Unexpected type: {type(props)}"
                        )
                        props[exposed_field.param] = determine_final_json_schema(
                            exposed_field.param,
                            base_type,
                            exposed_field.model_dump(),
                        )
                        param_docs[exposed_field.param] = (
                            exposed_field.description
                            if exposed_field.description
                            else "",
                            exposed_field.accepts_literal_input,
                            exposed_field.accepts_node_input,
                            exposed_field.accepts_place_input,
                            exposed_field.discoverable,
                        )

        raw_obj["description"] = create_model_description(
            description=adapter.description,
            param_docs=param_docs,
            returns=adapter.returns,
        )

        self.parse_raw_obj(klassname, raw_obj, path_parts)

    def parse_interface(
        self,
        path_parts: list[str],
        base_package: str,
        site_name: str,
        provider_name: str | None,
        interfaces: dict[str, ReferencesToDicts | ReferenceToObject | AdapterObject],
    ) -> None:
        """Parses the interface of a site or provider.

        Each interface entry maps an operation reference to an adapter that
        defines how that operation is executed at this site/provider.
        """
        for operation_ref, rhs in interfaces.items():
            if isinstance(rhs, AdapterObject):
                self.parse_adapter(
                    path_parts=[*path_parts, operation_ref],
                    base_package=base_package,
                    site_name=site_name,
                    provider_name=provider_name,
                    operation_ref=operation_ref,
                    adapter=rhs,
                )
            elif isinstance(rhs, ReferenceToObject):
                ref_model = self.resolve_object(rhs, AdapterObject)
                self.parse_adapter(
                    path_parts=[*path_parts, operation_ref],
                    base_package=base_package,
                    site_name=site_name,
                    provider_name=provider_name,
                    operation_ref=operation_ref,
                    adapter=ref_model,
                )
            else:
                assert isinstance(rhs, ReferencesToDicts), (
                    f"Unexpected type: {type(rhs)}"
                )
                for ref in rhs.root:
                    remote_dict = self.resolve_raw_obj(ref)
                    for remote_name, remote_obj in remote_dict.items():
                        ref_model = AdapterObject.model_validate(remote_obj)
                        self.parse_adapter(
                            path_parts=[*path_parts, remote_name],
                            base_package=base_package,
                            site_name=site_name,
                            provider_name=provider_name,
                            operation_ref=remote_name,
                            adapter=ref_model,
                        )

    def parse_provider(
        self,
        path_parts: list[str],
        base_package: str,
        site_name: str,
        provider_name: str,
        provider: ProviderObject,
    ) -> None:
        """Parses a provider within a site."""
        self.parse_interface(
            path_parts=[*path_parts, provider_name],
            base_package=base_package,
            site_name=site_name,
            provider_name=provider_name,
            interfaces=provider.interface,
        )

    def parse_providers(
        self,
        path_parts: list[str],
        base_package: str,
        site_name: str,
        providers: dict[str, ReferencesToDicts | ReferenceToObject | ProviderObject],
    ) -> None:
        """Parses all providers within a site."""
        for provider_name, rhs in providers.items():
            if isinstance(rhs, ProviderObject):
                self.parse_provider(
                    path_parts=[*path_parts, provider_name],
                    base_package=base_package,
                    site_name=site_name,
                    provider_name=provider_name,
                    provider=rhs,
                )
            elif isinstance(rhs, ReferenceToObject):
                ref_model = self.resolve_object(rhs, ProviderObject)
                self.parse_provider(
                    path_parts=[*path_parts, provider_name],
                    base_package=base_package,
                    site_name=site_name,
                    provider_name=provider_name,
                    provider=ref_model,
                )
            else:
                assert isinstance(rhs, ReferencesToDicts), (
                    f"Unexpected type: {type(rhs)}"
                )
                for ref in rhs.root:
                    remote_dict = self.resolve_raw_obj(ref)
                    for remote_name, remote_obj in remote_dict.items():
                        ref_model = ProviderObject.model_validate(remote_obj)
                        self.parse_provider(
                            path_parts=[*path_parts, remote_name],
                            base_package=base_package,
                            site_name=site_name,
                            provider_name=provider_name,
                            provider=ref_model,
                        )

    def parse_site(
        self, path_parts: list[str], base_package: str, site_name: str, site: SiteObject
    ) -> None:
        """Parses a site in the federation.

        A site has both a direct interface and providers that implement operations.
        """
        site_name = site_name_to_identifier(site_name)
        interface = site.interface
        self.parse_interface(
            path_parts=[*path_parts, site_name],
            base_package=base_package,
            site_name=site_name,
            provider_name=None,
            interfaces=interface,
        )
        self.parse_providers(
            path_parts=[*path_parts, site_name],
            base_package=base_package,
            site_name=site_name,
            providers=site.providers,
        )

    def parse_sites(
        self,
        path_parts: list[str],
        base_package: str,
        sites: dict[str, ReferencesToDicts | ReferenceToObject | SiteObject],
    ) -> None:
        """Parses all sites in the federation."""
        for site_name, rhs in sites.items():
            if isinstance(rhs, SiteObject):
                self.parse_site(
                    path_parts=[*path_parts, site_name],
                    base_package=base_package,
                    site_name=site_name,
                    site=rhs,
                )
            elif isinstance(rhs, ReferenceToObject):
                ref_model = self.resolve_object(rhs, SiteObject)
                self.parse_site(
                    path_parts=[*path_parts, site_name],
                    base_package=base_package,
                    site_name=site_name,
                    site=ref_model,
                )
            else:
                assert isinstance(rhs, ReferencesToDicts), (
                    f"Unexpected type: {type(rhs)}"
                )
                for ref in rhs.root:
                    remote_dict = self.resolve_raw_obj(ref)
                    for remote_name, remote_obj in remote_dict.items():
                        ref_model = SiteObject.model_validate(remote_obj)
                        self.parse_site(
                            path_parts=[*path_parts, remote_name],
                            base_package=base_package,
                            site_name=site_name,
                            site=ref_model,
                        )

    def parse_federation_object(self, federation_object: FederationObject) -> None:
        """Main entry point: parses a complete federation configuration.

        Processes the federation object in this order:
        1. Schemas (JSON Schema objects)
        2. Operations (API operations)
        3. Sites (federation nodes with their interfaces and providers)
        """
        # we are at the root, no path_parts
        path_parts = []

        # This is important. self.get_ref_model() looks
        # either at a file/url if the ref is a file or url, but if not,
        # it looks at self.raw_obj to traverse the ref.
        self.raw_obj: dict[str, Any] = federation_object.model_dump()

        self.parse_schemas(
            [*path_parts, "#/schemas"],
            federation_object.info.code_package,
            federation_object.schemas,
        )

        self.parse_operations(
            [*path_parts, "#/operations"],
            federation_object.info.code_package,
            federation_object.operations,
        )

        self.parse_sites(
            [*path_parts, "#/sites"],
            federation_object.info.code_package,
            federation_object.sites,
        )

    def _add_copyright_header(self, content: str):
        """Adds NVIDIA copyright header to generated code."""
        header = '''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
THIS FILE WAS GENERATED BY DFM-APIGEN. DO NOT MODIFY.
"""

'''
        return header + content

    def parse(self) -> list[tuple[Path, str]] | None:
        """Generates the final Python code with copyright headers."""
        if not self.results:
            print("WARNING: Parser has no contents to parse.")
            return None
        assert self.raw_obj is not None
        assert isinstance(self.raw_obj, dict)
        result: Any = super().parse()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(result, str):
            return [(Path("???"), self._add_copyright_header(result))]
        elif isinstance(result, dict):
            return [
                (Path(*file_parts), self._add_copyright_header(result.body))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                for file_parts, result in result.items()  # pyright: ignore[reportUnknownVariableType]
            ]
        else:
            raise ValueError(f"Unexpected type: {type(result)}")  # pyright: ignore[reportUnknownArgumentType]
