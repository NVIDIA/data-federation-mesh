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

import copy
import logging
from pathlib import Path
from typing import Any

import yaml
from datamodel_code_generator.types import DataType
from jinja2 import Environment, FileSystemLoader

from ._openapi_parser import DFMOpenAPIParser

"""
IMPORTANT TODO:
The challenge is this: when prance runs and sees a $ref in the document,
it "resolves" it by default. "Resolve" means, it's following the $ref part and finds the dict it
leads to (either in the same file, if the ref is like #/some/path, 
or in a different file or even a URL, if the ref is like http://some.url#/some/path).
It then copy&pastes this found dict into the original place, replacing the $ref object.
The problem is, that this copy&paste loses information. Example: an OpenAPI spec with
something like this:
components:
    schemas:
    A: # maybe a schema for the First name of a person
        type: string
    B: # maybe a schema for the Password of your credit card
        type: string
paths:
    Users:
        name:
            $ref: #/components/schemas/A

After resolving, it looks like this:
paths:
    Users:
        name:
            type: string # we don't know if this was supposed to be A or B

We can confiture prance to not resolve anything. However, that's also not quite what we want
because $ref resolution may require following file or URL paths plus caching 
and we don't want to implement this ourselves. 

Therefore, we configure prance to only resolve HTTP and FILE references, but NOT INTERNAL
references. However, THIS COMES WITH THE SEVERE LIMITATION THAT IT CAN ONLY HANDLE OpenAPI
SPECS THAT DON'T HAVE $refs OUTSIDE OF THE ROOT FILE. THIS NEEDS TO BE FIXED AT SOME POINT.

When an external file uses references itself, the resulting resolved dictionary will contain
local $refs (which won't be resolved), but we again don't know from what file they came from.

THOUGHT: It may be the case that the dict objects that prance returns actually reference each other
correctly, because if I dump such files to yaml I see inserted references like *id004.
"""


def find_path(data: dict[str, Any], path: str) -> Any:
    path = path.lstrip("#/")
    parts = path.split("/")
    current = data
    for part in parts:
        if part not in current:
            raise ValueError(
                f"Path {path} not found in dict. Part {part} failed in current dict {current.keys()}"
            )
        current = current[part]
    return current


def resolve_reference(specs: dict[str, Any], item: dict[str, Any]) -> dict:
    if "$ref" in item:
        return resolve_reference(specs, find_path(specs, item["$ref"]))
    else:
        return item


def resolve_references_for_obj(specs: dict[str, Any], obj: Any):
    if not isinstance(obj, dict) and not isinstance(obj, list):
        return obj
    if "$ref" in obj:
        return resolve_references_for_obj(specs, find_path(specs, obj["$ref"]))
    if isinstance(obj, dict):
        updated_obj = {}
        for o, obj_prop in obj.items():
            if o == "examples":
                continue
            updated_obj[o] = resolve_references_for_obj(specs, obj_prop)
        return updated_obj
    elif isinstance(obj, list):
        updated_obj = []
        for obj_prop in obj:
            updated_obj.append(resolve_references_for_obj(specs, obj_prop))
        return updated_obj
    else:
        return obj


class OpenApiGen:
    """
    A class for parsing OpenAPI specifications and generating various file types
    from the parsed specification.
    """

    def __init__(
        self,
        openapi_file_path: str | Path,
        api_package_name: str,
        adapter_package_prefix: str,
        generate_async: bool = True,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the OpenAPI importer with a file path.

        Args:
            openapi_file_path: Path to the OpenAPI specification file (YAML or JSON)
            api_package_name: Name of the API package. All federation config operations are prefixed with this name.
            adapter_package_prefix: Additional prefix for the adapter package. All generated adapters will be in {adapter_package_prefix}.{api_package_name}.AdapterName
            generate_async: Whether to generate async adapters.
            logger: Optional logger instance
        """
        self.openapi_file_path: Path = Path(openapi_file_path)
        self.api_package_prefix: str = f"{api_package_name}."
        self.adapter_package_prefix: str = f"{adapter_package_prefix}."
        self.generate_async: bool = generate_async
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        self._parsed_spec: dict[str, Any] | None = None
        self._generated_schemas: dict[str, Any] | None = None
        self._generated_operations: dict[str, Any] | None = None
        self._generated_site_interface: dict[str, Any] | None = None

        self._generated_adapters: dict[Path, str] | None = None

        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self._jinja_env: Environment = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=True,
        )

        if not self.openapi_file_path.exists():
            raise FileNotFoundError(f"OpenAPI file not found: {self.openapi_file_path}")

    def parse(self) -> dict[str, Any]:
        """
        Parse and resolve the OpenAPI specification file.

        Returns:
            A fully resolved OpenAPI specification as a Python dictionary

        Raises:
            FileNotFoundError: If the OpenAPI file doesn't exist
            ValueError: If the file is empty or invalid
            RuntimeError: If parsing fails
        """
        if self._parsed_spec is not None:
            return self._parsed_spec

        # Check if file exists and is not empty
        if not self.openapi_file_path.exists():
            raise FileNotFoundError(f"OpenAPI file not found: {self.openapi_file_path}")

        try:
            self._parser = DFMOpenAPIParser(str(self.openapi_file_path))
            if not self._parser:
                raise ValueError("Failed to create OpenAPI parser")
            self._schema_model = self._parser.parse()
            self._parsed_spec = self._parser.raw_obj
            assert self._parsed_spec is not None  # pyright: ignore[reportUnknownMemberType]
            self.logger.info("OpenAPI specification parsed and resolved successfully")
            return self._parsed_spec  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        except Exception as e:
            self.logger.error(f"Failed to parse OpenAPI specification: {e}")
            raise RuntimeError(f"Failed to parse OpenAPI specification: {e}")

    def write_current_federation_config_part(
        self, outpath: Path, filename: str, reset: bool = True
    ) -> None:
        """
        Writes whatever API parts have been generated so far. I.e. the schemas
        and/or operations and/or site interface. Resets the generated parts to None afterwards.
        This can be used to decide when and what to write to file, e.g. if everything should
        go into one file, call all the generate_XYZ methods and then write the partial config.
        If you want each part in a different file, call a generate_XYZ method followed by
        this write method.

        Note: this generator does NOT generate full federation config files. It only writes parts (templates)
              that can be $ref'd into an actual config.

        NOTE: this is only for the federation config. The adapter code is generated and written
        differently.

        Args:
            outpath: Base directory where generated files should be placed.
                     If None, no files are written. The result is stored
            filename: The filename to write the API part to.
            reset: Whether to reset the generated parts to None afterwards or to keep them.
        """

        outpath.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{filename}.dfm.part.yaml"
            if not filename.endswith(".dfm.part.yaml")
            else filename
        )

        with open(outpath / filename, mode="w", encoding="utf-8") as f:
            if self._generated_schemas is not None:
                yaml.dump(
                    self._generated_schemas,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2,
                )
            if self._generated_operations is not None:
                yaml.dump(
                    self._generated_operations,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2,
                )
            if self._generated_site_interface is not None:
                yaml.dump(
                    self._generated_site_interface,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2,
                )

        if reset:
            self._generated_schemas = None
            self._generated_operations = None
            self._generated_site_interface = None

    def generate_api_schemas(self) -> dict[str, Any]:
        """
        Generate API schemas from the OpenAPI specification.

        Returns:
            The dict with the api schemas part that can be dumped into a yaml file, for example.
        """

        if self._generated_schemas is not None:
            return self._generated_schemas

        spec = self.parse()
        schemas = spec.get("components", {}).get("schemas", {})
        if not schemas:
            self.logger.warning("No schemas found in OpenAPI specification")
            self._generated_schemas = {}
        else:
            self._generated_schemas = {
                "components": {"schemas": self._translate_api_schemas(schemas)}
            }

        return self._generated_schemas

    def generate_api_operations(self) -> dict[str, Any]:
        """
        Generate API operations templates from the OpenAPI specification.

        Returns:
            The dict with the api operations part that can be dumped into a yaml file, for example.
        """
        spec = self.parse()

        paths = spec.get("paths", {})
        if not paths:
            self.logger.warning("No paths found in OpenAPI specification")
            self._generated_operations = {}
        else:
            self._generated_operations = {
                "operations": self._translate_api_operations(paths)
            }

        return self._generated_operations

    def generate_site_interface(self) -> dict[str, Any]:
        """
        Generate site interface templates from the OpenAPI specification.

        Returns:
            The dict with the api schemas part that can be dumped into a yaml file, for example.
        """

        spec = self.parse()

        paths = spec.get("paths", {})
        if not paths:
            self.logger.warning("No paths found in OpenAPI specification")
            self._generated_site_interface = {}
        else:
            self._generated_site_interface = {
                "interface": self._translate_site_interface(paths)
            }

        return self._generated_site_interface

    def _write_init_file(self, outpath: Path) -> None:
        """Write __init__.py for adapters library"""
        init_path = outpath
        for adapter_package, submodules in self.adapter_modules.items():
            content = [
                f"from ._{module_name.strip('.')} import {class_name}\n"
                for module_name, class_name in submodules
            ]
            for p in adapter_package.split("."):
                init_path = init_path / p
            init_path.mkdir(parents=True, exist_ok=True)
            target_file = init_path / "__init__.py"
            _ = target_file.write_text("".join(content), encoding="utf-8")
            init_path = outpath

    def _write_datamodel_files(self, outpath: Path) -> None:
        """
        Write the pydantic models of schemas to files.
        """
        if self._schema_model is None:
            self.logger.warning("No schema model generated")
            return
        outpath.mkdir(parents=True, exist_ok=True)
        target_file = (
            outpath
            / self.adapter_package_prefix.replace(".", "/")
            / self.api_package_prefix.strip(".")
            / "data_models.py"
        )
        self.logger.info(f"Writing data models to {target_file}")
        target_file.parent.mkdir(parents=True, exist_ok=True)
        _ = target_file.write_text(self._schema_model, encoding="utf-8")

    def write_adapter_files(self, outpath: Path) -> None:
        """
        Write the federation config part to a file.
        """
        if self._generated_adapters is None:
            self.logger.warning("No adapters generated")
            return

        outpath.mkdir(parents=True, exist_ok=True)
        for filepath, content in self._generated_adapters.items():
            target_file = outpath / filepath
            self.logger.info(f"Writing adapter to {target_file}")
            target_file.parent.mkdir(parents=True, exist_ok=True)
            _ = target_file.write_text(content, encoding="utf-8")

        self._write_init_file(outpath=outpath)
        self._write_datamodel_files(outpath=outpath)

    def generate_adapters(self) -> dict[Path, str]:
        """
        Generate adapter classes from the OpenAPI specification.

        Returns:
            dict of relative file names and their content
        """
        spec = self.parse()
        self._generated_adapters = {}

        # Generate individual operation adapters
        self.adapter_modules = {}
        paths = spec.get("paths", {})
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.upper() in [
                    "GET",
                    "POST",
                    "PUT",
                    "DELETE",
                    "PATCH",
                    "HEAD",
                    "OPTIONS",
                ]:
                    operation_subclasses, operation_id = self._parse_operation_id(
                        operation.get("operationId", "")
                    )
                    adapter_class_name = f"{operation_id}Adapter"
                    adapter_package = f"{self.adapter_package_prefix}{self.api_package_prefix}{'.'.join(operation_subclasses)}".rstrip(
                        "."
                    )

                    # Convert CamelCase adapter_class_name to snake_case for the filename
                    import re

                    def camel_to_snake(name: str):
                        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
                        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

                    adapter_file = (
                        adapter_package.replace(".", "/")
                        + f"/_{camel_to_snake(adapter_class_name)}.py"
                    )

                    if adapter_package not in self.adapter_modules:
                        self.adapter_modules[adapter_package] = []
                    self.adapter_modules[adapter_package].append(
                        (camel_to_snake(adapter_class_name), adapter_class_name)
                    )

                    operation_adapter_content = self._generate_operation_adapter(
                        adapter_class_name=adapter_class_name,
                        adapter_package=adapter_package,
                        path=path,
                        method=method.upper(),
                        operation=operation,
                    )

                    self._generated_adapters[Path(adapter_file)] = (
                        operation_adapter_content
                    )
                    self.logger.info(f"Generated adapter: {adapter_file}")

        return self._generated_adapters

    def _parse_operation_id(self, operation_id: str) -> tuple[list[str], str]:
        """For the case when hierarchial naming is used for OperationID,
        separate ID of the operation from subclasses.
        Returns: subclasses and operation ID
        """
        subclasses = operation_id.split(".")
        if len(subclasses) == 1:
            return [], operation_id
        return subclasses[1:-1], subclasses[-1]

    def _translate_api_schemas(self, schemas: dict[str, Any]) -> dict[str, Any]:
        """Generate a dfm-style schema definition in YAML. Since the dfm spec is pretty much
        the same as OpenAPI for the schemas (they are just JSON Schema objects), we simply return
        the openapi schemas dict as the federation config schemas dict.
        However, it's well possible that later we will need to patch or fix or rewrite the
        openapi schema dict to make it a valid dfm federation config schema dict."""
        return schemas

    def _resolve_openapi_parameter_spec(
        self, parameter: dict[str, Any]
    ) -> dict[str, Any]:
        assert self._parsed_spec is not None
        if "$ref" in parameter:
            # resolve the path to the parameter spec
            param_spec = find_path(self._parsed_spec, parameter["$ref"])
        else:
            param_spec = parameter

        if "name" not in param_spec:
            raise ValueError(f"Parameter dict {parameter} has no name")
        if "schema" not in param_spec:
            raise ValueError(f"Parameter dict {parameter} has no schema")

        return param_spec

    def _translate_operation_parameter(
        self, parameter: dict[str, Any]
    ) -> tuple[str, dict[str, Any], bool]:
        """Generate a dfm-style parameter definition in YAML.
        Returns the name of the parameter and the JsonSchema spec.

        OpenAPI operation parameters are often referenced from the components/parameters section.
        And the schema part of a parameter is often referencing components/schemas entries.

        So the parameter dict is usually something like {'$ref': '#/components/parameters/name'}
        or { 'name': 'p1', 'schema': {'$ref': '#/components/schemas/s1'} }
        or { 'name': 'p1', 'schema': { ... inlined schema here ...} } or
        """
        assert self._parsed_spec is not None
        param_spec = self._resolve_openapi_parameter_spec(parameter)

        def resolve_ref(specs, schema):
            if "$ref" in schema:
                schema = find_path(specs, schema["$ref"])
            return schema

        dfm_param_name = param_spec["name"].replace("-", "_")
        dfm_param_schema = param_spec["schema"]
        dfm_param_schema = resolve_ref(self._parsed_spec, dfm_param_schema)
        if "items" in dfm_param_schema:
            dfm_param_schema["items"] = resolve_ref(
                self._parsed_spec, dfm_param_schema["items"]
            )

        # return a deep copy, otherwise the yaml inserts *id004 references. This could
        # actually be useful to solve the aliasing problem after resolving $refs, see long comment above.
        return (
            dfm_param_name,
            copy.deepcopy(dfm_param_schema),
            param_spec.get("required", True),
        )

    def _translate_api_operations(self, paths: dict[str, Any]) -> dict[str, Any]:
        """Generate the dict with the api operations part of the dfm federation config."""

        operations: dict[str, Any] = {}
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                # prefix the operation name with the api package prefix
                operation_id = operation.get("operationId", "").lstrip(".")
                op_name = f"{self.api_package_prefix}{operation_id}"
                description = f"From OpenAPI {method.upper()} {path}: {operation.get('description', '-')}"
                # translate each openapi parameter to a dfm parameter into a list of tuples
                parameters = [
                    self._translate_operation_parameter(parameter)
                    for parameter in operation.get("parameters", {})
                ]
                parameters.extend(
                    [
                        ("server", {"type": "string"}, True),
                        ("access_token", {"type": "string"}, False),
                    ]
                )
                required = [pname for pname, _, required in parameters if required]
                returns = "Any"  # we don't use the return information in the dfm at the moment, so no need to translate all the operation.get('responses', "")

                operations[op_name] = {
                    "description": description,
                    "parameters": {
                        pname: pschema for pname, pschema, _ in parameters
                    },  # translate the list into a dict
                    "required": required,
                    "returns": returns,
                }

        return self._postprocess_operation_refs(operations)

    def _translate_operation_arguments(
        self, parameter: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """similar to _translate_operation_parameter, but for arguments"""

        param_spec = self._resolve_openapi_parameter_spec(parameter)
        dfm_param_name = param_spec["name"].replace("-", "_")
        # for arguments that are already defined in the corresponding operation, we only need the
        # name of the parameter
        return dfm_param_name, {"from-param": dfm_param_name}

    def _translate_site_interface(self, paths: dict[str, Any]) -> dict[str, Any]:
        """Generate a dict with the site interface part of the dfm federation config."""

        interface: dict[str, Any] = {}
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                # prefix the operation name with the api package prefix
                operation_id = operation.get("operationId", "").lstrip(".")
                op_ref = f"#/operations/{self.api_package_prefix}{operation_id}"
                adapter_name = f"{self.adapter_package_prefix}{self.api_package_prefix}{operation_id}Adapter"
                description = f"From OpenAPI {method.upper()} {path}: {operation.get('description', '-')}"
                # translate each openapi parameter to a dfm parameter into a list of tuples
                args = [
                    self._translate_operation_arguments(parameter)
                    for parameter in operation.get("parameters", {})
                ]
                args.extend(
                    [
                        ("server", {"from-param": "server"}),
                        ("access_token", {"from-param": "access_token"}),
                    ]
                )

                interface[op_ref] = {
                    "adapter": adapter_name,
                    "description": description,
                    "is-async": "true" if self.generate_async else "false",
                    "args": {
                        aname: aschema for aname, aschema in args
                    },  # translate the list into a dict
                }

        return self._postprocess_operation_refs(interface)

    def _postprocess_operation_refs(self, operations: dict[str, Any]):
        """For the case of hierarchial operation naming, ensure that all operations
        are correctly prefixed with the api package prefix.
        """
        has_submodules = False
        for ref in operations.keys():
            has_submodules = len(ref.split(".")) > 2
            if has_submodules:
                break
        if not has_submodules:
            return operations
        revised_operations = {}
        for ref, data in operations.items():
            if len(ref.split(".")) > 2:
                revised_operations[ref] = data
            else:
                ref_upd = ref.split(".")
                ref_upd = (
                    f"{'.'.join(ref_upd[:-1])}.{self.api_package_prefix}{ref_upd[-1]}"
                )
                revised_operations[ref_upd] = data
        return revised_operations

    def _parse_operation_parameters(self, spec_parameters: list) -> list:
        operation_params = []
        operation_params_optional = []
        for parameter in spec_parameters:
            param_spec = self._resolve_openapi_parameter_spec(parameter)
            in_value = param_spec.get("in", "path")
            param_name = param_spec["name"].replace("-", "_")
            param_schema = param_spec.get("schema", {})
            param_required = param_spec.get("required", True)
            param_type = self._get_python_type(param_schema)
            if not param_required:
                param_type = f"Optional[{param_type}] = None"
                operation_params_optional.append((param_name, param_type, in_value))
            else:
                operation_params.append((param_name, param_type, in_value))
        return operation_params + operation_params_optional

    def _parse_operation_responses(
        self, operation_responses: dict[str | int, dict[str, DataType]]
    ) -> tuple[dict[str, tuple[str, str]], set]:
        assert operation_responses != {}

        response_models = {}
        content_types = set()

        for response_code, response in operation_responses.items():
            response_models[response_code] = []
            for content_type, content in response.items():
                content_types.add(content_type)
                if content.reference is not None:
                    response_models[response_code].append(
                        (content_type, content.reference.name)
                    )

        return response_models, content_types

    def _generate_operation_adapter(
        self,
        adapter_class_name: str,
        adapter_package: str,
        path: str,
        method: str,
        operation: dict[str, Any],
    ) -> str:
        """Generate adapter for a specific operation."""
        operation_id = operation.get(
            "operationId", f"{method}_{path.replace('/', '_').strip('_')}"
        )
        description = operation.get("description", f"{method} {path}")

        operation_params = self._parse_operation_parameters(
            operation.get("parameters", [])
        )

        parsed_operation = self._parser.parsed_operations.get(operation_id, None)
        if parsed_operation is None:
            raise RuntimeError(
                f"Operation {operation_id} not found in parsed operations"
            )

        response_models, content_types = self._parse_operation_responses(
            parsed_operation.responses
        )
        model_imports = f"from {self.adapter_package_prefix}{self.api_package_prefix}data_models import {', '.join(set(model for content in response_models.values() for _, model in content))}"

        template = self._jinja_env.get_template("operation_adapter.py.jinja2")
        return template.render(
            adapter_class_name=adapter_class_name,
            adapter_package=adapter_package,
            model_imports=model_imports,
            operation={
                "id": operation_id,
                "description": description,
                "path": path,
                "method": method,
                "params": operation_params,
            },
            response_models=response_models,
            content_types=", ".join(content_types),
        )

    def _get_python_type(self, prop_def: dict[str, Any]) -> str:
        """Convert OpenAPI property definition to Python type annotation."""
        prop_type = prop_def.get("type", "string")
        format_type = prop_def.get("format")

        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list[Any]",
            "object": "dict[str, Any]",
        }

        if format_type == "date-time" or format_type == "date":
            return "str"  # Could be datetime if needed

        return type_mapping.get(prop_type, "Any")
