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

"""
Pydantic models for DFM (Data Federation Mesh) configuration files.

This module defines the structure for federation configuration files that describe:
- Federation metadata (version, package info)
- Schemas (JSON Schema objects that get converted to Pydantic models)
- Operations (API operations with parameters and return types)
- Sites (federation nodes with interfaces and providers)
- Providers (service implementations within sites)

The main entry point is FederationObject, which contains all other configuration elements.
"""

import re
from typing import Any, Literal

from datamodel_code_generator.parser.jsonschema import JsonSchemaObject
from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel, field_validator

from nv_dfm_core.exec import FsspecConfig, SecretsVaultConfig
from nv_dfm_core.gen.irgen._fed_info import (
    ComputeCostInfo,
    SendCostInfo,
)


def validate_ref_key_is_references_to_dicts(v: dict[str, Any]) -> None:
    for key, value in v.items():
        # If value is ReferencesToDicts, key must be "$ref"
        if isinstance(value, ReferencesToDicts) and key != "$ref":
            raise ValueError(
                f"ReferencesToDicts can only be used with '$ref' key, found key: {key}"
            )

        # If key is "$ref", value must be ReferencesToDicts
        if key == "$ref" and not isinstance(value, ReferencesToDicts):
            raise ValueError("'$ref' key must have a ReferencesToDicts value")


def validate_key_is_reference_to_operation_or_ref(v: dict[str, Any]) -> None:
    for key in v.keys():
        if not key == "$ref" and not key.startswith("#/operations/"):
            raise ValueError(
                f"Keys in the interface must either be '$ref' or be references to operations of the form: '#/operations/some.Operation', got: {key}"
            )


def validate_python_identifier(v: str) -> str:
    if not v.isidentifier():
        raise ValueError(f"Must be a valid Python identifier, got: {v}")
    return v


def validate_python_path(v: str) -> str:
    parts = v.split(".")
    for part in parts:
        if not part.isidentifier():
            raise ValueError(
                f"Must be a valid Python path (identifiers separated by dots), got: {v}"
            )
    return v


def validate_version_string(v: str) -> str:
    if not re.match(r"^\d+\.\d+\.\d+$", v):
        raise ValueError(
            f"Version must be in format 'X.Y.Z' where X, Y, Z are numbers, got: {v}"
        )
    return v


class ReferenceToObject(BaseModel):
    """A json reference to a dict (json object) in the same file
    or a different file or a URL."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    ref: str = Field(..., alias="$ref")


class ReferencesToDicts(RootModel[list[str]]):
    """A list of json pointers that are treated as one large
    dict; that is, they get merged"""

    model_config = ConfigDict(frozen=True)  # pyright: ignore[reportUnannotatedClassAttribute]


class ConstructorCall(BaseModel):
    """Represents a call to construct a Python object with a fully qualified class path and arguments."""

    path: str  # fully qualified name of a Python class
    args: dict[str, JsonValue] = {}  # args for the constructor

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        return validate_python_path(v)


class FederationInfoObject(BaseModel):
    """The top level object of a dfm federation configuration."""

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    api_version: str = Field(..., alias="api-version")
    code_package: str = Field(..., alias="code-package")
    description: str | None = None

    @field_validator("code_package")
    @classmethod
    def validate_code_package_is_identifier(cls, v: str) -> str:
        return validate_python_identifier(v)

    @field_validator("api_version")
    @classmethod
    def validate_api_version(cls, v: str) -> str:
        return validate_version_string(v)


# Docker and Helm provisioning configuration models
# These define how the federation components are deployed


class DockerBuildSaveInfoObject(BaseModel):
    """The docker build save object of a dfm federation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    enabled: bool = Field(default=False, alias="enabled")
    file: str | None = None
    file: str | None = None


class DockerRegistryInfoObject(BaseModel):
    """The docker registry object of a dfm federation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    url: str = Field(..., alias="url")
    username: str = Field(default="", alias="username")
    password: str = Field(default="", alias="password")


class DockerPushInfoObject(BaseModel):
    """The docker push object of a dfm federation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    enabled: bool = Field(default=False, alias="enabled")
    registry: DockerRegistryInfoObject | None = None
    registry: DockerRegistryInfoObject | None = None


class DockerBuildInfoObject(BaseModel):
    """The docker build object of a dfm federation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    context: str = Field(..., alias="context")
    engine: Literal["podman", "docker"] = Field(..., alias="engine")
    arch: Literal["arm64", "amd64"] = Field(..., alias="arch")
    save: DockerBuildSaveInfoObject | None = None
    push: DockerPushInfoObject | None = None
    save: DockerBuildSaveInfoObject | None = None
    push: DockerPushInfoObject | None = None


class DockerfileInfoObject(BaseModel):
    """The dockerfile object of a dfm federation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    dir: str | None = None
    file: str | None = None


class DockerProvisionInfoObject(BaseModel):
    """The docker provision object of a dfm federation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    image: str = Field(..., alias="image")
    tag: str = Field(..., alias="tag")
    dockerfile: DockerfileInfoObject | None = None
    build: DockerBuildInfoObject | None = None


class HelmProvisionInfoObject(BaseModel):
    """The helm provision object of a dfm federation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=False, populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    path: str = Field(..., alias="path")
    name: str = Field(..., alias="name")
    chartVersion: str | None = Field(default=None, alias="chartVersion")
    appVersion: str | None = Field(default=None, alias="appVersion")
    startup_package_source: Literal["env", "workspace"] = Field(
        ..., alias="startup_package_source"
    )


class ProvisionInfoObject(BaseModel):
    """The provision object of a dfm federation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    docker: DockerProvisionInfoObject | None = None
    helm: HelmProvisionInfoObject | None = None


class OperationObject(JsonSchemaObject):
    """The OperationObject itself gets translated into a Pydantic model.
    Therefore, it is pretty much a JsonSchemaObject but with a few extra fields
    and some values fixed.

    Extra fields:

    In Operation:
    - description: str

    The properties dict can have a key "$ref" with a list of Json pointers that get
    inserted into the properties dict. (ReferencesToDicts)

    For each property (i.e. where the key is not "$ref" but a property name):
    - the value can be a {"$ref": <json pointer>} (ReferenceToObject)
    - or it is a JsonSchemaObject where:
        - type is a json schema type
        - accepts-literal-input: bool. Default: true. Adds the specified json schema as an option.
        - accepts-node-input: bool. Default: true. Adds the NodeParam as a type option.
        - accepts-place-input: bool. Default: true. Adds the PlaceParam as a type option.
        - discoverable: bool. Default: true. If true, adds Advise as a type option
    """

    model_config: ConfigDict = ConfigDict(extra="forbid", populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    type: Literal["object"] = "object"  # pyright: ignore[reportIncompatibleVariableOverride]
    properties: (  # pyright: ignore[reportIncompatibleVariableOverride]
        dict[str, ReferencesToDicts | ReferenceToObject | JsonSchemaObject] | None
    ) = Field(default=None, alias="parameters")
    returns: str

    @field_validator("returns")
    @classmethod
    def validate_returns(cls, v: str) -> str:
        return validate_python_path(v)


# Adapter argument types - these define how operation parameters are bound
# to values when operations are executed at specific sites/providers


class ExposedField(BaseModel):
    """Defines a field that can be exposed to users when calling an operation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    param: str
    type: str
    description: str | None = None
    accepts_literal_input: bool = Field(default=True, alias="accepts-literal-input")
    accepts_node_input: bool = Field(default=True, alias="accepts-node-input")
    accepts_place_input: bool = Field(default=True, alias="accepts-place-input")
    discoverable: bool = Field(default=True, alias="discoverable")

    @field_validator("param")
    @classmethod
    def validate_param(cls, v: str) -> str:
        return validate_python_identifier(v)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        return validate_python_path(v)


class OperationFieldArg(BaseModel):
    """The argument for the adapter call comes from an existing operation field passed
    by the user."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    from_param: str = Field(..., alias="from-param")
    doc: str | None = None

    @field_validator("from_param")
    @classmethod
    def validate_from_param(cls, v: str) -> str:
        return validate_python_identifier(v)


class SecretsArg(BaseModel):
    """The argument for the adapter call comes from the secrets vault."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    expose_as: ExposedField | None = Field(default=None, alias="expose-as")
    from_secrets: str = Field(..., alias="from-secrets")
    doc: str | None = None


class ConstantArg(BaseModel):
    """The argument for the adapter call is set to a constant by the site admin."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    expose_as: ExposedField | None = Field(default=None, alias="expose-as")
    const: JsonValue
    doc: str | None = None


class ProviderPropertyArg(BaseModel):
    """The argument for the adapter call comes from a provider property."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    expose_as: ExposedField | None = Field(default=None, alias="expose-as")
    # use format: <providername>.<propertyname>; example acme.base_url
    from_provider: str = Field(..., alias="from-provider")
    doc: str | None = None

    @field_validator("from_provider")
    @classmethod
    def validate_provider_format(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 2:
            raise ValueError(
                "from-provider must be in format '<providername>.<propertyname>', for example 'acme.base_url'"
            )
        for part in parts:
            if not part.isidentifier():
                raise ValueError(
                    f"Provider and property names must be valid Python identifiers, got: {v}"
                )
        return v


class SitePropertyArg(BaseModel):
    """The argument for the adapter call comes from a site property."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    expose_as: ExposedField | None = Field(default=None, alias="expose-as")
    from_site: str = Field(..., alias="from-site")
    doc: str | None = None

    @field_validator("from_site")
    @classmethod
    def validate_from_site(cls, v: str) -> str:
        return validate_python_identifier(v)


# Note: Make sure that the above models can be distinguished by their field names,
# otherwise Pydantic will not be able to correctly distinguish them
SiteAdapterCallArg = (
    OperationFieldArg | SecretsArg | ConstantArg | ProviderPropertyArg | SitePropertyArg
)


class AdapterObject(BaseModel):
    """Configuration for an operation adapter that binds an operation to a specific site/provider implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    adapter: str | ConstructorCall
    description: str | None = None
    returns: str | None = None
    is_async: bool = Field(default=True, alias="is-async")
    args: Literal["from-operation"] | dict[str, str | SiteAdapterCallArg] = {}
    compute_cost: ComputeCostInfo | None = None

    def extract_adapter_import(self) -> str:
        if isinstance(self.adapter, str):
            return self.adapter.rpartition(".")[0]
        else:
            return self.adapter.path.rpartition(".")[0]

    @field_validator("adapter")
    @classmethod
    def validate_adapter(cls, v: str | ConstructorCall) -> str | ConstructorCall:
        if isinstance(v, str):
            return validate_python_path(v)
        return v

    @field_validator("returns")
    @classmethod
    def validate_returns(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_python_path(v)

    @field_validator("args", mode="before")
    @classmethod
    def validate_interface_keys(
        cls, v: Literal["from-operation"] | dict[str, str | SiteAdapterCallArg]
    ) -> Any:
        """Checking manually if the args have a good chance to be valid SiteAdapterCallArgs."""
        discriminators = set(
            {
                "from-param",
                "from-secrets",
                "const",
                "from-provider",
                "from-site",
                "from_param",
                "from_secrets",
                "from_provider",
                "from_site",
            }
        )
        if v == "from-operation":
            return v
        assert isinstance(v, dict), (
            f"Invalid spec for adapter args field: expected a dict, got {type(v)}"
        )
        for key, value in v.items():
            if isinstance(value, dict):
                arg_keys: set[str] = set(value.keys())  # pyright: ignore[reportUnknownArgumentType]
                if not arg_keys.intersection(discriminators):
                    raise ValueError(
                        f"Invalid spec for arg '{key}': expected a discriminator in {discriminators}, but keys were {arg_keys}"
                    )
        return v


class ProviderObject(BaseModel):
    """Configuration for a provider that offers a set of operations within a site."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    info: dict[str, Any] | None = None
    impl: str | ConstructorCall | None = None
    interface: dict[
        str,  # a string, must be a local json pointer to an operation
        ReferencesToDicts | ReferenceToObject | AdapterObject,
    ] = {}

    @field_validator("impl")
    @classmethod
    def validate_impl(
        cls, v: str | ConstructorCall | None
    ) -> str | ConstructorCall | None:
        if v is None:
            return v
        if isinstance(v, str):
            return validate_python_path(v)
        return v

    @field_validator("interface")
    @classmethod
    def validate_ref_and_references_to_dicts(cls, v: dict[str, Any]) -> dict[str, Any]:
        validate_key_is_reference_to_operation_or_ref(v)
        validate_ref_key_is_references_to_dicts(v)
        return v


class SiteObject(BaseModel):
    """Configuration for a site in the federation with its interface, providers, and resources."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    info: dict[str, Any] | None = None
    impl: str | ConstructorCall | None = None
    cache_config: FsspecConfig = FsspecConfig(protocol="file")
    secrets_vault: SecretsVaultConfig = SecretsVaultConfig()
    interface: dict[
        str,  # a string, must be a local json pointer to an operation
        ReferencesToDicts | ReferenceToObject | AdapterObject,
    ] = {}
    providers: dict[str, ReferencesToDicts | ReferenceToObject | ProviderObject] = {}
    send_cost: dict[str, SendCostInfo] = {}

    def collect_all_adapter_imports(self) -> list[str]:
        """Collect all the adapter imports from the interface and providers."""
        imports: set[str] = set()
        for adapter in self.interface.values():
            assert isinstance(adapter, AdapterObject)
            imports.add(adapter.extract_adapter_import())
        for provider in self.providers.values():
            assert isinstance(provider, ProviderObject)
            for adapter in provider.interface.values():
                assert isinstance(adapter, AdapterObject)
                imports.add(adapter.extract_adapter_import())

        return sorted(list(imports))

    @field_validator("impl")
    @classmethod
    def validate_impl(
        cls, v: str | ConstructorCall | None
    ) -> str | ConstructorCall | None:
        if v is None:
            return v
        if isinstance(v, str):
            return validate_python_path(v)
        return v

    @field_validator("interface")
    @classmethod
    def validate_ref_and_references_to_dicts(cls, v: dict[str, Any]) -> dict[str, Any]:
        validate_key_is_reference_to_operation_or_ref(v)
        validate_ref_key_is_references_to_dicts(v)
        return v


class FederationObject(BaseModel):
    """Top-level configuration object representing an entire DFM federation with schemas, operations, and sites."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    dfm: Literal["1.0.0"]
    info: FederationInfoObject
    provision: ProvisionInfoObject | None = None
    # ReferenceToDicts can only be used as the value for the single "$ref" key
    schemas: dict[str, ReferencesToDicts | ReferenceToObject | JsonSchemaObject] = {}
    operations: dict[str, ReferencesToDicts | ReferenceToObject | OperationObject] = {}
    sites: dict[str, ReferencesToDicts | ReferenceToObject | SiteObject] = {}

    @field_validator("schemas", "operations", "sites")
    @classmethod
    def validate_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        validate_ref_key_is_references_to_dicts(v)
        return v
