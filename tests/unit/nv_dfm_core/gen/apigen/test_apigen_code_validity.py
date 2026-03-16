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

import ast
from pathlib import Path
from typing import Dict, List, Tuple, Union

from nv_dfm_core.gen.apigen import ApiGen
from nv_dfm_core.gen.apigen._config_models import (
    AdapterObject,
    ConstantArg,
    ConstructorCall,
    ExposedField,
    FederationInfoObject,
    FederationObject,
    JsonSchemaObject,
    OperationFieldArg,
    OperationObject,
    ProviderObject,
    ProviderPropertyArg,
    ReferencesToDicts,
    ReferenceToObject,
    SecretsArg,
    SiteObject,
    SitePropertyArg,
)
from nv_dfm_core.gen.irgen import ComputeCostInfo


def is_valid_python_code(code: str) -> bool:
    """Check if the given code string is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def get_generated_files(apigen: ApiGen) -> List[Tuple[Path, str]]:
    """Generate code and return list of (path, content) tuples."""
    api_files = apigen.generate_api(language="python") or []
    runtime_files = apigen.generate_runtime(language="python") or []
    return api_files + runtime_files


def test_minimal_valid_config():
    """Test that a minimal valid configuration generates valid Python code."""
    # Create minimal valid configuration
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    compute_cost=ComputeCostInfo(),
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    # Create ApiGen instance
    apigen = ApiGen(federation)

    # Generate code and check validity
    files = get_generated_files(apigen)

    # Check that all generated files contain valid Python code
    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_operation_with_parameters():
    """Test that operations with parameters generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Add": OperationObject(
            type="object",
            description="Add two numbers",
            returns="int",
            parameters={
                "a": JsonSchemaObject(type="integer", description="First number"),
                "b": JsonSchemaObject(type="integer", description="Second number"),
            },
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Add": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.AddAdapter",
                    description="Central site's addition adapter",
                    returns="int",
                    compute_cost=ComputeCostInfo(),
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_async_operation():
    """Test that async operations generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.AsyncOp": OperationObject(
            type="object",
            description="An async operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.AsyncOp": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.AsyncOpAdapter",
                    description="Central site's async adapter",
                    returns="str",
                    is_async=True,  # type: ignore
                    compute_cost=ComputeCostInfo(),
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_provider_interface():
    """Test that provider interfaces generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    compute_cost=ComputeCostInfo(),
                ),
            },
            providers={
                "provider1": ProviderObject(
                    interface={
                        "#/operations/utils.Greet": AdapterObject(
                            adapter="testfed.fed.runtime.central.provider1.GreetAdapter",
                            description="Provider1's greeting adapter",
                            returns="str",
                            compute_cost=ComputeCostInfo(),
                        ),
                    },
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_site_with_string_impl():
    """Test that sites with string implementation generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    compute_cost=ComputeCostInfo(),
                ),
            },
            impl="testfed.fed.runtime.central._this_site.CentralSiteImpl",
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_site_with_constructor_impl():
    """Test that sites with constructor implementation generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    compute_cost=ComputeCostInfo(),
                ),
            },
            impl=ConstructorCall(
                path="testfed.fed.runtime.central._this_site.CentralSiteImpl",
                args={"config": {"key": "value"}},
            ),
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_provider_with_string_impl():
    """Test that providers with string implementation generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    compute_cost=ComputeCostInfo(),
                ),
            },
            providers={
                "provider1": ProviderObject(
                    interface={
                        "#/operations/utils.Greet": AdapterObject(
                            adapter="testfed.fed.runtime.central.provider1.GreetAdapter",
                            description="Provider1's greeting adapter",
                            returns="str",
                            compute_cost=ComputeCostInfo(),
                        ),
                    },
                    impl="testfed.fed.runtime.central.provider1.Provider1Impl",
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_provider_with_constructor_impl():
    """Test that providers with constructor implementation generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    compute_cost=ComputeCostInfo(),
                ),
            },
            providers={
                "provider1": ProviderObject(
                    interface={
                        "#/operations/utils.Greet": AdapterObject(
                            adapter="testfed.fed.runtime.central.provider1.GreetAdapter",
                            description="Provider1's greeting adapter",
                            returns="str",
                            compute_cost=ComputeCostInfo(),
                        ),
                    },
                    impl=ConstructorCall(
                        path="testfed.fed.runtime.central.provider1.Provider1Impl",
                        args={"config": {"key": "value"}},
                    ),
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_adapter_with_constructor():
    """Test that adapters with constructor implementation generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter=ConstructorCall(
                        path="testfed.fed.runtime.central._this_site.GreetAdapter",
                        args={"config": {"key": "value"}},
                    ),
                    description="Central site's greeting adapter",
                    returns="str",
                    compute_cost=ComputeCostInfo(),
                ),
            },
            providers={
                "provider1": ProviderObject(
                    interface={
                        "#/operations/utils.Greet": AdapterObject(
                            adapter=ConstructorCall(
                                path="testfed.fed.runtime.central.provider1.GreetAdapter",
                                args={"config": {"key": "value"}},
                            ),
                            description="Provider1's greeting adapter",
                            returns="str",
                            compute_cost=ComputeCostInfo(),
                        ),
                    },
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_adapter_args_from_operation():
    """Test that adapters with 'from-operation' args generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
            parameters={
                "name": JsonSchemaObject(type="string", description="Name to greet"),
            },
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    args="from-operation",
                    compute_cost=ComputeCostInfo(),
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_adapter_args_with_strings():
    """Test that adapters with string args generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    args={
                        "name": "from-param",
                        "greeting": "Hello",
                    },
                    compute_cost=ComputeCostInfo(),
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_adapter_args_with_operation_field():
    """Test that adapters with OperationFieldArg args generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
            parameters={
                "name": JsonSchemaObject(type="string", description="Name to greet"),
            },
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    args={
                        "name": OperationFieldArg(
                            from_param="name",  # type: ignore
                            doc="Name to greet",
                        ),
                    },
                    compute_cost=ComputeCostInfo(),
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_adapter_args_with_secrets():
    """Test that adapters with SecretsArg args generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    args={
                        "api_key": SecretsArg(
                            from_secrets="api_key",  # type: ignore
                            expose_as=ExposedField(  # type: ignore
                                param="api_key",
                                type="string",
                                description="API key for the greeting service",
                            ),
                            doc="API key from secrets vault",
                        ),
                    },
                    compute_cost=ComputeCostInfo(),
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_adapter_args_with_constant():
    """Test that adapters with ConstantArg args generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    args={
                        "greeting": ConstantArg(
                            const="Hello",
                            expose_as=ExposedField(  # type: ignore
                                param="greeting",
                                type="string",
                                description="Greeting to use",
                            ),
                            doc="Default greeting",
                        ),
                    },
                    compute_cost=ComputeCostInfo(),
                ),
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_adapter_args_with_provider_property():
    """Test that adapters with ProviderPropertyArg args generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    args={
                        "base_url": ProviderPropertyArg(
                            from_provider="provider1.base_url",  # type: ignore
                            expose_as=ExposedField(  # type: ignore
                                param="base_url",
                                type="string",
                                description="Base URL for the greeting service",
                            ),
                            doc="Base URL from provider properties",
                        ),
                    },
                    compute_cost=ComputeCostInfo(),
                ),
            },
            providers={
                "provider1": ProviderObject(
                    info={
                        "base_url": "https://api.example.com",
                    },
                ),
            },  # type: ignore
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )


def test_adapter_args_with_site_property():
    """Test that adapters with SitePropertyArg args generate valid Python code."""
    operations: Dict[
        str, Union[ReferencesToDicts, ReferenceToObject, OperationObject]
    ] = {
        "utils.Greet": OperationObject(
            type="object",
            description="A simple greeting operation",
            returns="str",
        ),
    }
    sites: Dict[str, Union[ReferencesToDicts, ReferenceToObject, SiteObject]] = {
        "central": SiteObject(
            interface={
                "#/operations/utils.Greet": AdapterObject(
                    adapter="testfed.fed.runtime.central._this_site.GreetAdapter",
                    description="Central site's greeting adapter",
                    returns="str",
                    args={
                        "timeout": SitePropertyArg(
                            from_site="timeout",  # type: ignore
                            expose_as=ExposedField(  # type: ignore
                                param="timeout",
                                type="integer",
                                description="Timeout in seconds",
                            ),
                            doc="Timeout from site properties",
                        ),
                    },
                    compute_cost=ComputeCostInfo(),
                ),
            },
            info={
                "timeout": 30,
            },
        ),
    }
    federation = FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            api_version="1.0.0",  # type: ignore
            code_package="testfed",  # type: ignore
        ),
        operations=operations,
        sites=sites,
    )

    apigen = ApiGen(federation)
    files = get_generated_files(apigen)

    for path, content in files:
        assert is_valid_python_code(content), (
            f"Generated code in {path} is not valid Python"
        )
