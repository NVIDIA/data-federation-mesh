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

import pytest
from pydantic import ValidationError

from nv_dfm_core.gen.apigen._config_models import (
    AdapterObject,
    ConstructorCall,
    ExposedField,
    FederationObject,
    OperationFieldArg,
    OperationObject,
    ProviderObject,
    ProviderPropertyArg,
    ReferencesToDicts,
    SiteObject,
    SitePropertyArg,
    validate_python_path,
)


def test_allows_mixing_refs_and_inline_in_schemas():
    config = {
        "dfm": "1.0.0",
        "info": {"api-version": "1.0.0", "code-package": "atlantis"},
        "schemas": {
            "$ref": ["remote.site#/schemas/SomeSchema"],
            "MySchema": {"type": "object"},
        },
    }
    obj = FederationObject.model_validate(config)
    assert isinstance(obj.schemas["$ref"], ReferencesToDicts)


def test_operation_requires_returns():
    config = {}
    with pytest.raises(ValidationError):
        OperationObject.model_validate(config)


def test_operation_stores_parameters_as_properties():
    config = {"parameters": {"param1": {"type": "object"}}, "returns": "None"}
    op = OperationObject.model_validate(config)
    assert op.properties
    assert "param1" in op.properties


def test_allows_mixing_refs_and_inline_in_operations():
    config = {
        "dfm": "1.0.0",
        "info": {"api-version": "1.0.0", "code-package": "atlantis"},
        "operations": {
            "$ref": ["remote.site#/operations/SomeOp"],
            "MyOp": {"returns": "None"},
        },
    }
    obj = FederationObject.model_validate(config)
    assert isinstance(obj.operations["$ref"], ReferencesToDicts)


def test_prevents_ReferencesToDicts_with_non_ref_key_in_schemas():
    config = {
        "dfm": "1.0.0",
        "info": {"version": "1.0.0", "code-package": "atlantis"},
        "schemas": {
            "MySchema": ["remote.site#/schemas/SomeSchema"],
        },
    }
    with pytest.raises(ValidationError):
        FederationObject.model_validate(config)


def test_prevents_ReferencesToDicts_with_non_ref_key_in_operations():
    config = {
        "dfm": "1.0.0",
        "info": {"version": "1.0.0", "code-package": "atlantis"},
        "operations": {
            "myOp": ["remote.site#/operations/SomeOp"],
        },
    }
    with pytest.raises(ValidationError):
        FederationObject.model_validate(config)


def test_requires_ReferencesToDicts_with_ref_key_in_schemas():
    config = {
        "dfm": "1.0.0",
        "info": {"version": "1.0.0", "code-package": "atlantis"},
        "schemas": {
            "$ref": {"type": "object"},
        },
    }
    with pytest.raises(ValidationError):
        FederationObject.model_validate(config)


def test_requires_ReferencesToDicts_with_ref_key_in_operations():
    config = {
        "dfm": "1.0.0",
        "info": {"version": "1.0.0", "code-package": "atlantis"},
        "operations": {
            "$ref": {"returns": "None"},
        },
    }
    with pytest.raises(ValidationError):
        FederationObject.model_validate(config)


def test_requires_dfm_version():
    config = {
        "info": {"version": "1.0.0", "code-package": "atlantis"},
        "schemas": {
            "MySchema": ["remote.site#/schemas/SomeSchema"],
        },
    }
    with pytest.raises(ValidationError):
        FederationObject.model_validate(config)


def test_requires_info_version():
    config = {
        "dfm": "1.0.0",
        "info": {"code-package": "atlantis"},
        "schemas": {
            "MySchema": ["remote.site#/schemas/SomeSchema"],
        },
    }
    with pytest.raises(ValidationError):
        FederationObject.model_validate(config)


def test_requires_code_package():
    config = {
        "dfm": "1.0.0",
        "info": {"version": "1.0.0"},
    }
    with pytest.raises(ValidationError):
        FederationObject.model_validate(config)


def test_requires_valid_code_package_identifier():
    config = {
        "dfm": "1.0.0",
        "info": {"version": "1.0.0", "code-package": "invalid-identifier"},
    }
    with pytest.raises(ValidationError):
        FederationObject.model_validate(config)


def test_requires_valid_api_version():
    config = {
        "dfm": "1.0.0",
        "info": {"version": "1.0", "code-package": "atlantis"},  # Missing patch version
    }
    with pytest.raises(ValidationError):
        FederationObject.model_validate(config)


def test_validate_python_path():
    validate_python_path("valid.path")


def test_invalid_python_path():
    with pytest.raises(ValueError):
        validate_python_path("invalid#.path")


def test_requires_valid_operation_returns():
    config = {"parameters": {"param1": {"type": "object"}}, "returns": "invalid#.path"}
    with pytest.raises(ValueError):
        OperationObject.model_validate(config)


def test_requires_valid_exposed_field_param():
    config = {
        "param": "invalid-param",
        "type": "str",
    }
    with pytest.raises(ValueError):
        ExposedField.model_validate(config)


def test_requires_valid_exposed_field_type():
    config = {
        "param": "valid_param",
        "type": "invalid#.type",
    }
    with pytest.raises(ValueError):
        ExposedField.model_validate(config)


def test_requires_valid_operation_field_arg():
    config = {
        "from-param": "invalid-param",
    }
    with pytest.raises(ValueError):
        OperationFieldArg.model_validate(config)


def test_requires_valid_provider_property_arg():
    config = {
        "from-provider": "valid.provider",
    }
    ProviderPropertyArg.model_validate(config)

    config = {
        "from-provider": "provider.invalid-property",
    }
    with pytest.raises(ValidationError):
        ProviderPropertyArg.model_validate(config)

    config = {
        "from-provider": "not.valid.provider",
    }
    with pytest.raises(ValidationError):
        ProviderPropertyArg.model_validate(config)


def test_requires_valid_site_property_arg():
    config = {
        "from-site": "invalid-property",
    }
    with pytest.raises(ValidationError):
        SitePropertyArg.model_validate(config)


def test_requires_valid_constructor_call_path():
    config = {
        "path": "invalid@.path",
    }
    with pytest.raises(ValidationError):
        ConstructorCall.model_validate(config)


def test_requires_valid_adapter_path():
    config = {
        "adapter": "invalid-adapter.path",
        "returns": "str",
    }
    with pytest.raises(ValidationError):
        AdapterObject.model_validate(config)

    # Test with valid path
    config = {
        "adapter": "valid.adapter.path",
        "returns": "str",
    }
    adapter = AdapterObject.model_validate(config)
    assert adapter.adapter == "valid.adapter.path"


def test_requires_valid_adapter_returns():
    config = {
        "adapter": "valid.adapter.path",
        "returns": "invalid-.returns.path",
    }
    with pytest.raises(ValidationError):
        AdapterObject.model_validate(config)

    config = {
        "adapter": "valid.adapter.path",
        "returns": "valid.returns.path",
    }
    AdapterObject.model_validate(config)


def test_requires_valid_provider_impl():
    config = {
        "impl": "invalid.impl^.path",
    }
    with pytest.raises(ValidationError):
        ProviderObject.model_validate(config)

    # Test with valid path
    config = {
        "impl": "valid.impl.path",
    }
    provider = ProviderObject.model_validate(config)
    assert provider.impl == "valid.impl.path"


def test_requires_valid_site_impl():
    config = {
        "impl": "invalid.impl.path?",
    }
    with pytest.raises(ValidationError):
        SiteObject.model_validate(config)

    # Test with valid path
    config = {
        "impl": "valid.impl.path",
    }
    site = SiteObject.model_validate(config)
    assert site.impl == "valid.impl.path"
