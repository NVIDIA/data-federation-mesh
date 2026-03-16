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

from pathlib import Path

from nv_dfm_core.gen.apigen._apigen import ApiGen
from nv_dfm_core.gen.apigen._config_models import (
    AdapterObject,
    FederationInfoObject,
    FederationObject,
    JsonSchemaObject,
    OperationObject,
    ProviderObject,
    ReferencesToDicts,
    ReferenceToObject,
    SiteObject,
)


def create_test_federation() -> FederationObject:
    """Create a simple test federation object."""
    return FederationObject(
        dfm="1.0.0",
        info=FederationInfoObject(
            **{
                "api-version": "1.0.0",
                "code-package": "test_package",
                "description": "Test federation",
            }
        ),
        schemas={},
        operations={},
        sites={},
    )


def test_canonicalize_simple_schemas_reference():
    """Test canonicalization of a simple reference."""
    # Create a test federation with a reference
    fed = create_test_federation()
    fed.schemas["test_schema"] = ReferenceToObject(
        **{"$ref": "canonicalize_sincle_objects_remote.yaml#/schemas/other_schema"}
    )

    # Create a parser that will resolve the reference
    apigen = ApiGen(fed, base_path=Path("tests/assets/inputs/apigen_testfiles"))

    # Canonicalize the federation
    canonical = apigen._canonicalize_federation_object()

    # The reference should be replaced with the actual schema
    assert "test_schema" in canonical.schemas
    assert not isinstance(canonical.schemas["test_schema"], ReferenceToObject)
    assert isinstance(canonical.schemas["test_schema"], JsonSchemaObject)


def test_canonicalize_simple_site_references():
    """Test canonicalization of nested references."""
    # Create a test federation with nested references
    fed = create_test_federation()
    fed.sites["test_site"] = ReferenceToObject(
        **{"$ref": "canonicalize_sincle_objects_remote.yaml#/sites/other_site"}
    )

    # Create a parser that will resolve the references
    apigen = ApiGen(fed, base_path=Path("tests/assets/inputs/apigen_testfiles"))

    # Canonicalize the federation
    canonical = apigen._canonicalize_federation_object()

    # The reference should be replaced with the actual site
    assert "test_site" in canonical.sites
    assert not isinstance(canonical.sites["test_site"], ReferenceToObject)
    assert isinstance(canonical.sites["test_site"], SiteObject)


def test_canonicalize_simple_operations_references():
    """Test canonicalization of ReferencesToDicts."""
    # Create a test federation with ReferencesToDicts
    fed = create_test_federation()
    fed.operations["test_op"] = ReferenceToObject(
        **{"$ref": "canonicalize_sincle_objects_remote.yaml#/operations/other_op"}
    )

    # Create a parser that will resolve the references
    apigen = ApiGen(fed, base_path=Path("tests/assets/inputs/apigen_testfiles"))

    # Canonicalize the federation
    canonical = apigen._canonicalize_federation_object()

    # The ReferencesToDicts should be replaced with a merged dict
    assert "test_op" in canonical.operations
    assert not isinstance(canonical.operations["test_op"], ReferencesToDicts)
    assert isinstance(canonical.operations["test_op"], OperationObject)


def test_canonicalize_simple_site_interface_references():
    """Test that non-reference values are preserved during canonicalization."""
    # Create a test federation with various non-reference values
    fed = create_test_federation()
    fed.operations["some.TestOp"] = OperationObject(
        type="object",
        returns="string",
    )
    fed.sites["test_site"] = SiteObject(
        info={"name": "Test Site"},
        interface={
            "#/operations/some.TestOp": ReferenceToObject(
                **{
                    "$ref": "canonicalize_sincle_objects_remote.yaml#/adapters/other_adapter"
                }
            )
        },
        providers={},
    )

    # Create a parser
    apigen = ApiGen(fed, base_path=Path("tests/assets/inputs/apigen_testfiles"))

    # Canonicalize the federation
    canonical = apigen._canonicalize_federation_object()

    # All non-reference values should be preserved
    assert "test_site" in canonical.sites
    assert isinstance(canonical.sites["test_site"], SiteObject)
    assert "#/operations/some.TestOp" in canonical.sites["test_site"].interface
    assert isinstance(
        canonical.sites["test_site"].interface["#/operations/some.TestOp"],
        AdapterObject,
    )


def test_canonicalize_simple_provider_interface_references():
    """Test that non-reference values are preserved during canonicalization."""
    # Create a test federation with various non-reference values
    fed = create_test_federation()
    fed.operations["some.TestOp"] = OperationObject(
        type="object",
        returns="string",
    )
    fed.sites["test_site"] = SiteObject(
        info={"name": "Test Site"},
        providers={
            "other_provider": ReferenceToObject(
                **{
                    "$ref": "canonicalize_sincle_objects_remote.yaml#/providers/other_provider"
                }
            )
        },
        interface={},
    )

    # Create a parser
    apigen = ApiGen(fed, base_path=Path("tests/assets/inputs/apigen_testfiles"))

    # Canonicalize the federation
    canonical = apigen._canonicalize_federation_object()

    # All non-reference values should be preserved
    assert "test_site" in canonical.sites
    assert isinstance(canonical.sites["test_site"], SiteObject)
    assert isinstance(
        canonical.sites["test_site"].providers["other_provider"], ProviderObject
    )


def test_canonicalize_preserves_non_references():
    """Test that non-reference values are preserved during canonicalization."""
    # Create a test federation with various non-reference values
    fed = create_test_federation()
    fed.operations["some.TestOp"] = OperationObject(
        type="object",
        returns="string",
    )
    fed.sites["test_site"] = SiteObject(
        info={"name": "Test Site"},
        interface={
            "#/operations/some.TestOp": AdapterObject(
                adapter="test_adapter",
                args={"from-operation": "test_op"},
            )
        },
        providers={},
    )

    # Create a parser
    apigen = ApiGen(fed, base_path=Path("tests/assets/inputs/apigen_testfiles"))

    # Canonicalize the federation
    canonical = apigen._canonicalize_federation_object()

    # All non-reference values should be preserved
    assert "test_site" in canonical.sites
    assert isinstance(canonical.sites["test_site"], SiteObject)
    assert "#/operations/some.TestOp" in canonical.sites["test_site"].interface
    assert isinstance(
        canonical.sites["test_site"].interface["#/operations/some.TestOp"],
        AdapterObject,
    )
