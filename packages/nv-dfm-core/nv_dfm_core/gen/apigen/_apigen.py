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

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Literal

import yaml
from datamodel_code_generator.parser.jsonschema import JsonSchemaObject
from jinja2 import (
    ChoiceLoader,
    Environment,
    FileSystemLoader,
    PackageLoader,
    select_autoescape,
)

from nv_dfm_core.exec._helpers import site_name_to_identifier
from nv_dfm_core.gen.irgen._fed_info import (
    ComputeCostInfo,
    FedInfo,
    OperationInfo,
    ProviderInfo,
    SiteInfo,
)

from ._config_models import (
    AdapterObject,
    FederationObject,
    OperationObject,
    ProviderObject,
    ReferencesToDicts,
    ReferenceToObject,
    SiteObject,
)
from ._config_parser import ConfigParser

license_header = """# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

init_py = "__init__.py"


class ApiGen:
    """Generator for federation API code from configuration files.

    ApiGen takes a federation configuration and generates Python code including
    schema models, operation classes, and site-specific adapters.
    """

    def __init__(
        self,
        federation_object: FederationObject,
        base_path: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Base path is used to resolve relative refs in the config file.
        """
        self._federation_object: FederationObject = federation_object
        self._base_path: Path | None = base_path
        self._logger: logging.Logger = logger if logger else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    def generate_api(
        self,
        language: Literal["python"],
        outpath: Path | None = None,
        delete_generated_packages_first: bool = True,
    ) -> list[tuple[Path, str]] | None:
        """
        Outpath: the path where to write the generated files. If None,
        no files are written and only the results are returned.

        Returns a list or (Path, str) tuples, one for each file that got generated.
        The returned paths are relative paths.
        None means there was nothing to generate.
        """
        if not language == "python":
            raise ValueError(f"Language {language} not supported")  # pyright: ignore[reportUnreachable]

        parser = ConfigParser(
            base_path=self._base_path.absolute() if self._base_path else None
        )

        parser.parse_federation_object(self._federation_object)
        result = parser.parse()

        if outpath:
            code_package = self._federation_object.info.code_package
            packages_with_generated_code = ["fed"]
            # first a sanity check to make sure all our returned paths are
            # only in the expected subpackages
            for path, _ in result:
                path_parts = path.parts
                if len(path_parts) < 2:
                    raise ValueError(
                        f"Path {path} of generated file is unexpectedly short. Expected generated code to have at least two parts: <code_package> and the '/fed' subpackage."
                    )
                if not path_parts[0] == code_package:
                    raise ValueError(
                        f"Path {path} of generated file does not start with the expected code package: {code_package}"
                    )
                if (
                    path_parts[1] != init_py
                    and path_parts[1] not in packages_with_generated_code
                ):
                    raise ValueError(
                        f"Path {path} of generated file is not in one of the expected subpackages: {packages_with_generated_code}"
                    )

            # now we know that we are only working in the expected subpackages
            # so we can delete all of them first
            if delete_generated_packages_first:
                for pckg in packages_with_generated_code:
                    pckg_path = outpath.joinpath(
                        self._federation_object.info.code_package, pckg
                    )
                    if pckg_path.exists():
                        self._logger.info(f"Deleting existing folder {pckg_path}")
                        shutil.rmtree(pckg_path)

            # and write the files
            outpath.mkdir(parents=True, exist_ok=True)
            for path, content in result:
                # We technically only generate code in the expected subpackages.
                # But the result will also contain an __init__.py file for
                # the root package. We don't want to overwrite that if it exists,
                # only if it doesn't exist, because that's not counting as a generated file.
                if (
                    len(path.parts) == 2
                    and path.parts[1] == init_py
                    and outpath.joinpath(path).exists()
                ):
                    self._logger.info(
                        f"Root package's __init__.py {path} already exists, skipping."
                    )
                    continue
                target_file = outpath.joinpath(path)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                _ = target_file.write_text(content, encoding="utf-8")
            self._logger.info(f"Generated {len(result)} files in {outpath}")

    def generate_runtime(
        self,
        language: Literal["python"],
        outpath: Path | None = None,
        delete_generated_packages_first: bool = True,
        generate_for_sites: list[str] | None = None,
    ) -> list[tuple[Path, str]]:
        """Generate runtime code for each site in the federation including site-specific adapters and resources."""
        if not language == "python":
            raise ValueError(f"Language {language} not supported")  # pyright: ignore[reportUnreachable]

        templates = self._load_runtime_templates()
        canonical_federation = self._canonicalize_federation_object()
        result = self._generate_runtime_files(
            canonical_federation, templates, generate_for_sites
        )

        if outpath:
            self._write_runtime_files(
                result, outpath, canonical_federation, delete_generated_packages_first
            )

        _ = self._generate_fed_info(outpath)

        return result

    def _load_runtime_templates(self) -> dict[str, Any]:
        """Load Jinja2 templates for runtime code generation."""
        template_dir = Path(__file__).parent / "templates"
        choice_loader = ChoiceLoader(
            [
                PackageLoader(
                    "nv_dfm_core", "gen/apigen/templates"
                ),  # Tries package resources first
                FileSystemLoader(template_dir),  # Falls back to local filesystem
            ]
        )
        _jinja_env = Environment(loader=choice_loader, autoescape=select_autoescape())
        return {
            "this_site": _jinja_env.get_template("runtime_this_site.py.jinja2"),
            "site_init": _jinja_env.get_template("runtime_site_init.py.jinja2"),
        }

    def _generate_runtime_files(
        self,
        canonical_federation: FederationObject,
        templates: dict[str, Any],
        generate_for_sites: list[str] | None,
    ) -> list[tuple[Path, str]]:
        """Generate runtime files for all sites."""
        result: list[tuple[Path, str]] = []
        fed_package = Path(canonical_federation.info.code_package, "fed")
        runtime_package = fed_package / "runtime"

        # Create package init files
        result.extend(self._create_package_init_files(fed_package, runtime_package))

        # Generate site-specific files
        for site_name, site in canonical_federation.sites.items():
            if generate_for_sites and site_name not in generate_for_sites:
                continue
            if not isinstance(site, SiteObject):
                raise ValueError(
                    f"Site {site_name} is not a SiteObject. Currently, cannot handle refs yet."
                )

            site_files = self._generate_site_files(
                canonical_federation, site_name, site, runtime_package, templates
            )
            result.extend(site_files)

        return result

    def _create_package_init_files(
        self, fed_package: Path, runtime_package: Path
    ) -> list[tuple[Path, str]]:
        """Create __init__.py files for fed and runtime packages."""
        result: list[tuple[Path, str]] = []

        # create empty <code_package>.fed.__init__.py if it doesn't exist
        if not Path(fed_package, init_py).exists():
            result.append((fed_package / init_py, license_header))

        # create empty fed.runtime.__init__.py
        result.append((runtime_package / init_py, license_header))

        return result

    def _generate_site_files(
        self,
        canonical_federation: FederationObject,
        site_name: str,
        site: SiteObject,
        runtime_package: Path,
        templates: dict[str, Any],
    ) -> list[tuple[Path, str]]:
        """Generate runtime files for a single site."""
        site_identifier = site_name_to_identifier(site_name)
        site_package = runtime_package / site_identifier

        template_context = {
            "federation": canonical_federation,
            "site_name": site_identifier,
            "site": site,
            "code_package": canonical_federation.info.code_package,
            "json": json,
        }

        result: list[tuple[Path, str]] = []

        # Generate _this_site.py
        content = templates["this_site"].render(**template_context)
        result.append((site_package / "_this_site.py", content))

        # Generate __init__.py
        content = templates["site_init"].render(**template_context)
        result.append((site_package / init_py, content))

        return result

    def _write_runtime_files(
        self,
        result: list[tuple[Path, str]],
        outpath: Path,
        canonical_federation: FederationObject,
        delete_generated_packages_first: bool,
    ) -> None:
        """Write generated runtime files to disk."""
        runtime_package = Path(canonical_federation.info.code_package, "fed", "runtime")

        if delete_generated_packages_first:
            pckg_path = outpath / runtime_package
            if pckg_path.exists():
                self._logger.info(f"Deleting existing folder {pckg_path}")
                shutil.rmtree(pckg_path)

        for path, content in result:
            target_file = outpath.joinpath(path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            _ = target_file.write_text(content, encoding="utf-8")

        self._logger.info(f"Generated {len(result)} files in {outpath}")

    def _generate_fed_info(self, outpath: Path | None = None) -> FedInfo:
        """Generate a dict representation of the federation info object.
        The FedInfo object is used during irgen and gets the information from
        the api generation time over to runtime, where irgen uses it to make its decisions.
        """
        # Canonicalize the federation object to resolve all references
        canonical_federation = self._canonicalize_federation_object()

        def generate_interface_info(
            interface: dict[str, Any],
        ) -> dict[str, OperationInfo]:
            """Convert adapter interface to operation info with compute costs and async flags."""
            result: dict[str, OperationInfo] = {}
            for op_name, op in interface.items():
                assert isinstance(op, AdapterObject)
                if not op.compute_cost:
                    self._logger.warning(
                        f"Compute cost not set for operation {op_name}. Using suboptimal default values"
                    )
                    compute_cost = ComputeCostInfo(
                        fixed_time=1_000_000, fixed_size=10_000_000
                    )  # 10MB fixed and runs forever
                else:
                    compute_cost = op.compute_cost

                real_op_name = op_name.split("/")[-1]
                result[real_op_name] = OperationInfo(
                    operation=real_op_name,
                    compute_cost=compute_cost,
                    is_async=op.is_async,
                )
            return result

        site_infos: dict[str, SiteInfo] = {}
        for site_name, site in canonical_federation.sites.items():
            assert isinstance(site, SiteObject)
            providers: dict[str, ProviderInfo] = {}
            for provider_name, provider in site.providers.items():
                assert isinstance(provider, ProviderObject)
                provider_info = ProviderInfo(
                    interface=generate_interface_info(provider.interface),
                )
                providers[provider_name] = provider_info
            site_infos[site_name_to_identifier(site_name)] = SiteInfo(
                interface=generate_interface_info(site.interface),
                providers=providers,
                # we don't add a default send cost here, if it's missing. Instead, FedInfo
                # will use a default cost if it's missing.
                send_cost=site.send_cost,
            )
        fed_info = FedInfo(sites=site_infos)

        if outpath:
            target_file = (
                outpath
                / canonical_federation.info.code_package
                / "fed"
                / "runtime"
                / "fed_info.json"
            )
            _ = target_file.write_text(
                fed_info.model_dump_json(indent=2), encoding="utf-8"
            )

        return fed_info

    @staticmethod
    def from_dict(conf_dict: dict[str, Any]):
        config_obj = FederationObject.model_validate(conf_dict)
        return ApiGen(config_obj)

    @staticmethod
    def from_yaml_file(filename: str | Path):
        """Create an ApiGen instance from a YAML configuration file."""
        with open(filename, mode="r", encoding="utf-8") as f:
            conf_dict = yaml.load(f, Loader=yaml.SafeLoader)
        return ApiGen.from_dict(conf_dict)

    @staticmethod
    def from_yaml_string(yaml_str: str):
        conf_dict = yaml.load(yaml_str, Loader=yaml.SafeLoader)
        return ApiGen.from_dict(conf_dict)

    @staticmethod
    def from_json_file(filename: str | Path):
        """Create an ApiGen instance from a JSON configuration file."""
        with open(filename, mode="r", encoding="utf-8") as f:
            conf_dict = json.load(f)
        return ApiGen.from_dict(conf_dict)

    @staticmethod
    def from_json_string(json_str: str):
        conf_dict = json.loads(json_str)
        return ApiGen.from_dict(conf_dict)

    def _canonicalize_federation_object(self) -> FederationObject:
        """Canonicalizes the federation object by resolving all references.

        Returns a new FederationObject where all ReferenceToObject and ReferencesToDicts
        have been replaced with their corresponding pydantic models.
        """
        parser = ConfigParser(
            base_path=self._base_path.absolute() if self._base_path else None
        )
        parser.raw_obj = self._federation_object.model_dump()

        def traverse_fed_object(fed: FederationObject) -> FederationObject:
            """Recursively traverse a FederationObject, processing all fields."""
            return fed.model_copy(
                update={
                    "schemas": traverse_schemas(fed.schemas),
                    "operations": traverse_operations(fed.operations),
                    "sites": traverse_sites(fed.sites),
                }
            )

        def traverse_schemas(
            schemas: dict[
                str, ReferencesToDicts | ReferenceToObject | JsonSchemaObject
            ],
        ) -> dict[str, JsonSchemaObject]:
            """Recursively traverse a dict of schemas, processing all fields."""
            """Schemas are just plain JsonSchema objects"""
            result: dict[str, JsonSchemaObject] = {}
            for schema_name, rhs in schemas.items():
                if isinstance(rhs, JsonSchemaObject):
                    result[schema_name] = rhs
                elif isinstance(rhs, ReferenceToObject):
                    ref_model = parser.resolve_object(rhs, JsonSchemaObject)
                    result[schema_name] = ref_model
                elif isinstance(rhs, ReferencesToDicts):  # pyright: ignore[reportUnnecessaryIsInstance]
                    for ref in rhs.root:
                        remote_dict = parser.resolve_raw_obj(ref)
                        for remote_name, remote_obj in remote_dict.items():
                            ref_model = JsonSchemaObject.model_validate(remote_obj)
                            result[remote_name] = ref_model
                else:
                    raise ValueError(f"Unexpected type: {type(rhs)}")
            return result

        def traverse_operations(
            operations: dict[
                str, ReferencesToDicts | ReferenceToObject | OperationObject
            ],
        ) -> dict[str, OperationObject]:
            """Recursively traverse a dict of operations, processing all fields."""
            result: dict[str, OperationObject] = {}
            for op_name, rhs in operations.items():
                if isinstance(rhs, OperationObject):
                    result[op_name] = rhs
                elif isinstance(rhs, ReferenceToObject):
                    ref_model = parser.resolve_object(rhs, OperationObject)
                    result[op_name] = ref_model
                elif isinstance(rhs, ReferencesToDicts):  # pyright: ignore[reportUnnecessaryIsInstance]
                    for ref in rhs.root:
                        remote_dict = parser.resolve_raw_obj(ref)
                        for remote_name, remote_obj in remote_dict.items():
                            ref_model = OperationObject.model_validate(remote_obj)
                            result[remote_name] = ref_model
                else:
                    raise ValueError(f"Unexpected type: {type(rhs)}")
            return result

        def traverse_sites(
            sites: dict[str, ReferencesToDicts | ReferenceToObject | SiteObject],
        ) -> dict[str, SiteObject]:
            """Recursively traverse a dict of sites, processing all fields."""
            result: dict[str, SiteObject] = {}
            for op_name, rhs in sites.items():
                if isinstance(rhs, SiteObject):
                    result[op_name] = traverse_site(rhs)
                elif isinstance(rhs, ReferenceToObject):
                    ref_model = parser.resolve_object(rhs, SiteObject)
                    result[op_name] = traverse_site(ref_model)
                elif isinstance(rhs, ReferencesToDicts):  # pyright: ignore[reportUnnecessaryIsInstance]
                    for ref in rhs.root:
                        remote_dict = parser.resolve_raw_obj(ref)
                        for remote_name, remote_obj in remote_dict.items():
                            ref_model = SiteObject.model_validate(remote_obj)
                            result[remote_name] = traverse_site(ref_model)
                else:
                    raise ValueError(f"Unexpected type: {type(rhs)}")
            return result

        def traverse_site(site: SiteObject) -> SiteObject:
            """Recursively traverse a SiteObject, processing all fields."""
            return site.model_copy(
                update={
                    "interface": traverse_interface(site.interface),
                    "providers": traverse_providers(site.providers),
                }
            )

        def traverse_interface(
            interface: dict[str, ReferencesToDicts | ReferenceToObject | AdapterObject],
        ) -> dict[str, AdapterObject]:
            """Recursively traverse a dict of interfaces, processing all fields."""
            result: dict[str, AdapterObject] = {}
            for op_name, rhs in interface.items():
                if isinstance(rhs, AdapterObject):
                    result[op_name] = rhs
                elif isinstance(rhs, ReferenceToObject):
                    ref_model = parser.resolve_object(rhs, AdapterObject)
                    result[op_name] = ref_model
                elif isinstance(rhs, ReferencesToDicts):  # pyright: ignore[reportUnnecessaryIsInstance]
                    for ref in rhs.root:
                        remote_dict = parser.resolve_raw_obj(ref)
                        for remote_name, remote_obj in remote_dict.items():
                            ref_model = AdapterObject.model_validate(remote_obj)
                            result[remote_name] = ref_model
                else:
                    raise ValueError(f"Unexpected type: {type(rhs)}")
            return result

        def traverse_providers(
            providers: dict[
                str, ReferencesToDicts | ReferenceToObject | ProviderObject
            ],
        ) -> dict[str, ProviderObject]:
            """Recursively traverse a dict of providers, processing all fields."""
            result: dict[str, ProviderObject] = {}
            for op_name, rhs in providers.items():
                if isinstance(rhs, ProviderObject):
                    result[op_name] = traverse_provider(rhs)
                elif isinstance(rhs, ReferenceToObject):
                    ref_model = parser.resolve_object(rhs, ProviderObject)
                    result[op_name] = traverse_provider(ref_model)
                elif isinstance(rhs, ReferencesToDicts):  # pyright: ignore[reportUnnecessaryIsInstance]
                    for ref in rhs.root:
                        remote_dict = parser.resolve_raw_obj(ref)
                        for remote_name, remote_obj in remote_dict.items():
                            ref_model = ProviderObject.model_validate(remote_obj)
                            result[remote_name] = traverse_provider(ref_model)
                else:
                    raise ValueError(f"Unexpected type: {type(rhs)}")
            return result

        def traverse_provider(provider: ProviderObject) -> ProviderObject:
            """Recursively traverse a ProviderObject, processing all fields."""
            return provider.model_copy(
                update={
                    "interface": traverse_interface(provider.interface),
                }
            )

        return traverse_fed_object(self._federation_object)
