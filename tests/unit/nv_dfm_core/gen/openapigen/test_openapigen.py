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

import yaml
from nv_dfm_core.gen.openapigen import OpenApiGen


def test_parsing_ogc_edr_spec():
    gen = OpenApiGen(
        openapi_file_path="tests/assets/inputs/openapigen/ogcapi-environmental-data-retrieval-1-oas31.bundled.json",
        # openapi_file_path="tests/assets/inputs/openapigen/tiny_spec/main.yaml",
        api_package_name="edr",
        adapter_package_prefix="atlantis.lib",
    )
    spec = gen.parse()
    assert spec is not None
    # print(yaml.dump(spec, indent=2))

    # generate the template parts of the federation config
    schemas = gen.generate_api_schemas()
    print(yaml.dump(schemas, indent=2))

    ops = gen.generate_api_operations()
    print(yaml.dump(ops, indent=2))

    # save the schemas and ops to their own file
    gen.write_current_federation_config_part(
        Path("tests/assets/outputs/openapigen/edr"), "api"
    )

    interface = gen.generate_site_interface()
    print(yaml.dump(interface, indent=2))

    # just for the fun of it, write the interface parts to its own file
    gen.write_current_federation_config_part(
        Path("tests/assets/outputs/openapigen/edr"), "interface"
    )

    files = gen.generate_adapters()
    # and write the adapters to their files
    gen.write_adapter_files(Path("tests/assets/outputs/openapigen/edr/lib"))

    for path, content in files.items():
        print("-" * 100)
        print(f"File: {path}")
        print(content)
        print("-" * 100)
