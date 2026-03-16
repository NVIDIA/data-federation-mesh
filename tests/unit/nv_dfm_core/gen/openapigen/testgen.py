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

import yaml
import json
import random
from typing import Any
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def read_operations_from_yaml(yaml_path: str) -> dict[str, Any]:
    """
    Reads operation names and their parameters from a YAML file.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        dict: Dictionary where keys are operation names and values are parameter dicts
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    operations = data.get("operations", {})

    result = {}
    for op_name, op_data in operations.items():
        result[op_name] = {
            "description": op_data.get("description", ""),
            "parameters": op_data.get("parameters", {}),
            "required": op_data.get("required", []),
            "returns": op_data.get("returns", "Any"),
        }

    return result


def generate_import_from_operation_name(
    operation_name: str, api_package_path: str
) -> tuple[str, str]:
    """
    Converts an operation name to an import statement.

    Rule: if operation is 'this.is.Operation', then import should be
    'from ogc_api_test.fed.api.this.is import Operation'

    Args:
        operation_name: String like 'ogc_tiles.getAPICollections'
        api_path: str - path to the api package: 'ogc_api_test.fed.api'

    Returns:
        tuple: (module_path, class_name) for the import
    """
    parts = operation_name.split(".")
    if len(parts) == 1:
        # No dots, just the operation name
        class_name = parts[0][0].upper() + parts[0][1:] if parts[0] else parts[0]
        return api_package_path, class_name

    # Last part is the class name, rest is the module path
    class_name = parts[-1]
    # Capitalize first letter of class name
    class_name = class_name[0].upper() + class_name[1:] if class_name else class_name
    module_parts = parts[:-1]
    module_path = f"{api_package_path}.{'.'.join(module_parts)}"

    return module_path, class_name


def yaml_type_to_python_type(yaml_type: str) -> str:
    """
    Converts YAML type to Python type annotation.

    Args:
        yaml_type: YAML type string (e.g., 'string', 'integer', 'array')

    Returns:
        str: Python type annotation
    """
    type_mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }
    return type_mapping.get(yaml_type, "Any")


def process_parameters(parameters: dict) -> list:
    """
    Processes parameter definitions and extracts parameter names and their metadata.

    Args:
        parameters: Dict of parameter definitions from YAML

    Returns:
        list: List of parameter info dicts with name, type, required, etc.
    """
    param_list = []
    for param_name, param_spec in parameters.items():
        yaml_type = param_spec.get("type", "Any")
        python_type = yaml_type_to_python_type(yaml_type)

        param_info = {
            "name": param_name,
            "type": python_type,
            "required": param_spec.get("required", True),  # Default to required
            "enum": param_spec.get("enum"),
            "items": param_spec.get("items"),
            "minimum": param_spec.get("minimum"),
        }
        param_list.append(param_info)

    return param_list


def generate_test_value(param_name, param_spec):
    """
    Generates a test value for a parameter based on its specification.

    Args:
        param_name: Name of the parameter
        param_spec: Parameter specification dict from YAML

    Returns:
        Appropriate test value based on type and constraints
    """
    # Special case for server parameter
    if param_name == "server":
        return "https://maps.ecere.com/ogcapi"

    # If enum exists, use the first value
    if "enum" in param_spec and param_spec["enum"]:
        return param_spec["enum"][0]

    # Get the type
    param_type = param_spec.get("type", "string")

    # Generate value based on type
    if param_type == "string":
        # Generate a sample string
        return f"test_{param_name}"

    elif param_type == "integer":
        # Use minimum if specified, otherwise a random value
        minimum = param_spec.get("minimum", 0)
        maximum = param_spec.get("maximum", minimum + 100)
        return random.randint(minimum, maximum)

    elif param_type == "number":
        # Use minimum if specified, otherwise a random float
        minimum = param_spec.get("minimum", 0.0)
        maximum = param_spec.get("maximum", minimum + 100.0)
        return round(random.uniform(minimum, maximum), 2)

    elif param_type == "boolean":
        return random.choice([True, False])

    elif param_type == "array":
        # Generate array based on items specification if available
        items = param_spec.get("items", {})
        if items:
            # Generate a sample array with one or two items
            sample_values = []
            for _ in range(random.randint(1, 2)):
                sample_values.append(generate_test_value(f"{param_name}_item", items))
            return sample_values
        return []

    elif param_type == "object":
        # Return empty object
        return {}

    # Default: return None
    return None


def generate_test_parameters_json(yaml_path, output_json_path=None):
    """
    Generates a JSON file with test parameter values for all operations.

    Args:
        yaml_path: Path to the YAML configuration file
        output_json_path: Optional path to write the JSON file

    Returns:
        dict: Dictionary with operation names as keys and parameter dicts as values
    """
    # Read operations from YAML
    operations = read_operations_from_yaml(yaml_path)

    # Generate test parameters for each operation
    test_data = {}
    for op_name, op_data in operations.items():
        params = op_data["parameters"]
        test_params = {}

        for param_name, param_spec in params.items():
            if param_name in op_data["required"]:
                test_params[param_name] = generate_test_value(param_name, param_spec)

        test_data[op_name] = test_params

    # Optionally write to JSON file
    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(test_data, f, indent=2)

    return test_data


def generate_python_script(
    yaml_path: str, api_package_path: str, template_path: str, output_path=None
) -> str:
    """
    Generates a Python script with imports for all operations in the YAML file.

    Args:
        yaml_path: Path to the YAML configuration file
        template_path: Path to the Jinja2 template file
        output_path: Optional path to write the generated script

    Returns:
        str: Generated Python script content
    """
    # Read operations from YAML
    operations = read_operations_from_yaml(yaml_path)

    # Generate import statements and operation calls
    imports = []
    for op_name, op_data in operations.items():
        module_path, class_name = generate_import_from_operation_name(
            op_name, api_package_path
        )
        if class_name[:4] == "Post":
            continue
        alias = "_".join(module_path.split(".")[3:] + [class_name.lower()])

        # Process parameters for this operation
        params = process_parameters(op_data["parameters"])

        imports.append(
            {
                "alias": alias,
                "module": module_path,
                "name": class_name,
                "full_name": op_name,
                "data": op_data,
                "parameters": [p for p in params if p["name"] in op_data["required"]],
            }
        )

    # Load and render template
    template_dir = Path(template_path).parent
    template_name = Path(template_path).name

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    rendered = template.render(imports=imports, operations=operations)

    # Optionally write to file
    if output_path:
        with open(output_path, "w") as f:
            f.write(rendered)

    return rendered


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        raise RuntimeError("Missing command line parameter.")

    # Example usage
    config_path = sys.argv[
        1
    ]  # Path(__file__).parent.parent / 'configs' / '_api.dfm.part.yaml'
    template_path = (
        Path(__file__).parent / "templates" / "tests_pipeline.jinja2"
    )  # Path(__file__).parent / 'templates' / 'tests_pipeline.jinja2'
    output_path = sys.argv[2]  # Path(__file__).parent / 'generated_tests.py'
    json_output_path = sys.argv[3]  # Path(__file__).parent / 'test_parameters.json'
    api_package_path = sys.argv[4]  #'ogc_api_test.fed.api'

    # Generate Python script
    script = generate_python_script(
        yaml_path=config_path,
        api_package_path=api_package_path,
        template_path=template_path,
        output_path=output_path,
    )
    print(f"Generated Python script: {output_path}")

    # Generate test parameters JSON
    test_params = generate_test_parameters_json(config_path, json_output_path)
    print(f"Generated test parameters JSON: {json_output_path}")
    print(f"\nTest parameters:\n{json.dumps(test_params, indent=2)}")
