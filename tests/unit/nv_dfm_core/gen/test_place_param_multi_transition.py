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
Test for PlaceParam variable naming in multi-transition scenarios.

This test validates that PlaceParam variable names are correctly generated when
operations on the same site are split across multiple transitions due to cross-site
dependencies.

Bug scenario:
1. Site1 runs op1, op2 (transition t1)
2. Data flows to Site2 for op3 (transition t2)
3. Data returns to Site1 for op4, op5 which use PlaceParams (transition t3)

The bug: In t3, the generated code extracts PlaceParam values with node-specific names
(e.g., `node4_param = _data_['node4_param']`) but then uses the original PlaceParam
names in operation calls (e.g., `param` instead of `node4_param`), causing NameError.

This test directly inspects the generated code rather than running pipelines with
golden files.
"""

from collections.abc import Generator
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from nv_dfm_core.api import Pipeline, Yield
from nv_dfm_core.api._place_param import PlaceParam
from nv_dfm_core.gen.irgen import IRGen
from nv_dfm_core.gen.modgen import ModGen

from tests.assets.testfed.fed.api.ops import UnaryOp, BinaryOp
from tests._builder_helpers import FedInfoBuilder


def build_fed_info() -> FedInfoBuilder:
    """Build federation info for two sites with UnaryOp and BinaryOp."""
    b = FedInfoBuilder(federation_module_name="tests.assets.testfed")
    _ = (
        b.homesite("site1")
        .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
        .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
        .site("site2")
        .op("ops.UnaryOp", fixed_sec=10.0, fixed_size=10)
        .op("ops.BinaryOp", fixed_sec=10.0, fixed_size=10)
    )
    return b


def generate_code_for_pipeline(
    pipeline: Pipeline, fed_info: FedInfoBuilder
) -> dict[str, str]:
    """Generate code for all sites in a pipeline and return as dict[site, code]."""
    federation_module_name, homesite, built_fed_info = fed_info.build()

    irgen = IRGen()
    prepped = irgen.prepare(
        pipeline=pipeline,
        candidate_sites=["site1", "site2"],
        federation_module_name=federation_module_name,
        fed_info=built_fed_info,
        homesite=homesite,
        debug=False,
    )

    modgen = ModGen()
    generated_code: dict[str, str] = {}

    for site, netir in prepped.net_irs().items():
        runtime_module: ModuleType = MagicMock(spec=ModuleType)
        runtime_module.__name__ = f"{federation_module_name}.fed.runtime.{site}"
        generated = modgen._generate_python_code(
            this_site_runtime_module=runtime_module,
            netir=netir,
        )
        generated_code[site] = generated

    return generated_code


class TestPlaceParamMultiTransitionCodeGen:
    """Test PlaceParam variable naming when same-site ops are split across transitions.

    These tests directly inspect generated code to verify correct variable naming,
    without requiring golden files or full pipeline execution.
    """

    def test_place_param_naming_after_cross_site_hop(self):
        """Test that PlaceParams in later transitions use correct data variable names.

        Pipeline:
        - site1: op1(shared_param) - first transition (t1)
        - site2: op2 - cross-site transition
        - site1: op3(op2_result, shared_param) - third transition (t3)

        Bug: In t3, the generated code should use the data variable extracted from
        _data_ (e.g., 'node2_p2') but instead uses the original param name
        ('shared_param') which refers to self.shared_param (a Place object), not
        the actual data value.
        """
        with Pipeline(name="place_param_multi_transition") as p:
            # First transition: use 'shared_param'
            op1 = UnaryOp(
                site="site1",
                p1=PlaceParam(place="shared_param", multiuse=True),
            )

            # Cross to site2
            op2 = UnaryOp(
                site="site2",
                p1=op1,
            )

            # Back to site1 with SAME param name in new transition
            op3 = BinaryOp(
                site="site1",
                p1=op2,
                p2=PlaceParam(place="shared_param", multiuse=True),
            )

            _ = Yield(value=op3)

        fed_info = build_fed_info()
        generated = generate_code_for_pipeline(p, fed_info)

        # Get site1 generated code
        site1_code = generated["site1"]

        # Parse the code to find fire functions
        lines = site1_code.split("\n")

        # Find all t*_fire functions and analyze them
        in_fire_func = False
        current_func_name = None
        current_func_lines = []
        fire_functions = {}

        for line in lines:
            if "async def t" in line and "_fire" in line:
                # Save previous function if any
                if current_func_name:
                    fire_functions[current_func_name] = current_func_lines
                # Start new function
                current_func_name = line.split("(")[0].split()[-1]
                current_func_lines = [line]
                in_fire_func = True
            elif in_fire_func:
                if (
                    line.strip()
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                ):
                    # End of function (new class-level definition)
                    if current_func_name:
                        fire_functions[current_func_name] = current_func_lines
                    in_fire_func = False
                    current_func_name = None
                    current_func_lines = []
                else:
                    current_func_lines.append(line)

        # Save last function
        if current_func_name:
            fire_functions[current_func_name] = current_func_lines

        # Check each fire function for the bug pattern
        for func_name, func_lines in fire_functions.items():
            func_code = "\n".join(func_lines)

            # Find all variables extracted from _data_
            extracted_vars = set()
            for line in func_lines:
                if "_data_[" in line and "=" in line:
                    var_name = line.split("=")[0].strip()
                    extracted_vars.add(var_name)

            # Find all operation calls
            for line in func_lines:
                if "call_ops_" in line or "call_" in line:
                    # Check for usage of 'shared_param' directly in calls
                    # The bug: uses 'shared_param' (which references self.shared_param)
                    # instead of the extracted variable (e.g., 'node2_p2')
                    if "=shared_param" in line or "= shared_param" in line:
                        # This is the bug! Using 'shared_param' instead of extracted var
                        # Check if 'shared_param' was actually extracted in this function
                        if "shared_param" not in extracted_vars:
                            pytest.fail(
                                f"PlaceParam variable naming bug detected in {func_name}!\n"
                                f"Operation call uses 'shared_param' directly: {line.strip()}\n"
                                f"But 'shared_param' was NOT extracted from _data_ in this function.\n"
                                f"Extracted variables: {extracted_vars}\n"
                                f"The generated code incorrectly references self.shared_param "
                                f"(a Place object) instead of the data variable.\n"
                                f"Full function code:\n{func_code}"
                            )

    def test_same_param_name_in_different_transitions(self):
        """Test when the same PlaceParam name is used in both initial and later transitions.

        This reproduces the EPFL bug where 'experiment_name' was used multiple times.
        The bug is: in a later transition, the generated code references the original
        PlaceParam name (e.g., 'shared_param') instead of the data variable that was
        extracted from _data_.
        """
        with Pipeline(name="place_param_same_name") as p:
            # First transition: use 'shared_param'
            op1 = UnaryOp(
                site="site1",
                p1=PlaceParam(place="shared_param", multiuse=True),
            )

            # Cross to site2
            op2 = UnaryOp(
                site="site2",
                p1=op1,
            )

            # Back to site1 with SAME param name in new transition
            op3 = BinaryOp(
                site="site1",
                p1=op2,
                p2=PlaceParam(place="shared_param", multiuse=True),
            )

            _ = Yield(value=op3)

        fed_info = build_fed_info()
        generated = generate_code_for_pipeline(p, fed_info)

        site1_code = generated["site1"]

        # Parse fire functions
        lines = site1_code.split("\n")

        # Track whether we found the bug
        found_bug = False
        bug_details = []

        in_fire_func = False
        current_func_name = None
        current_func_lines = []

        for i, line in enumerate(lines):
            if "async def t" in line and "_fire" in line:
                # Process previous function first
                if current_func_name and current_func_lines:
                    # Check this function for the bug
                    func_code = "\n".join(current_func_lines)
                    extracted_vars = set()
                    for fline in current_func_lines:
                        if "_data_[" in fline and "=" in fline:
                            var_name = fline.split("=")[0].strip()
                            extracted_vars.add(var_name)

                    for fline in current_func_lines:
                        if ("call_ops_" in fline or "call_" in fline) and "(" in fline:
                            # Check if 'shared_param' is used in the call
                            if "=shared_param" in fline or "= shared_param" in fline:
                                # Bug: using shared_param but it wasn't extracted
                                if "shared_param" not in extracted_vars:
                                    found_bug = True
                                    bug_details.append(
                                        f"Function {current_func_name}: uses 'shared_param' "
                                        f"but only extracted {extracted_vars}"
                                    )

                # Start tracking new function
                current_func_name = line.split("(")[0].split()[-1]
                current_func_lines = [line]
                in_fire_func = True
            elif in_fire_func:
                current_func_lines.append(line)

        # Process last function
        if current_func_name and current_func_lines:
            extracted_vars = set()
            for fline in current_func_lines:
                if "_data_[" in fline and "=" in fline:
                    var_name = fline.split("=")[0].strip()
                    extracted_vars.add(var_name)

            for fline in current_func_lines:
                if ("call_ops_" in fline or "call_" in fline) and "(" in fline:
                    if "=shared_param" in fline or "= shared_param" in fline:
                        if "shared_param" not in extracted_vars:
                            found_bug = True
                            bug_details.append(
                                f"Function {current_func_name}: uses 'shared_param' "
                                f"but only extracted {extracted_vars}"
                            )

        if found_bug:
            pytest.fail(
                f"PlaceParam variable naming bug detected!\n"
                f"Details: {bug_details}\n"
                f"The generated code uses 'shared_param' which references "
                f"self.shared_param (a Place object) instead of the actual "
                f"data value extracted from _data_.\n"
                f"Generated code:\n{site1_code}"
            )


class TestPlaceParamCodeGenRegression:
    """Regression tests for PlaceParam code generation correctness."""

    def test_all_extracted_variables_are_used(self):
        """Verify that all variables extracted from _data_ are actually used in calls."""
        with Pipeline(name="param_usage_check") as p:
            op1 = UnaryOp(
                site="site1",
                p1=PlaceParam(place="p1", multiuse=True),
            )
            op2 = BinaryOp(
                site="site2",
                p1=op1,
                p2=PlaceParam(place="p2", multiuse=True),
            )
            op3 = BinaryOp(
                site="site1",
                p1=op2,
                p2=PlaceParam(place="p3", multiuse=True),
            )
            _ = Yield(value=op3)

        fed_info = build_fed_info()
        generated = generate_code_for_pipeline(p, fed_info)

        for site, code in generated.items():
            lines = code.split("\n")

            # Find all variable extractions from _data_
            extractions = {}
            for i, line in enumerate(lines):
                if "_data_[" in line and "=" in line:
                    var_name = line.split("=")[0].strip()
                    extractions[var_name] = line

            # For each extraction, verify the variable is used somewhere else in the code
            for var_name, extraction_line in extractions.items():
                # Count occurrences - should be at least 2 (definition + usage)
                occurrences = code.count(var_name)

                if occurrences < 2:
                    pytest.fail(
                        f"Variable '{var_name}' extracted but never used!\n"
                        f"Site: {site}\n"
                        f"Extraction: {extraction_line}\n"
                        f"This indicates a code generation bug where the wrong variable name "
                        f"is used in operation calls."
                    )
