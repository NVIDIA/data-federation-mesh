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

# pyright: reportPrivateUsage=false
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from nv_dfm_core.gen.modgen import ModGen
from nv_dfm_core.gen.modgen._modgen import _sanitize_version_for_identifier
from nv_dfm_core.gen.modgen.ir import (
    START_PLACE_NAME,
    ActivateWhenPlacesReady,
    AdapterCallStmt,
    InPlace,
    NetIR,
    StmtRef,
    TokenSend,
    Transition,
)


class TestSanitizeVersionForIdentifier:
    """Tests for _sanitize_version_for_identifier function."""

    def test_simple_version(self):
        """Test a simple semantic version without any suffixes."""
        assert _sanitize_version_for_identifier("3.0.10") == "3_0_10"

    def test_version_with_commit_hash(self):
        """Test version with commit hash (tagged release on exact commit)."""
        assert (
            _sanitize_version_for_identifier("3.0.10+gabc12345") == "3_0_10_gabc12345"
        )

    def test_version_with_dev_and_hash(self):
        """Test dev version with commit hash (common during development)."""
        result = _sanitize_version_for_identifier("3.0.11.dev1+gdc558fc93")
        assert result == "3_0_11_gdc558fc9"

    def test_version_with_dev_hash_and_date(self):
        """Test full setuptools-scm dev version with date."""
        result = _sanitize_version_for_identifier("3.0.11.dev1+gdc558fc93.d20260203")
        assert result == "3_0_11_gdc558fc9"

    def test_version_hash_truncation(self):
        """Test that long commit hashes are truncated to 8 chars."""
        result = _sanitize_version_for_identifier("1.2.3+gabcdef1234567890")
        assert result == "1_2_3_gabcdef12"

    def test_result_is_valid_identifier(self):
        """Test that output is always a valid Python identifier component."""
        test_cases = [
            "3.0.10",
            "3.0.10+gabc12345",
            "3.0.11.dev1+gdc558fc93.d20260203",
            "1.0.0.post1+g1234abcd",
            "0.0.1",
        ]
        for version in test_cases:
            result = _sanitize_version_for_identifier(version)
            # Check it only contains valid identifier characters
            assert result.replace("_", "").isalnum(), (
                f"Invalid identifier for {version}: {result}"
            )
            # Check it doesn't start with a digit (though our prefix handles this)
            # The function output is used as a suffix, so this is less critical

    def test_no_dots_or_plus_in_output(self):
        """Test that output contains no dots or plus signs."""
        result = _sanitize_version_for_identifier("3.0.11.dev1+gdc558fc93.d20260203")
        assert "." not in result
        assert "+" not in result


@pytest.fixture
def artifacts_dir() -> Path:
    d = Path("artifacts", "outputs", "modgen")
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def net() -> NetIR:
    return NetIR(
        pipeline_name="test",
        site="somesite",
        transitions=[
            Transition(
                control_place=InPlace(
                    name=START_PLACE_NAME,
                    origin="external",
                    kind="control",
                    flavor="seq_control",
                    type="nv_dfm_core.exec.FlowInfo",
                ),
                data_places=[
                    InPlace(
                        name="p1",
                        origin="external",
                        kind="data",
                        flavor="scoped",
                        type="Any",
                    ),
                    InPlace(
                        name="p2",
                        origin="external",
                        kind="data",
                        flavor="scoped",
                        type="Any",
                    ),
                ],
                try_activate_func=ActivateWhenPlacesReady(),
                fire_body=[
                    AdapterCallStmt(
                        stmt_id="x",
                        has_users=True,
                        adapter="op1",
                        literal_params={},
                        stmt_params={},
                        is_async=True,
                    ),
                    TokenSend(
                        job=None,
                        site="somesite",
                        place="p2",
                        data=StmtRef(stmt_id="x"),
                        node_id=None,
                        is_yield=False,
                        kind="data",
                    ),
                ],
                signal_error_body=[
                    TokenSend(
                        job=None,
                        site="somesite",
                        place="p2",
                        data=StmtRef(stmt_id="_error_"),
                        node_id=None,
                        is_yield=False,
                        kind="data",
                    ),
                ],
                signal_stop_body=[],
            )
        ],
    )


def read_golden_file(test_name: str) -> str:
    """Read the golden file for a test case."""
    golden_path = Path("tests") / "assets" / "golden" / "modgen" / f"{test_name}.py"
    return golden_path.read_text()


def write_artifact(artifacts_dir: Path, test_name: str, content: str) -> None:
    _ = (artifacts_dir / f"{test_name}.py").write_text(content)


def test_simple_transition(artifacts_dir: Path, net: NetIR):
    """Test generation of a simple transition with one required place."""
    # Create a mock runtime module
    runtime_module = MagicMock(spec=ModuleType)
    runtime_module.__name__ = "examplefed.fed.runtime.reception"

    # Generate the code
    modgen = ModGen()
    generated = modgen._generate_python_code(runtime_module, net)

    write_artifact(artifacts_dir, "test_simple_transition", generated)

    # Compare with golden file
    expected = read_golden_file("test_simple_transition")
    assert generated == expected


# NOTE: the below unit tests should be well covered by the end-to-end gen tests, which are much
# easier to maintain.

# def test_transition_with_adapter(artifacts_dir: Path):
#     """Test generation of a transition that uses an adapter."""
#     runtime_module = MagicMock(spec=ModuleType)
#     runtime_module.__name__ = "examplefed.fed.runtime.reception"

#     net = NetIR(
#         places=[
#             InPlace(name=START_PLACE_NAME, kind="special", optional=False),
#             InPlace(name="input", kind="param", optional=False),
#         ],
#         trans=[
#             Transition(
#                 act=ActivationCondition(data_places=["input"], opt=[]),
#                 body=[
#                     ReadPlaceStmt(ssa_def="data", place="input"),
#                     AdapterCallStmt(
#                         ssa_def="processed",
#                         adapter="users_GreetMe",
#                         litparams={"param1": any_object_to_tagged_json_value("value1")},
#                         ssaparams={"name": StmtRef(ssa_def="data")},
#                         is_async=True,
#                     ),
#                     SendStmt(
#                         site=None,
#                         place="output",
#                         data=StmtRef(ssa_def="processed"),
#                     ),
#                 ],
#                 out=[OutPlace(to_site=None, to_place="output")],
#             )
#         ],
#     )

#     modgen = ModGen()
#     logger = MagicMock()
#     generated = modgen._generate_python_code(runtime_module, net, logger)

#     write_artifact(artifacts_dir, "test_transition_with_adapter", generated)

#     expected = read_golden_file("test_transition_with_adapter")
#     assert generated == expected

# def test_multiple_transitions(artifacts_dir: Path):
#     """Test generation of multiple transitions."""
#     runtime_module = MagicMock(spec=ModuleType)
#     runtime_module.__name__ = "examplefed.fed.runtime.reception"

#     net = NetIR(
#         places=[
#             InPlace(name=START_PLACE_NAME, kind="special", optional=False),
#             InPlace(name="input1", kind="param", optional=False),
#             InPlace(name="input2", kind="param", optional=False),
#         ],
#         trans=[
#             Transition(
#                 act=ActivationCondition(data_places=["input1"], opt=[]),
#                 body=[
#                     ReadPlaceStmt(ssa_def="data1", place="input1"),
#                     SendStmt(
#                         site=None,
#                         place="output1",
#                         data=StmtRef(ssa_def="data1"),
#                     ),
#                 ],
#                 out=[OutPlace(to_site=None, to_place="output1")],
#             ),
#             Transition(
#                 act=ActivationCondition(data_places=["input2"], opt=[]),
#                 body=[
#                     ReadPlaceStmt(ssa_def="data2", place="input2"),
#                     SendStmt(
#                         job="abcd1234",
#                         site="somesite",
#                         place="output2",
#                         data=StmtRef(ssa_def="data2"),
#                     ),
#                 ],
#                 out=[OutPlace(job="abcd1234", to_site="somesite", to_place="output2")],
#             ),
#         ],
#     )

#     modgen = ModGen()
#     logger = MagicMock()
#     generated = modgen._generate_python_code(runtime_module, net, logger)

#     write_artifact(artifacts_dir, "test_multiple_transitions", generated)

#     expected = read_golden_file("test_multiple_transitions")
#     assert generated == expected


# def test_discovery_transitions(artifacts_dir: Path):
#     """Test generation of multiple transitions."""
#     runtime_module = MagicMock(spec=ModuleType)
#     runtime_module.__name__ = "examplefed.fed.runtime.reception"

#     net = NetIR(
#         places=[
#             InPlace(name=START_PLACE_NAME, kind="special", optional=False),
#         ],
#         trans=[
#             Transition(
#                 act=ActivationCondition(data_places=[START_PLACE_NAME], opt=[]),
#                 body=[
#                     AdapterDiscoveryStmt(
#                         prov="someprovider",
#                         ssa_def="a1",
#                         nodeid=1,
#                         adapter="users_GreetMe",
#                         litparams={"param1": any_object_to_tagged_json_value("value1")},
#                         is_async=True,
#                     ),
#                     AdapterDiscoveryStmt(
#                         prov=None,
#                         ssa_def="a2",
#                         nodeid="my_node_id",
#                         adapter="users_GreetMe",
#                         litparams={"param1": any_object_to_tagged_json_value("value1")},
#                         is_async=True,
#                     ),
#                     SendDiscoveryStmt(),
#                 ],
#                 out=[OutPlace(to_site=None, to_place="discovery")],
#             ),
#         ],
#     )

#     modgen = ModGen()
#     logger = MagicMock()
#     generated = modgen._generate_python_code(runtime_module, net, logger)

#     write_artifact(artifacts_dir, "test_discovery_transitions", generated)

#     expected = read_golden_file("test_discovery_transitions")
#     assert generated == expected


# def test_invalid_net_no_required_place():
#     """Test that validation fails when a transition has no required places."""
#     runtime_module = MagicMock(spec=ModuleType)
#     runtime_module.__name__ = "test_runtime"

#     net = NetIR(
#         places=[
#             InPlace(name=START_PLACE_NAME, kind="special", optional=False),
#             InPlace(name="input", kind="param", optional=True),
#         ],
#         trans=[
#             Transition(
#                 act=ActivationCondition(data_places=[], opt=["input"]),
#                 body=[],
#                 out=[],
#             )
#         ],
#     )

#     # This should be caught by NetIR.validate()
#     with pytest.raises(AssertionError, match="At least one required place is needed"):
#         net.validate_net_ir()

#     # But let's verify ModGen also fails
#     modgen = ModGen()
#     logger = MagicMock()
#     with pytest.raises(AssertionError):
#         modgen._generate_python_code(runtime_module, net, logger)


# def test_invalid_net_undefined_variable():
#     """Test that validation fails when a variable is used before definition."""
#     runtime_module = MagicMock(spec=ModuleType)
#     runtime_module.__name__ = "test_runtime"

#     net = NetIR(
#         places=[
#             InPlace(name=START_PLACE_NAME, kind="special", optional=False),
#             InPlace(name="input", kind="param", optional=False),
#             InPlace(name="output", kind="param", optional=True),
#         ],
#         trans=[
#             Transition(
#                 act=ActivationCondition(data_places=["input"], opt=[]),
#                 body=[
#                     SendStmt(
#                         site="somesite",
#                         place="output",
#                         data=StmtRef(ssa_def="data"),  # data not defined
#                     ),
#                 ],
#                 out=[OutPlace(to_site="somesite", to_place="output")],
#             )
#         ],
#     )

#     # This should be caught by NetIR.validate()
#     with pytest.raises(ValueError, match="SSA definition data does not exist"):
#         net.validate_net_ir()

#     # But let's verify ModGen also fails
#     modgen = ModGen()
#     logger = MagicMock()
#     with pytest.raises(ValueError, match="SSA definition data does not exist"):
#         modgen._generate_python_code(runtime_module, net, logger)


# def test_invalid_net_missing_out_place():
#     """Test that validation fails when sending to a non-out place."""
#     runtime_module = MagicMock(spec=ModuleType)
#     runtime_module.__name__ = "test_runtime"

#     net = NetIR(
#         places=[
#             InPlace(name=START_PLACE_NAME, kind="special", optional=False),
#             InPlace(name="input", kind="param", optional=False),
#             InPlace(name="output", kind="param", optional=True),
#         ],
#         trans=[
#             Transition(
#                 act=ActivationCondition(data_places=["input"], opt=[]),
#                 body=[
#                     ReadPlaceStmt(ssa_def="data", place="input"),
#                     SendStmt(
#                         site="somesite",
#                         place="output",
#                         data=StmtRef(ssa_def="data"),
#                     ),
#                 ],
#                 out=[OutPlace(to_site="somesite", to_place="wrong")],  # wrong place
#             )
#         ],
#     )

#     # This should be caught by NetIR.validate()
#     with pytest.raises(
#         AssertionError,
#         match="Send to site somesite, place output is missing in the out_places of the transition.",
#     ):
#         net.validate_net_ir()

#     # But let's verify ModGen also fails
#     modgen = ModGen()
#     logger = MagicMock()
#     with pytest.raises(AssertionError):
#         modgen._generate_python_code(runtime_module, net, logger)
