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
import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal
from unittest.mock import MagicMock

from nv_dfm_core.api import Pipeline
from nv_dfm_core.api._prepared_pipeline import PreparedPipeline
from nv_dfm_core.exec._frame import Frame
from nv_dfm_core.gen.irgen import FedInfo, IRGen
from nv_dfm_core.gen.irgen.graph import Graph, GraphState
from nv_dfm_core.gen.modgen import ModGen
from nv_dfm_core.gen.modgen.ir import IRStmt, NetIR
from nv_dfm_core.session._job import JobStatus
from nv_dfm_core.session._session import Session
from tests._builder_helpers import FedInfoBuilder


@dataclass
class PipelineTestCase:
    # used as a subfolder for the golden sample
    test_name: str | None
    test_variant: str | None
    pipeline: Pipeline
    candidate_sites: list[str]
    fed_info: FedInfoBuilder
    expected_net_irs: dict[str, NetIR] | None
    # set to None if you don't want to run the pipeline
    run_params: dict[str, Any] | list[dict[str, Any]] | None
    delayed_params: list[dict[str, Any]] | None = None
    debug: bool = False


@dataclass
class TokenResponse:
    from_site: str
    from_node: int | str | None
    frame: Frame
    to_place: str
    data: Any


def compare_body(name: str, found: list[IRStmt], expected: list[IRStmt], debug: bool):
    assert len(found) == len(expected), (
        f"{name} statements do not match: {len(found)} != {len(expected)}"
    )
    for found_stmt, expected_stmt in zip(found, expected):
        if debug:
            print("-" * 120)
            print(f"found_stmt: {repr(found_stmt)}")
            print(f"expected_stmt: {repr(expected_stmt)}")
            print(f"found_stmt == expected_stmt: {found_stmt == expected_stmt}")
            print("-" * 120)
        assert found_stmt == expected_stmt, (
            f"{name} statements do not match: {found_stmt} != {expected_stmt}"
        )


def compare_net_irs(found: NetIR, expected: NetIR, debug: bool):
    if debug:
        print("=" * 120)
        print("Comparing NetIRs")
        print("found:")
        print(f"{repr(found)}")
        print("-" * 120)
        print("expected:")
        print(f"{repr(expected)}")
        print("=" * 120)
    assert found.site == expected.site, (
        f"UnitTest - Site does not match: {found.site} != {expected.site}"
    )
    assert len(found.transitions) == len(expected.transitions), (
        f"UnitTest - Number of transitions do not match: {len(found.transitions)} != {len(expected.transitions)}"
    )
    for found_trans, expected_trans in zip(found.transitions, expected.transitions):
        assert found_trans.control_place == expected_trans.control_place, (
            f"UnitTest - Control place does not match: found '{found_trans.control_place}' != expected '{expected_trans.control_place}'"
        )
        assert set(found_trans.data_places) == set(expected_trans.data_places), (
            f"UnitTest - Required places do not match: found {found_trans.data_places} != expected {expected_trans.data_places}"
        )
        compare_body(
            "UnitTest - Fire body",
            found_trans.fire_body,
            expected_trans.fire_body,
            debug,
        )
        compare_body(
            "UnitTest - Signal error body",
            found_trans.signal_error_body,
            expected_trans.signal_error_body,
            debug,
        )
        compare_body(
            "UnitTest - Signal stop body",
            found_trans.signal_stop_body,
            expected_trans.signal_stop_body,
            debug,
        )


class PipelineCompilationTestBase(ABC):
    """Base class for pipeline compilation tests.

    NOTE: Don't inherit from unittest.TestCase, it's not needed and it doesn't work with pytest.mark.parametrize.

    This class can be configured at the class level using the following parameters:

    debug: bool
        Whether to enable debug output during test execution. Defaults to False.
    """

    def check_built_graph(self, _graph: Graph) -> None:
        pass

    def check_solved_graph(self, _graph: Graph) -> None:
        pass

    def check_pruned_graph(self, _graph: Graph) -> None:
        pass

    def check_empty_sites_removed_graph(self, _graph: Graph) -> None:
        pass

    def check_leaders_selected_graph(self, _graph: Graph) -> None:
        pass

    def check_exits_are_synced_graph(self, _graph: Graph) -> None:
        pass

    def check_empty_regions_removed_graph(self, _graph: Graph) -> None:
        pass

    def check_deadlocks_resolved_graph(self, _graph: Graph) -> None:
        pass

    def check_cut_graph(self, _graph: Graph) -> None:
        pass

    def check_verified_graph(self, _graph: Graph) -> None:
        pass

    def golden_file_path(self, test_name: str | None, site: str) -> Path:
        if test_name is None:
            return (
                Path("tests")
                / "assets"
                / "golden"
                / "gen"
                / f"{self.__class__.__name__}"
                / f"{site}.py"
            )
        else:
            return (
                Path("tests")
                / "assets"
                / "golden"
                / "gen"
                / f"{self.__class__.__name__}"
                / f"{test_name}"
                / f"{site}.py"
            )

    def read_golden_file(self, test_name: str | None, site: str) -> str:
        """Read the golden file for a test case."""
        return self.golden_file_path(test_name, site).read_text()

    def single_pipeline_test(
        self,
        test_case: PipelineTestCase,
    ) -> list[TokenResponse] | None:
        """Set up test fixtures before each test method."""
        test_name = test_case.test_name
        pipeline = test_case.pipeline
        candidate_sites = test_case.candidate_sites
        fed_info = test_case.fed_info
        expected_net_irs = test_case.expected_net_irs
        debug = test_case.debug

        self._federation_module_name: str
        self._homesite: str
        self._fed_info: FedInfo
        self._federation_module_name, self._homesite, self._fed_info = fed_info.build()
        self._expected_net_irs: dict[str, NetIR] | None = expected_net_irs

        if debug:
            print()
            print("\033[94m" + "█" * 80 + "\033[0m")
            print(
                f"\033[94m█\033[0m Running test {self.__class__.__name__} with pipeline named '{pipeline.name}'"
            )
            print("\033[94m" + "█" * 80 + "\033[0m")

        logger = logging.getLogger(__name__)
        if debug:
            logger.setLevel(logging.NOTSET)
        else:
            logger.setLevel(logging.ERROR)

        deserialized_pipeline = Pipeline.model_validate_json(pipeline.model_dump_json())
        if not deserialized_pipeline.model_dump() == pipeline.model_dump():
            before_jsons = pipeline.model_dump_json(indent=2)
            after_jsons = deserialized_pipeline.model_dump_json(indent=2)
            assert before_jsons != after_jsons, (
                "Would have expected different jsons if the models are different?!?"
            )
            print("+" * 120)
            print("Pipeline json serialization error")
            print("pipeline before:")
            print("+" * 120)
            print(before_jsons)
            print("+" * 120)
            print("pipeline after:")
            print(after_jsons)
            print("+" * 120)
            import difflib

            diff = difflib.ndiff(before_jsons.splitlines(), after_jsons.splitlines())
            print("".join(diff))
            assert False, "Pipeline json serialization error"

        if debug:
            print("✅ Pipeline json serialization and deserialization works")

        irgen = IRGen()

        # We want to call irgen, but we also want to be able to investigate
        # the graph after each pass, so we patch the _solve_graph method
        # of the irgen object with the function below.
        def _solve_graph(
            _slf: IRGen,
            mode: Literal["execute", "discovery"],
            graph: Graph,
            debug: bool = False,
        ):
            self.check_built_graph(graph)

            check_funcs: dict[GraphState, Callable[[Graph], None]] = {
                GraphState.SOLVED: self.check_solved_graph,
                GraphState.PRUNED: self.check_pruned_graph,
                GraphState.EMPTY_SITES_REMOVED: self.check_empty_sites_removed_graph,
                GraphState.LEADERS_SELECTED: self.check_leaders_selected_graph,
                GraphState.EXITS_ARE_SYNCED: self.check_exits_are_synced_graph,
                GraphState.EMPTY_REGIONS_REMOVED: self.check_empty_regions_removed_graph,
                GraphState.DEADLOCKS_RESOLVED: self.check_deadlocks_resolved_graph,
                GraphState.CROSS_CUT: self.check_cut_graph,
                GraphState.VERIFIED: self.check_verified_graph,
            }
            # Apply each pass in order
            for i, pass_instance in enumerate(graph.get_transformation_passes()):
                if mode == "discovery" and pass_instance.target_state() in (
                    GraphState.SOLVED,
                    GraphState.PRUNED,
                ):
                    # don't run solve and prune in discovery
                    continue
                pass_instance.apply()

                if debug:
                    print("+" * 120)
                    print(f"Pass {i + 1}: {pass_instance.target_state()}")
                    print("+" * 120)
                    print(graph.to_graphviz_by_site())
                    print("+" * 120)

                if pass_instance.target_state() in check_funcs:
                    check_funcs[pass_instance.target_state()](graph)
                    if debug:
                        print(
                            f"✅ Pass {pass_instance.target_state()} has been applied successfully"
                        )

            if debug:
                print("#" * 120)
                print("Final graph, by region:")
                print("#" * 120)
                print(graph.to_graphviz_by_region())
                print("#" * 120)

        # Patch the _solve_graph method of the irgen object with the function above
        irgen._solve_graph = _solve_graph.__get__(irgen, type(irgen))

        # run irgen
        if debug:
            print("+" * 120)
            print("Running irgen.prepare")
            print("+" * 120)
        prepped: PreparedPipeline = irgen.prepare(
            pipeline=pipeline,
            candidate_sites=candidate_sites,
            federation_module_name=self._federation_module_name,
            fed_info=self._fed_info,
            homesite=self._homesite,
            logger=logger,
            debug=debug,
        )

        if debug:
            for net in prepped.net_irs().values():
                print(f"NetIR for site: {net.site}")
                print("+" * 120)
                print(repr(net))
                print("+" * 120)
                print("+" * 120)
            print("✅ Graph has been created successfully")

        # check they serialize well
        for netir in prepped.net_irs().values():
            assert NetIR.model_validate_json(netir.model_dump_json()) == netir, (
                "Site net IR validation error"
            )
        if debug:
            print("✅ Site net IRs serialize well.")

        # if the test has expectations, check them
        if self._expected_net_irs:
            assert set(self._expected_net_irs.keys()) == set(
                prepped.net_irs().keys()
            ), (
                f"Expected net IRs keys {self._expected_net_irs.keys()} do not match net IRs keys in the prepped pipeline {prepped.net_irs().keys()}"
            )
            for site, expected_net_ir in self._expected_net_irs.items():
                compare_net_irs(
                    found=prepped.net_irs()[site],
                    expected=expected_net_ir,
                    debug=debug,
                )
            if debug:
                print("✅ Expected net IRs match generated net IRs.")

        assert prepped.api_version == pipeline.api_version, (
            "API version does not match expected API version"
        )
        if debug:
            print("✅ API version matches expected API version.")

        # create python code
        modgen = ModGen()
        for site, netir in prepped.net_irs().items():
            self._runtime_module: ModuleType = MagicMock(spec=ModuleType)
            self._runtime_module.__name__ = (
                f"{self._federation_module_name}.fed.runtime.{site}"
            )
            generated = modgen._generate_python_code(
                this_site_runtime_module=self._runtime_module,
                netir=netir,
            )
            if debug:
                print("+" * 120)
                print("Generated")
                print("+" * 120)
                print(generated)
                print("+" * 120)
            expected = self.read_golden_file(test_name, site)
            if generated != expected:
                if not debug:
                    print("+" * 120)
                    print("Generated")
                    print("+" * 120)
                    print(generated)
                    print("+" * 120)
                print("+" * 120)
                print(f"DIFF with golden file {self.golden_file_path(test_name, site)}")
                found_diff = False
                for i, (g, e) in enumerate(
                    zip(generated.splitlines(), expected.splitlines())
                ):
                    if g == e:
                        continue
                    else:
                        found_diff = True
                        print(f"Difference in Line {i + 1}:")
                        print(f"g: {repr(g)}")
                        print(f"e: {repr(e)}")
                assert len(generated.splitlines()) == len(expected.splitlines()), (
                    f"Number of generated and expected lines do not match: {len(generated.splitlines())} != {len(expected.splitlines())}"
                )
                if not found_diff:
                    assert False, "Something fishy is going on, no differences found?!?"
                print("+" * 120)
            assert generated == expected, (
                f"Generated code does not match expected code from file {self.golden_file_path(test_name, site)}"
            )

        if test_case.run_params is None:
            # don't want to run the pipeline
            return None

        # run the test input
        replies: list[TokenResponse] = []

        def collect_tokens_callback(
            from_site: str,
            from_node: int | str | None,
            frame: Frame,
            to_place: str,
            data: Any,
        ):
            if debug:
                logger.warning(
                    f"Received yield token: {data} from {from_site} at place {to_place}, frame {frame}"
                )
            replies.append(
                TokenResponse(
                    from_site=from_site,
                    from_node=from_node,
                    frame=frame,
                    to_place=to_place,
                    data=data,
                )
            )

        assert self._federation_module_name == "tests.assets.testfed", (
            f"For tests that should actually be executed during unit testing, you must use the 'tests.assets.testfed' federation. Was {self._federation_module_name}"
        )

        session = Session(
            federation_name=self._federation_module_name,
            homesite=self._homesite,
            target="local",
            logger=logger,
        )
        session._delegate.load_fed_info = MagicMock(return_value=self._fed_info)
        session.connect()

        logger.info("Running test input with params: %s", test_case.run_params)
        # by default, Pipeline() doesn't set the api_version but the session does, if it's missing.
        # so we need to set it here.
        import tests.assets.testfed.fed.runtime.site1

        prepped.api_version = tests.assets.testfed.fed.runtime.site1.API_VERSION

        autostop = test_case.delayed_params is None
        job = session.execute(
            pipeline=prepped,
            input_params=test_case.run_params,
            default_callback=collect_tokens_callback,
            autostop=autostop,
            # we don't want to accidentally load old cached code during testing
            force_modgen=True,
        )

        if test_case.delayed_params:
            for params in test_case.delayed_params:
                job.send_input_params(params)
            job.send_stop_frame()

        res = job.wait_until_finished(timeout=20.0)
        assert job.get_status() == JobStatus.FINISHED, "Job did not finish successfully"
        assert res, "Job did not finish"

        session.close()
        return replies
