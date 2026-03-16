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

import logging
from logging import Logger
from typing import Literal

from nv_dfm_core.api import (
    Pipeline,
    PreparedPipeline,
)
from nv_dfm_core.api._yield import STATUS_PLACE_NAME
from nv_dfm_core.gen.irgen.graph._graph import GraphState
from nv_dfm_core.gen.modgen.ir import (
    NetIR,
    YieldPlace,
)

from ._fed_info import FedInfo
from .graph import Graph
from .visitors import (
    GraphToDiscoveryIRTranslator,
    GraphToExecuteIRTranslator,
    PipelineInfoExtractorVisitor,
    PipelineToGraphVisitor,
)


class IRGen:
    """Intermediate Representation Generator that transforms pipelines into executable IR.

    IRGen takes a pipeline and federation information, then optimizes and generates
    the intermediate representation (NetIRs) for each participating site.
    """

    def prepare(
        self,
        pipeline: Pipeline,
        candidate_sites: list[str] | None,
        federation_module_name: str,
        fed_info: FedInfo,
        homesite: str,
        logger: Logger | None = None,
        debug: bool = False,
    ) -> PreparedPipeline:
        """
        Prepares and optimizes the pipeline for execution.
        The result will contain net IRs for each participating site.
        On the site, the modgen will finalize the nets for execution.
        """
        if not logger:
            logger = logging.getLogger(__name__)

        graph = self._create_graph_from_pipeline(
            candidate_sites=candidate_sites,
            fed_info=fed_info,
            homesite=homesite,
            pipeline=pipeline,
            logger=logger,
            debug=debug,
        )
        if debug:
            print("+" * 120)
            print("Built")
            print("+" * 120)
            print(graph.to_graphviz_by_site())
            print("+" * 120)

        self._solve_graph(mode=pipeline.mode, graph=graph, debug=debug)

        # create the net IRs
        net_irs, yield_places = self._create_net_irs(
            pipeline=pipeline,
            graph=graph,
            logger=logger,
            debug=debug,
        )
        # create the prepared pipeline
        prepared_pipeline = self._create_prepared_pipeline(
            pipeline=pipeline,
            federation_module_name=federation_module_name,
            homesite=homesite,
            net_irs=net_irs,
            yield_places=yield_places,
        )
        return prepared_pipeline

    def _create_graph_from_pipeline(
        self,
        candidate_sites: list[str] | None,
        fed_info: FedInfo,
        homesite: str,
        pipeline: Pipeline,
        logger: Logger,
        debug: bool = False,
    ) -> Graph:
        """Returns the graph"""

        # extracts information about yield and param places as well as used sites.
        # It also will verify the correct usage of multiuse=True for yield and param places.
        info_visitor = PipelineInfoExtractorVisitor()
        pipeline.accept(info_visitor)

        if debug:
            print("+" * 120)
            print("IRGen._create_graph_from_pipeline")

        # if no candidate sites are provided, use all known sites
        sites_to_use: set[str] = (
            set(candidate_sites) if candidate_sites else set(fed_info.sites.keys())
        )

        if len(sites_to_use) == 0:
            raise ValueError("No candidate sites provided.")

        if debug:
            print("+" * 120)
            print(
                f"Passed candidate sites were {candidate_sites}. Sites I will use are {sites_to_use}"
            )
            print("Running PipelineToGraphVisitor")

        visitor = PipelineToGraphVisitor(
            fed_info=fed_info,
            pipeline=pipeline,
            homesite=homesite,
            candidate_sites=list(sites_to_use),
            logger=logger,
            debug=debug,
        )
        pipeline.accept(visitor)
        if debug:
            print("+" * 120)
            print("Graph created")
            print("+" * 120)
            print(visitor.graph.to_graphviz_by_site())
            print("+" * 120)

        return visitor.graph

    def _solve_graph(
        self,
        mode: Literal["execute", "discovery"],
        graph: Graph,
        debug: bool = False,
    ):
        """Apply graph transformation passes to optimize the execution graph for the given mode."""
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

        if debug:
            print("#" * 120)
            print("Final graph, by region:")
            print("#" * 120)
            print(graph.to_graphviz_by_region())
            print("#" * 120)

    def _create_net_irs(
        self,
        pipeline: Pipeline,
        graph: Graph,
        logger: Logger,
        debug: bool = False,
    ) -> tuple[dict[str, NetIR], list[YieldPlace]]:
        """Translate the optimized graph into NetIRs for each site and collect yield places."""
        if pipeline.mode == "execute":
            translator = GraphToExecuteIRTranslator(graph=graph, logger=logger)
        else:
            translator = GraphToDiscoveryIRTranslator(graph=graph, logger=logger)

        translator.translate_graph()

        site_net_irs: dict[str, NetIR] = translator.get_netirs()
        yield_places: list[YieldPlace] = translator.get_yield_places()

        # make sure we have the special status place, which is used to send error and stop tokens to.
        found_status_place: bool = False
        for yp in yield_places:
            if yp.place == STATUS_PLACE_NAME:
                found_status_place = True
                break
        if not found_status_place:
            yield_places.append(
                YieldPlace(
                    kind="control",
                    place=STATUS_PLACE_NAME,
                    type="Any",
                )
            )

        for net_ir in site_net_irs.values():
            if debug:
                print("+" * 120)
                print("IRGen prepped NetIR:")
                print("+" * 120)
                print(net_ir.model_dump_json(indent=2))
                print("+" * 120)

        if len(site_net_irs) == 0:
            logger.warning(
                "No site net IRs found, this is unusual, unless the pipeline was empty. Pipeline was: %s",
                pipeline.model_dump_json(indent=2),
            )

        return site_net_irs, yield_places

    def _create_prepared_pipeline(
        self,
        pipeline: Pipeline,
        federation_module_name: str,
        homesite: str,
        net_irs: dict[str, NetIR],
        yield_places: list[YieldPlace],
    ) -> PreparedPipeline:
        """Create a PreparedPipeline object from the pipeline metadata and generated NetIRs."""
        prepared_pipeline = PreparedPipeline(
            api_version=pipeline.api_version,
            federation_module_name=federation_module_name,
            homesite=homesite,
            pipeline_name=pipeline.name,
            net_irs=list(net_irs.values()),
            yield_places=yield_places,
        )
        return prepared_pipeline
