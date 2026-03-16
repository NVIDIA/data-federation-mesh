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

# pyright: reportPrivateUsage=false, reportImportCycles=false
import logging
from abc import ABC, abstractmethod
from enum import Enum
from logging import Logger
from typing import Literal, TypeVar

from pydantic import JsonValue

from nv_dfm_core.api import (
    STATUS_PLACE_NAME,
    Advise,
    BestOf,
    NodeId,
    NodeRef,
    Pipeline,
    PlaceParam,
)
from nv_dfm_core.api import (
    If as IfStmt,
)
from nv_dfm_core.api import (
    Operation as OperationStmt,
)
from nv_dfm_core.api import (
    Yield as YieldStmt,
)
from nv_dfm_core.gen.irgen._fed_info import FedInfo, OperationInfo
from nv_dfm_core.gen.modgen.ir._in_place import START_PLACE_NAME

from ._graph_elements import (
    BoolValue,
    Branch,
    CannotReach,
    EndFork,
    ExitNode,
    FlowEdge,
    Jump,
    Loop,
    Next,
    Node,
    Operation,
    ParForEach,
    Place,
    Region,
    Send,
    SeqForEach,
    Stop,
    Yield,
    global_id,
)

NodeT = TypeVar("NodeT", bound=Node)


class GraphState(Enum):
    """States representing different stages of graph optimization."""

    SOLVED = "SOLVED"
    PRUNED = "PRUNED"
    EMPTY_SITES_REMOVED = "EMPTY_SITES_REMOVED"
    LEADERS_SELECTED = "LEADERS_SELECTED"
    EXITS_ARE_SYNCED = "EXITS_ARE_SYNCED"
    EMPTY_REGIONS_REMOVED = "EMPTY_REGIONS_REMOVED"
    DEADLOCKS_RESOLVED = "DEADLOCKS_RESOLVED"
    CROSS_CUT = "CROSS_CUT"
    VERIFIED = "VERIFIED"


class GraphTransformPass(ABC):
    """Abstract base class for graph transformation passes that optimize the execution graph."""

    def __init__(self, graph: "Graph", logger: Logger):
        self._graph: "Graph" = graph
        self.logger: Logger = logger

    @abstractmethod
    def apply(self) -> None:
        pass

    @abstractmethod
    def target_state(self) -> GraphState:
        pass


class Graph:
    """
    TODO:
    - Implement a pass to remove redundant send/place pairs
    - Implement ParallelForEach (needs additional node types)
    - Implement Type checking
    """

    def __init__(
        self,
        pipeline: Pipeline,
        homesite: str,
        fed_info: FedInfo,
        candidate_sites: list[str],
        logger: Logger | None,
    ):
        self._pipeline: Pipeline = pipeline
        self._homesite: str = homesite
        self._fed_info: FedInfo = fed_info
        if homesite not in fed_info.sites:
            raise ValueError(
                f"Homesite {homesite} not in fed info sites: {fed_info.sites.keys()}"
            )
        self._candidate_sites: list[str] = candidate_sites
        for site in self._candidate_sites:
            if site not in fed_info.sites:
                raise ValueError(
                    f"Site {site} not in fed info sites: {fed_info.sites.keys()}"
                )

        self._nodes_by_global_id: dict[str, Node] = {}
        self._nodes_by_logid: dict[str, list[Node]] = {}
        self._edges: list[FlowEdge] = []
        # create a special yield region, representing the yield places on the homesite
        self._yield_region: Region = Region(
            graph=self,
            site=self._homesite,
            site_local_logid="__YIELD_REGION__",
            is_loop_head=True,
        )
        _ = CannotReach(self, self._yield_region)
        self.logger: Logger = (
            logger if logger is not None else logging.getLogger(__name__)
        )

    def _get_fresh_per_nodetype_logid(self, nodetype: type[Node]) -> str:
        count = 0
        for nodes in self._nodes_by_logid.values():
            if len(nodes) > 0 and isinstance(nodes[0], nodetype):
                count += 1
        return f"{nodetype.__name__}{count}".lower()

    def _register_node(self, node: Node):
        """Register a node in the graph's tracking dictionaries by global ID and logical ID."""
        assert node.global_id() not in self._nodes_by_global_id, (
            f"Node {node.global_id()} already in graph"
        )
        self._nodes_by_global_id[node.global_id()] = node
        if node.site_local_logid() not in self._nodes_by_logid:
            self._nodes_by_logid[node.site_local_logid()] = []
        if len(self._nodes_by_logid[node.site_local_logid()]) > 0:
            assert type(self._nodes_by_logid[node.site_local_logid()][0]) is type(
                node
            ), (
                f"Node {node.global_id()} has same logid {node.site_local_logid()} as {self._nodes_by_logid[node.site_local_logid()][0].global_id()}, but different type"
            )
        self._nodes_by_logid[node.site_local_logid()].append(node)

    def _deregister_node(self, node: Node, force: bool = False):
        """Remove a node from the graph's tracking dictionaries with optional safety checks."""
        if not force:
            if isinstance(node, Place):
                assert node.site_local_logid() != START_PLACE_NAME, (
                    f"Cannot deregister start place {node.global_id()}"
                )
            assert not isinstance(node, (Region, ExitNode)), (
                f"Cannot deregister region or exit node {node.global_id()}"
            )
        del self._nodes_by_global_id[node.global_id()]
        self._nodes_by_logid[node.site_local_logid()].remove(node)
        if len(self._nodes_by_logid[node.site_local_logid()]) == 0:
            del self._nodes_by_logid[node.site_local_logid()]

    def _register_edge(self, edge: FlowEdge):
        """Register an edge in the graph after validating it exists in source and destination node slots."""
        assert edge in edge.src()._outgoing_edges, (
            f"Edge {edge.src()} -> {edge.dst()} not in outgoing edges of {edge.src().node().global_id()}"
        )
        assert edge in edge.dst()._incoming_edges, (
            f"Edge {edge.src()} -> {edge.dst()} not in incoming edges of {edge.dst().node().global_id()}"
        )
        assert (
            self._nodes_by_global_id[edge.src().node().global_id()] == edge.src().node()
        ), (
            f"Edge {edge.src()} -> {edge.dst()} has src node {edge.src().node().global_id()} not in graph"
        )
        assert (
            self._nodes_by_global_id[edge.dst().node().global_id()] == edge.dst().node()
        ), (
            f"Edge {edge.src()} -> {edge.dst()} has dst node {edge.dst().node().global_id()} not in graph"
        )
        assert (
            edge.src().node()
            in self._nodes_by_logid[edge.src().node().site_local_logid()]
        )
        assert (
            edge.dst().node()
            in self._nodes_by_logid[edge.dst().node().site_local_logid()]
        )
        assert edge not in self._edges, (
            f"Edge {edge.src()} -> {edge.dst()} already in graph"
        )
        self._edges.append(edge)

    def _deregister_edge(self, edge: FlowEdge):
        """Remove an edge from the graph's edge list after validation."""
        assert edge in self._edges, (
            f"Edge {edge.src().node().global_id()}:{edge.src().name()} -> {edge.dst().node().global_id()}:{edge.dst().name()} not in graph"
        )
        len_before = len(self._edges)
        self._edges.remove(edge)
        assert edge not in self._edges, (
            f"Edge {edge.src().node().global_id()}:{edge.src().name()} -> {edge.dst().node().global_id()}:{edge.dst().name()} still in graph"
        )
        len_after = len(self._edges)
        assert len_before - 1 == len_after, (
            f"Edge {edge.src().node().global_id()}:{edge.src().name()} -> {edge.dst().node().global_id()}:{edge.dst().name()} not removed"
        )

    def _locations_to_try_for_operation(
        self, operation: OperationStmt
    ) -> dict[str, OperationInfo]:
        """Returns a dict for site/provider pairs that can be used."""
        if isinstance(operation.site, str):
            # user fixed the site
            try_sites = [operation.site]
        else:
            assert isinstance(operation.site, (Advise, BestOf))
            # we are nice and accept an Advise(), even though it shouldn't be used
            try_sites = (
                operation.site.sites if isinstance(operation.site, BestOf) else None
            )
            if try_sites is None:
                # try all sites
                try_sites = self.candidate_sites()

        if not all(s in self.candidate_sites() for s in try_sites):
            raise ValueError(
                f"Operation {operation} is associated with sites {try_sites}, some of which are not in site candidates: {self.candidate_sites()}"
            )

        # now for each site, check if it has the operation for the given provider
        op_infos: dict[str, OperationInfo] = {}
        for site in try_sites:
            assert site in self._fed_info.sites, (
                f"Site {site} not in fed info: {self._fed_info.sites.keys()}"
            )
            site_info = self._fed_info.sites[site]
            if operation.provider is None:
                # explicitly no provider, use the site's interface
                if operation.__api_name__ in site_info.interface:
                    op_infos[site] = site_info.interface[operation.__api_name__]
            elif isinstance(operation.provider, str):  # pyright: ignore[reportUnnecessaryIsInstance]
                # explicitly a provider, use that provider
                if operation.provider in site_info.providers:
                    provider_info = site_info.providers[operation.provider]
                    if operation.__api_name__ in provider_info.interface:
                        op_infos[site] = provider_info.interface[operation.__api_name__]
            else:
                raise ValueError(
                    f"Provider field for Operation {operation} was set to Advise() during IR generation, but Advise() is not allowed"
                )
        # TODO: here we should really run discovery to see which site can handle this operation
        return op_infos

    def pipeline(self) -> Pipeline:
        return self._pipeline

    def homesite(self) -> str:
        return self._homesite

    def candidate_sites(self) -> list[str]:
        return self._candidate_sites

    def yield_region(self) -> Region:
        return self._yield_region

    def num_operation_nodes_by_site(self) -> dict[str, int]:
        res: dict[str, int] = {}
        for site in self.candidate_sites():
            res[site] = 0

        for node in self._nodes_by_global_id.values():
            assert node.site() in self.candidate_sites(), (
                f"Node {node.global_id()} is not in sites"
            )
            if isinstance(node, Operation):
                res[node.site()] += 1
        return res

    def get_all_edges(self) -> list[FlowEdge]:
        return self._edges

    def get_all_nodes_copy(self) -> list[Node]:
        return list(self._nodes_by_global_id.values())

    def has_node(self, global_id: str) -> bool:
        return global_id in self._nodes_by_global_id

    def get_nodes_by_logids(self) -> dict[str, list[Node]]:
        return self._nodes_by_logid

    def get_nodes_for_logid(self, site_local_logid: str, T: type[NodeT]) -> list[NodeT]:
        return [
            node
            for node in self._nodes_by_logid[site_local_logid]
            if isinstance(node, T)
        ]

    def get_node(self, site: str, site_local_logid: str, T: type[NodeT]) -> NodeT:
        glob_id = global_id(site, site_local_logid)
        node = self._nodes_by_global_id[glob_id]
        assert isinstance(node, T), f"Node {glob_id} is not a {T.__name__}"
        return node

    def has_node_on_site(self, site: str, site_local_logid: str) -> bool:
        return self.has_node(global_id(site, site_local_logid))

    def get_regions_copy(self) -> list[Region]:
        """Return a list of all regions in the graph."""
        regions: list[Region] = []
        for node in self._nodes_by_global_id.values():
            if isinstance(node, Region):
                regions.append(node)
        return regions

    def non_yield_regions_for_site(self, site: str) -> list[Region]:
        """Return all regions for a specific site excluding the yield region."""
        res: list[Region] = []
        for node in self._nodes_by_global_id.values():
            if (
                isinstance(node, Region)
                and node.site() == site
                and node != self.yield_region()
            ):
                res.append(node)
        return res

    def non_yield_places_for_site(self, site: str) -> list[Place]:
        """Return all places for a specific site excluding those in the yield region."""
        res: list[Place] = []
        for node in self._nodes_by_global_id.values():
            if (
                isinstance(node, Place)
                and node.site() == site
                and node.region() != self.yield_region()
            ):
                res.append(node)
        return res

    def get_transformation_passes(self) -> list[GraphTransformPass]:
        """Get the list of transformation passes to apply to this graph.

        Returns:
            A list of GraphTransformPass instances to apply in order
        """
        from .passes._cut_cross_region_edges_pass import CutCrossRegionEdgesPass
        from .passes._prune_pass import PrunePass
        from .passes._remove_empty_regions_pass import RemoveEmptyRegionsPass
        from .passes._remove_empty_sites_pass import RemoveEmptySitesPass
        from .passes._resolve_deadlock_pass import ResolveDeadlocksPass
        from .passes._select_leaders_pass import SelectLeadersPass
        from .passes._solve_pass import SolvePass
        from .passes._sync_exit_nodes_pass import SyncExitNodesPass
        from .passes._verify_result_graph_pass import VerifyResultGraphPass

        return [
            SolvePass(self, self.logger),
            PrunePass(self, self.logger),
            RemoveEmptySitesPass(self, self.logger),
            SelectLeadersPass(self, self.logger),
            SyncExitNodesPass(self, self.logger),
            RemoveEmptyRegionsPass(self, self.logger),
            ResolveDeadlocksPass(self, self.logger),
            CutCrossRegionEdgesPass(self, self.logger),
            VerifyResultGraphPass(self, self.logger),
        ]

    def create_log_start_region(self) -> str:
        """Creates a logical (log) pipeline start region node, which is really
        one region node on every site. The start region comes with a
        START_PLACE_NAME input place plus connecting edge: START_PLACE_NAME.__data__ -> __pipeline__start__.__flow__"""

        start_region_logid = "_START_"
        for site in self.candidate_sites():
            start = Region(
                graph=self,
                site=site,
                site_local_logid=start_region_logid,
                is_loop_head=False,
            )
            place = Place(
                graph=self,
                region=start,
                site_local_logid=START_PLACE_NAME,
                kind="control",
                origin="external",
                flavor="control",
                type="nv_dfm_core.exec.FlowInfo",
            )
            _ = FlowEdge(
                src=place.out_slot("__data__"),
                dst=start.in_slot("__flow__"),
                send_cost_info=self._fed_info.find_send_cost(
                    place.site(), start.site(), logger=self.logger
                ),
                is_weak=False,
            )

        return start_region_logid

    def create_place_param_node_if_needed(self, place: PlaceParam) -> str:
        # pace params are associated with the start region. But they are "global" in the sense
        # that we may access them directly from other regions.
        place_logid = place.place

        # if already exists, don't create a new one
        if place_logid in self._nodes_by_logid:
            return place_logid

        # create one place per site/start region
        for start_region in self._nodes_by_logid["_START_"]:
            assert isinstance(start_region, Region)
            _place = Place(
                graph=self,
                region=start_region,
                site_local_logid=place_logid,
                kind="data",
                origin="external",
                flavor="sticky",
                type="Any",
            )

        return place_logid

    def create_yield_place_node_if_needed(self, place: str) -> str:
        if place in self._nodes_by_logid:
            assert isinstance(self._nodes_by_logid[place][0], Place)
            assert (
                self._nodes_by_logid[place][0].region().site_local_logid()
                == "__YIELD_REGION__"
            )
            return place

        yield_region = self.yield_region()

        _ = Place(
            graph=self,
            region=yield_region,
            site_local_logid=place,
            kind="data",
            origin="internal",
            flavor="yield",
            type="Any",
        )

        return place

    def create_log_place_node(
        self,
        region_logid: str,
        place: str,
        kind: Literal["data", "control"],
        origin: Literal["internal", "external"],
        flavor: Literal["control", "scoped", "sticky", "framecount", "yield"],
        type: str,
    ) -> str:
        logid: str | None = None
        for site in self.candidate_sites():
            place_obj = Place(
                graph=self,
                region=self.get_node(site, region_logid, Region),
                site_local_logid=place,
                kind=kind,
                origin=origin,
                flavor=flavor,
                type=type,
            )
            logid = place_obj.site_local_logid()  # just pick any, are all the same
        assert logid is not None, "logid is None"
        return logid

    def create_log_region_node(self, is_loop_head: bool) -> str:
        start_region_logid = self._get_fresh_per_nodetype_logid(Region)
        for site in self.candidate_sites():
            _ = Region(
                graph=self,
                site=site,
                site_local_logid=start_region_logid,
                is_loop_head=is_loop_head,
            )
        return start_region_logid

    def create_log_jump_node(self, region_logid: str) -> str:
        logid: str | None = None
        for site in self.candidate_sites():
            jump = Jump(graph=self, region=self.get_node(site, region_logid, Region))
            logid = jump.site_local_logid()  # just pick any, are all the same
        assert logid is not None, "logid is None"
        return logid

    def create_log_loop_node(self, region_logid: str) -> str:
        logid: str | None = None
        for site in self.candidate_sites():
            loop = Loop(graph=self, region=self.get_node(site, region_logid, Region))
            logid = loop.site_local_logid()  # just pick any, are all the same
        assert logid is not None, "logid is None"
        return logid

    def create_log_end_fork_node(self, region_logid: str) -> str:
        logid: str | None = None
        for site in self.candidate_sites():
            end_fork = EndFork(
                graph=self, region=self.get_node(site, region_logid, Region)
            )
            logid = end_fork.site_local_logid()  # just pick any, are all the same
        assert logid is not None, "logid is None"
        return logid

    def create_log_stop_node(
        self, region_logid: str, yield_place: str = STATUS_PLACE_NAME
    ) -> str:
        logid: str | None = None
        for site in self.candidate_sites():
            jump = Stop(
                graph=self,
                region=self.get_node(site, region_logid, Region),
                yield_place=yield_place,
            )
            logid = jump.site_local_logid()  # just pick any, are all the same
        assert logid is not None, "logid is None"
        return logid

    def create_log_branch_node(self, region_logid: str) -> str:
        logid: str | None = None
        for site in self.candidate_sites():
            branch = Branch(
                graph=self, region=self.get_node(site, region_logid, Region)
            )
            logid = branch.site_local_logid()  # just pick any, are all the same
        assert logid is not None, "logid is None"
        return logid

    def create_log_seq_foreach_node(
        self, region_logid: str, foreach_node_id: NodeId
    ) -> str:
        logid: str | None = None
        for site in self.candidate_sites():
            seq_iterate = SeqForEach(
                graph=self,
                region=self.get_node(site, region_logid, Region),
                foreach_node_id=foreach_node_id,
            )
            logid = seq_iterate.site_local_logid()  # just pick any, are all the same
        assert logid is not None, "logid is None"
        return logid

    def create_log_par_foreach_node(
        self, region_logid: str, foreach_node_id: NodeId
    ) -> str:
        logid: str | None = None
        for site in self.candidate_sites():
            par_iterate = ParForEach(
                graph=self,
                region=self.get_node(site, region_logid, Region),
                foreach_node_id=foreach_node_id,
            )
            logid = par_iterate.site_local_logid()  # just pick any, are all the same
        assert logid is not None, "logid is None"
        return logid

    def create_log_operation_node_for_execute(
        self, region_logid: str, operation: OperationStmt
    ) -> str:
        logid: str | None = None

        # find out where we could place the operation
        candidates = self._locations_to_try_for_operation(operation)
        if len(candidates) == 0:
            raise ValueError(
                f"No candidate locations found for operation {operation} in region {region_logid}."
                + f" Looked at interfaces of candidate sites: {self.candidate_sites()}"
                + f" for operation API name: {operation.__api_name__} and operation provider: {operation.provider}"
                + " but couldn't find any match."
            )

        for site, op_info in candidates.items():
            region_node = self.get_node(site, region_logid, Region)
            # create operation
            op = Operation(
                graph=self,
                region=region_node,
                operation=operation,
                operation_info=op_info,
            )
            logid = op.site_local_logid()  # just pick any, are all the same
        assert logid is not None, "logid is None"
        return logid

    def create_log_yield_node_where_needed(
        self, region_logid: str, yld_stmt: YieldStmt
    ) -> str:
        """Only create yield nodes on sites where the input node is present"""
        if isinstance(yld_stmt.value, NodeRef):
            stmt_input = yld_stmt.value.id.as_identifier()
        else:
            stmt_input = None
        logid: str = yld_stmt.dfm_node_id.as_identifier()
        for site in self.candidate_sites():
            # if the Yield value is a constant or there is an input node on this site, create a yield node
            if not stmt_input or self.has_node_on_site(site, stmt_input):
                _ = Yield(
                    graph=self,
                    region=self.get_node(site, region_logid, Region),
                    yld=yld_stmt,
                    send_cost_info=self._fed_info.find_send_cost(
                        site, self._homesite, logger=self.logger
                    ),
                )
        return logid

    def create_log_send_node(
        self,
        region_logid: str,
        node_id: str,
        data_kind: Literal["data", "control"],
        type: str,
        literal_data: tuple[str, JsonValue] | None = None,
    ) -> str:
        for site in self.candidate_sites():
            _ = Send(
                graph=self,
                region=self.get_node(site, region_logid, Region),
                site_local_logid=node_id,
                data_kind=data_kind,
                type=type,
                literal_data=literal_data,
            )
        return node_id

    def create_log_bool_value_node(
        self, region_logid: str, if_stmt: IfStmt, params: set[NodeId]
    ) -> str:
        logid: str | None = None
        for site in self.candidate_sites():
            node = BoolValue(
                graph=self,
                region=self.get_node(site, region_logid, Region),
                site_local_logid=if_stmt.dfm_node_id.as_identifier(),
                bool_expr=if_stmt.cond,
                params=params,
            )
            logid = node.site_local_logid()  # just pick any, are all the same
        assert logid is not None, "logid is None"
        return logid

    def create_log_next_node_where_needed(
        self,
        region_logid: str,
        foreach_node_id: NodeId,
        iterator_logid: str,
        in_type: str,
        out_type: str,
        is_async: bool,
    ) -> str:
        logid: str = f"{foreach_node_id.as_identifier()}_NEXT"
        for site in self.candidate_sites():
            if self.has_node_on_site(site, iterator_logid):
                _ = Next(
                    graph=self,
                    region=self.get_node(site, region_logid, Region),
                    site_local_logid=logid,
                    in_type=in_type,
                    out_type=out_type,
                    is_async=is_async,
                )
        return logid

    def create_site_internal_flow_edges(
        self,
        src_logid: str,
        src_out_slot: str,
        dst_logid: str,
        dst_in_slot: str,
        is_weak: bool = False,
    ) -> None:
        for site in self.candidate_sites():
            # the assumption is that src and tgt either both exist on this site, or both don't exist.
            if not self.has_node_on_site(site, src_logid) or not self.has_node_on_site(
                site, dst_logid
            ):
                continue
            src = self.get_node(site, src_logid, Node)
            dst = self.get_node(site, dst_logid, Node)
            _ = FlowEdge(
                src=src.out_slot(src_out_slot),
                dst=dst.in_slot(dst_in_slot),
                send_cost_info=self._fed_info.find_send_cost(
                    src.site(), dst.site(), logger=self.logger
                ),
                is_weak=is_weak,
            )

    def create_cross_site_flow_edges(
        self,
        src_logid: str,
        src_out_slot: str,
        dst_logid: str,
        dst_in_slot: str,
        is_weak: bool = False,
    ) -> None:
        for src in self.get_nodes_for_logid(src_logid, Node):
            assert src.site() in self.candidate_sites(), (
                f"Source node {src.global_id()} is not in sites"
            )
            for dst in self.get_nodes_for_logid(dst_logid, Node):
                assert dst.site() in self.candidate_sites(), (
                    f"Destination node {dst.global_id()} is not in sites"
                )
                _ = FlowEdge(
                    src=src.out_slot(src_out_slot),
                    dst=dst.in_slot(dst_in_slot),
                    send_cost_info=self._fed_info.find_send_cost(
                        src.site(), dst.site(), logger=self.logger
                    ),
                    is_weak=is_weak,
                )

    def to_graphviz_by_site(self) -> str:
        """Returns a graphviz representation of the graph."""
        lines: list[str] = []
        lines.append("digraph G {")
        lines.append("    rankdir=TB;")

        # add subgraphs
        for site in self.candidate_sites():
            lines.append(f"    subgraph cluster_{site} {{")
            for node in self._nodes_by_global_id.values():
                if node.site() == site:
                    node.to_graphviz(lines)
            lines.append("    }")

        # Add edges
        for edge in self._edges:
            edge.to_graphviz(lines)

        lines.append("}")
        return "\n".join(lines)

    def to_graphviz_by_region(self) -> str:
        """Returns a graphviz representation of the graph."""
        lines: list[str] = []
        lines.append("digraph G {")
        lines.append("    rankdir=TB;")

        for region in self.get_regions_copy():
            # a subgraph for the region containing another subgraph for the places and the body
            lines.append(f'    subgraph "cluster_{region.global_id()}" {{')

            lines.append(f'    subgraph "cluster_{region.global_id()}_places" {{')
            lines.append(
                '    color = "transparent"; style = "filled"; fillcolor = "salmon";'
            )
            for place in region.places():
                place.to_graphviz(lines)
            lines.append("    }")

            lines.append(f'    subgraph "cluster_{region.global_id()}_body" {{')
            lines.append(
                '    color = "transparent"; style = "filled"; fillcolor = "khaki";'
            )
            region.to_graphviz(lines, show_region_edges=False)
            for node in region.nodes_copy():
                node.to_graphviz(lines)
            region.exit_node().to_graphviz(lines)
            lines.append("    }")

            lines.append("    }")
        # Add edges
        for edge in self._edges:
            edge.to_graphviz(lines)

        lines.append("}")
        return "\n".join(lines)
