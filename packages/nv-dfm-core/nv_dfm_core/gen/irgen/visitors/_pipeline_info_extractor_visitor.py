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

from typing_extensions import override

from nv_dfm_core.api import (
    ApiVisitor,
    ForEach,
    If,
    Operation,
    PlaceParam,
    Statement,
    TryFromCache,
    WriteToCache,
    Yield,
)
from nv_dfm_core.api._advise import Advise
from nv_dfm_core.api._best_of import BestOf
from nv_dfm_core.api._pipeline import Pipeline


class PipelineInfoExtractorVisitor(ApiVisitor):
    def __init__(self):
        self.mentioned_sites: set[str | None] = set()
        # collect all yield places and verify that they are using multiuse correctly
        self.yield_places: set[str] = set()
        self.multiuse_yield_places: set[str] = set()
        # same for param places
        self.param_places: set[str] = set()
        self.multiuse_param_places: set[str] = set()

    @override
    def visit_pipeline(self, pipeline: Pipeline) -> None:
        for stmt in pipeline.dfm_body:
            stmt.accept(self)

    def _handle_site(self, site: str | BestOf | Advise) -> None:
        # record all concrete mentions of sites
        if isinstance(site, str):
            self.mentioned_sites.add(site)
        elif isinstance(site, BestOf):
            if site.sites is None:
                self.mentioned_sites.add(None)
            else:
                for s in site.sites:
                    self.mentioned_sites.add(s)
        else:
            assert isinstance(site, Advise)
            self.mentioned_sites.add(None)

    def _handle_param_places(self, stmt: Statement) -> None:
        # find all used param places in any field in the statement
        fields = stmt.get_noderef_and_placeparam_pydantic_fields()
        for _name, value in fields:
            if isinstance(value, PlaceParam):
                self._handle_param_place(value)

    def _handle_param_place(self, pp: PlaceParam) -> None:
        # record a single place param
        if pp.place in self.yield_places:
            raise ValueError(
                f"PlaceParam {pp.place} is also a yield place. Yield places and param places must be disjoint."
            )

        if pp.place not in self.param_places:
            # seen for the first time, all good
            self.param_places.add(pp.place)
            if pp.multiuse:
                self.multiuse_param_places.add(pp.place)
        elif pp.multiuse:
            # we have seen this place before. param stmt is multiuse, make sure was marked as multiuse before too
            if pp.place not in self.multiuse_param_places:
                raise ValueError(
                    f"PlaceParam {pp.place} is reused multiple times. All uses for this param place must be flagged as multiuse=True."
                )
        else:
            # we have seen this place before. But it is marked as single use, which is always an error
            raise ValueError(
                f"PlaceParam {pp.place} is reused multiple times. All uses for this param place must be flagged as multiuse=True."
            )

    @override
    def visit_yield(self, yield_stmt: Yield) -> None:
        self._handle_param_places(yield_stmt)
        if yield_stmt.place in self.param_places:
            raise ValueError(
                f"Yield place {yield_stmt.place} is also a param place. Yield places and param places must be disjoint."
            )

        if yield_stmt.place not in self.yield_places:
            # seen for the first time, all good
            self.yield_places.add(yield_stmt.place)
            if yield_stmt.multiuse:
                self.multiuse_yield_places.add(yield_stmt.place)
        elif yield_stmt.multiuse:
            # we have seen this place before. yield stmt is multiuse, make sure was marked as multiuse before too
            if yield_stmt.place not in self.multiuse_yield_places:
                raise ValueError(
                    f"Yield place {yield_stmt.place} is reused multiple times. All uses for this yield place must be flagged as multiuse=True."
                )
        else:
            # we have seen this place before. But it is marked as single use, which is always an error
            raise ValueError(
                f"Yield place {yield_stmt.place} is reused multiple times. All uses for this yield place must be flagged as multiuse=True."
            )

    @override
    def visit_if(self, if_stmt: If) -> None:
        self._handle_param_places(if_stmt)
        for stmt in if_stmt.dfm_body:
            stmt.accept(self)

    @override
    def visit_for_each(self, for_each_stmt: ForEach) -> None:
        self._handle_param_places(for_each_stmt)
        for stmt in for_each_stmt.dfm_body:
            stmt.accept(self)

    @override
    def visit_try_from_cache(self, try_from_cache_stmt: TryFromCache) -> None:
        self._handle_param_places(try_from_cache_stmt)
        for stmt in try_from_cache_stmt.dfm_body:
            stmt.accept(self)

    @override
    def visit_write_to_cache(self, write_to_cache_stmt: WriteToCache) -> None:
        self._handle_param_places(write_to_cache_stmt)

    @override
    def visit_operation(self, operation: Operation) -> None:
        self._handle_param_places(operation)
        self._handle_site(operation.site)
