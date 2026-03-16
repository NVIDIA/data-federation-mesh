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

from typing import Any, Optional

from nv_dfm_core.api import Pipeline, StopToken, STATUS_PLACE_NAME, DISCOVERY_PLACE_NAME
from nv_dfm_core.session import Session, JobStatus
from nv_dfm_core.exec import Frame


class PipelineHelper:
    def __init__(self, session: Session):
        self.session = session
        self.place_callbacks = {
            DISCOVERY_PLACE_NAME: self.generic_callback,
            STATUS_PLACE_NAME: self.generic_callback,
            "yield": self.generic_callback,
        }
        self.callback_results = {
            "default_place": [],
            DISCOVERY_PLACE_NAME: [],
            STATUS_PLACE_NAME: [],
            "yield": [],
        }
        self.prepared_pipeline = None

    def generic_callback(
        self,
        _from_site: str,
        _node: int | str | None,
        _frame: Frame,
        target_place: str,
        data: Any,
    ):
        if not isinstance(data, StopToken):
            self.callback_results[target_place].append(data)

    def prepare(self, pipeline: Pipeline):
        self.prepared_pipeline = self.session.prepare(pipeline)

    def add_callback(self, place_name: str, callback: Any):
        self.place_callbacks[place_name] = callback

    def print_callback_results(self):
        for key, value in self.callback_results.items():
            print(f"{key}: {value}")

    def clear_callback_results(self):
        for key in self.callback_results.keys():
            self.callback_results[key] = []

    def run(self, input_params: dict, timeout_s: float = 60.0):
        if self.prepared_pipeline is None:
            raise ValueError("Pipeline not prepared")
        job = self.session.execute(
            self.prepared_pipeline,
            input_params=input_params,
            place_callbacks=self.place_callbacks,
            default_callback=self.generic_callback,
        )
        job.wait_until_finished(timeout=timeout_s)
        if job.get_status() != JobStatus.FINISHED:
            raise ValueError(
                f"Pipeline execution failed: {job.get_status()}, {self.callback_results[STATUS_PLACE_NAME]}"
            )
        if self.callback_results[STATUS_PLACE_NAME] != []:
            raise ValueError(
                f"Pipeline execution had errors: {self.callback_results[STATUS_PLACE_NAME]}"
            )
