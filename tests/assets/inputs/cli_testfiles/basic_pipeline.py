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

import argparse

from nv_dfm_core.api import Pipeline
from examplefed.apps.concierge import GreetMe, Yield


def get_pipeline(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="World")
    parser.add_argument("--site", type=str, default="reception")
    parser.add_argument("--target", type=str, default="flare")
    args = parser.parse_args(args)

    with Pipeline() as pipeline:
        greet = GreetMe(site=args.site, name=args.name)
        Yield(value=greet)
    return pipeline


def get_pipeline_parameters(args: list[str] | None = None):
    return {}


if __name__ == "__main__":
    print(get_pipeline())
    print(get_pipeline_parameters())
