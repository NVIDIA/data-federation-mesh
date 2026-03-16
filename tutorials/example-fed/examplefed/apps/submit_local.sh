#!/bin/bash
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

# An example script that uses the concierge to run a pipeline locally.
# This uses the "local" target which emulates the distributed federation
# on a single machine using multiprocessing - great for development and testing.
#
# Usage:
#   ./submit_local.sh                           # Uses default args
#   ./submit_local.sh --name "Alice"            # Custom name
#   ./submit_local.sh --greetme-from server     # Greet from server site
#
# To enable telemetry output:
#   DFM_TELEMETRY_ENABLED=true ./submit_local.sh

# Assume default arguments
args="--name World --greetme-from reception --target local"

if [[ -n "$@" ]]; then
    echo "Using custom args: $@"
    args="--target local $@"
fi

echo "Running concierge app in LOCAL mode..."
echo "Command: uv run python tutorials/example-fed/examplefed/apps/concierge.py $args"
echo "---"

uv run python tutorials/example-fed/examplefed/apps/concierge.py $args

