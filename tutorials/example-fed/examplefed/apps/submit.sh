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

# An example script that uses the concierge to submit a pipeline to the federation.
# demonstrates usage of DFM CLI to submit a pipeline using a script.

# Get the project root (parent of examplefed)
PROJECT_ROOT=$(realpath $(dirname $0)/../../../../)

# Add project root to PYTHONPATH so examplefed module is importable
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PROJECT_ROOT"

FEDERATION_WORKSPACE_PATH=${PROJECT_ROOT}/workspace/examplefed_poc/examplefed/

if [ -z "$DFM_FEDERATION_WORKSPACE_PATH" ]; then
    echo "DFM_FEDERATION_WORKSPACE_PATH is not set. Using default path: $FEDERATION_WORKSPACE_PATH"
else
    FEDERATION_WORKSPACE_PATH=$DFM_FEDERATION_WORKSPACE_PATH
fi

# Assume default arguments
args="--name World --greetme-from reception"

if [[ -n "$@" ]]; then
    echo "Using job args: $@"
    args="$@"
fi

uv run python tutorials/example-fed/examplefed/apps/concierge.py --federation-path $FEDERATION_WORKSPACE_PATH $args
