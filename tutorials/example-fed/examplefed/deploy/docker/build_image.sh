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

set -e

PLATFORM="linux/arm64"
FILE="tutorials/example-fed/examplefed/deploy/docker/Dockerfile"
CONTEXT="."
IMAGE="dfm-examplefed"
TAG="$IMAGE_TAG"
DOCKER="podman"
FORMAT="--format=docker"

if [[ ! -f $FILE ]]; then
    echo "Dockerfile not found at $FILE"
    echo "Did you forget to change directory?"
    exit 1
fi

# Build image
$DOCKER build -t $IMAGE:$TAG  --platform $PLATFORM $FORMAT -f $FILE $CONTEXT