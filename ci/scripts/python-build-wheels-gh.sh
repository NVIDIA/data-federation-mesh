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

source ci/scripts/common-python.sh

BUILD_WHEEL_OUTPUT_DIR=${BUILD_WHEEL_OUTPUT_DIR:-artifacts/dist}

# Build all workspace packages
PACKAGES=$(list_packages)

echo "Building packages:"
echo "$PACKAGES" | while read pkg; do echo "  - $pkg"; done
echo ""

for PACKAGE in $PACKAGES; do
    echo "Building $PACKAGE..."
    uv build --package "$PACKAGE" --out-dir="$BUILD_WHEEL_OUTPUT_DIR"
done

echo ""
echo "All packages built in $BUILD_WHEEL_OUTPUT_DIR"
ls -la "$BUILD_WHEEL_OUTPUT_DIR/"
