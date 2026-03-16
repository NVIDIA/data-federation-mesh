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

uv sync --group docs

rm -rf artifacts/docs

# Copy tutorials into docs/ so Sphinx sees them as first-class source
# documents and resolves relative cross-references correctly.
cp -r tutorials/zero-to-thirty  docs/tutorials-zero-to-thirty
cp -r tutorials/weather-fed     docs/tutorials-weather-fed
cp -r tutorials/example-fed     docs/tutorials-example-fed

DOC_VERSION="${DOC_VERSION:-main}" uv run sphinx-build docs artifacts/docs

# Clean up copies (they are in .gitignore, but keep the tree tidy)
rm -rf docs/tutorials-zero-to-thirty docs/tutorials-weather-fed docs/tutorials-example-fed
