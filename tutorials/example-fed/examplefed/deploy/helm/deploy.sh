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

# Initialize values arguments
VALUES_ARGS=""
INSTALL=false
UNINSTALL=false
HELM_ARGS=""

# Process command line value files
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--values)
            VALUES_ARGS="$VALUES_ARGS --values $(realpath $2)"
            shift 2
            ;;
        -u|--uninstall)
            UNINSTALL=true
            shift
            ;;
        -i|--install)
            INSTALL=true
            shift
            ;;
        --)
            shift
            # Collect all remaining arguments for helm
            HELM_ARGS="$@"
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-i|--install] [-u|--uninstall] [-f|--values FILE] [-- HELM_ARGS...]"
            exit 1
            ;;
    esac
done

if [[ "$INSTALL" == false && "$UNINSTALL" == false ]]; then
    echo "No action requested, exiting"
    exit 0
fi


# No namespace specified, use default namespace
NAMESPACE_ARGS=""

# Uninstall if requested
if [[ "$UNINSTALL" == true ]]; then
    echo "Uninstalling existing release..."
    helm uninstall examplefed --ignore-not-found $NAMESPACE_ARGS
fi

# Clean up old secrets

if kubectl $NAMESPACE_ARGS get secret flare-workspace-server-init &> /dev/null; then
    kubectl $NAMESPACE_ARGS delete secret flare-workspace-server-init
fi


if kubectl $NAMESPACE_ARGS get secret flare-workspace-reception-init &> /dev/null; then
    kubectl $NAMESPACE_ARGS delete secret flare-workspace-reception-init
fi


# If we are only uninstalling, exit here
if [[ "$UNINSTALL" == true && "$INSTALL" == false ]]; then
    echo "Only uninstallation requested, exiting"
    exit 0
fi



SERVER_PACKAGE_PATH=$DFM_WORKSPACE_SERVER_INIT_PATH

if [[ ! -f "$SERVER_PACKAGE_PATH" ]]; then
    echo "Server startup package not found: $SERVER_PACKAGE_PATH"
    exit 1
fi
# Create new secrets for DFM initialization
echo "Creating secret for server from $SERVER_PACKAGE_PATH"
kubectl $NAMESPACE_ARGS create secret generic flare-workspace-server-init --from-file=$SERVER_PACKAGE_PATH




CLIENT_PACKAGE_PATH=$DFM_WORKSPACE_RECEPTION_INIT_PATH

if [[ ! -f "$CLIENT_PACKAGE_PATH" ]]; then
    echo "Client startup package not found: $CLIENT_PACKAGE_PATH"
    exit 1
fi
# Create new secrets for DFM initialization
echo "Creating secret for reception from $CLIENT_PACKAGE_PATH"
kubectl $NAMESPACE_ARGS create secret generic flare-workspace-reception-init --from-file=$CLIENT_PACKAGE_PATH



# Deploy the Helm chart
cd "$(dirname "$0")"

echo "Installing examplefed from examplefed"
helm upgrade --install examplefed examplefed $NAMESPACE_ARGS $VALUES_ARGS $HELM_ARGS

