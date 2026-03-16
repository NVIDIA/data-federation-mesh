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


class Constant:
    # task name defaults
    TASK_START_EXECUTION = "nv_dfm_core.start_execution"

    # default component config values
    TASK_TIMEOUT = 100
    TASK_CHECK_INTERVAL = 0.1
    JOB_STATUS_CHECK_INTERVAL = 0.1
    MAX_CLIENT_OP_INTERVAL = 90.0
    WORKFLOW_PROGRESS_TIMEOUT = 3600.0

    # message topics
    TOPIC_SEND_TO_PLACE = "nv_dfm_core.send_to_place"
    # commands
    CMD_RETRIEVE_TOKENS = "nv_dfm_core.retrieve_tokens"
    CMD_SEND_TO_PLACE = "nv_dfm_core.send_to_place"

    # keys for Shareable between client and server
    MSG_KEY_TOKEN_PACKAGE_DICT = "nv_dfm_core.send_to_place_package_dict"

    ERROR = "nv_dfm_core.reply.error"

    EXIT_CODE_CANT_START = 101
    EXIT_CODE_FATAL_ERROR = 102

    APP_CTX_FL_CONTEXT = "nv_dfm_core.fl_context"
