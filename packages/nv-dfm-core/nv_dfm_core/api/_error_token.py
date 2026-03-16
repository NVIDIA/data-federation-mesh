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

import traceback
from typing import Literal

from pydantic import BaseModel


class ErrorInfo(BaseModel, frozen=True):
    """Information about a single error including type, message, and stack trace."""

    type: str
    message: str
    stack_trace: str


class ErrorToken(BaseModel, frozen=True):
    """A token representing one or more errors that occurred during pipeline execution."""

    token: Literal["@dfm-error-token"] = "@dfm-error-token"
    errors: list[ErrorInfo]

    def print(self):
        print(f"ErrorToken with {len(self.errors)} errors:")
        for error in self.errors:
            print(f"Error: {error.type}: {error.message}")
            print(f"Stack trace: {error.stack_trace}")

    @classmethod
    def from_exception(cls, error: Exception) -> "ErrorToken":
        return cls(
            errors=[
                ErrorInfo(
                    type=type(error).__name__,
                    message=str(error),
                    stack_trace=traceback.format_exc(),
                )
            ]
        )

    @classmethod
    def from_error_tokens(cls, error_tokens: list["ErrorToken"]) -> "ErrorToken":
        assert len(error_tokens) > 0, (
            "Cannot create an ErrorToken from an empty list of error tokens"
        )
        if len(error_tokens) == 1:
            return error_tokens[0]
        return cls(
            errors=[
                ErrorInfo(
                    type=type(e).__name__,
                    message=str(e),
                    stack_trace=traceback.format_exc(),
                )
                for e in error_tokens
            ]
        )
