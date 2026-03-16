#!/usr/bin/env python3
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

import json
from pathlib import Path
from uuid import uuid4


class ResultsManager:
    """Manages the storage and retrieval of job results.

    This class maintains two main data structures:
    1. _results: Maps result IDs to their corresponding result data
    2. _sent: Tracks which results have been sent to the client

    The class provides methods to:
    - Add new results
    - Mark results as sent
    - Retrieve results (all, unsent, or by ID)
    - Save results to disk
    """

    def __init__(self):
        """Initialize a new ResultsManager with empty result and sent tracking dictionaries."""
        self._results = {}  # Maps result IDs to their data
        self._sent = {}  # Tracks which results have been sent to the client

    def add_result(self, id: uuid4, result: dict):
        """Add a new result to the manager.

        Args:
            id: Unique identifier for the result
            result: The result data to store

        Note:
            New results are automatically marked as unsent.
        """
        self._results[id] = result
        self._sent[id] = False

    def mark_as_sent(self, id: uuid4):
        """Mark a result as having been sent to the client.

        Args:
            id: The ID of the result to mark as sent
        """
        self._sent[id] = True

    def get_result(self, id: uuid4) -> dict:
        """Retrieve a specific result by its ID.

        Args:
            id: The ID of the result to retrieve

        Returns:
            The result data

        Note:
            This method does not affect the sent status of the result.
        """
        return self._results[id]

    def is_sent(self, id: uuid4) -> bool:
        """Check if a result has been sent to the client.

        Args:
            id: The ID of the result to check

        Returns:
            True if the result has been sent, False otherwise
        """
        return self._sent[id]

    def get_all(self) -> list[dict]:
        """Retrieve all results and mark them as sent.

        Returns:
            List of all results

        Note:
            This method automatically marks all results as sent to prevent
            them from being returned by get_all_unsent in subsequent calls.
        """
        results = [self._results[id] for id in self._results]
        # Mark all results as sent, otherwise get_all_unsent will return them
        for id in self._results:
            self._sent[id] = True
        return results

    def get_all_unsent(self) -> list[dict]:
        """Retrieve all results that haven't been sent yet.

        Returns:
            List of unsent results

        Note:
            This method automatically marks the returned results as sent
            to prevent them from being returned again in subsequent calls.
        """
        unsent_ids = [id for id in self._results if not self._sent[id]]
        unsent_results = [self._results[id] for id in unsent_ids]
        # Mark the returned results as sent
        for id in unsent_ids:
            self._sent[id] = True
        return unsent_results

    def save_all(self, directory: Path | str):
        """Save all results to JSON files in the specified directory.

        Args:
            directory: Path to the directory where results should be saved

        Raises:
            RuntimeError: If the specified directory does not exist

        Note:
            Each result is saved as a separate JSON file named with its ID.
            The files are formatted with indentation for readability.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for id, result in self._results.items():
            with open(directory / f"{id}.json", "w") as f:
                json.dump(result, f, indent=4)
