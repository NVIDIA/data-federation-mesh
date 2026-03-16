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

from typing import Optional

import torch


def available_devices() -> list[str]:
    """
    Get the available devices.
    """
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]


def setup_device(device: Optional[str], logger) -> torch.device:
    """
    Setup and validate device for operations.

    Args:
        device: Device specification string
        logger: Logger instance for logging device selection

    Returns:
        torch.device: The configured device

    Raises:
        ValueError: If device specification is invalid or device is not available
    """
    if device is not None:
        # User specified a device
        if device.startswith("cuda:"):
            # Specific CUDA device (e.g., "cuda:0", "cuda:1")
            device_id = int(device.split(":")[1])
            if device_id >= torch.cuda.device_count():
                available_devices = list(range(torch.cuda.device_count()))
                raise ValueError(
                    f"CUDA device {device_id} not available. Available devices: {available_devices}"
                )
            selected_device = torch.device(device)
            logger.info(f"Using specified CUDA device: {device}")
        elif device == "cuda":
            # Use default CUDA device (cuda:0)
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available on this system")
            selected_device = torch.device("cuda:0")
            logger.info("Using default CUDA device: cuda:0")
        elif device == "cpu":
            # Force CPU usage
            selected_device = torch.device("cpu")
            logger.info("Using CPU device")
        else:
            raise ValueError(
                f"Invalid device specification: {device}. Use 'cuda', 'cuda:N', or 'cpu'"
            )
    else:
        # Auto-select device
        if torch.cuda.is_available():
            # Check if multiple GPUs are available
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                # Use the first available GPU by default
                selected_device = torch.device("cuda:0")
                logger.info(
                    f"Multiple GPUs detected ({gpu_count}). Using cuda:0 by default. Available: {list(range(gpu_count))}"
                )
            else:
                selected_device = torch.device("cuda:0")
                logger.info("Auto-selected CUDA device: cuda:0")
        else:
            selected_device = torch.device("cpu")
            logger.info("Auto-selected CPU device")

    return selected_device
