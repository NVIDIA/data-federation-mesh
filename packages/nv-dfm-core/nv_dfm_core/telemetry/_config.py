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

"""Telemetry configuration management.

Configuration is read from environment variables with sensible defaults.
Telemetry is disabled by default and must be explicitly enabled.
"""

import os
import uuid
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class ExporterType(str, Enum):
    """Supported telemetry exporters."""

    CONSOLE = "console"
    FILE = "file"
    OTLP = "otlp"
    NONE = "none"


def _get_user_telemetry_base_dir() -> Path:
    """Get the base per-user directory for telemetry data.

    Uses XDG_STATE_HOME (for state/log-like data) with fallback to ~/.local/state.
    This ensures telemetry files are user-private and follow OS conventions.
    """
    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / "dfm" / "telemetry"
    return Path.home() / ".local" / "state" / "dfm" / "telemetry"


def _get_session_telemetry_dir() -> Path:
    """Get a unique directory for this session's telemetry data.

    Creates a UUID-based subdirectory to guarantee uniqueness across all runs.
    """
    base_dir = _get_user_telemetry_base_dir()
    session_id = uuid.uuid4().hex[:12]  # 12 chars is plenty unique, easier to read
    return base_dir / session_id


# Default values for configuration (used in TelemetryConfig and get_telemetry_config)
_DEFAULT_SERVICE_NAME = "dfm"
_DEFAULT_TELEMETRY_DIR = _get_session_telemetry_dir()
_DEFAULT_FILE_PATH = str(_DEFAULT_TELEMETRY_DIR / "telemetry.jsonl")
_DEFAULT_STREAMING_FILE_TEMPLATE = str(_DEFAULT_TELEMETRY_DIR / "{job_id}_{site}.jsonl")
_DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"


class TelemetryConfig(BaseModel):
    """Telemetry configuration settings.

    All settings can be overridden via environment variables prefixed with DFM_TELEMETRY_.
    """

    enabled: bool = False
    """Whether telemetry collection is enabled."""

    service_name: str = _DEFAULT_SERVICE_NAME
    """Service name used in spans and metrics."""

    exporter: ExporterType = ExporterType.CONSOLE
    """Which exporter to use for telemetry data."""

    file_path: str = _DEFAULT_FILE_PATH
    """Path for file exporter output (JSON Lines format)."""

    include_attributes: bool = True
    """Whether to include detailed attributes in spans."""

    sample_rate: float = 1.0
    """Sampling rate for traces (0.0 to 1.0). 1.0 = all traces."""

    flush_on_job_complete: bool = True
    """Whether to automatically flush telemetry when a job completes."""

    max_spans_per_batch: int = 1000
    """Maximum number of spans to include in a single batch."""

    # Streaming mode settings for long-running jobs
    streaming_enabled: bool = True
    """Whether to stream spans to local file as they complete (recommended for long jobs)."""

    streaming_file_template: str = _DEFAULT_STREAMING_FILE_TEMPLATE
    """Template for streaming file path. {job_id} and {site} are replaced."""

    flush_interval_seconds: float = 10.0
    """Seconds between automatic flushes to homesite (0 = only flush on complete).
    
    Lower values provide better telemetry reliability but slightly more overhead.
    For short jobs (<30s), consider 1-5 seconds to ensure spans are captured."""

    flush_threshold_spans: int = 100
    """Number of spans that triggers an automatic flush (0 = only time-based)."""

    buffer_max_spans: int = 1000
    """Maximum spans to keep in memory. Oldest dropped if exceeded (streaming saves them)."""

    # OTLP exporter settings
    otlp_endpoint: str = _DEFAULT_OTLP_ENDPOINT
    """OTLP HTTP endpoint for trace export (e.g., Jaeger, Tempo, OTel Collector)."""

    otlp_timeout: float = 10.0
    """HTTP timeout in seconds for OTLP requests."""


def _parse_bool(value: str) -> bool:
    """Parse a boolean from an environment variable string."""
    return value.lower() in ("true", "1", "yes", "on")


def _parse_float(value: str, default: float, clamp_01: bool = False) -> float:
    """Parse a float from an environment variable string.

    Args:
        value: String to parse
        default: Default value if parsing fails
        clamp_01: If True, clamp result to [0.0, 1.0] range
    """
    try:
        result = float(value)
        if clamp_01:
            return max(0.0, min(1.0, result))
        return max(0.0, result)  # At least non-negative
    except ValueError:
        return default


def _parse_int(value: str, default: int) -> int:
    """Parse an int from an environment variable string."""
    try:
        return max(1, int(value))
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_telemetry_config() -> TelemetryConfig:
    """Get the telemetry configuration from environment variables.

    Environment variables:
        DFM_TELEMETRY_ENABLED: Enable telemetry (default: false)
        DFM_TELEMETRY_SERVICE_NAME: Service name (default: dfm)
        DFM_TELEMETRY_EXPORTER: Exporter type: console, file, otlp, none (default: console)
        DFM_TELEMETRY_FILE_PATH: File path for file exporter (default: /tmp/dfm_telemetry.jsonl)
        DFM_TELEMETRY_INCLUDE_ATTRIBUTES: Include detailed attributes (default: true)
        DFM_TELEMETRY_SAMPLE_RATE: Trace sampling rate 0.0-1.0 (default: 1.0)
        DFM_TELEMETRY_FLUSH_ON_JOB_COMPLETE: Flush on job complete (default: true)
        DFM_TELEMETRY_MAX_SPANS_PER_BATCH: Max spans per batch (default: 1000)
        DFM_TELEMETRY_STREAMING_ENABLED: Stream to local file (default: true)
        DFM_TELEMETRY_STREAMING_FILE_TEMPLATE: Template for streaming file path
        DFM_TELEMETRY_FLUSH_INTERVAL_SECONDS: Seconds between homesite flushes (default: 60)
        DFM_TELEMETRY_FLUSH_THRESHOLD_SPANS: Spans before forced flush (default: 100)
        DFM_TELEMETRY_BUFFER_MAX_SPANS: Max spans in memory (default: 1000)
        DFM_TELEMETRY_OTLP_ENDPOINT: OTLP HTTP endpoint (default: http://localhost:4318)
        DFM_TELEMETRY_OTLP_TIMEOUT: OTLP request timeout in seconds (default: 10.0)

    Returns:
        TelemetryConfig with values from environment or defaults.
    """
    enabled = _parse_bool(os.environ.get("DFM_TELEMETRY_ENABLED", "false"))

    exporter_str = os.environ.get("DFM_TELEMETRY_EXPORTER", "console").lower()
    try:
        exporter = ExporterType(exporter_str)
    except ValueError:
        exporter = ExporterType.CONSOLE

    return TelemetryConfig(
        enabled=enabled,
        service_name=os.environ.get(
            "DFM_TELEMETRY_SERVICE_NAME", _DEFAULT_SERVICE_NAME
        ),
        exporter=exporter,
        file_path=os.environ.get("DFM_TELEMETRY_FILE_PATH", _DEFAULT_FILE_PATH),
        include_attributes=_parse_bool(
            os.environ.get("DFM_TELEMETRY_INCLUDE_ATTRIBUTES", "true")
        ),
        sample_rate=_parse_float(
            os.environ.get("DFM_TELEMETRY_SAMPLE_RATE", "1.0"), 1.0, clamp_01=True
        ),
        flush_on_job_complete=_parse_bool(
            os.environ.get("DFM_TELEMETRY_FLUSH_ON_JOB_COMPLETE", "true")
        ),
        max_spans_per_batch=_parse_int(
            os.environ.get("DFM_TELEMETRY_MAX_SPANS_PER_BATCH", "1000"), 1000
        ),
        # Streaming mode settings
        streaming_enabled=_parse_bool(
            os.environ.get("DFM_TELEMETRY_STREAMING_ENABLED", "true")
        ),
        streaming_file_template=os.environ.get(
            "DFM_TELEMETRY_STREAMING_FILE_TEMPLATE",
            _DEFAULT_STREAMING_FILE_TEMPLATE,
        ),
        flush_interval_seconds=_parse_float(
            os.environ.get("DFM_TELEMETRY_FLUSH_INTERVAL_SECONDS", "10.0"), 10.0
        ),
        flush_threshold_spans=_parse_int(
            os.environ.get("DFM_TELEMETRY_FLUSH_THRESHOLD_SPANS", "100"), 100
        ),
        buffer_max_spans=_parse_int(
            os.environ.get("DFM_TELEMETRY_BUFFER_MAX_SPANS", "1000"), 1000
        ),
        # OTLP settings
        otlp_endpoint=os.environ.get(
            "DFM_TELEMETRY_OTLP_ENDPOINT", _DEFAULT_OTLP_ENDPOINT
        ),
        otlp_timeout=_parse_float(
            os.environ.get("DFM_TELEMETRY_OTLP_TIMEOUT", "10.0"), 10.0
        ),
    )


def telemetry_enabled() -> bool:
    """Quick check if telemetry is enabled.

    This is the primary check used throughout the codebase to
    guard telemetry operations.
    """
    return get_telemetry_config().enabled


def reset_config_cache() -> None:
    """Reset the cached configuration.

    Useful for testing when environment variables change.
    """
    get_telemetry_config.cache_clear()
