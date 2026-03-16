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

"""Tests for telemetry configuration."""

import os
from unittest.mock import patch

import pytest

from nv_dfm_core.telemetry._config import (
    ExporterType,
    TelemetryConfig,
    get_telemetry_config,
    reset_config_cache,
    telemetry_enabled,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the config cache before and after each test."""
    reset_config_cache()
    yield
    reset_config_cache()


class TestTelemetryConfig:
    """Tests for TelemetryConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TelemetryConfig()
        assert config.enabled is False
        assert config.service_name == "dfm"
        assert config.exporter == ExporterType.CONSOLE
        assert config.sample_rate == 1.0
        assert config.streaming_enabled is True
        assert config.flush_interval_seconds == 10.0
        assert config.flush_threshold_spans == 100
        assert config.buffer_max_spans == 1000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TelemetryConfig(
            enabled=True,
            service_name="my-service",
            exporter=ExporterType.OTLP,
            sample_rate=0.5,
            flush_interval_seconds=30.0,
        )
        assert config.enabled is True
        assert config.service_name == "my-service"
        assert config.exporter == ExporterType.OTLP
        assert config.sample_rate == 0.5
        assert config.flush_interval_seconds == 30.0


class TestGetTelemetryConfig:
    """Tests for get_telemetry_config function."""

    def test_default_config(self):
        """Test default config when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            reset_config_cache()
            config = get_telemetry_config()
            assert config.enabled is False
            assert config.exporter == ExporterType.CONSOLE

    def test_enabled_from_env(self):
        """Test enabling telemetry via environment variable."""
        with patch.dict(os.environ, {"DFM_TELEMETRY_ENABLED": "true"}, clear=True):
            reset_config_cache()
            config = get_telemetry_config()
            assert config.enabled is True

    def test_exporter_from_env(self):
        """Test setting exporter via environment variable."""
        with patch.dict(os.environ, {"DFM_TELEMETRY_EXPORTER": "otlp"}, clear=True):
            reset_config_cache()
            config = get_telemetry_config()
            assert config.exporter == ExporterType.OTLP

    def test_invalid_exporter_defaults_to_console(self):
        """Test invalid exporter value defaults to console."""
        with patch.dict(os.environ, {"DFM_TELEMETRY_EXPORTER": "invalid"}, clear=True):
            reset_config_cache()
            config = get_telemetry_config()
            assert config.exporter == ExporterType.CONSOLE

    def test_sample_rate_clamped(self):
        """Test sample rate is clamped to 0-1 range."""
        with patch.dict(os.environ, {"DFM_TELEMETRY_SAMPLE_RATE": "1.5"}, clear=True):
            reset_config_cache()
            config = get_telemetry_config()
            assert config.sample_rate == 1.0

        with patch.dict(os.environ, {"DFM_TELEMETRY_SAMPLE_RATE": "-0.5"}, clear=True):
            reset_config_cache()
            config = get_telemetry_config()
            assert config.sample_rate == 0.0

    def test_streaming_settings(self):
        """Test streaming configuration from environment."""
        env = {
            "DFM_TELEMETRY_STREAMING_ENABLED": "false",
            "DFM_TELEMETRY_FLUSH_INTERVAL_SECONDS": "120",
            "DFM_TELEMETRY_FLUSH_THRESHOLD_SPANS": "50",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            config = get_telemetry_config()
            assert config.streaming_enabled is False
            assert config.flush_interval_seconds == 120.0
            assert config.flush_threshold_spans == 50

    def test_otlp_settings(self):
        """Test OTLP configuration from environment."""
        env = {
            "DFM_TELEMETRY_OTLP_ENDPOINT": "http://jaeger:4318",
            "DFM_TELEMETRY_OTLP_TIMEOUT": "30.0",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_config_cache()
            config = get_telemetry_config()
            assert config.otlp_endpoint == "http://jaeger:4318"
            assert config.otlp_timeout == 30.0

    def test_config_is_cached(self):
        """Test that config is cached and reused."""
        with patch.dict(os.environ, {"DFM_TELEMETRY_ENABLED": "true"}, clear=True):
            reset_config_cache()
            config1 = get_telemetry_config()
            config2 = get_telemetry_config()
            assert config1 is config2


class TestTelemetryEnabled:
    """Tests for telemetry_enabled helper function."""

    def test_disabled_by_default(self):
        """Test telemetry is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            reset_config_cache()
            assert telemetry_enabled() is False

    def test_enabled_when_set(self):
        """Test telemetry enabled when env var set."""
        with patch.dict(os.environ, {"DFM_TELEMETRY_ENABLED": "true"}, clear=True):
            reset_config_cache()
            assert telemetry_enabled() is True

    def test_various_true_values(self):
        """Test various truthy string values."""
        for value in ["true", "True", "TRUE", "1", "yes", "on"]:
            with patch.dict(os.environ, {"DFM_TELEMETRY_ENABLED": value}, clear=True):
                reset_config_cache()
                assert telemetry_enabled() is True, f"Failed for value: {value}"

    def test_various_false_values(self):
        """Test various falsy string values."""
        for value in ["false", "False", "0", "no", "off", ""]:
            with patch.dict(os.environ, {"DFM_TELEMETRY_ENABLED": value}, clear=True):
                reset_config_cache()
                assert telemetry_enabled() is False, f"Failed for value: {value}"
