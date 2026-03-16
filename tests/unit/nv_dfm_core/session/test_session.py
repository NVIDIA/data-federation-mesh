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

# pyright: reportPrivateUsage=false
"""
Tests for the dfm.session._session module.
"""

from unittest.mock import MagicMock, patch

import pytest

from nv_dfm_core.api._pipeline import Pipeline
from nv_dfm_core.api._prepared_pipeline import PreparedPipeline
from nv_dfm_core.session._session import Session

FEDERATION_NAME = "examplefed"
HOMESITE_NAME = "concierge"


@pytest.fixture
def session():
    """Create a Session instance with default parameters."""
    with patch(
        "nv_dfm_core.session._session.load_site_runtime_module"
    ) as load_module_mock:
        load_module_mock.return_value = MagicMock()
        session = Session(FEDERATION_NAME, HOMESITE_NAME)
        yield session


@pytest.fixture
def pipeline():
    """Create a Pipeline instance."""
    return Pipeline()


@pytest.fixture
def prepared_pipeline():
    """Create a PreparedPipeline instance."""
    with patch("nv_dfm_core.session._session.PreparedPipeline") as mock:
        mock.controller_site = HOMESITE_NAME
        yield mock


class TestSession:
    """Tests for the Session class."""

    def test_init_default_values(self):
        """Test Session initialization with default values."""
        session = Session(FEDERATION_NAME, HOMESITE_NAME)
        assert session._target == "flare"
        assert session._logger is not None
        assert session._delegate is not None

    def test_init_custom_values(self):
        """Test Session initialization with custom values."""
        session = Session(
            federation_name="test_federation",
            homesite="test_app",
            target="local",
            debug=True,
        )

        assert session._target == "local"
        assert session._delegate is not None

    def test_init_invalid_values(self):
        """Test Session initialization with invalid values."""
        with pytest.raises(
            ValueError, match="Federation module and homesite are required"
        ):
            Session("", HOMESITE_NAME)

        with pytest.raises(
            ValueError, match="Federation module and homesite are required"
        ):
            Session(FEDERATION_NAME, "")

    def test_connect(self, session: Session):
        """Test connecting to the target."""
        with patch.object(session._delegate, "connect") as mock_connect:
            session.connect()
            mock_connect.assert_called_once()

    def test_runtime_module_property(self, session: Session):
        """Test the runtime_module property."""
        mock_module = MagicMock()
        mock_module.API_VERSION = "1.0.0"

        with patch(
            "nv_dfm_core.session._session.load_site_runtime_module"
        ) as load_mock:
            load_mock.return_value = mock_module
            runtime_module = session.runtime_module

            assert runtime_module == mock_module
            assert runtime_module.API_VERSION == "1.0.0"

    def test_runtime_module_property_cached(self, session: Session):
        """Test that runtime_module property is cached."""
        mock_module = MagicMock()

        with patch(
            "nv_dfm_core.session._session.load_site_runtime_module"
        ) as load_mock:
            load_mock.return_value = mock_module

            # First call should load the module
            runtime_module1 = session.runtime_module
            # Second call should use cached version
            runtime_module2 = session.runtime_module

            assert runtime_module1 == runtime_module2
            load_mock.assert_called_once()

    def test_prepare_pipeline_homesite(self, session: Session):
        """Test preparing a pipeline for homesite execution."""
        pipeline = Pipeline()
        prepared = session.prepare(pipeline, restrict_to_sites="homesite")
        assert isinstance(prepared, PreparedPipeline)

    def test_prepare_pipeline_specific_sites(self, session: Session):
        """Test preparing a pipeline for specific sites."""
        pipeline = Pipeline()
        prepared = session.prepare(pipeline, restrict_to_sites=["server", "concierge"])
        assert isinstance(prepared, PreparedPipeline)

    def test_prepare_pipeline_no_restriction(self, session: Session):
        """Test preparing a pipeline without site restrictions."""
        pipeline = Pipeline()
        prepared = session.prepare(pipeline)
        assert isinstance(prepared, PreparedPipeline)

    def test_prepare_pipeline_with_api_version(self, session: Session):
        """Test preparing a pipeline that already has API version."""
        pipeline = Pipeline(api_version="2.0.0")
        prepared = session.prepare(pipeline)
        assert isinstance(prepared, PreparedPipeline)

    def test_execute_pipeline(self, session: Session, prepared_pipeline: MagicMock):
        """Test executing a pipeline."""
        with patch.object(session._delegate, "execute") as mock_execute:
            mock_job = MagicMock()
            mock_execute.return_value = mock_job

            job = session.execute(prepared_pipeline, {})

            assert job == mock_job
            mock_execute.assert_called_once()

    def test_execute_pipeline_with_callbacks(
        self, session: Session, prepared_pipeline: MagicMock
    ):
        """Test executing a pipeline with callbacks."""
        default_callback = MagicMock()
        place_callbacks = {"site1": MagicMock()}

        with patch.object(session._delegate, "execute") as mock_execute:
            mock_job = MagicMock()
            mock_execute.return_value = mock_job

            job = session.execute(
                prepared_pipeline,
                {},
                default_callback=default_callback,
                place_callbacks=place_callbacks,
            )

            assert job == mock_job
            mock_execute.assert_called_once()

    def test_debug_show_code(self, session: Session, prepared_pipeline: MagicMock):
        """Test debug_show_code method."""
        prepared_pipeline.net_irs.return_value = {
            "homesite": MagicMock(),
            "server": MagicMock(),
        }
        with patch("nv_dfm_core.gen.modgen.ModGen") as mock_modgen_class:
            mock_modgen = MagicMock()
            mock_modgen_class.return_value = mock_modgen
            mock_modgen._generate_python_code.return_value = "test_code"

            result = session.debug_show_code(prepared_pipeline)

            assert isinstance(result, dict)
            assert mock_modgen._generate_python_code.call_count == 2

    def test_reattach_validates_callbacks(self, session: Session):
        """Test that reattach validates callback combinations."""
        # Should raise if place_callbacks without default_callback
        with pytest.raises(ValueError, match="default_callback must be provided"):
            session.reattach(
                job_id="some_job",
                default_callback=None,
                place_callbacks={"place1": MagicMock()},
            )

        # Should not raise with valid combinations
        with patch.object(session._delegate, "reattach") as mock_reattach:
            mock_reattach.return_value = MagicMock()

            # No callbacks (info only)
            session.reattach(job_id="job1")
            mock_reattach.assert_called_once()
            mock_reattach.reset_mock()

            # Default callback only
            session.reattach(job_id="job2", default_callback=MagicMock())
            mock_reattach.assert_called_once()
            mock_reattach.reset_mock()

            # Both callbacks
            session.reattach(
                job_id="job3",
                default_callback=MagicMock(),
                place_callbacks={"place1": MagicMock()},
            )
            mock_reattach.assert_called_once()
