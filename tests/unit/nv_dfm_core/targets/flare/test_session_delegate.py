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

# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false, reportMissingImports=false
"""
Tests for the dfm.targets.flare._session_delegate module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nv_dfm_core.exec._frame import Frame
from nv_dfm_core.session import DirectDispatcher
from nv_dfm_core.targets.flare._flare_options import FlareOptions
from nv_dfm_core.targets.flare._session_delegate import FlareSessionDelegate

FEDERATION_NAME = "examplefed"
HOMESITE_NAME = "concierge"


@pytest.fixture
def mock_flare_session():
    """Create a mock Flare session."""
    with patch("nvflare.fuel.flare_api.flare_api.new_secure_session") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_session():
    """Create a mock Session instance."""
    session = MagicMock()
    session.runtime_module = MagicMock()
    return session


@pytest.fixture
def flare_delegate(mock_session):
    """Create a FlareSessionDelegate instance."""
    delegate = FlareSessionDelegate(
        session=mock_session,
        federation_name=FEDERATION_NAME,
        homesite=HOMESITE_NAME,
        logger=MagicMock(),
    )
    return delegate


@pytest.fixture
def prepared_pipeline():
    """Create a PreparedPipeline instance."""
    pipeline = MagicMock()
    pipeline.pipeline_name = "test"
    pipeline.get_net_irs.return_value = {"site1": MagicMock()}
    return pipeline


class TestFlareSessionDelegate:
    """Tests for the FlareSessionDelegate class."""

    def test_init_default_values(self, mock_session):
        """Test FlareSessionDelegate initialization with default values.

        Default paths use appdirs.user_state_dir("nv_dfm_core"); we patch it
        to return a fixed path so expected values are deterministic.
        (When appdirs returns a path with a space, the delegate uses
        ~/.nv_dfm_core instead so Flare submit_job does not get ERROR_SYNTAX.)
        """
        state_dir = "/tmp"
        with patch("appdirs.user_state_dir", return_value=state_dir):
            delegate = FlareSessionDelegate(
                session=mock_session,
                federation_name=FEDERATION_NAME,
                homesite=HOMESITE_NAME,
                logger=MagicMock(),
            )

        assert delegate._user == "admin@nvidia.com"
        assert delegate._flare_workspace == Path(f"{state_dir}/nvflare/poc").absolute()
        assert delegate._job_workspace == Path(f"{state_dir}/dfm_workspace").absolute()
        assert (
            delegate._admin_package
            == Path(f"{state_dir}/nvflare/poc/admin@nvidia.com").absolute()
        )
        assert delegate._flare_session is None

    def test_init_custom_values(self, mock_session):
        """Test FlareSessionDelegate initialization with custom values."""
        user = "test@nvidia.com"
        flare_workspace = Path("/custom/flare/workspace")
        job_workspace = Path("/custom/job/workspace")
        admin_package = Path("/custom/admin/package")

        delegate = FlareSessionDelegate(
            session=mock_session,
            federation_name=FEDERATION_NAME,
            homesite=HOMESITE_NAME,
            logger=MagicMock(),
            user=user,
            flare_workspace=flare_workspace,
            job_workspace=job_workspace,
            admin_package=admin_package,
        )

        assert delegate._user == user
        assert delegate._flare_workspace == flare_workspace.absolute()
        assert delegate._job_workspace == job_workspace.absolute()
        assert delegate._admin_package == admin_package.absolute()

    def test_connect(
        self, flare_delegate: FlareSessionDelegate, mock_flare_session: MagicMock
    ):
        """Test connecting to Flare."""
        flare_delegate.connect(debug=False)
        mock_flare_session.assert_called_once_with(
            flare_delegate._user,
            flare_delegate._admin_package.as_posix(),
            debug=False,
        )
        assert flare_delegate._flare_session is not None

    def test_connect_already_connected(
        self, flare_delegate: FlareSessionDelegate, mock_flare_session: MagicMock
    ):
        """Test connecting when already connected."""
        flare_delegate._flare_session = MagicMock()
        flare_delegate.connect(debug=False)
        mock_flare_session.assert_not_called()

    def test_connect_failure(self, flare_delegate: FlareSessionDelegate):
        """Test connecting failure."""
        with patch(
            "nvflare.fuel.flare_api.flare_api.new_secure_session"
        ) as mock_session:
            mock_session.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                flare_delegate.connect(debug=False)

    def test_provision_only(
        self, flare_delegate: FlareSessionDelegate, prepared_pipeline: MagicMock
    ):
        """Test provisioning a pipeline."""
        input_params = [(Frame.start_frame(num=0), {"param1": "value1"})]

        with patch("nv_dfm_core.targets.flare.FlareApp") as mock_app_class:
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app
            mock_app.export.return_value = Path("/tmp/test_job")

            result = flare_delegate.provision_only(
                prepared_pipeline,
                input_params,
                options=FlareOptions(),
            )

            assert result == Path("/tmp/test_job")
            mock_app_class.assert_called_once()
            mock_app.export.assert_called_once_with(flare_delegate._job_workspace)

    def test_provision_only_failure(
        self, flare_delegate: FlareSessionDelegate, prepared_pipeline: MagicMock
    ):
        """Test provisioning failure."""
        input_params = [(Frame.start_frame(num=0), {"param1": "value1"})]

        with patch("nv_dfm_core.targets.flare.FlareApp") as mock_app_class:
            mock_app_class.side_effect = Exception("Provisioning failed")

            with pytest.raises(Exception, match="Provisioning failed"):
                flare_delegate.provision_only(
                    prepared_pipeline, input_params, options=FlareOptions()
                )

    def test_execute(
        self, flare_delegate: FlareSessionDelegate, prepared_pipeline: MagicMock
    ):
        """Test executing a pipeline."""
        # Setup connected session
        mock_session = MagicMock()
        mock_session.submit_job.return_value = "job_123"
        flare_delegate._flare_session = mock_session
        assert flare_delegate._flare_session is not None

        input_params = [(Frame.start_frame(num=0), {"param1": "value1"})]
        default_callback = MagicMock()
        place_callbacks = {"site1": MagicMock()}

        # Create callback runner
        dispatcher = DirectDispatcher()
        callback_runner = dispatcher.create_runner(
            default_callback=default_callback,
            place_callbacks=place_callbacks,
        )

        with patch("nv_dfm_core.targets.flare._job.Job") as mock_job_class:
            mock_job = MagicMock()
            mock_job_class.return_value = mock_job

            with patch.object(flare_delegate, "provision_only") as mock_provision:
                mock_provision.return_value = Path("/tmp/test_job")

                result = flare_delegate.execute(
                    pipeline=prepared_pipeline,
                    next_frame=Frame.start_frame(num=1),
                    input_params=input_params,
                    callback_runner=callback_runner,
                    debug=False,
                    options=FlareOptions(),
                )

                assert result == mock_job
                kwargs = mock_provision.call_args.kwargs
                assert kwargs["pipeline"] is prepared_pipeline
                assert kwargs["input_params"] == input_params
                assert kwargs["force_modgen"] is False
                assert "options" in kwargs
                assert isinstance(kwargs["options"], FlareOptions)
                flare_delegate._flare_session.submit_job.assert_called_once_with(
                    "/tmp/test_job"
                )
                mock_job_class.assert_called_once()

    def test_execute_not_connected(
        self, flare_delegate: FlareSessionDelegate, prepared_pipeline: MagicMock
    ):
        """Test executing when not connected."""
        input_params = [(Frame.start_frame(num=0), {"param1": "value1"})]

        # Create callback runner
        dispatcher = DirectDispatcher()
        callback_runner = dispatcher.create_runner()

        with pytest.raises(ValueError, match="Flare session not connected"):
            _ = flare_delegate.execute(
                pipeline=prepared_pipeline,
                next_frame=Frame.start_frame(num=1),
                input_params=input_params,
                callback_runner=callback_runner,
                debug=False,
            )
