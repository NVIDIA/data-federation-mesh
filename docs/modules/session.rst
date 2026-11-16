.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

:mod:`nv_dfm_core.session`: Session and Job Management
=======================================================

.. automodule:: nv_dfm_core.session
    :no-members:
    :no-inherited-members:

.. currentmodule:: nv_dfm_core.session

Session
-------

The main entry point for connecting to a federation and executing pipelines.

.. autosummary::
    :nosignatures:
    :toctree: generated/session/
    :template: class.rst

    Session
    SessionDelegate

Jobs
----

.. autosummary::
    :nosignatures:
    :toctree: generated/session/
    :template: class.rst

    Job
    JobStatus

Callbacks
---------

DFM delivers pipeline results to your application through callbacks. The callback
types and dispatcher strategies below control how and when callbacks are invoked.

.. autosummary::
    :nosignatures:
    :toctree: generated/session/
    :template: class.rst

    DfmDataCallback
    DfmDataCallbackSync
    DfmDataCallbackAsync
    CallbackDispatcher
    DirectDispatcher
    ManualDispatcher
    AsyncioDispatcher
    CallbackRunner
    ManualCallbackRunner

.. autosummary::
    :nosignatures:
    :toctree: generated/session/
    :template: function.rst

    callbacks_kind
    call_callbacks
    call_callbacks_async
    configure_session_logging
