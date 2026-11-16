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

:mod:`nv_dfm_core.targets`: Execution Targets
==============================================

DFM supports two execution targets that share the same core execution layer.
Choose a target when constructing a :class:`~nv_dfm_core.session.Session`.

.. _targets.flare:

``nv_dfm_core.targets.flare``: NVIDIA Flare Target
---------------------------------------------------

.. py:module:: nv_dfm_core.targets.flare

    Runs the federation over NVIDIA Flare's distributed infrastructure. Suitable
    for production deployments where sites run on separate machines.

.. currentmodule:: nv_dfm_core.targets.flare

.. autosummary::
    :nosignatures:
    :toctree: generated/targets/
    :template: class.rst

    FlareSessionDelegate
    FlareApp
    FlareOptions
    FlareRouter
    Job

.. _targets.local:

``nv_dfm_core.targets.local``: Local Target
--------------------------------------------

.. py:module:: nv_dfm_core.targets.local

    Emulates a distributed federation on a single machine using Python
    multiprocessing. Ideal for development and testing without external
    infrastructure.

.. currentmodule:: nv_dfm_core.targets.local

.. autosummary::
    :nosignatures:
    :toctree: generated/targets/
    :template: class.rst

    LocalSessionDelegate
    FederationRunner
    LocalJob
    JobRunner
    JobExecution
    JobHandle
    JobSubmission
    LocalRouter
