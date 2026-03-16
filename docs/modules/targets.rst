:mod:`nv_dfm_core.targets`: Execution Targets
==============================================

DFM supports two execution targets that share the same core execution layer.
Choose a target when constructing a :class:`~nv_dfm_core.session.Session`.

.. _targets.flare:

``nv_dfm_core.targets.flare``: NVIDIA Flare Target
---------------------------------------------------

Runs the federation over NVIDIA Flare's distributed infrastructure. Suitable
for production deployments where sites run on separate machines.

.. automodule:: nv_dfm_core.targets.flare
    :no-members:
    :no-inherited-members:

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

Emulates a distributed federation on a single machine using Python
multiprocessing. Ideal for development and testing without external
infrastructure.

.. automodule:: nv_dfm_core.targets.local
    :no-members:
    :no-inherited-members:

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
