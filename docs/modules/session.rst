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
