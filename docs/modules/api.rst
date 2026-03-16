:mod:`nv_dfm_core.api`: Pipeline API
=====================================

.. automodule:: nv_dfm_core.api
    :no-members:
    :no-inherited-members:

.. currentmodule:: nv_dfm_core.api

Core
----

The building blocks for constructing DFM pipelines.

.. autosummary::
    :nosignatures:
    :toctree: generated/api/
    :template: class.rst

    Pipeline
    PreparedPipeline
    Operation
    Yield
    PlaceParam
    NodeRef

Control Flow
------------

Constructs for branching, looping, and caching within a pipeline.

.. autosummary::
    :nosignatures:
    :toctree: generated/api/
    :template: class.rst

    ForEach
    If
    Block
    TryFromCache
    WriteToCache
    BestOf
    Advise

Boolean Expressions
-------------------

Used with :class:`If` to define conditional logic.

.. autosummary::
    :nosignatures:
    :toctree: generated/api/
    :template: class.rst

    BooleanExpression
    ComparisonExpression
    And
    Or
    Not
    Equal
    NotEqual
    GreaterThan
    GreaterThanOrEqual
    LessThan
    LessThanOrEqual
    Atom

Tokens and Signals
------------------

Internal data units that flow through the pipeline.

.. autosummary::
    :nosignatures:
    :toctree: generated/api/
    :template: class.rst

    StopToken
    ErrorToken

Advanced
--------

Lower-level types and utilities for framework authors and advanced users.

.. autosummary::
    :nosignatures:
    :toctree: generated/api/
    :template: class.rst

    PipelineBuildHelper
    NodeId
    NodeParam
    Located
    Expression
    Statement
    PickledObject
    ApiVisitor
    BooleanExpressionVisitor

.. autosummary::
    :nosignatures:
    :toctree: generated/api/
    :template: function.rst

    make_auto_id
    well_known_id
