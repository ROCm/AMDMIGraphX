.. meta::
   :description: Issue Triaging Guide for MIGraphX
   :keywords: MIGraphX, issues, debugging, triaging, environment variables

========================================
Issue Triaging Guide for MIGraphX
========================================

This guide helps diagnose MIGraphX issues by systematically disabling components to isolate the root cause.

Overview
========

Systematic approach to identify problems from:

- MLIR backend compilation
- Graph fusion optimizations
- MIOpen integration
- GEMM provider implementations
- Specific operations or passes

Step-by-Step Diagnostic Process
===============================

Step 1: Disable MLIR Backend
-----------------------------

**Variable**: ``MIGRAPHX_DISABLE_MLIR=1``

**Purpose**: Test if issue is MLIR-related by using native GPU backend

**Command**:

.. code-block:: bash

   export MIGRAPHX_DISABLE_MLIR=1

**Result**:

- Issue persists → Continue to Step 2
- Issue resolves → MLIR problem, use rocMLIR triage guide

Step 2: Reduce Graph Complexity
--------------------------------

**Tool**: ``migraphx-driver --reduce`` or ``-r``

**Purpose**: Find minimal failing case early in process

**Commands**:

.. code-block:: bash

   migraphx-driver compile model.onnx --reduce
   migraphx-driver run program.mxr --reduce

Step 3: Disable Fusion Passes
------------------------------

Test each fusion type individually:

**Pointwise Fusion**: ``MIGRAPHX_DISABLE_POINTWISE_FUSION=1``

- Disables element-wise operations fusion (add, mul, relu)

**LayerNorm Fusion**: ``MIGRAPHX_DISABLE_LAYERNORM_FUSION=1``

- Disables layer normalization fusion

**Reduce Fusion**: ``MIGRAPHX_DISABLE_REDUCE_FUSION=1``

- Disables reduction operations fusion (sum, mean)

**MIOpen Fusion**: ``MIGRAPHX_DISABLE_MIOPEN_FUSION=1``

- Disables MIOpen-based kernel fusion

Step 4: Test MIOpen Components
-------------------------------

**Pooling**: ``MIGRAPHX_DISABLE_MIOPEN_POOLING=1``

- Forces MIGraphX pooling instead of MIOpen
- Use for MaxPool, AvgPool, GlobalAvgPool issues

Step 5: Test GEMM Providers
----------------------------

**Variables**:

- ``MIGRAPHX_SET_GEMM_PROVIDER=rocblas``
- ``MIGRAPHX_SET_GEMM_PROVIDER=hipblaslt``
- ``MIGRAPHX_ENABLE_CK=1`` (with ``MIGRAPHX_DISABLE_MLIR=1``)

**Purpose**: Isolate GEMM library issues

**Commands**:

.. code-block:: bash

   export MIGRAPHX_SET_GEMM_PROVIDER=rocblas
   export MIGRAPHX_SET_GEMM_PROVIDER=hipblaslt

Step 6: Granular MLIR Control
------------------------------

**Variable**: ``MIGRAPHX_MLIR_USE_SPECIFIC_OPS``

**Purpose**: Enable/disable MLIR for specific operations

**Examples**:

.. code-block:: bash

   export MIGRAPHX_MLIR_USE_SPECIFIC_OPS=dot,convolution      # Enable for specific ops
   export MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention          # Disable for attention
   export MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention,~softmax # Disable multiple

Debugging and Tracing
=====================

**Compilation Tracing**:

- ``MIGRAPHX_TRACE_MLIR=1`` or ``2``        # MLIR compilation steps
- ``MIGRAPHX_TRACE_PASSES=1``           # Print compilation passes
- ``MIGRAPHX_GPU_COMPILE_PARALLEL=1``   # Disable parallel compilation

**Performance Tracing**:

- ``MIGRAPHX_TRACE_BENCHMARKING=3``     # Kernel benchmarking process

Systematic Workflow
===================

1. **Start with MLIR**: Test ``MIGRAPHX_DISABLE_MLIR=1``
2. **Reduce complexity**: Use ``--reduce`` flag to find minimal case
3. **Disable fusions**: Test each type individually
4. **Check MIOpen**: Disable components if applicable
5. **Test GEMM providers**: Try rocblas vs hipblaslt
6. **Fine-tune MLIR**: Use operation-specific control
7. **Enable tracing**: Add diagnostic variables

This systematic approach helps maintainers quickly understand and fix root causes.