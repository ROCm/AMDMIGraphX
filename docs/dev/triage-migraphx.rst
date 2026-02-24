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
- Issue resolves → MLIR problem, use rocMLIR triage guide (https://github.com/ROCm/AMDMIGraphX/blob/develop/docs/dev/triage-rocmlir.rst)

Step 2: Bisect to Find Problematic Operation
---------------------------------------------

**Tool**: ``migraphx-driver --bisect``

**Purpose**: Quickly identify the specific operation causing the failure using binary search

**Commands**:

.. code-block:: bash

   # Bisect an ONNX model
   migraphx-driver compile model.onnx --bisect

**What this does**: Uses binary search to systematically disable operations until it finds the exact operation that causes the failure. Much faster than ``--reduce`` for pinpointing issues.

Step 3: Disable Fusion Passes
------------------------------

**Purpose**: Isolate optimization-related issues by testing each fusion type individually

Test each fusion type individually:

**Pointwise Fusion**: ``MIGRAPHX_DISABLE_POINTWISE_FUSION=1``

- Disables element-wise operations fusion (add, mul, relu)

**LayerNorm Fusion**: ``MIGRAPHX_DISABLE_LAYERNORM_FUSION=1``

- Disables layer normalization fusion

**Reduce Fusion**: ``MIGRAPHX_DISABLE_REDUCE_FUSION=1``

- Disables reduction operations fusion (sum, mean)

**MIOpen Fusion**: ``MIGRAPHX_DISABLE_MIOPEN_FUSION=1``

- Disables MIOpen-based kernel fusion

Step 4: Reduce Graph Complexity
--------------------------------

**Tool**: ``migraphx-driver --reduce`` or ``-r``

**Purpose**: Find minimal failing case by creating smaller versions of the program

**Commands**:

.. code-block:: bash

   migraphx-driver compile model.onnx --reduce
   migraphx-driver run program.mxr --reduce

**When to use**: Use after bisect if you need a smaller program for detailed analysis or bug reporting.

Step 5: Test MIOpen Components
-------------------------------

**Purpose**: Isolate MIOpen integration issues by forcing MIGraphX native implementations

**Pooling**: ``MIGRAPHX_DISABLE_MIOPEN_POOLING=1``

- Forces MIGraphX pooling instead of MIOpen
- Use for MaxPool, AvgPool, GlobalAvgPool issues

Step 6: Test GEMM Providers
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

Step 7: Granular MLIR Control
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

This systematic approach helps maintainers quickly understand and fix root causes.