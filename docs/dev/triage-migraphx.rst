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

**Purpose**: Isolate MIOpen integration issues by comparing MIOpen and MIGraphX native implementations

**Pooling**: ``MIGRAPHX_ENABLE_MIOPEN_POOLING=1``

- Forces MIOpen pooling instead of MIGraphX (the default)
- Use for MaxPool, AvgPool, GlobalAvgPool issues to compare against MIOpen behavior

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

Binary Cache
============

Compiled kernels are shared within a single compile, so a kernel that appears many times in a
model is compiled only once. Setting a directory makes that reuse survive across runs:

.. code-block:: bash

   export MIGRAPHX_BINARY_CACHE=$HOME/.cache/migraphx

Entries are grouped by a directory naming the entry format, the HIP compiler, a digest of the
embedded kernel headers, and the rocMLIR build, so entries a build cannot use are never
consulted. The compiler is identified by compiling a small probe that records
``__clang_version__`` into the object and reading it back, because the device compiler is loaded
at runtime and need not be the one MIGraphX was built with. Reclaim space by deleting the
directories for builds you no longer use:

.. code-block:: bash

   ls $HOME/.cache/migraphx        # directories are named after the build that wrote them
   rm -r $HOME/.cache/migraphx/v1-hip22.0.*

The same settings are available as backend options, which take precedence over the environment
and are how tests configure the cache:

.. code-block:: python

   model.compile(migraphx.get_target("gpu"),
                 advance_backend_options={"binary_cache": "/tmp/cache",
                                          "binary_cache_verify": True})

Entries are keyed on everything handed to the backend compiler. If a kernel is suspected of
being reused when it should not be, set ``binary_cache_verify``. That compiles even when a
result could be reused and fails loudly if the two disagree, which is the only way an
incomplete key shows up other than as wrong results. It is slower than compiling without a
cache at all, so use it to diagnose rather than routinely.
