.. meta::
  :description: Tune MIGraphX model performance with environment variables
  :keywords: MIGraphX, performance, tuning, environment variables, ROCm

********************************************************************
Tune model performance
********************************************************************

MIGraphX exposes environment variables that change compilation behavior and
kernel selection at runtime. Set these variables before you compile or run a
model to experiment with performance options without changing application
code.

For the complete list of environment variables, including developer-only
tracing and debugging options, see
:doc:`MIGraphX environment variables <../reference/MIGraphX-dev-env-vars>`.

Set environment variables
====================================================================

Export a variable in your shell before running ``migraphx-driver`` or your
application:

.. code-block:: shell

   export MIGRAPHX_SET_GEMM_PROVIDER=hipblaslt
   /opt/rocm/bin/migraphx-driver perf model.onnx --onnx --gpu

Replace ``model.onnx`` with your model path.

Variables that take comma-separated lists must not include spaces between
entries. Quote the value when your shell requires it.

Common tuning variables
====================================================================

The following variables are the most commonly used for model performance
tuning. Each entry describes accepted values and default behavior documented
in the MIGraphX source tree.

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Environment variable
     - Values
   * - ``MIGRAPHX_GPU_OPTIONS``
     - JSON object of backend options (for example, ``{convolution_layout:channels_last}``). Quotes around keys and values are optional. Unrecognized options are ignored.
   * - ``MIGRAPHX_SET_GEMM_PROVIDER``
     - ``hipblaslt`` or ``rocblas``. Default: ``rocblas`` on gfx90a; ``hipblaslt`` on all other architectures.
   * - ``MIGRAPHX_ENABLE_GEMM_TUNING``
     - ``1`` enables exhaustive GEMM tuning even when ``--exhaustive-tune`` is not set. ``0`` returns to default behavior.
   * - ``MIGRAPHX_ENABLE_HIP_GEMM_TUNING``
     - ``1`` enables exhaustive hipBLASLt tuning. ``0`` returns to default behavior.
   * - ``MIGRAPHX_MLIR_TUNE_EXHAUSTIVE``
     - ``1`` enables exhaustive MLIR tuning. ``0`` returns to default behavior.
   * - ``MIGRAPHX_MLIR_USE_SPECIFIC_OPS``
     - Comma-separated list of operations (for example, ``attention``, ``convolution``, ``dot``, ``fused_dot``, ``fused_convolution``, ``fused``). Prefix an entry with ``~`` to negate it.
   * - ``MIGRAPHX_DISABLE_MLIR``
     - ``1`` disables rocMLIR. ``0`` returns to default behavior.
   * - ``MIGRAPHX_ENABLE_CK``
     - ``1`` enables Composable Kernel. Use with ``MIGRAPHX_DISABLE_MLIR=1``.
   * - ``MIGRAPHX_USE_FAST_SOFTMAX``
     - ``1`` enables fast softmax optimization. ``0`` returns to default behavior.
   * - ``MIGRAPHX_ENABLE_LAYERNORM_FUSION``
     - ``1`` enables layernorm fusion. ``0`` returns to default behavior.
   * - ``MIGRAPHX_FLASH_DECODING_ENABLED``
     - ``1`` enables flash decoding for attention fusion. Default: ``0``.
   * - ``MIGRAPHX_SKIP_BENCHMARKING``
     - ``1`` skips kernel benchmarking and compiles with the first available solution. ``0`` returns to default behavior.

Tuning examples
====================================================================

Select a GEMM provider
--------------------------------------------------------------------

Set the general matrix multiply (GEMM) provider before compilation:

.. code-block:: shell

   export MIGRAPHX_SET_GEMM_PROVIDER=hipblaslt
   /opt/rocm/bin/migraphx-driver perf model.onnx --onnx --gpu

Replace ``model.onnx`` with your model path.

Enable exhaustive GEMM tuning
--------------------------------------------------------------------

Search for the fastest GEMM kernel configuration:

.. code-block:: shell

   export MIGRAPHX_ENABLE_GEMM_TUNING=1
   /opt/rocm/bin/migraphx-driver perf model.onnx --onnx --gpu --exhaustive-tune

You can also pass ``--exhaustive-tune`` on the command line. The environment
variable enables exhaustive GEMM tuning even when that flag is omitted.

Specify MLIR operations
--------------------------------------------------------------------

Force specific operations to lower through MLIR. The list is comma-separated:

.. code-block:: shell

   export MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention,dot
   /opt/rocm/bin/migraphx-driver compile model.onnx --onnx --gpu --text

Replace ``model.onnx`` with your model path.

Measure the effect
====================================================================

Compare performance before and after changing a variable:

.. code-block:: shell

   /opt/rocm/bin/migraphx-driver perf model.onnx --onnx --gpu -n 50

Replace ``model.onnx`` with your model path and adjust ``-n`` for the number
of timing iterations.

See also
====================================================================

* :doc:`MIGraphX driver <../migraphx-driver>` for ``perf`` and ``compile`` commands.
* :doc:`MIGraphX environment variables <../reference/MIGraphX-dev-env-vars>` for
  the full variable reference, including pass controls and compilation tracing.
