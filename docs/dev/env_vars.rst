Environment Variables
=====================

For parsing
---------------

.. envvar:: MIGRAPHX_TRACE_ONNX_PARSER

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print debugging traces for the onnx parser.
Prints: initializers (if used), ONNX node operators, added MIGraphX instructions

.. envvar:: MIGRAPHX_DISABLE_FP16_INSTANCENORM_CONVERT

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables the conversion from fp16 to fp32 for the InstanceNormalization ONNX operator that MIGX does as a workaround for accuracy issues with reduce_mean/variance.
See ``parse_instancenorm.cpp`` for more details.


Matchers
------------

.. envvar:: MIGRAPHX_TRACE_MATCHES

Set to "1" to print the matcher that matches an instruction and the matched instruction.
Set to "2" and use the ``MIGRAPHX_TRACE_MATHCES_FOR`` flag to filter out results.

.. envvar:: MIGRAPHX_TRACE_MATCHES_FOR

Set to the name of any matcher and only traces for that matcher will be printed out.

.. envvar:: MIGRAPHX_VALIDATE_MATCHES

Set to "1", "enable", "enabled", "yes", or "true" to use.
Validate the module after finding the matches (runs ``module.validate()``).

Program Execution 
---------------------

.. envvar:: MIGRAPHX_TRACE_EVAL

Set to "1", "2", or "3" to use.
"1" prints the instruction run and the time taken.
"2" prints everything in "1" and a snippet of the output argument and some statistics (ex. min, max, mean) of the output.
"3" prints everything in "1" and the full output buffers.


Program Verification
------------------------

.. envvar:: MIGRAPHX_VERIFY_ENABLE_ALLCLOSE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Uses ``allclose`` with the given ``atol`` and ``rtol`` for verifying ranges with ``driver verify`` or the tests that use ``migraphx/verify.hpp``.


Pass debugging or Pass controls
-----------------------------------

.. envvar:: MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS

Set to "1", "enable", "enabled", "yes", or "true" to use.
Debug print the instructions that have input ``contiguous`` instructions removed.

.. envvar:: MIGRAPHX_DISABLE_POINTWISE_FUSION

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables the ``fuse_pointwise`` compile pass.

.. envvar:: MIGRAPHX_DEBUG_MEMORY_COLORING

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print debug statements for the ``memory_coloring`` pass.

.. envvar:: MIGRAPHX_TRACE_SCHEDULE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print debug statements for the ``schedule`` pass.

.. envvar:: MIGRAPHX_TRACE_PROPAGATE_CONSTANT

Set to "1", "enable", "enabled", "yes", or "true" to use.
Traces instructions replaced with a constant.

.. envvar:: MIGRAPHX_INT8_QUANTIZATION_PARAMS

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print the quantization parameters in only the main module.

.. envvar:: MIGRAPHX_DISABLE_DNNL_POST_OPS_WORKAROUND

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disable the DNNL post ops workaround.

.. envvar:: MIGRAPHX_DISABLE_MIOPEN_FUSION

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disable MIOpen fusions.

.. envvar:: MIGRAPHX_DISABLE_SCHEDULE_PASS

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disable the ``schedule`` pass.

.. envvar:: MIGRAPHX_DISABLE_REDUCE_FUSION

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disable the ``fuse_reduce`` pass.

.. envvar:: MIGRAPHX_ENABLE_NHWC

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enable the ``layout_nhwc`` pass.

.. envvar:: MIGRAPHX_ENABLE_CK

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enable using the Composable Kernels library.
Should be used in conjunction with ``MIGRAPHX_DISABLE_MLIR=1``.

.. envvar:: MIGRAPHX_DISABLE_MLIR*
Set to "1", "enable", "enabled", "yes", or "true" to use.
Disable using the rocMLIR library.

.. envvar:: MIGRAPHX_ENABLE_EXTRA_MLIR
Set to "1", "enable", "enabled", "yes", or "true" to use.
Enables additional opportunities to use MLIR that may improve performance.

.. envvar:: MIGRAPHX_COPY_LITERALS

Set to "1", "enable", "enabled", "yes", or "true" to use.
Use ``hip_copy_to_gpu`` with a new ``literal`` instruction rather than use ``hip_copy_literal{}``.

Compilation traces
----------------------

.. envvar:: MIGRAPHX_TRACE_FINALIZE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Debug print instructions during the ``module.finalize()`` step.

.. envvar:: MIGRAPHX_TRACE_COMPILE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print trace information for the graph compilation process.

.. envvar:: MIGRAPHX_TRACE_PASSES

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print the compile pass and the program after the pass.

.. envvar:: MIGRAPHX_TIME_PASSES

Set to "1", "enable", "enabled", "yes", or "true" to use.
Time the compile passes.


GPU Kernels JIT compilation debugging (applicable for both hiprtc and hipclang)
-----------------------------------------

.. envvar:: MIGRAPHX_TRACE_CMD_EXECUTE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print commands executed by the MIGraphX ``process``.

.. envvar:: MIGRAPHX_TRACE_HIPRTC

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print HIPRTC options and C++ file executed.

.. envvar:: MIGRAPHX_DEBUG_SAVE_TEMP_DIR

Set to "1", "enable", "enabled", "yes", or "true" to use.
Make it so the created temporary directories are not deleted.

.. envvar:: MIGRAPHX_GPU_DEBUG

Set to "1", "enable", "enabled", "yes", or "true" to use.
Internally, this adds the option ``-DMIGRAPHX_DEBUG`` when compiling GPU kernels. It enables assertions and capture of source locations for the errors. 

.. envvar:: MIGRAPHX_GPU_DEBUG_SYM

Set to "1", "enable", "enabled", "yes", or "true" to use.
Adds the option ``-g`` when compiling HIPRTC.

.. envvar:: MIGRAPHX_GPU_DUMP_SRC

Set to "1", "enable", "enabled", "yes", or "true" to use.
Dump the HIPRTC source files compiled.

.. envvar:: MIGRAPHX_GPU_DUMP_ASM

Set to "1", "enable", "enabled", "yes", or "true" to use.
Dump the hip-clang assembly.

.. envvar:: MIGRAPHX_GPU_OPTIMIZE

Set the optimization mode for GPU compile (``-O`` option).
Defaults to ``-O3``.

.. envvar:: MIGRAPHX_GPU_COMPILE_PARALLEL

Set to the number of threads to use.
Compile GPU code in parallel with the given number of threads.

.. envvar:: MIGRAPHX_TRACE_NARY

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print the ``nary`` device functions used.

.. envvar:: MIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enable HIPRTC workarounds for bugs in HIPRTC.

.. envvar:: MIGRAPHX_USE_FAST_SOFTMAX

Set to "1", "enable", "enabled", "yes", or "true" to use.
Use the fast softmax optimization.

.. envvar:: MIGRAPHX_ENABLE_NULL_STREAM

Set to "1", "enable", "enabled", "yes", or "true" to use.
Allow using null stream for miopen and hipStream.

.. envvar:: MIGRAPHX_NSTREAMS

Set to the number of streams to use.
Defaults to 1.

.. envvar:: MIGRAPHX_TRACE_BENCHMARKING

Set to "1" to print benchmarching trace.
Set to "2" to print benchmarching trace with more detail.

MLIR vars
-------------

.. envvar:: MIGRAPHX_TRACE_MLIR

Set to "1" to trace MLIR and print any failures.
Set to "2" to additionally print all MLIR operations.

.. envvar:: MIGRAPHX_MLIR_USE_SPECIFIC_OPS

Set to the name of the operations you want to always use MLIR regardless of GPU architecture.
Accepts a list of operators separated by commas (ex: "fused", "convolution", "dot").

.. envvar:: MIGRAPHX_MLIR_TUNING_DB

Set to the path of the MLIR tuning database to load.

.. envvar:: MIGRAPHX_MLIR_TUNING_CFG

Set to the path of the tuning configuration.
Appends to tuning cfg file that could be used with rocMLIR tuning scripts.

.. envvar:: MIGRAPHX_MLIR_TUNE_EXHAUSTIVE

Set to "1", "enable", "enabled", "yes", or "true" to use.
Do exhaustive tuning for MLIR.


CK vars
-----------

.. envvar:: MIGRAPHX_LOG_CK_GEMM

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print Composable Kernels GEMM traces.

.. envvar:: MIGRAPHX_CK_DEBUG

Set to "1", "enable", "enabled", "yes", or "true" to use.
Always add the ``-DMIGRAPHX_CK_CHECK=1`` for compiling Composable Kernels operators.

.. envvar:: MIGRAPHX_TUNE_CK

Set to "1", "enable", "enabled", "yes", or "true" to use.
Use tuning for Composable Kernels.

Testing 
------------

.. envvar:: MIGRAPHX_TRACE_TEST_COMPILE

Set to the target that you want to trace the compilation of (ex. "gpu", "cpu").
Prints the compile trace for the given target for the verify tests.
This flag shouldn't be used in conjunction with ``MIGRAPHX_TRACE_COMPILE``.
For the verify tests only use ``MIGRAPHX_TRACE_TEST_COMPILE``.

.. envvar:: MIGRAPHX_TRACE_TEST

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints the reference and target programs even if the verify passed successfully.

.. envvar:: MIGRAPHX_DUMP_TEST

Set to "1", "enable", "enabled", "yes", or "true" to use.
Dumps verify tests to ``.mxr`` files.
