Environment Variables
=====================

For parsing
---------------

**MIGRAPHX_TRACE_ONNX_PARSER**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print debugging traces for the onnx parser.
Prints: initializers (if used), ONNX node operators, added MIGraphX instructions

**MIGRAPHX_DISABLE_FP16_INSTANCENORM_CONVERT**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables the conversion from fp16 to fp32 for the InstanceNormalization ONNX operator that MIGX does as a workaround for accuracy issues with reduce_mean/variance.
See ``parse_instancenorm.cpp`` for more details.


Matchers
------------

**MIGRAPHX_TRACE_MATCHES**

Set to "1" to print the matcher that matches an instruction and the matched instruction.
Set to "2" and use the ``MIGRAPHX_TRACE_MATHCES_FOR`` flag to filter out results.

**MIGRAPHX_TRACE_MATCHES_FOR**

Set to a string of what you want to filter matched results of.
TODO: need to figure out what this can do

**MIGRAPHX_VALIDATE_MATCHES**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Validate the module after finding the matches (runs ``module.validate()``).

Program Execution 
---------------------

**MIGRAPHX_TRACE_EVAL**

Set to "1", "2", or "3" to use.
"1" prints the instruction run and the time taken.
"2" prints everything in "1" and a snippet of the output argument and some statistics (ex. min, max, mean) of the output.
"3" prints everything in "1" and the full output buffers.


Program Verification
------------------------

**MIGRAPHX_VERIFY_ENABLE_ALLCLOSE**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Uses ``allclose`` with the given ``atol`` and ``rtol`` for verifying ranges with ``driver verify`` or the tests that use ``migraphx/verify.hpp``.


Pass debugging or Pass controls
-----------------------------------

**MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Debug print the instructions that have input ``contiguous`` instructions removed.

**MIGRAPHX_DISABLE_POINTWISE_FUSION**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disables the ``fuse_pointwise`` compile pass.

**MIGRAPHX_DEBUG_MEMORY_COLORING**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print debug statements for the ``memory_coloring`` pass.

**MIGRAPHX_TRACE_SCHEDULE**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print debug statements for the ``schedule`` pass.

**MIGRAPHX_TRACE_PROPAGATE_CONSTANT**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Traces instructions replaced with a constant.

**MIGRAPHX_INT8_QUANTIZATION_PARAMS**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print the quantization parameters in only the main module.

**MIGRAPHX_DISABLE_DNNL_POST_OPS_WORKAROUND**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disable the DNNL post ops workaround.

**MIGRAPHX_DISABLE_MIOPEN_FUSION**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disable MIOpen fusions.

**MIGRAPHX_DISABLE_SCHEDULE_PASS**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disable the ``schedule`` pass.

**MIGRAPHX_DISABLE_REDUCE_FUSION**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Disable the ``fuse_reduce`` pass.

**MIGRAPHX_ENABLE_NHWC**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enable the ``layout_nhwc`` pass.

**MIGRAPHX_ENABLE_CK**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enable using the Composable Kernels library.

**MIGRAPHX_ENABLE_MLIR**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Enable using the rocMLIR library.

**MIGRAPHX_COPY_LITERALS**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Use ``hip_copy_to_gpu`` with a new ``literal`` instruction rather than use ``hip_copy_literal{}``.

Compilation traces
----------------------

**MIGRAPHX_TRACE_FINALIZE**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Debug print instructions during the ``module.finalize()`` step.

**MIGRAPHX_TRACE_COMPILE**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print trace information for the graph compilation process.

**MIGRAPHX_TRACE_PASSES**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Print the compile pass and the program after the pass.

**MIGRAPHX_TIME_PASSES**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Time the compile passes.


GPU Kernels JIT compilation debugging
-----------------------------------------

**MIGRAPHX_TRACE_CMD_EXECUTE**

**MIGRAPHX_TRACE_HIPRTC**

**MIGRAPHX_DEBUG_SAVE_TEMP_DIR**

**MIGRAPHX_GPU_DEBUG**

**MIGRAPHX_GPU_DEBUG_SYM**

**MIGRAPHX_GPU_DUMP_SRC**

**MIGRAPHX_GPU_DUMP_ASM**

**MIGRAPHX_GPU_OPTIMIZE**

**MIGRAPHX_GPU_COMPILE_PARALLEL**

**MIGRAPHX_TRACE_NARY**

**MIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS**

**MIGRAPHX_USE_FAST_SOFTMAX**

**MIGRAPHX_ENABLE_NULL_STREAM**

**MIGRAPHX_NSTREAMS**


MLIR vars
-------------

**MIGRAPHX_TRACE_MLIR**

**MIGRAPHX_MLIR_USE_SPECIFIC_OPS**

**MIGRAPHX_MLIR_TUNING_DB**

**MIGRAPHX_MLIR_TUNING_CFG**

**MIGRAPHX_TUNE_EXHAUSTIVE**


CK vars
-----------

**MIGRAPHX_LOG_CK_GEMM**

**MIGRAPHX_CK_DEBUG**

**MIGRAPHX_TUNE_CK**


Testing 
------------

**MIGRAPHX_TRACE_TEST_COMPILE**

Set to the target that you want to trace the compilation of (ex. "gpu", "cpu").
Prints the compile trace for the given target for the verify tests.

**MIGRAPHX_TRACE_TEST**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Prints the reference and target programs even if the verify passed successfully.

**MIGRAPHX_DUMP_TEST**

Set to "1", "enable", "enabled", "yes", or "true" to use.
Dumps verify tests to ``.mxr`` files.
