.. meta::
  :description: MIGraphX environment variables for developers
  :keywords: MIGraphX, code base, contribution, developing, env vars, environment variables

========================================================
MIGraphX environment variables
========================================================

The MIGraphX environment variables can be used to customize tuning, verification, and tracing.

Most users will only need to use the following variables:

| ``MIGRAPHX_ENABLE_NHWC``
| ``MIGRAPHX_DISABLE_MLIR``
| ``MIGRAPHX_ENABLE_CK``
| ``MIGRAPHX_SET_GEMM_PROVIDER``
| ``MIGRAPHX_DISABLE_LAYERNORM_FUSION``
| ``MIGRAPHX_DISABLE_MIOPEN_POOLING``   
| ``MIGRAPHX_USE_FAST_SOFTMAX``
| ``MIGRAPHX_MLIR_USE_SPECIFIC_OPS``
| ``MIGRAPHX_MLIR_TUNE_EXHAUSTIVE``
| ``MIGRAPHX_ENABLE_MLIR_INPUT_FUSION``
| ``MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION``
| ``MIGRAPHX_MLIR_ENABLE_SPLITK``

Model performance tunable variables
************************************

Model performance tunable variables change the compilation behavior of a model.
 
 .. list-table:: 
  :widths: 20 40 40
  :header-rows: 1

  * - Environment variable
    - Description
    - Values
  * 
    - | ``MIGRAPHX_ENABLE_NHWC``
    - When set to ``1``, forces the model to use the NHWC layout.
    - | ``0``: Does not force the use of the NHWC layout.
      | ``1``: Forces the use of the NHWC layout.
      | Default: ``0``

  * - ``MIGRAPHX_DISABLE_MLIR``
    -  When set to 1, disables the use of the rocMLIR library.
    - | ``0``: Use of the rocMLIR library is enabled.
      | ``1``: Use of the rocMLIR library is disabled. 
      | Default: 0

  * - ``MIGRAPHX_ENABLE_CK``
    - | Enables the use of the Composable Kernels library when set to 1. 
      | Use with ``MIGRAPHX_DISABLE_MLIR``=1.
    - | ``0``: Use of the Composable Kernel libraries is disabled.
      | ``1``: Use of the Composable Kernel libraries is enabled.
      | Default: ``0``

  * - ``MIGRAPHX_SET_GEMM_PROVIDER``
    - Sets the GEMM provider.
    - | ``hipblaslt``: hipBLASLt is set as the GEMM provider.
      | ``rocblas``: rocBLAS is set as the GEMM provider.
      |  Default: ``rocblas`` on gfx90a; ``hipblaslt`` on all other architectures.

  * - ``MIGRAPHX_DISABLE_LAYERNORM_FUSION``
    - Disables layernorm fusion.
    - | ``0``: Layernorm fusion is enabled.
      | ``1``: Layernorm fusion is disabled.
      | Default: ``0``
  
  * - ``MIGRAPHX_DISABLE_MIOPEN_POOLING``   
    - When set to ``1``, MIOpen pooling is disabled and MIGraphX pooling is enabled.
    - | ``0``: MIOpen pooling is enabled.
      | ``1``: MIGraphX pooling is enabled.
      | Default: ``0``

  * - ``MIGRAPHX_USE_FAST_SOFTMAX``
    - Enables fast softmax optimization to speed up softmax computations.
    - | 0: Softmax optimization isn't used.
      | 1: Softmax optimization is used.
      | Default: 0

  * - ``MIGRAPHX_MLIR_USE_SPECIFIC_OPS``
    - Specifies the MLIR operations to use regardless of GPU architecture.  
    - | A comma-separated list of operations must be provided.
      | For example: ``MIGRAPHX_MLIR_USE_SPECIFIC_OP=fused,convolution,dot``.
      | Default: unset

  * - ``MIGRAPHX_MLIR_TUNE_EXHAUSTIVE``
    - Enables exhaustive tuning for MLIR to find the optimal configuration.
    - | 0: No tuning is performed.
      | 1: Exhaustive tuning is performed.
      | Default: 0

  * - ``MIGRAPHX_ENABLE_MLIR_INPUT_FUSION``
    - Enables input fusions in MLIR.
    - | 0: Input fusion is disabled.
      | 1: Input fusion is enabled.
      | Default: 0

  * - ``MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION``
    - Enables reduction fusions in MLIR.
    - | 0: Reduction fusions are disabled.
      | 1: Reduction fusions are enabled.
      | Default: 0

  * - ``MIGRAPHX_MLIR_ENABLE_SPLITK``
    - Enables Split-k performance configurations during MLIR tuning.
    - | 0: Split-k performance configurations are disabled.
      | 1: Split-k performance configurations are enabled.
      | Default: 0


The following environment variables only need to be used when the core MIGraphX APIs are being modified.

Parsing
******************

.. list-table:: 
  :widths: 30 70
  :header-rows: 0

  * - ``MIGRAPHX_TRACE_ONNX_PARSER``
    - | Set to ``1`` to turn on debugging traces for the ONNX parser. Traces include initializers, ONNX node operators, and MIGraphX. 
      | Off by default.                                                                  

  * - ``MIGRAPHX_TRACE_ONNX_PARSER`` 
    - | Set to ``1`` to turn on debugging traces for the ONNX parser. Traces include initializers, ONNX node operators, and MIGraphX. 
      | Off by default.  

  * - ``MIGRAPHX_DISABLE_FP16_INSTANCENORM_CONVERT``
    - | Set to ``1`` to disable conversion from FP16 to FP32 in the ``InstanceNormalization`` ONNX operator. 
      | FP16 to F32 conversion is enabled by default. 


Matching
**********

.. list-table:: 
  :widths: 30 70
  :header-rows: 0

  * - ``MIGRAPHX_TRACE_MATCHES``
    - | Set to ``1`` to print the matcher used and the matched instruction. 
      | Set it to ``2`` to print additional filtered results. 
      | ``MIGRAPHX_TRACE_MATCHES_FOR`` must be set when ``MIGRAPHX_TRACE_MATCHES`` is set to ``2``. 

  * - ``MIGRAPHX_TRACE_MATCHES_FOR``
    - | Prints traces for the specified matcher. 
      | ``MIGRAPHX_TRACE_MATCHES`` must be set to ``2`` for ``MIGRAPHX_TRACE_MATCHES_FOR`` to have an effect.
    
  * - ``MIGRAPHX_VALIDATE_MATCHES``
    - Set to ``1`` to run ``module.validate()`` and validate the module after finding the matches.

Running
*********

.. list-table:: 
  :widths: 30 70
  :header-rows: 0

  * - ``MIGRAPHX_TRACE_EVAL``
    - | Trace evaluation verbosity. 
      | Set to ``1`` to print the run instruction and the time taken.
      | Set to ``2`` to print the run instructions, time taken, a snippet of the output, and statistics.
      | Set to ``3`` to print to print the run instructions, time taken, a snippet of the output, and statistics for all output buffers.
    
Verification
*************

.. list-table:: 
  :widths: 30 70
  :header-rows: 0

  * - ``MIGRAPHX_VERIFY_ENABLE_ALLCLOSE``
    - Set to 1 to verify using allclose with specified atol and rtol for range verification with the driver or tests using migraphx/verify.hpp. 

Pass debugging controls
************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 0

  * - ``MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS``
    - Set to 1 to print debug statements for instructions that have contiguous input instructions removed.

  * - ``MIGRAPHX_DISABLE_POINTWISE_FUSION``
    - Set to 1 to disable the fuse_pointwise compile pass.
  
  * - ``MIGRAPHX_DEBUG_MEMORY_COLORING``
    - Set to 1 to print debug statements for the memory-coloring pass.

  * - ``MIGRAPHX_TRACE_SCHEDULE``
    - Set to 1 to print debug statements for the schedule pass.

  * - ``MIGRAPHX_TRACE_PROPAGATE_CONSTANT``
    - Set to 1 to trace instructions that have been replaced with a constant.

  * - ``MIGRAPHX_TRACE_QUANTIZATION``
    - Set to 1 to print traces for any passes run during quantization.

  * - ``MIGRAPHX_8BITS_QUANTIZATION_PARAMS``
    - Set to 1 to print quantization parameters in the main module only.

  * - ``MIGRAPHX_DISABLE_DNNL_POST_OPS_WORKAROUND``
    - Set to 1 to disable the DNNL post-ops workaround.

  * - ``MIGRAPHX_DISABLE_MIOPEN_FUSION``
    - Set to 1 to disable MIOpen fusions.
  
  * - ``MIGRAPHX_DISABLE_SCHEDULE_PASS``
    - Set to 1 to disable the schedule pass.

  * - ``MIGRAPHX_DISABLE_REDUCE_FUSION``
    - Set to 1 to disable the fuse_reduce pass.

  * - ``MIGRAPHX_ENABLE_REWRITE_DOT``
    - Set to 1 to enable the rewrite_dot pass.

  * - ``MIGRAPHX_SPLIT_REDUCE_SIZE``
    - Minimum size of a reduction to perform a split reduce. The minimum size must be an integer. Set to -1 to disable split reduce.

Model performance tuning
****************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 0  

  * - ``MIGRAPHX_COPY_LITERALS``
    - Uses ``hip_copy_to_gpu`` with a new literal instruction instead of ``hip_copy_literal{}``.

Compilation tracing
************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 0

  * - ``MIGRAPHX_TRACE_FINALIZE`` 
    - Set to 1 to prints graph instructions during the module.finalize() step.

  * - ``MIGRAPHX_TRACE_COMPILE`` 
    - Set to 1 to trace the compilation of a graph.

  * - ``MIGRAPHX_TRACE_PASSES``
    - Set to 1 to print the compile pass and the program after the pass.

  * - ``MIGRAPHX_TIME_PASSES``
    - Set to 1 to time the compile passes.

  * - ``MIGRAPHX_DISABLE_PASSES``
    - Skips the specified passes. A comma-separated list of passes must be provided. For example, ``MIGRAPHX_DISABLE_PASSES=rewrite_pooling,rewrite_gelu``.

GPU kernel JIT debugging
**************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 0

  * - ``MIGRAPHX_TRACE_CMD_EXECUTE``
    - Set to 1 to print commands run by the MIGraphX process.

  * - ``MIGRAPHX_TRACE_HIPRTC``
    - Set to 1 to print the HIPRTC options and C++ file used.
    
  * - ``MIGRAPHX_DEBUG_SAVE_TEMP_DIR``
    - Set to 1 to prevent the deletion of temporary directories.

  * - ``MIGRAPHX_GPU_DEBUG``
    - Set to 1 to add the ``-DMIGRAPHX_DEBUG`` directive when compiling GPU kernels. ``DMIGRAPHX_DEBUG`` enables assertions and source location capture.
  
  * - ``MIGRAPHX_GPU_DEBUG_SYM``
    - Set to 1 to add the ``-g`` option when compiling HIPRTC for debugging purposes.

  * - ``MIGRAPHX_GPU_DUMP_SRC``
    - Set to 1 to dump the compiled HIPRTC source files for inspection.

  * - ``MIGRAPHX_GPU_DUMP_ASM``
    - Set to 1 to dump the hip-clang assembly output for further analysis.

  * - ``MIGRAPHX_GPU_OPTIMIZE``
    - Sets the GPU compiler optimization mode. A valid optimization mode must be passed to the variable. For example, ``MIGRAPHX_GPU_OPTIMIZE=O3``
  
  * - ``MIGRAPHX_GPU_COMPILE_PARALLEL``
    - Set this to the number of threads to use for parallel GPU code compilation. This must be set to a positive integer value.

  * - ``MIGRAPHX_TRACE_NARY``
    - Set to 1 to print the nary device functions used during execution.

  * - ``MIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS``
    - Set to 1 to enable HIPRTC workarounds for known bugs in HIPRTC.

  * - ``MIGRAPHX_ENABLE_NULL_STREAM``
    - Set to 1 to allow the use of a null stream for MIOpen and HIP stream handling.
  
  * - ``MIGRAPHX_NSTREAMS``
    - Set this to the number of HIP streams to use in the GPU. If not set, one stream will be used. The value passed must be a positive integer.

  * - ``MIGRAPHX_TRACE_BENCHMARKING``
    - | Sets the verbosity of benchmarking traces. 
      | Set to 1 for basic trace.
      | Set to 2 for detailed trace. 
      | Set to 3 for compiled traces.

  * - ``MIGRAPHX_PROBLEM_CACHE``
    - Set this to the JSON file from which the problem cache will be saved to and loaded from. Must be set to the path of a valid JSON file. For example, ``MIGRAPHX_PROBLEM_CACHE="path/to/cache_file.json"``

  * - ``MIGRAPHX_BENCHMARKING_BUNDLE``
    - Set this to the number of configurations to run in a bundle during benchmarking. This must be set to a positive integer value.

  * - ``MIGRAPHX_BENCHMARKING_NRUNS``
    - Set this to the number of timing runs for each config bundle being benchmarked. This must be set to a positive integer.


MLIR
**************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 0

  * - ``MIGRAPHX_TRACE_MLIR``
    - | Sets the MLIR trace level.
      | Set to 1 to trace MLIR and print failures. 
      | Set to 2 to print all MLIR operations in addition to tracing MLIR and printing failures.

  * - ``MIGRAPHX_MLIR_TUNING_DB``
    - The path of the tuning database. 

  * - ``MIGRAPHX_MLIR_TUNING_CFG``
    - The path to the tuning configuration file to use with rocMLIR tuning scripts. For example, ``MIGRAPHX_MLIR_TUNING_CFG="path/to/config_file.cfg"``

  * - ``MIGRAPHX_MLIR_TUNE_LIMIT``
    - Set to the maximum number of solutions available for MLIR tuning. Must be set to an integer greater than 1

  * - ``MIGRAPHX_MLIR_DUMP_TO_MXR``
    - Sets the directory where the MXR files the MLIR modules are written to are saved. For example, ``MIGRAPHX_MLIR_DUMP_TO_MXR="/path/to/save_mxr_file/`` 

  * - ``MIGRAPHX_MLIR_DUMP``
    - Sets the directory where the .mlir files the MLIR modules are written to are saved.

Composable Kernel
**************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 0
  
  * - ``MIGRAPHX_LOG_CK_GEMM``
    - Set to 1 to print composable kernels GEMM traces.

  * - ``MIGRAPHX_CK_DEBUG``
    - Set to 1 to add ``-DMIGRAPHX_CK_CHECK=1`` to the composable kernel operator compilation options.

  * - ``MIGRAPHX_TUNE_CK``
    - Set to 1 to perform tuning for composable kernels.

hipBLASLt
**************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 0

  * - ``MIGRAPHX_ENABLE_HIP_GEMM_TUNING``
    - Set to 1 to perform exhaustive tuning for hipBLASLt.


Testing
**************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 0

  * - ``MIGRAPHX_TRACE_TEST_COMPILE``
    - | Set to the target to be traced and prints the compile trace for verify tests on the given target. 
      | Set to ``cpu`` to trace for the CPU target. 
      | Set to ``GPU`` to trace the GPU target. 
      | This flag cannot be used in conjunction with ``MIGRAPHX_TRACE_COMPILE``.

  * - ``MIGRAPHX_TRACE_TEST``
    - Set to 1 to print the reference and target programs even if the verify tests pass.

  * - ``MIGRAPHX_DUMP_TEST``
    - Writes the results of verify tests to MXR files.

  * - ``MIGRAPHX_VERIFY_DUMP_DIFF``
    - Writes the output of the test results, as well as the reference, when they differ.

  