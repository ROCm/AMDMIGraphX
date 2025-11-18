.. meta::
  :description: MIGraphX environment variables for developers
  :keywords: MIGraphX, code base, contribution, developing, env vars, environment variables

========================================================
MIGraphX environment variables
========================================================

The MIGraphX environment variables can be used by contributors to the MIGraphX code base to customize tuning, verification, and tracing.


Model performance tunable variables
************************************

Model performance tunable variables change the compilation behavior of a model. These are the most commonly used variables.
 
.. list-table:: 
  :widths: 40 60
  :header-rows: 1

  * - Environment variable
    - Values
  
  * - | ``MIGRAPHX_ENABLE_NHWC``
      | Forces the model to use the NHWC layout.
      
    - | ``1``: Forces the use of the NHWC layout.
      | ``0``: Returns to default behavior.

      | Default: The use of the NHWC layout isn't forced.

  * - | ``MIGRAPHX_DISABLE_MLIR``
      | When set, the rocMLIR library won't be used.
      
    - | ``1``: The rocMLIR library won't be used.
      | ``0``: Returns to default behavior.

      | Default: The rocMLIR library is used.   

  * - | ``MIGRAPHX_ENABLE_CK``
      | When set, the Composable Kernel library is used. 
      
    - | Use with ``MIGRAPHX_DISABLE_MLIR = 1``.
      
      | ``1``: The Composable Kernel library is used.
      | ``0``: Returns to default behavior.

      | Default: Composable Kernel library isn't used.

  * - | ``MIGRAPHX_SET_GEMM_PROVIDER``
      | Sets the GEMM provider to be either rocBLAS or hipBLASlt.
      
    - | ``hipblaslt``: hipBLASLt is used as the GEMM provider.
      | ``rocblas``: rocBLAS is used as the GEMM provider.

      | Default: ``rocblas`` on gfx90a; ``hipblaslt`` on all other architectures.

  * - | ``MIGRAPHX_ENABLE_LAYERNORM_FUSION``
      | When set, layernorm fusion is used.
      
    - | ``1``: Layernorm fusion will be used.
      | ``0``: Returns to default behavior.

      | Default: Layernorm fusion is not used.
  
  * - | ``MIGRAPHX_DISABLE_MIOPEN_POOLING``   
      | When set, MIGraphX pooling is used instead of MIOpen pooling.
      
    - | ``1``: Use MIGraphX pooling.
      | ``0``: Returns to default behavior.

      | Default: MIOpen pooling is used.

  * - | ``MIGRAPHX_USE_FAST_SOFTMAX``
      | Turns on fast softmax optimization to speed up softmax computations.
      
    - | ``1``: Turns on Softmax optimization.
      | ``0``: Returns to default behavior.

      | Default: Softmax optimization is turned off.

  * - | ``MIGRAPHX_DISABLE_FP32_SOFTMAX``
      | Disables upcasting to fp32 when computing softmax for lower precision graphs.
      
    - | ``1``: Disables forcing full precision computation of softmax
      | ``0``: Returns to default behavior.

      | Default: Upcasting to FP32 is turned on.

  * - | ``MIGRAPHX_MLIR_USE_SPECIFIC_OPS``
      | Specifies the MLIR operations to use regardless of GPU architecture.  
      
    - | Takes a comma-separated list of operations. Operations can be any of the following:
      
      | ``attention``: Use attention fusion. This is used by default on MI300, but must be specified on other architectures.

      | ``convolution``: Use MLIR generated kernels for all convolutions. MIOpen is used by default otherwise.

      | ``convolution_backwards``: Use MLIR generated kernels for backward-convolution. MIOpen is used by default otherwise.
      
      | ``dot``: Use MLIR generated kernels for all GEMMs. hipBLASlt is used otherwise.
      
      | ``fused_convolution``: Use MLIR generated kernels for convolutions when doing so enables extra fusions.
      
      | ``fused_dot``: Use MLIR generated kernels for GEMMs when doing so enables extra fusions.
      
      | ``fused``: Equivalent to setting both ``fused_dot`` and ``fused_convolution``.
      
      | For example: ``MIGRAPHX_MLIR_USE_SPECIFIC_OP=fuse, convolution,dot``.
      
      | A tilde (``~``) can be used to negate an operation.

      | For example, setting ``MIGRAPHX_MLIR_USE_SPECIFIC_OP=~convolution`` specifies that MLIR generated kernels should never be used.
      
  * - | ``MIGRAPHX_MLIR_TUNE_EXHAUSTIVE``
      | When set, exhaustive tuning for MLIR is used to find the optimal configuration.
      
    - | ``1``: Exhaustive tuning is used.
      | ``0``: Returns to default behavior.

      | Default: No MLIR tuning is used.

  * - | ``MIGRAPHX_ENABLE_HIP_GEMM_TUNING``
      | When set, exhaustive tuning for hipBLASLt is used to find the optimal configuration.

    - | ``1``: Exhaustive tuning is used.
      | ``0``: Returns to default behavior.

      | Default: Exhaustive hipBLASLt tuning isn't used.

  * - | ``MIGRAPHX_ENABLE_MLIR_INPUT_FUSION``
      | Turns on input fusions in MLIR.
      
    - | ``1``: Turns on input fusions.  
      | ``0``: Returns to default behavior.

      | Default: Input fusions are turned off.

  * - | ``MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION``
      | Turns on reduction fusions in MLIR.
    
    - | ``1``: Turns on reduction fusions.
      | ``0``: Returns to default behavior.

      | Default: Reduction fusions are turned off.

  * - | ``MIGRAPHX_ENABLE_MLIR_GEG_FUSION``
      | Turns on GEMM+GEMM fusions in MLIR.
    
    - | ``1``: Turns on G+G fusions.
      | ``0``: Returns to default behavior.

      | Default: GEMM+GEMM fusions are turned off.

  * - | ``MIGRAPHX_MLIR_ENABLE_SPLITK``
      | Turns on Split-k performance configurations during MLIR tuning.
      
    - | ``1``: Turns on Split-k performance configurations.
      | ``0``: Returns to default behavior.

      | Default: Split-k performance configurations are turned off.

  * - | ``MIGRAPHX_DISABLE_FP16_INSTANCENORM_CONVERT``
      | When set, FP16 is not converted to FP32 in the ``InstanceNormalization`` ONNX operator. 

    - | ``1``: FP16 is not converted to F32.
      | ``0``: Returns to default behavior.
    
      | Default: FP16 is converted to F32.

  * - | ``MIGRAPHX_ENABLE_REWRITE_DOT``
      | When set, the ``rewrite_dot`` pass is run.
            
    - | ``1``: Runs the ``rewrite_dot`` pass
      | ``0``: Returns to default behavior.

      | Default: The ``rewrite_dot`` pass isn't run.

  * - | ``MIGRAPHX_SPLIT_REDUCE_SIZE``
      | Minimum size of a reduction to perform a split reduce. 
      
    - | The minimum size must be an integer. 
    
      | Set to ``-1`` to disable split reduce.

  * - | ``MIGRAPHX_COPY_LITERALS``
      | When set, literals won't be stored on the GPU but will only be copied over when needed.    
    
    - | ``1``: Literals are copied over to the GPU as needed.
      | ``0``: Returns to default behavior.

      | Default: Literals are stored on the GPU.

  * - | ``MIGRAPHX_VERIFY_ENABLE_ALLCLOSE``
      | When set, the range tolerance is verified using ``allclose``.

    - | ``1``: The range tolerance is verified using ``allclose``. 
      | ``0``: Returns to the default behavior.

      | Default: Range tolerance isn't verified.
                                             
  * - | ``MIGRAPHX_LOG_CK_GEMM``
      | Turns on printing of Composable Kernel GEMM traces.

    - | ``1``: Composable Kernel GEMM traces will be printed.
      | ``0``: Returns to default behavior.

      | Default: Composable Kernel GEMM traces aren't printed.

  * - | ``MIGRAPHX_CK_DEBUG``
      | When set, ``-DMIGRAPHX_CK_CHECK=1`` is added to the Composable Kernel operator compilation options.

    - | ``1``: ``-DMIGRAPHX_CK_CHECK=1`` is added to the compilation options.
      | Default: Compilation is run without ``-DMIGRAPHX_CK_CHECK=1``.

  * - | ``MIGRAPHX_TUNE_CK``
      | Turns on tuning for composable kernels.

    - | ``1``: Composable kernel tuning is done.
      | ``0``: Returns to default behavior.

      | Default: No tuning is done for composable kernels.

  * - | ``MIGRAPHX_REWRITE_LRN``
      | Turns on LRN-to-pooling lowering in the rewrite_pooling pass.
      
    - | ``1``: Turns on LRN-to-pooling lowering.
      | ``0``: Returns to default behavior.

      | Default: LRN-to-pooling lowering is turned off.
               
Matching
**********

Debug settings for matchers. Matchers are responsible for finding optimizations in the graph compilation stage.

.. list-table:: 
  :widths: 40 60
  :header-rows: 1

  * - Environment variable
    - Values

  * - | ``MIGRAPHX_TRACE_MATCHES``
      | When set, prints the name of matchers that have found a valid pattern match. 

    - | ``1``: Prints the name of the matchers that have found a valid match.
      | ``2``: When used with ``MIGRAPHX_TRACE_MATCHES_FOR``, prints the names of matchers that have been tried but which have not necessarily found a match.
      | ``0``: Returns to default behavior.

      | Default: Nothing is printed.

  * - | ``MIGRAPHX_TRACE_MATCHES_FOR``
      | Turns on the printing of traces for the specified matcher if a string is found in the matcher's ``file-name``, ``function-name``, or ``matcher-name``.

    - Takes a string to match.  
    
  * - | ``MIGRAPHX_VALIDATE_MATCHES``
      | When set, ``module.validate()`` is used to validate the module after finding matches.

    - | ``1``: Runs ``module.validate()``.
      | ``0``: Returns to default behavior.

      | Default: ``module.validate()`` isn't run.

  * - | ``MIGRAPHX_TIME_MATCHERS``
      | When set, prints the time spent on a matcher. This helps identify time-consuming patterns.
    
    - | ``1`: Prints the time spent on the matcher.
      | ``0``: Returns to default behavior.

      | Default: The time is not printed.


Pass controls
************************

Debug settings for passes.

.. list-table:: 
  :widths: 30 70
  :header-rows: 1

  * - Environment variable
    - Values

  * - | ``MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS``
      | Turns on the printing of debug statements for ``eliminate contiguous instruction`` passes.
    
    - | ``1``: Debug statements are printed for ``eliminate contiguous instructions`` passes.
      | ``0``: Returns to default behavior.

      | Default: Debug statements aren't printed for ``eliminate contiguous instructions`` passes.
      
  * - | ``MIGRAPHX_DISABLE_POINTWISE_FUSION``
      | When set, the ``fuse_pointwise compile`` pass isn't run.
    
    - | ``1``: The ``fuse_pointwise compile`` pass isn't run.
      | ``0``: Returns to default behavior.

      | Default: The ``fuse_pointwise compile`` pass is run.

  * - | ``MIGRAPHX_DEBUG_MEMORY_COLORING``
      | Turns on the printing of debug statements for the ``memory-coloring`` pass.

    - | ``1``: Debug statements for the ``memory-coloring`` pass are printed.
      | ``0``: Returns to default behavior.

      | Default: Debug statements for the ``memory-coloring`` pass aren't printed.

  * - | ``MIGRAPHX_TRACE_SCHEDULE``
      | Turns on the printing of debug statements for the ``schedule`` pass.

    - | ``1``: Debug statements for the ``schedule`` pass are printed.
      | ``0``: Returns to default behavior.

      | Default: Debug statements for the ``memory-coloring`` pass aren't printed.

  * - | ``MIGRAPHX_TRACE_PROPAGATE_CONSTANT``
      | Turns on tracing of instructions that have been replaced with a constant.
    
    - | ``1``: Instructions that have been replaced with a constant are traced.
      | ``0``: Returns to default behavior.

      | Default: Instructions that have been replaced with a constant aren't traced.
    
  * - | ``MIGRAPHX_DISABLE_DNNL_POST_OPS_WORKAROUND``
      | When set, the DNNL post-ops workaround isn't used.

    - | ``1``: The DNNL post-ops workaround ins't used.
      | ``0``: Returns to default behavior.

      | Default: The DNNL post-ops workaround is used.

  * - | ``MIGRAPHX_DISABLE_MIOPEN_FUSION``
      | When set, MIOpen fusions aren't used.

    - | ``1``: MIOpen fusions aren't used.
      | ``0``: Returns to default behavior.

      | Default: MIOpen fusions are used.

  * - | ``MIGRAPHX_DISABLE_SCHEDULE_PASS``
      | When set, the ``schedule`` pass isn't run.

    - | ``1``: The ``schedule`` pass isn't run.
      | ``0``: Returns to default behavior.

      | Default: The ``schedule`` pass is run.

  * - | ``MIGRAPHX_DISABLE_REDUCE_FUSION``
      | When set, the ``fuse_reduce`` pass isn't run.

    - | ``1``: The ``fuse_reduce`` pass isn't run.
      | ``0``: Returns to default behavior.

      | Default: The ``fuse_reduce`` pass is run.

  * - | ``MIGRAPHX_TRACE_PASSES``
      | Turns on printing of the compile passes and the program after the passes.

    - | ``1``: Prints the compile passes.
      | ``0``: Returns to the default behavior.

      | Default: The compile pass traces aren't printed.

  * - | ``MIGRAPHX_TIME_PASSES``
      | When set, the compile passes are timed.

    - | ``1``: Compile passes are timed.
      | ``0``: Returns to the default behavor.

      | Default: Compile passes aren't timed.

  * - | ``MIGRAPHX_DISABLE_PASSES``
      | Specifies passes that are to be skipped.  
      
    - | Takes a comma-separated list of passes. 
      | For example:
      | ``MIGRAPHX_DISABLE_PASSES=rewrite_pooling,rewrite_gelu``.
  

Compilation tracing
************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 1

  * - Environment variable
    - Values

  * - | ``MIGRAPHX_TRACE_FINALIZE`` 
      | Turns on printing of graph instructions during the ``module.finalize()`` step.

    - | ``1``: Graph instructions will be printed.
      | ``0``: Returns to default behavior.

      | Default: Graph instructions won't be printed.

  * - | ``MIGRAPHX_TRACE_COMPILE`` 
      | Turns on graph compilation tracing.

    - | ``1``: Turns on graph compilation tracing.
      | ``0``: Returns to default behavior.

      | Default: Graph compilation isn't traced.
  
  * - | ``MIGRAPHX_TRACE_ONNX_PARSER``
      | Turns on node-by-node tracing for the ONNX parser. 
      
    - | ``1``: Node-by-node tracing is turned on.
      | ``0``: Returns to the default behavior.

      | Default: There is no node-by-node tracing of the ONNX parser.

  * - | ``MIGRAPHX_TRACE_EVAL``
      | Turns on model evaluation tracing and sets its tracing level. 
      
    - | ``1``: Print the run instructions and the time taken to complete the evaluation.
      | ``2``: Print the run instructions, time taken, a snippet of the output, and some statistics.
      | ``3``: Print the run instructions, time taken, a snippet of the output, and statistics for all output buffers.

  * - | ``MIGRAPHX_TRACE_QUANTIZATION``
      | Turns on the printing of the traces for passes run during quantization.  

    - | ``1``: Traces for passes run during quantization will be printed.
      | ``0``: Returns to default behavior.

      | Default: The traces for passes run during quantization won't be printed out.

  * - | ``MIGRAPHX_8BITS_QUANTIZATION_PARAMS``
      | Turns on the printing of the quantization parameters in the main module only.

    - | ``1``: Only the quantization parameters in the main module are printed.
      | ``0``: Returns to default behavior.

      | Default:

MLIR
**************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 1

  * - Environment variable
    - Values

  * - | ``MIGRAPHX_TRACE_MLIR``
      | Sets the MLIR trace level.
      
    - | ``1``: MLIR trace failures are printed. 
      | ``2``: MLIR trace failures are printed and all MLIR operations are printed as well.

  * - | ``MIGRAPHX_MLIR_TUNING_DB``
      | The path of the tuning database. 

    - Takes the path to the tuning database.

  * - | ``MIGRAPHX_MLIR_TUNING_CFG``
      | Sets the path to the tuning configuration file to use with rocMLIR tuning scripts. 
      
    - | Takes the path to the configuration file.
      | For example: 
      | ``MIGRAPHX_MLIR_TUNING_CFG="path/to/config_file.cfg"``

  * - | ``MIGRAPHX_MLIR_TUNE_LIMIT``
      | Sets the maximum number of solutions available for MLIR tuning. 

    - | Takes an integer greater than 1.

  * - | ``MIGRAPHX_MLIR_DUMP_TO_MXR``
      | Sets the location to where the MXR files that the MLIR modules are written to are saved. 
      
    - | Takes the path to the directory where the files should be saved.
      | For example: 
      | ``MIGRAPHX_MLIR_DUMP_TO_MXR="/path/to/save_mxr_file/`` 

  * - | ``MIGRAPHX_MLIR_DUMP``
      | Sets the the location where the MLIR files that the MLIR modules are written to are saved.

    - | Takes the path to the directory where the files should be saved.
      | For example: 
      | ``MIGRAPHX_MLIR_DUMP="/path/to/save_mlir_file/``


Testing
**************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 1

  * - Environment variable
    - Values

  * - | ``MIGRAPHX_TRACE_TEST_COMPILE``
      | Sets the target to be traced, and turns on printing of the compile trace for verify tests on the given target. 
      | This flag cannot be used if ``MIGRAPHX_TRACE_COMPILE`` is used.
      
    - | ``cpu``: Turns on traces for the CPU target. 
      | ``GPU``: Turns on traces for the GPU target. 
      |  Default: 

  * - | ``MIGRAPHX_TRACE_TEST``
      | When set, the reference and target programs are printed even if the verify tests pass.

    - | ``1``: The reference and target programs are printed when the verify tests pass.
      | ``0``: Returns to default behavior.

      | Default: Reference and target programs aren't printed if the verify tests pass.

  * - | ``MIGRAPHX_DUMP_TEST``
      | When set, the model that is being verified using ``test-verify`` is output to an MXR file. 

    - | ``1``: The model that is being verified is output to an MXR file.
      | ``0``: Returns to default behavior.

      | Default: The model isn't output to file.

  * - | ``MIGRAPHX_VERIFY_DUMP_DIFF``
      | When set, writes out the output of the test results, as well as the reference, when they differ.

    - | ``1``: Test results are written out when they differ.
      | ``0``: Returns to default behavior.

      | Default: The results and the reference aren't written out when they differ.
  
Advanced settings
**************************

.. list-table:: 
  :widths: 30 70
  :header-rows: 1

  * - Environment variable
    - Values

  * - | ``MIGRAPHX_TRACE_CMD_EXECUTE``
      | When set, commands run by the MIGraphX process will be printed.

    - | ``1``: Printing of commands is turned on.
      | ``0``: Returns to default behavior.

      | Default: Commands aren't printed.

  * - | ``MIGRAPHX_TRACE_HIPRTC``
      | When set, the HIPRTC options and C++ file used will be printed.
    
    - | ``1``: HIPRTC options and C++ file will be printed.
      | ``0``: Returns to default behavior.

      | Default: HIPRTC options and C++ file aren't printed.

  * - | ``MIGRAPHX_DEBUG_SAVE_TEMP_DIR``
      | When set, temporary directories won't be deleted.
    
    - | ``1``: Temporary directories aren't deleted.
      | ``0``: Returns to default behavior.

      | Default: Temporary directories are deleted.

  * - | ``MIGRAPHX_GPU_DEBUG``
      | When set, the ``-DMIGRAPHX_DEBUG`` option is used when compiling GPU kernels. ``-DMIGRAPHX_DEBUG`` enables assertions and source location capture.
  
    - | ``1``: The ``-DMIGRAPHX_DEBUG`` option is used when compiling GPU kernels.

      | Default: Compilation is run without ``-DMIGRAPHX_DEBUG``.

  * - | ``MIGRAPHX_GPU_DEBUG_SYM``
      | When set, the ``-g`` option is used when compiling HIPRTC for debugging purposes.

    - | ``1``: The ``-g`` option is used when compiling HIPRTC.

      | Default: Compilation is run without the ``-g`` option.

  * - | ``MIGRAPHX_GPU_DUMP_SRC``
      | The compiled HIPRTC source files is written out for further analysis.

    - | ``1``: HIPRTC source files are written out.
      | ``0``: Returns to default behavior.

      | Default: HIPRTC source files aren't written out.

  * - | ``MIGRAPHX_GPU_DUMP_ASM``
      | When set, the hip-clang assembly output is written out for further analysis.

    - | ``1``: The hip-clang assembly output is written out.
      | ``0``: Returns to default behavior.

      | Default: The hip-clang assembly output isn't written out.

  * - | ``MIGRAPHX_GPU_HIP_FLAGS``
      | When set, the hip-clang compiler appends these extra flags for compilation.

    - | Takes a valid string, a valid hip compile option, e.g. "-Wno-error".

      | Default: The compiler will not append any extra flags for compilation.

  * - | ``MIGRAPHX_GPU_OPTIMIZE``
      | Sets the GPU compiler optimization mode. 
  
    - | Takes a valid optimization mode such as ``O3``.
      | Default: No compiler optimization is used.

  * - | ``MIGRAPHX_GPU_COMPILE_PARALLEL``
      | Sets the number of threads to use for parallel GPU code compilation. 
      
    - | Takes a positive integer value.
      | Default: Number of threads is equal to number of processing units (`nproc`).

  * - | ``MIGRAPHX_TRACE_NARY``
      | When set, the nary device functions used during execution are printed out.

    - | ``1``: The nary device functions are printed out.
      | ``0``: Returns to default behavior.

      | Default: nary device functions aren't printed out.

  * - | ``MIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS``
      | When set, the workarounds for known bugs in HIPRTC are used.

    - | ``1``: HIPRTC workarounds are used.
      | ``0``: Returns to default behavior.

      | Default: HIPRTC workarounds aren't used.

  * - | ``MIGRAPHX_ENABLE_NULL_STREAM``
      | Whem set, a null stream can be used for MIOpen and HIP stream handling.
  
    - | ``1``: A null stream can be used for stream handling. 
      | ``0``: Returns to default behavior.

      | Default: A null stream can't be used for stream handling.

  * - | ``MIGRAPHX_NSTREAMS``
      | Sets the number of HIP streams to use in the GPU. 
      
    - | Takes a positive integer.
      | Default: one stream will be used.

  * - | ``MIGRAPHX_TRACE_BENCHMARKING``
      | Sets the verbosity of benchmarking traces. 
      
    - | ``1``: Basic trace 
      | ``2``: Detailed trace 
      | ``3``: Compiled traces

  * - | ``MIGRAPHX_PROBLEM_CACHE``
      | Sets the JSON file that the problem cache will be saved to and loaded from. 
      
    - | Takes a fully qualified path to a valid JSON file. 
      | For example: 
      | ``MIGRAPHX_PROBLEM_CACHE="path/to/cache_file.json"``

  * - | ``MIGRAPHX_BENCHMARKING_BUNDLE``
      | Sets the number of configurations to run in a bundle during benchmarking. 
      
    - Takes a positive integer.

  * - | ``MIGRAPHX_BENCHMARKING_NRUNS``
      | Sets the number of timing runs for each configuration bundle being benchmarked. 
      
    - Takes a positive integer.

