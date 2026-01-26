# Changelog for MIGraphX

Full documentation for MIGraphX is available at
[https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/).

## Develop Branch

### Added

* Added a dedicated logger for MIGraphX.

### Changed

### Resolved issues

### Optimized

### Removed

## MIGraphX 2.15 for ROCm 7.2.0

### Added

* Added MXFP4 support for Quark and Brevitas quantized models.
* Added dynamic shape support for DepthToSpace Op.
* Added `bias` and `key_mask_padding` inputs for the `MultiHeadAttention` operator.
* Added GEMM+GEMM fusions.
* Added the `dim_params` input parameter to the `parse_onnx` python call.
* Created an API to query supported ONNX Operators `get_onnx_operators()`.
* Added right pad masking mode for Multihead Attention.
* Added support for Flash Decoding.
* Added Torch-MIGraphX installation instructions.
* Added Operator Builders with supporting documentation.
* Added index range check to the Gather operator.


### Changed

* Updated the Resize operator to support linear mode for Dynamic shapes.
* Switch to `--input-dim` instead of `--batch`  to set any dynamic dimensions when using `migraphx-driver`.
* Different stride sizes are now supported in ONNX `if` branches.
* ONNX version change to 1.18.0 to support PyTorch 2.9.
* Refactor GroupQueryAttention.
* Enable PipelineRepoRef parameter in CI.
* Hide LLVM symbols that come from ROCmlir and provide option for stripping in release mode.
* Model compilation failures now produce an mxr file for debugging the failure.
* Bump SQlite3 to 3.50.4.


### Resolved issues

* Quiet nrvo and noreturn warnings (#4429).
* Fixed `pointwise: Wrong number of arguments` error when quantizing certain models to `int8` (#4398).
* TopK exception bugfix (#4329).
* Updated SD3 example for change in optimum-onnx[onnxruntime] (#4344).
* Fixed an issue with Torch-MIGraphX where the model compilation would fail (#4388)
* Fixed an issue where a reduction was broadcast with different dimensions than the input (#4408).
* Resolved a path name issue stopping some files being created on Windows for debugging (#4420).
* Fix "reduce_sum: axes: value out of range" error in simplify_reshapes (#4443).
* Updated README `rbuild` installation instructions to use python venv to avoid warning (#4405).
* Ensured directories exist when generating files for debugging (#4383).
* Resolved a compilation hang issue (#4428).


### Optimized

* Converted the `LRN` operator to an optimized `pooling` operator.
* Streamlined the `find_matches` function.
* Reduce the number of splits used for `split_reduce`.
* Improve layout propagation in poinwise fusion when using broadcasted inputs.


### Removed



## MIGraphX 2.14 for ROCm 7.1.0

### Added

* Added Python 3.13 support.
* Added PyTorch wheels to the Dockerfile.
* Added Python API for returning serialized bytes.
* Added `fixed_pad` operator for padding dynamic shapes to the maximum static shape.
* Added matcher to upcast base `Softmax` operations.
* Added support for the `convolution_backwards` operator through rocMLIR.
* Added `LSE` output to attention fusion.
* Added flags to `EnableControlFlowGuard` due to BinSkim errors.
* Added new environment variable documentation and reorganized structure.
* Added `stash_type` attribute for `LayerNorm` and expanded test coverage.
* Added operator builders (phase 2).
* Added `MIGRAPHX_GPU_HIP_FLAGS` to allow extra HIP compile flags.

### Changed

* Updated C API to include `current()` caller information in error reporting.
* Updated documentation dependencies:
  * **rocm-docs-core** bumped from 1.21.1 → 1.25.0 across releases.
  * **Doxygen** updated to 1.14.0.
  * **urllib3** updated from 2.2.2 → 2.5.0.
* Updated `src/CMakeLists.txt` to support `msgpack` 6.x (`msgpack-cxx`).
* Updated model zoo test generator to fix test issues and add summary logging.
* Updated `rocMLIR` and `ONNXRuntime` mainline references across commits.
* Updated module sorting algorithm for improved reliability.
* Restricted FP8 quantization to `dot` and `convolution` operators.
* Moved ONNX Runtime launcher script into MIGraphX and updated build scripts.
* Simplified ONNX `Resize` operator parser for correctness and maintainability.
* Updated `any_ptr` assertion to avoid failure on default HIP stream.
* Print kernel and module information on compile failure.

### Resolved issues

* Fixed error in `MIGRAPHX_GPU_COMPILE_PARALLEL` documentation (#4337).
* Fixed rocMLIR `rewrite_reduce` issue (#4218).
* Fixed bug with `invert_permutation` on GPU (#4194).
* Fixed compile error when `MIOPEN` is disabled (missing `std` includes) (#4281).
* Fixed ONNX `Resize` parsing when input and output shapes are identical (#4133, #4161).
* Fixed issue with MHA in attention refactor (#4152).
* Fixed synchronization issue from upstream ONNX Runtime (#4189).
* Fixed spelling error in “Contiguous” (#4287).
* Fixed tidy complaint about duplicate header (#4245).
* Fixed `reshape`, `transpose`, and `broadcast` rewrites between pointwise and reduce operators (#3978).
* Fixed extraneous include file in HIPRTC-based compilation (#4130).
* Fixed CI Perl dependency issue for SLES builds (#4254).
* Fixed compiler warnings for ROCm 7.0 of ``error: unknown warning option '-Wnrvo'``(#4192).

### Optimized

* Reduced nested visits in reference operators to improve compile time.
* Avoided dynamic memory allocation during kernel launches.
* Removed redundant NOP instructions for GFX11/12 platforms.
* Improved `Graphviz` output (node color and layout updates).
* Optimized interdependency checking during compilation.
* Skip hipBLASLt solutions requiring workspace size larger than 128 MB for efficient memory utilization.

### Removed

* Removed Perl dependency from SLES builds.
* Removed redundant includes and unused internal dependencies.


## MIGraphX 2.13 for ROCm 7.0.0

### Added

* Support for OCP `FP8` on AMD Instinct MI350X accelerators.
* Support for PyTorch 2.7 via Torch-MIGraphX.
* Support for the Microsoft ONNX Contrib Operators (Self) Attention, RotaryEmbedding, QuickGelu, BiasAdd, BiasSplitGelu, SkipLayerNorm.
* Support for Sigmoid and AddN TensorFlow operators.
* Added GroupQuery Attention support for LLMs.
* Added support for edge mode in the ONNX Pad operator.
* Added ONNX runtime Python driver.
* Added FLUX e2e example.
* Added C++ and Python APIs to save arguments to a graph as a msgpack file, and then read the file back.
* Added rocMLIR fusion for kv-cache attention.
* Introduced a check for file-write errors.

### Changed

* `quantize_bf16` for quantizing the model to `BF16` has been made visible in the MIGraphX user API.
* Print additional kernel/module information in the event of compile failure.
* Use hipBLASLt instead of rocBLAS on newer GPUs.
* 1x1 convolutions are now rewritten to GEMMs.
* `BF16::max` is now represented by its encoding rather than its expected value.
* Direct warnings now go to `cout` rather `cerr`.
* `FP8` uses hipBLASLt rather than rocBLAS.
* ONNX models are now topologically sorted when nodes are unordered.
* Improved layout of Graphviz output.
* Enhanced debugging for migraphx-driver: consumed environment variables are printed, timestamps and duration are added to the summary.
* Add a trim size flag to the verify option for migraphx-driver.
* Node names are printed to track parsing within the ONNX graph when using the `MIGRAPHX_TRACE_ONNX_PARSER` flag.
* Update accuracy checker to output test data with the `--show-test-data` flag.
* The `MIGRAPHX_TRACE_BENCHMARKING` option now allows the problem cache file to be updated after finding the best solution. 

### Removed

* `ROCM_USE_FLOAT8` macro.
* The BF16 GEMM test was removed for Navi21, as it is unsupported by rocBLAS and hipBLASLt on that platform.

### Optimized

* Use common average in `compile_ops` to reduce run-to-run variations when tuning.
* Improved the performance of the TopK operator.
* Conform to a single layout (NHWC or NCHW) during compilation rather than combining two.
* Slice Channels Conv Optimization (slice output fusion)
* Horizontal fusion optimization after pointwise operations.
* Reduced the number of literals used in `GridSample` linear sampler. 
* Fuse multiple outputs for pointwise operations.
* Fuse reshapes on pointwise inputs for MLIR output fusion.
* MUL operation not folded into the GEMM when the GEMM is used more than once.
* Broadcast not fused after convolution or GEMM MLIR kernels.
* Avoid reduction fusion when operator data-types mismatch.

### Resolved issues

* Compilation workaround ICE in clang 20 when using `views::transform`.
* Fix bug with `reshape_lazy` in MLIR.
* Quantizelinear fixed for Nearbyint operation.
* Check for empty strings in ONNX node inputs for operations like Resize.
* Parse Resize fix: only check `keep_aspect_ratio_policy` attribute for sizes input.
* Nonmaxsuppression: fixed issue where identical boxes/scores not ordered correctly.
* Fixed a bug where events were created on the wrong device in a multi-gpu scenario.
* Fixed out of order keys in value for comparisons and hashes when caching best kernels.
* Fixed Controlnet MUL types do not match error.
* Fixed check for scales if ROI input is present in Resize operation.
* Einsum: Fixed a crash on empty squeeze operations.

## MIGraphX 2.12 for ROCm 6.4.0

### Added

* Support for gfx1200 and gfx1201
* hipBLASLt support for contiguous transpose GEMM fusion and GEMM pointwise fusions for improved performance
* Support for hardware specific FP8 datatypes (FP8 OCP and FP8 FNUZ)
* Add support for the BF16 datatype
* ONNX Operator Support for `com.microsoft.MultiHeadAttention`, `com.microsoft.NhwcConv`, and `com.microsoft.MatMulIntgerFloat`
* migraphx-driver can now produce output for use with Netron
* migraphx-driver now includes a `time` parameter (similar to `perf`) that is more accurate for very fast kernels
* An end-to-end Stable Diffusion 3 example with option to disable T5 encoder on VRAM-limited GPUs has been added
* Added support to track broadcast axes in `shape_transform_descriptor`
* Added support for unsigned types with `rocMLIR`
* Added a script to convert mxr files to ONNX models
* Added the `MIGRAPHX_SET_GEMM_PROVIDER` environment variable to choose between rocBLAS and hipBLASLt. Set `MIGRAPHX_SET_GEMM_PROVIDER` to `rocblas` to use rocBLAS, or to `hipblaslt` to use hipBLASLt.


### Changed

* With the exception of gfx90a, switched to using hipBLASLt instead of rocBLAS
* Included the min/max/median of the `perf` run as part of the summary report
* Enable non-packed inputs for `rocMLIR`
* Always output a packed type for q/dq after determining non-packed tensors were inefficient
* Even if using NHWC, MIGraphX will always convert group convolutions to NCHW for best performance 
* Renamed the `layout_nhwc` to `layout_convolution` and ensured that either the weights are the same layout as the inputs or set the input and weights to NHWC
* Minimum version of Cmake is now 3.27


### Removed

* Removed `fp8e5m2fnuz` rocBLAS support
* `__AMDGCN_WAVEFRONT_SIZE` has been deprecated.
* Removed a warning that printed to stdout when using FP8 types
* Remove zero point parameter for dequantizelinear when its zero


### Optimized

* Prefill buffers when MLIR produces a multioutput buffer
* Improved the resize operator performance which should improve overall performance of models that use it
* Allow the `reduce` operator to be split across an axis to improve fusion performance.  The `MIGRAPHX_SPLIT_REDUCE_SIZE` environment variable has been added to allow the minimum size of the reduction to be adjusted for a possible model specific performance improvement
* Added `MIGRAPHX_DISABLE_PASSES` environment variable for debugging
* Added `MIGRAPHX_MLIR_DUMP` environment variable to be set to a folder where individual final rocMLIR modules can be saved for investigation
* Improved the C++ API to allow onnxruntime access to fp8 quantization



### Resolved Issues

* Fixed multistream execution with larger models (#3757)
* Peephole LSTM Error (#3768)
* Fixed BertSquad example that could include a broken tokenizers package (#3556)
* Fixed Attention fusion ito not error with a shape mismatch when a trailing pointwise contains a literal (#3758)
* Fixed instruction::replace() logic to handle more complex cases (#3574)
* MatMulNBits could fail with a shape error (#3698)
* Fixed a bug were some models could fail to compile with an error `flatten: Shapes are not in standard layout` (#3579)



## MIGraphX 2.11 for ROCm 6.3.0

### Added

* Initial code to run on Windows
* Support for gfx120x GPU
* Support for FP8, and INT4
* Support for the Log2 internal operator
* Support for the GCC 14 compiler
* The BitwiseAnd, Scan, SoftmaxCrossEntropyLoss, GridSample, and NegativeLogLikelihoodLoss ONNX operators
* The MatMulNBits, QuantizeLinear/DequantizeLinear, GroupQueryAttention, SkipSimplifiedLayerNormalization, and SimpliedLayerNormalization Microsoft Contrib operators
* Dymamic batch parameter support to OneHot operator
* Split-K as an optional performance improvement
* Scripts to validate ONNX models from the ONNX Model Zoo
* GPU Pooling Kernel
* --mlir flag to the migraphx-driver program to offload entire module to rocMLIR
* Fusing split-reduce with MLIR
* Multiple outputs for the MLIR + Pointwise fusions
* Pointwise fusions with MLIR across reshape operations
* MIGRAPHX_MLIR_DUMP environment variable to dump MLIR modules to MXRs
* The 3 option to MIGRAPHX_TRACE_BENCHMARKING to print the MLIR program for improved debug output
* MIGRAPHX_ENABLE_HIPBLASLT_GEMM environment variable to call hipBlasLt libaries
* MIGRAPHX_VERIFY_DUMP_DIFF to improve the debugging of accuracy issues
* reduce_any and reduce_all options to the Reduce operation via Torch MIGraphX
* Examples for RNNT, and ControlNet


### Changed

* Switched to MLIR's 3D Convolution operator.
* MLIR is now used for Attention operations by default on gfx942 and newer ASICs.
* Names and locations for VRM specific libraries have changed.
* Use random mode for benchmarking GEMMs and convolutions.
* Python version is now printed with an actual version number.


### Removed

* Disabled requirements for MIOpen and rocBlas when running on Windows.
* Removed inaccuracte warning messages when using exhaustive-tune.
* Remove the hard coded path in MIGRAPHX_CXX_COMPILER allowing the compiler to be installed in different locations.


### Optimized

* Improved:
    * Infrastructure code to enable better Kernel fusions with all supported data types
    * Subsequent model compile time by creating a cache for already performant kernels
    * Use of Attention fusion with models
    * Performance of the Softmax JIT kernel and of the Pooling opterator
    * Tuning operations through a new 50ms delay before running the next kernel
    * Performance of several convolution based models through an optimized NHWC layout
    * Performance for the FP8 datatype
    * GPU utilization
    * Verification tools
    * Debug prints
    * Documentation, including gpu-driver utility documentation
    * Summary section of the migrahx-driver perf command
* Reduced model compilation time
* Reordered some compiler passes to allow for more fusions
* Preloaded tiles into LDS to improve performance of pointwise transposes
* Exposed the external_data_path property in onnx_options to set the path from onnxruntime


### Resolved Issues

* Fixed a bug with gfx1030 that overwrote dpp_reduce.
* Fixed a bug in 1arg dynamic reshape that created a failure.
* Fixed a bug with dot_broadcast and inner_broadcast that caused compile failures.
* Fixed a bug where some configs were failing when using exhaustive-tune.
* Fixed the ROCM Install Guide URL.
* Fixed an issue while building a whl package due to an apostrophe.
* Fixed the BERT Squad example requirements file to support different versions of Python.
* Fixed a bug that stopped the Vicuna model from compiling.
* Fixed failures with the verify option of migraphx-driver that would cause the application to exit early.


## MIGraphX 2.10 for ROCm 6.2.0

### Additions

* Added support for ONNX Runtime MIGraphX EP on Windows
* Added FP8 Python API 
* Added examples for SD 2.1 and SDXL
* Improved Dynamic Batch to support BERT
* Added a `--test` flag in migraphx-driver to validate the installation
* Added support for ONNX Operator: Einsum
* Added uint8 support in ONNX Operators
* Added fusion for group convolutions
* Added rocMLIR conv3d support 
* Added rocgdb to the Dockerfile


### Optimizations

* Improved ONNX Model Zoo coverage
* Reorganized memcpys with ONNX Runtime to improve performance
* Replaced scaler multibroadcast + unsqueeze with just a multibroadcast
* Improved MLIR kernel selection for multibroadcasted GEMMs
* Improved details of the perf report
* Enable mlir by default for GEMMs with small K
* Allow specifying dot or convolution fusion for mlir with environmental flag
* Improve performance on small reductions by doing multiple reduction per wavefront 
* Add additional algebraic simplifications for mul-add-dot  sequence of operations involving constants
* Use MLIR attention kernels in more cases
* Enables MIOpen and CK fusions for MI300 gfx arches
* Support for QDQ quantization patterns from Brevitas which have explicit cast/convert nodes before and after QDQ pairs
* Added Fusion of "contiguous + pointwise" and "layout + pointwise" operations which may result in performance gains in certain cases
* Added Fusion for "pointwise + layout" and "pointwise + contiguous" operations which may result in performance gains when using NHWC layout
* Added Fusion for "Pointwise + concat" operation which may help in performance in certain cases
* Fixes a bug in "concat + pointwise" fusion where output shape memory layout wasn't maintained 
* Simplifies "slice + concat" pattern in SDXL UNet
* eliminates ZeroPoint/Shift in QuantizeLinear or DeQuantizeLinear ops if zero points values are zeros
* Improved inference performance by fusing Reduce to Broadcast
* Added additional information when printing the perf report
* Improve scalar fusions when not all strides are 0
* Added support for multi outputs in pointwise ops
* Improve reduction fusion with reshape operators
* Use the quantized output when an operator is used again


### Resolved issues

* Super Resolution model verification failed with FP16
* Suppressed confusing messages when compiling the model
* Mod operator failed to compile with int8 and int32 inputs
* Prevented spawning too many threads for constant propagation when parallel STL is not enabled
* Fixed a bug when running migraphx-driver with the --run 1 option
* Layernorm Accuracy fix: calculations in FP32
* Update Docker generator script to ROCm 6.1 to point at Jammy
* Floating Point exception fix for dim (-1) in reshape operator
* Fixed issue with int8 accuracy and models which were failing due to requiring a fourth bias input
* Fixed missing inputs not previously handled for quantized bias for the weights, and data values of the input matrix
* Fixed order of operations for int8 quantization which were causing inaccuracies and slowdowns
* Removed list initializer of prefix_scan_sum which was causing issues during compilation and resulting in the incorrect constructor to be used at compile
* Fixed the MIGRAPHX_GPU_COMPILE_PARALLEL flag to enable users to control number of threads used for parallel compilation



### Changes

* Changed default location of libraries with release specific ABI changes
* Reorganized documentation in GitHub


### Removals

* Removed the `--model` flag with migraphx-driver



## MIGraphX 2.9 for ROCm 6.1.0

### Additions

* Added beta version of FP8, functional, not performant
* Created a dockerfile with MIGraphX+ONNX Runtime EP+Torch
* Added support for the `Hardmax`, `DynamicQuantizeLinear`, `Qlinearconcat`, `Unique`, `QLinearAveragePool`, `QLinearSigmoid`, `QLinearLeakyRelu`, `QLinearMul`, `IsInf` operators
* Created web site examples for `Whisper`, `Llama-2`, and `Stable Diffusion 2.1`
* Created examples of using the ONNX Runtime MIGraphX Execution Provider with the `InceptionV3` and `Resnet50` models
* Updated operators to support ONNX Opset 19
* Enable fuse_pointwise and fuse_reduce in the driver
* Add support for dot-(mul)-softmax-dot offloads to MLIR
* Added Blas auto-tuning for GEMMs
* Added dynamic shape support for the multinomial operator
* Added fp16 to accuracy checker
* Added initial code for running on Windows OS

### Optimizations

* Improved the output of migraphx-driver command
* Documentation now shows all environment variables
* Updates needed for general stride support
* Enabled Asymmetric Quantization
* Added ScatterND unsupported reduction modes
* Rewrote softmax for better performance
* General improvement to how quantization is performed to support INT8
* Used problem_cache for gemm tuning
* Improved performance by always using rocMLIR for quantized convolution
* Improved group convolutions by using rocMLIR
* Improved accuracy of fp16 models
* ScatterElements unsupported reduction
* Added concat fusions
* Improved INT8 support to include UINT8
* Allow reshape ops between dq and quant_op
* Improve dpp reductions on navi
* Have the accuracy checker print the whole final buffer
* Added support for handling dynamic Slice and ConstantOfShape ONNX operators
* Add support for the dilations attribute to Pooling ops
* Add layout attribute support for LSTM operator
* Improved performance by removing contiguous for reshapes
* Handle all slice input variations
* Add scales attribute parse in upsample for older opset versions
* Added support for uneven Split operations
* Improved unit testing to run in python virtual environments

### Resolved issues

* Fixed outstanding issues in autogenerated documentation
* Update model zoo paths for examples
* Fixed promote_literals_test by using additional if condition
* Fixed export API symbols from dynamic library
* Fixed bug in pad operator from dimension reduction
* Fixed using the LD to embed files and enable by default when building shared libraries on linux
* fixed get_version()
* Fixed Round operator inaccuracy
* Fixed wrong size check when axes not present for slice
* Set the .SO version correctly


### Changes

* Cleanup LSTM and RNN activation functions
* Placed gemm_pointwise at a higher priority than layernorm_pointwise
* Updated README to mention the need to include GPU_TARGETS when building MIGraphX


### Removals

* Removed unused device kernels from Gather and Pad operators
* Removed int8x4 format



## MIGraphX 2.8 for ROCm 6.0.0

### Additions

* Support for MI300 GPUs
* Support for TorchMIGraphX via PyTorch
* Boosted overall performance by integrating rocMLIR
* INT8 support for ONNX Runtime
* Support for ONNX version 1.14.1
* Added new operators: `Qlinearadd`, `QlinearGlobalAveragePool`, `Qlinearconv`, `Shrink`, `CastLike`,
  and `RandomUniform`
* Added an error message for when `gpu_targets` is not set during MIGraphX compilation
* Added parameter to set tolerances with `migraphx-driver` verify
* Added support for MXR files > 4 GB
* Added `MIGRAPHX_TRACE_MLIR` flag
* BETA added capability for using ROCm Composable Kernels via the `MIGRAPHX_ENABLE_CK=1`
  environment variable

### Optimizations

* Improved performance support for INT8
* Improved time precision while benchmarking candidate kernels from CK or MLIR
* Removed contiguous from reshape parsing
* Updated the `ConstantOfShape` operator to support Dynamic Batch
* Simplified dynamic shapes-related operators to their static versions, where possible
* Improved debugging tools for accuracy issues
* Included a print warning about `miopen_fusion` while generating `mxr`
* General reduction in system memory usage during model compilation
* Created additional fusion opportunities during model compilation
* Improved debugging for matchers
* Improved general debug messages

### Resolved issues

* Fixed scatter operator for nonstandard shapes with some models from ONNX Model Zoo
* Provided a compile option to improve the accuracy of some models by disabling Fast-Math
* Improved layernorm + pointwise fusion matching to ignore argument order
* Fixed accuracy issue with `ROIAlign` operator
* Fixed computation logic for the `Trilu` operator
* Fixed support for the DETR model

### Changes

* Changed MIGraphX version to 2.8
* Extracted the test packages into a separate deb file when building MIGraphX from source

### Removals

* Removed building Python 2.7 bindings

## MIGraphX 2.7 for ROCm 5.7.0

### Additions

* hipRTC no longer requires dev packages for MIGraphX runtime and allows the ROCm install to be in a
   different directory than build time
* Added support for multi-target execution
* Added Dynamic Batch support with C++/Python APIs
* Added `migraphx.create_argument` to Python API
* Added dockerfile example for Ubuntu 22.04
* Added TensorFlow supported ops in driver similar to exist onnx operator list
* Added a MIGRAPHX_TRACE_MATCHES_FOR env variable to filter the matcher trace
* Improved debugging by printing max,min,mean and stddev values for TRACE_EVAL = 2
* You can now use the ` fast_math` flag instead of `ENV` for GELU
* Print message from driver if offload copy is set for compiled program

### Optimizations

* Optimized for ONNX Runtime 1.14.0
* Improved compile times by only building for the GPU on the system
* Improved performance of pointwise/reduction kernels when using NHWC layouts
* Loaded specific version of the `migraphx_py` library
* Annotated functions with the block size so the compiler can do a better job of optimizing
* Enabled reshape on nonstandard shapes
* Used half HIP APIs to compute max and min
* Added support for broadcasted scalars to unsqueeze operator
* Improved multiplies with dot operator
* Handled broadcasts across dot and concat
* Added verify namespace for better symbol resolution

### Resolved issues

* Resolved accuracy issues with FP16 resnet50
* Updated cpp generator to handle inf from float
* Fixed assertion error during verify and made DCE work with tuples
* Fixed convert operation for NaNs
* Fixed shape typo in API test
* Fixed compile warnings for shadowing variable names
* Added missing specialization for the `nullptr` hash function

### Changees

* Bumped version of half library to 5.6.0
* Bumped CI to support ROCm 5.6
* Made building tests optional
* Replaced `np.bool` with `bool` per NumPy request

### Removals

* Removed int8x4 rocBlas calls due to deprecation
* Removed `std::reduce` usage because not all operating systems support it

## MIGraphX 2.5 for ROCm 5.5.0

### Additions

* Y-Model feature will store tuning information with the optimized model
* Added Python 3.10 bindings
* Accuracy checker tool based on ONNX runtime
* ONNX operators parse_split, and Trilu
* Build support for ROCm MLIR
* Added the `migraphx-driver` flag to print optimizations in Python (--python)
* Added JIT implementation of the Gather and Pad operators, which results in better handling for
  larger tensor sizes

### Optimizations

* Improved performance of Transformer-based models
* Improved performance of the `Pad`, `Concat`, `Gather`, and `Pointwise` operators
* Improved ONNX/pb file loading speed
* Added a general optimize pass that runs several passes, such as `simplify_reshapes`, algebra, and DCE
  in a loop

### Resolved issues

* Improved parsing for TensorFlow Protobuf files
* Resolved various accuracy issues with some ONNX models
* Resolved a gcc-12 issue with MIVisionX
* Improved support for larger sized models and batches
* Use `--offload-arch` instead of `--cuda-gpu-arch` for the HIP compiler
* Changes inside JIT to use float accumulator for large reduce ops of half type to avoid overflow
* Changes inside JIT to temporarily use cosine to compute sine function

### Changes

* Changed version and location of third-party build dependencies in order to pick up fixes
