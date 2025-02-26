# Changelog for MIGraphX

Full documentation for MIGraphX is available at
[https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/).

## MIGraphX 2.12 for ROCm 6.4.0

### Added

* Support for gfx1200 and gfx1201
* hipBlasLt support 
  * Added contiguous transpose gemm fusion for hipBLASLt
  * Added gemm pointwise fusion for hipBLASLt
  * Made gfx90a use rocBLAS by default
* Support for rdware specific FP8 datatypes
  * Driver quantize fp8 update
  * FP8 OCP to FP8 FNUZ on hardware with only FP8 FNUZ support
  * Added the bit_cast operator for fp8 OCP
  * Added the fp8e5m2fnuz data type
  * Added support on gfx120x for the FP8 OCP format
  * Enable GEMM/dot for FP8 using hipblasLT
* Add support for the BF16 datatype
* ONNX Operator Support
  * Add onnx support for com.microsoft.MultiHeadAttention
  * Add onnx support for com.microsoft.NhwcConv
  * Add onnx support for com.microsoft.MatMulIntgerFloat
* migraphx-driver improvements 
  * can now produce output for use with Netron
  * Added a `time` for better accuracy of very fast kernels
  * Included percentile details to summary report
* end-to-end Stable Diffusion 3 example with option to disable T5 encoder on VRAM-limited GPUs
* Fusions...
  * Fuse transposes in pointwise and reduce fusions
  * Fuse reshapes across concat
  * Fuse unpack_int4 across concat
  * Horizontally fuse elementwise operators with more then 2 inputs across concat
  * Fuse all pointwise inputs with mlir not just the first one found
* Enable non-packed inputs for mlir
* Track broadcast axes in the shape_transform_descriptor
* Disable dot/mul optimizations when there is int4 weights
* Add support for unsigned types with mlir
* Added a script to convert mxr files to ONNX models
* New environment variable to choose between rocBLAS and hipBLASLT; MIGRAPHX_SET_GEMM_PROVIDER=rocblas|hipblaslt for supported architectures


### Changed

* Switched to using hipBLASLt as default instead of rocBLAS for hipBLASLt supported architectures.
* Always output a packed type for q/dq
* Removed a warning that printed to stdout when using FP8 types
* Set migraphx version to 2.12
* Always use NCHW for group convolutions
* Remove zero point parameter for dequantizelinear when its zero
* Dont use mixed layouts with convolutions
* Update Cmake to 3.27.x since ORT 1.21 required a newer CMake to not break pybind


### Removed

* Disable fp8e5m2fnuz with rocBLAS
* Removed __AMDGCN_WAVEFRONT_SIZE for deprecation
* Removed environment variable MIGRAPHX_ENABLE_HIPBLASLT_GEMM.


### Optimized

* Refactor GPU math functions for an accuracy improvement 
* catch python buffer unsupported types
* Prefill buffers when MLIR produces a multioutput buffer
* Improved the performance of the resize operator
* Enable split reduce by default
* Added a MIGRAPHX_DISABLE_PASSES enviroment variable for debugging
* Added a MIGRAPHX_MLIR_DUMP_TO_FILE flag to capture the final mlir module to a file
* Move qlinear before concat to allow output fusion
* use reshape to handle Flatten operator
* Improved documentation by cleaning up links and adding a Table Of Contents
* Updated cpp code guideline checks
* Brokeout the fp8_quantization functions via our API to allow onnxruntime to use fp8 quantization



### Resolved Issues

* Fixed multistream execution with larger models
* Fixed broken links in the documentation
* Peephole LSTM Error
* Fixed BertSquad example that could include a broken tokenizers package 
* Fixed Attention fusion ito not error with a shape mismatch when a trailing pointwise contains a literal
* Fixed instruction::replace() logic to handle more complex cases
* MatMulNBits could fail with a shape error



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
* --mlir flag to the migraphx-driver program to offload entire module to mlir
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


### Fixes

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

### Fixes

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

### Fixes

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

### Fixes

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

### Fixes

* Improved parsing for TensorFlow Protobuf files
* Resolved various accuracy issues with some ONNX models
* Resolved a gcc-12 issue with MIVisionX
* Improved support for larger sized models and batches
* Use `--offload-arch` instead of `--cuda-gpu-arch` for the HIP compiler
* Changes inside JIT to use float accumulator for large reduce ops of half type to avoid overflow
* Changes inside JIT to temporarily use cosine to compute sine function

### Changes

* Changed version and location of third-party build dependencies in order to pick up fixes
