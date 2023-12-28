# Changelog for MIGraphX

Full documentation for MIGraphX is available at
[https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/).

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
