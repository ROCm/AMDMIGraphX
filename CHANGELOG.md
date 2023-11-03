# Change Log for MIGraphX

Full documentation for MIGraphX is available at [MIGraphX Documentation](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/).

## MIGraphX 2.8 for ROCm 6.0.0
### Added
- Support for MI300 GPUs
- Support for TorchMIGraphX via PyTorch
- Boosted overall performance by integrating rocMLIR 
- INT8 support for ONNX Runtime
- Support for ONNX version 1.14.1
- Added operators Qlinearadd, QlinearGlobalAveragePool, Qlinearconv, Shrink, CastLike, and RandomUniform operators
- Added an error message when gpu_targets is not set when compiling migraphx
- Added parameter to set tolerances with migraphx-driver verify 
- Added support for MXR files >4 GB 
- Added MIGRAPHX_TRACE_MLIR flag
- BETA added capability to use ROCm Composable Kernels via environment variable MIGRAPHX_ENABLE_CK=1

### Optimizations
- Improved performance support for INT8
- Improved time percision while benchmarking candidate kernels from CK or MLIR 
- Remove contiguous from reshape parsing
- Updated ConstantOfShape operator to support Dynamic Batch
- Simplifies dynamic shapes related operators to their static versions if possible
- Improved debugging tools for accuracy issues
- Print warning about miopen_fusion while generating mxr 
- General reduction in system memory usage during model compilation
- Created additional fusion opportunities during model compilation
- Improved debugging for matchers
- Improved general debug messages 

### Fixed
- Fixed scatter operator for nonstandard shapes with some models from ONNX Model Zoo
- Provided a compile option to improve accuracy of some models by disabling Fast-Math
- Improved layernorm + pointwise fusion matching to ignore arguments order
- Fixed accuracy issue with ROIAlign operator 
- Fixed Trilu operator computation logic
- Fixed support for the DETR model 

### Changed
- Changed migraphx version to 2.8
- Extracted test packages as its own separate deb file when building migraphx from source

### Removed
- Removed building Python 2.7 bindings


## MIGraphX 2.7 for ROCm 5.7.0
### Added
- Enabled hipRTC to not require dev packages for migraphx runtime and allow the ROCm install to be in a different directory than it was during build time
- Add support for multi-target execution
- Added Dynamic Batch support with C++/Python APIs
- Add migraphx.create_argument to python API
- Added dockerfile example for Ubuntu 22.04
- Add TensorFlow supported ops in driver similar to exist onnx operator list
- Add a MIGRAPHX_TRACE_MATCHES_FOR env variable to filter the matcher trace
- Improved debugging by printing max,min,mean and stddev values for TRACE_EVAL = 2
- use fast_math flag instead of ENV flag for GELU
- Print message from driver if offload copy is set for compiled program
### Optimizations
- Optimized for ONNX Runtime 1.14.0
- Improved compile times by only building for the GPU on the system
- Improve performance of pointwise/reduction kernels when using NHWC layouts
- Load specific version of the migraphx_py library
- Annotate functions with the block size so the compiler can do a better job of optimizing 
- Enable reshape on nonstandard shapes
- Use half HIP APIs to compute max and min
- Added support for broadcasted scalars to unsqueeze operator
- Improved multiplies with dot operator
- Handle broadcasts across dot and concat
- Add verify namespace for better symbol resolution
### Fixed
- Resolved accuracy issues with FP16 resnet50
- Update cpp generator to handle inf from  float
- Fix assertion error during verify and make DCE work with tuples
- Fix convert operation for NaNs
- Fix shape typo in API test
- Fix compile warnings for shadowing variable names
- Add missing specialization for the `nullptr` for the hash function
### Changed
- Bumped version of half library to 5.6.0
- Bumped CI to support rocm 5.6
- Make building tests optional
- replace np.bool with bool as per numpy request
### Removed
- Removed int8x4 rocBlas calls due to deprecation
- removed std::reduce usage since not all OS' support it


## MIGraphX 2.5 for ROCm 5.5.0
### Added
- Y-Model feature to store tuning information with the optimized model
- Added Python 3.10 bindings 
- Accuracy checker tool based on ONNX Runtime
- ONNX Operators parse_split, and Trilu 
- Build support for ROCm MLIR
- Added migraphx-driver flag to print optimizations in python (--python)
- Added JIT implementation of the Gather and Pad operator which results in better handling of larger tensor sizes.
### Optimizations
- Improved performance of Transformer based models
- Improved performance of the Pad, Concat, Gather, and Pointwise operators
- Improved onnx/pb file loading speed
- Added general optimize pass which runs several passes such as simplify_reshapes/algebra and DCE in loop.
### Fixed
- Improved parsing Tensorflow Protobuf files 
- Resolved various accuracy issues with some onnx models
- Resolved a gcc-12 issue with mivisionx
- Improved support for larger sized models and batches
- Use --offload-arch instead of --cuda-gpu-arch for the HIP compiler
- Changes inside JIT to use float accumulator for large reduce ops of half type to avoid overflow.
- Changes inside JIT to temporarily use cosine to compute sine function.
### Changed
- Changed version/location of 3rd party build dependencies to pick up fixes
