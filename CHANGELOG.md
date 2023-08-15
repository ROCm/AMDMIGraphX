# Change Log for MIGraphX

Full documentation for MIGraphX is available at [MIGraphX Documentation](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/).

## MIGraphX 2.7 for ROCm 5.7.0

### Added
- Enabled hipRTC allowing the ROCm install to not be required in /opt/rocm specifically 
- Add support for multi-target execution
- Added Dynamic Batch support with C++/Python APIs
- Enabled native int32 type support
- Add migraphx.create_argument to python API
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
- Resolved accuracy issues with resnet50
- Update cpp generator to handle inf from  float
- Fix assertion error during verify and make DCE work with tuples
- Fix convert operation for NaNs
- Fix shape typo in API test
- Fix compile warnings for shadowing variable names
- Add missing specialization for the `nullptr` for the hash function

### Changed
- Bumped version of half library to 5.6.0
- Use clang-format to format
- Bumped CI to support rocm 5.6
- Make building tests optional
- Update install prereqs python fix
- replace np.bool with bool as per numpy request

### Removed
- Removed int8x4 rocBlas calls due to deprecation
- removed std::reduce usage since not all OS' support it
