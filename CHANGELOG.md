# Change Log for MIGraphX

Full documentation for MIGraphX is available at [MIGraphX Documentation](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/).

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
