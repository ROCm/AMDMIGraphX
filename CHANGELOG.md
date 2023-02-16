# Change Log for MIGraphX

Full documentation for MIGraphX is available at [MIGraphX Github Pages](https://rocmsoftwareplatform.github.io/AMDMIGraphX/doc/html/).

## MIGraphX 2.5 for ROCm 5.5.0

### Added
- Y-Model feature to store tuning information with the optimized model
- Added an accuracy checker tool based on ONNX Runtime
- Added the ONNX Operator parse_split, and Trilu 
- Added build support for ROCm MLIR
- Added timestamps to tracing prints
- Added migraphx-driver flag to print optimizations in python (--python)


### Optimizations
- Improved performance of Transformer based models
- Improved performance of the Pad, concat, Gather, and Pointwise operators
- Improved onnx/pb file reading speed


### Fixed
- Improved Tensorflow Protobuf files 
- Resolved various accuracy issues with some onnx models
- Resolved a gcc-12 issue with mivisionx
- Improved support for larger sized models and batches
- Use --offload-arch instead of --cuda-gpu-arch for the HIP compiler

### Changed
- Upgraded CI environment to RCOm 5.3.0, onnx 1.10.3
- Changed version/location of 3rd party build dependencies to pick up fixes
