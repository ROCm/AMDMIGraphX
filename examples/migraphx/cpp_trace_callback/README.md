# Trace Callback

## Description
This example demonstrates how to use `program::run_trace` to inspect operator output buffers during program evaluation. The callback receives the instruction index, name, and a host-side copy of each operator's result.

The example shows three filtering modes:
1. **No filter** -- callback fires for every operator
2. **Name filter** -- callback fires only for operators whose name contains a substring (e.g. `"concat_kernel"`)
3. **Instruction index filter** -- callback fires only for a specific instruction identified by its `@N` index in the compiled graph

## Building
```
mkdir build && cd build
cmake ..
make
```

By default CMakeLists.txt expects the MIGraphX source tree at `../../..` and the build directory at `<source>/build`. Override with `-DMIGRAPHX_SRC_DIR=<path>` and `-DMIGRAPHX_BUILD_DIR=<path>` if needed.

## Running
```
./trace_callback_example
```
