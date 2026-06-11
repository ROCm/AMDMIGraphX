# MIGraphX Windows Build: HIP SDK vs TheRock Comparison

## Build Environment

All builds used:
- **Branch**: `uai-develop`
- **rbuild**: `windows` branch installed from source (`C:\develop\rbuild`)
- **x64 Native Tools Command Prompt for VS 2022** (or vcvars64 sourced for INCLUDE/LIB)
- **Python 3.12** with rbuild venv
- **GPU target**: gfx1201

## Critical Discovery: Environment Variable Setup

The key to building MIGraphX with TheRock is setting `HIP_PATH` to the **top-level** TheRock install directory (not `lib\llvm`). This enables the Clang driver to find `.hipVersion` and auto-include `__clang_hip_runtime_wrapper.h`, which provides device-side math function overloads (`isfinite`, `nextafter`, etc.).

### Correct environment setup (works for both TheRock 7.11 and 7.13):
```cmd
set HIP_PATH=C:\opt\rocm
set ROCM_PATH=C:\opt\rocm
set HIP_CLANG_PATH=C:\opt\rocm\lib\llvm\bin
set HIP_DEVICE_LIB_PATH=C:\opt\rocm\lib\llvm\amdgcn\bitcode
set HIP_PLATFORM=amd
set LIB=%LIB%;C:\opt\rocm\lib
```

### Build command (same for all):
```cmd
rbuild build -d depend -B build -DGPU_TARGETS=gfx1201 -DCMAKE_C_COMPILER="C:/opt/rocm/lib/llvm/bin/clang.exe" -DCMAKE_CXX_COMPILER="C:/opt/rocm/lib/llvm/bin/clang++.exe" -DCLANG_FORMAT=CLANG_FORMAT-NOTFOUND
```

Note: No `--rocm-path` or `-DCMAKE_CXX_FLAGS` needed. The environment variables handle path resolution.

## Comparison Results

| Aspect | HIP SDK 7.2 (Clang 21) | TheRock 7.11 (Clang 22) | TheRock 7.13 (Clang 23) |
|--------|------------------------|------------------------|------------------------|
| **Source changes needed** | None | **None** | 1 file (comment `-mno-ms-bitfields`) |
| **`-mno-ms-bitfields`** | Works fine | Works fine | Fatal error (Itanium ABI conflict) |
| **`std::isfinite` / `std::nextafter`** | Links correctly | Links correctly (with env var fix) | Links correctly (with env var fix) |
| **rocMLIR** | Builds successfully | Builds successfully | Builds successfully |
| **Build result** | Success | **Success (zero changes)** | Success (1 line commented) |
| **Runtime (real model)** | Works (2,649 inf/s) | **Works (2,137 inf/s)** | Hangs at HIPRTC compilation |
| **HIPRTC/comgr** | Works | **Works** | Hangs (bug in amd_comgr0702.dll) |

## TheRock 7.11 (Clang 22) - Fully Working

TheRock 7.11 with the correct environment variable setup is the cleanest TheRock path:
- **Zero source code changes** to MIGraphX
- **Zero patches** to TheRock headers
- Build and runtime both work end-to-end
- Tested with real model (`clc-v3-fp16-512x512.onnx`): 2,137 inferences/sec, completed in 1.95s

## TheRock 7.13 (Clang 23) - Build Works, Runtime Broken

### Issue 1: `-mno-ms-bitfields` (Clang 23 only)
- **File**: `src/CMakeLists.txt` line 156
- **Error**: `error: Itanium-compatible layout for the Microsoft C++ ABI is not yet supported`
- **Fix**: Comment out `target_compile_options(migraphx PUBLIC "-mno-ms-bitfields")`
- **Root cause**: Clang 23 rejects this flag with MSVC ABI; Clang 21 and 22 do not

### Issue 2: HIPRTC/comgr runtime hang (Clang 23 only)
- **Symptom**: `migraphx-driver perf` hangs indefinitely on real models
- **Details**: Simple MLIR-only models (e.g. `add_scalar_test.onnx` with just `dot`) work fine. Any model with pointwise/fused operations that require HIPRTC compilation hangs. `migraphx-hiprtc-driver` child processes spawn but stall with near-zero CPU usage.
- **Root cause**: Bug in TheRock 7.13's `amd_comgr0702.dll` / `hiprtc0702.dll` on Windows

### Previously Misidentified Issues (now resolved)

The `isfinite`/`nextafter` linker errors and the `__clang_cuda_complex_builtins.h` `max` identifier error were both caused by incorrect `HIP_PATH` configuration. When `HIP_PATH` pointed to `lib\llvm` instead of the top-level directory, the Clang driver failed to detect the HIP version and skipped auto-inclusion of `__clang_hip_runtime_wrapper.h`. Setting `HIP_PATH` to the top-level directory resolves both issues for all TheRock versions without any source code modifications.

## Directory Layout Differences

| Path | HIP SDK 7.2 | TheRock 7.11/7.13 |
|------|------------|-------------------|
| Clang | `bin/clang++.exe` | `lib/llvm/bin/clang++.exe` |
| Bitcode | `amdgcn/bitcode/` | `lib/llvm/amdgcn/bitcode/` |
| `.hipVersion` | `bin/.hipVersion` | `bin/.hipVersion` |
| `lib/llvm/` | Does not exist | Exists (full LLVM install) |
| Clang version dir | `lib/clang/21/` | `lib/llvm/lib/clang/{22,23}/` |
| cmake packages | `lib/cmake/` | `lib/cmake/` (same) |

## Conclusion

1. **TheRock 7.11 (Clang 22)** is fully functional with zero MIGraphX source changes when using the correct environment variable setup. This is the recommended TheRock version for Windows builds.

2. **TheRock 7.13 (Clang 23)** builds successfully with one minor change (commenting `-mno-ms-bitfields`) but has a runtime bug in its comgr/HIPRTC DLLs that causes kernel JIT compilation to hang on real models. This needs to be reported as a bug against TheRock.

3. The critical fix across all TheRock versions is setting `HIP_PATH` to the top-level install directory (e.g. `C:\opt\rocm`), not to the nested `lib\llvm` directory. This enables proper HIP version detection and auto-inclusion of `__clang_hip_runtime_wrapper.h`.

4. rocMLIR was successfully built and used by all three builds (HIP SDK, TheRock 7.11, TheRock 7.13).
