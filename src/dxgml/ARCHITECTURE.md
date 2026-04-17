# DxGML Frontend — Architecture

## Overview

The DxGML frontend converts **DxGML MLIR dialect** into
MIGraphX `program` objects, which then follow the standard MIGraphX GPU compilation
pipeline.  No DxGML ops ever reach the rocMLIR / GPU layer; all dialect-specific
nodes are translated to MIGraphX built-in ops during frontend parsing.

```
    .mlir
      │
      ▼
 parse_dxgml()            ← public entry point (src/dxgml/dxgml.cpp)
      │
      ▼
 dxgml_parser             ← hand-rolled text parser (src/dxgml/dxgml_parser.cpp)
      │  extracts entry_point sig + body, walks SSA assignments
      │
      ▼
 parse_dxgml_op()         ← per-op converter (src/dxgml/parse_ops.cpp)
      │  maps each dxgml_op.* name to migraphx::make_op(...)
      │
      ▼
 migraphx::program        ← standard MIGraphX IR (migraphx::module + instruction list)
      │
      ▼
 program::compile(target) ← standard GPU path: optimization → rocMLIR lowering → ISA
```

---

## File Map

| File | Role |
|------|------|
| `src/include/migraphx/dxgml.hpp` | Public API: `parse_dxgml()`, `parse_dxgml_string()`, `dxgml_options` |
| `src/dxgml/dxgml.cpp` | Entry points; orchestrates parser, applies dump flags |
| `src/dxgml/dxgml_parser.hpp` | Internal `dxgml_parser` class declaration |
| `src/dxgml/dxgml_parser.cpp` | Text parser: type parsing, attribute helpers, body walk |
| `src/dxgml/parse_ops.cpp` | Op-by-op conversion from DxGML names to MIGraphX ops |
| `src/dxgml/CMakeLists.txt` | Build rules for `amdxgml` shared library |
| `test/dxgml/` | Unit tests and embedded MLIR fixtures |

---

## Parser Design

### Why a hand-rolled text parser?

DxGML `.mlir` files use custom dialect shorthand syntax throughout:
  
```mlir
dxgml.module {
  dxgml.entry_point @name(%arg0: !dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<...>
  {
    %w = dxgml_op.constant(#dxgml.constant_resource<_conv1.weight : !dxgml.tensor<...>>)
    %0 = dxgml_op.convolution(%arg0, %w) { strides = ... } : (...) -> ...
    dxgml.return %0 : ...
  }
}
```

This syntax requires dialect-registered ops (`dxgml_op.*`, `dxgml.*`) to be parsed
by the MLIR C API.  The DxGML dialect registration library (`dxgml_ir.dll`) is built
against a different LLVM version than the rocMLIR install, making dynamic linking
impractical.  The custom text parser avoids this entirely.

### Parser stages

1. **Strip dialect resources** — the `{-# dialect_resources #-}` block at the end
   is removed before parsing (it contains binary weight placeholders, not ops).

2. **Extract `dxgml.entry_point`** — locate the entry point signature
   `(arg_list) -> ret_type attributes? { body }` using bracket matching.

3. **Register entry-point arguments** — each `%argN: !dxgml.tensor<...>` becomes a
   named `@param` instruction via `module::add_parameter`.  Args are added in reverse
   order so that `arg0` ends up at index 0 (since `add_parameter` prepends).

4. **Flatten the body** — strip line comments, join all body lines into one string
   for uniform scanning.

5. **Walk SSA assignments** — scan for:
   - `%result = dxgml_op.<name>(<operands>) { attrs } : type_sig`
   - `dxgml.return %val : type`

   For each `dxgml_op.*` assignment, dispatch to `parse_dxgml_op()`.

6. **Register the return** — `dxgml.return` maps to `module::add_return()`.

### Constant parameters

`dxgml_op.constant(#dxgml.constant_resource<NAME : TYPE>)` represents a named
weight tensor.  By default these become `@param` instructions (callers supply the
weight data at runtime).  When a resources file is provided (see below), matching
constants become `@literal` instructions with the decoded weight data embedded
directly in the program.  Constants are inserted with
`insert_parameter(module::end(), ...)` so they appear in source order, after the
entry-point argument parameters.

### External weight loading (`--dxgml-resources`)

Large models store weight tensor data in a companion `resources.mlir` file whose
`{-# dialect_resources #-}` block contains one hex-encoded blob per weight:

```
{-#
  dialect_resources: {
    dxgml: {
      _conv1.weight: "0x01000000AABBCC...",
      _fc1.bias:     "0x010000003F800000",
      ...
    }
  }
#-}
```

**Hex blob format** (MLIR dense_resource_blob):
- Starts with `0x`
- First **4 bytes** are the MLIR alignment header (`01 00 00 00`) — always skipped
- Remaining bytes are the raw little-endian tensor payload, matching `shape.bytes()` exactly

**How the parser loads it:**

`dxgml_parser::load_resources_from_stream()` reads the file **line by line** —
it never loads the whole file into memory.  For each entry line it:
1. Extracts the resource NAME (up to the `:`)
2. Finds the quoted hex string
3. Decodes the hex in-place, skipping the 4-byte MLIR header
4. Stores the raw bytes in `resource_map[NAME]`

`dxgml_parser::load_resources()` (used for inline text) wraps the string in an
`istringstream` and calls the same function.

In `parse_from_string()`, resources are loaded in two passes before op parsing begins:
1. Inline: `load_resources(mlir_text)` — handles models that embed the block internally
2. External file: stream directly from `opts.resources_file` via `load_resources_from_stream()`

Then in `parse_constant()`, after computing the shape:

```
resource_map.find(NAME)
  found + size matches shape.bytes() → mm->add_literal(literal{sh, raw.data()})
  found + size mismatch              → stderr warning, fall through to @param
  not found                          → @param (random data at runtime)
```

Size mismatches occur for int4-packed quantized weights, where the resource stores
two int4 values per byte but the shape describes the logical int8 element count.

**Using it from the driver:**

```bat
REM Parse only — verify weights load as @literal
migraphx-driver.exe read --dxgml model.mlir --dxgml-resources resources.mlir --text

REM GPU verify with real weights
migraphx-driver.exe verify --dxgml model.mlir --dxgml-resources resources.mlir --gpu --atol 1e-2 --rtol 1e-2
```

**Using it from the GPU test script:**

`run_dxgml_gpu_tests.ps1` passes `--dxgml-resources` automatically for models that
have a companion resources file.  The `RunTest` function accepts an `$ExtraArgs`
parameter; the phi_silica_qdq entry uses it as follows:

```powershell
$phiModel = "$MlirDir\phi_silica_qdq\model.mlir"
$phiRes   = "$MlirDir\phi_silica_qdq\resources.mlir"
$phiExtra = if (Test-Path $phiRes) { @("--dxgml-resources", $phiRes) } else { @() }
RunTest $phiModel "phi_silica_qdq" -atol "1e-2" -rtol "1e-2" -timeoutSec 600 -ExtraArgs $phiExtra
```

If `resources.mlir` is absent the test still runs, using random `@param` data.

**Performance note:** a typical resources file for a large model is several GB.
The streaming parser keeps peak memory usage proportional to the largest single
tensor's decoded byte array, not the total file size.

---

## Op Mapping

### Unary elementwise

| DxGML | MIGraphX |
|-------|----------|
| `relu` | `relu` |
| `sigmoid` | `sigmoid` |
| `tanh` | `tanh` |
| `erf` | `erf` |
| `exp` | `exp` |
| `log` | `log` |
| `sqrt` | `sqrt` |
| `abs` | `abs` |
| `ceil` | `ceil` |
| `floor` | `floor` |
| `neg` | `neg` |
| `rsqrt` | `rsqrt` |
| `recip` | `recip` |

### Binary elementwise

| DxGML | MIGraphX |
|-------|----------|
| `add` | `add` |
| `subtract` | `sub` |
| `multiply` | `mul` |
| `divide` | `div` |
| `pow` | `pow` |
| `max` | `max` |
| `min` | `min` |

### Structural ops

| DxGML | MIGraphX | Notes |
|-------|----------|-------|
| `convolution` | `convolution` | First 2 inputs only (input + filter); bias operand is ignored by MIGraphX conv |
| `gemm` / `dot` | `dot` | — |
| `reshape` | `reshape` | Output shape from type signature |
| `transpose` | `transpose` | `perm` attribute |
| `cast` | `convert` | Target type from type signature |
| `softmax` | `softmax` | `axis` attribute |
| `log_softmax` | `logsoftmax` | `axis` attribute |
| `max_pooling` | `pooling(mode=max)` | `window_size`, `strides`, `padding` |
| `average_pooling` | `pooling(mode=average)` | — |
| `global_avg_pool` | `pooling(mode=average)` | Lengths from input spatial dims |
| `concat` | `concat` | `axis` attribute |
| `slice` | `slice` | `axes`, `starts`, `ends` |
| `reduce_sum` | `reduce_sum` | `axes`, `keepdims` |
| `reduce_mean` | `reduce_mean` | — |
| `reduce_max` | `reduce_max` | — |
| `reduce_min` | `reduce_min` | — |
| `reduce_prod` | `reduce_prod` | — |
| `squeeze` | `squeeze` | `axes` |
| `unsqueeze` | `unsqueeze` | `axes` |
| `flatten` | `flatten` | `axis` |
| `constant` | `@literal` / `@param` | Named weight — `@literal` when `--dxgml-resources` supplies data; `@param` otherwise |

---

## Data Flow After Parsing

Once `parse_dxgml()` returns a `migraphx::program`, the caller follows the standard
MIGraphX pipeline:

```cpp
migraphx::program prog = migraphx::parse_dxgml("model.mlir");

// Optional: inspect the MIGraphX op graph
// std::cout << prog;

migraphx::compile_options compile_opts;
prog.compile(migraphx::gpu::target{}, compile_opts);

// prog is now GPU-executable
```

The `dxgml_options::dump_*` flags can be set to print intermediate representations
to `stderr`:

| Flag | When it fires | What is printed |
|------|---------------|-----------------|
| `dump_migraphx_ops` | After `parse_dxgml_string()` returns | MIGraphX op graph (before any lowering) |
| `dump_migraphx_dialect` | After GPU lowering pass | MIGraphX MLIR dialect (set by the caller after `compile()`) |
| `dump_gpu` | After `compile()` | Compiled GPU program |
| `dump_isa` | After code-object emission | GCN/RDNA assembly |

`dump_migraphx_dialect`, `dump_gpu`, and `dump_isa` require a compiled program and
must be acted upon by the calling layer (e.g. a driver or test harness) after
`program::compile()` completes.  The flags are exposed in `dxgml_options` so that
tooling wrappers can inspect them.

---

## Build

The frontend is a standalone shared library (`amdxgml`).  It has no
dependency on MLIR C API libraries or `dxgml_ir.dll`.

Use **`generate_migraphx.bat`** at the repo root.  It supports two generators:

### Ninja (default)

```bat
REM Full configure + build (WinRelWithDebInfo preset)
generate_migraphx.bat

REM Rebuild only after code change (skip re-configure)
generate_migraphx.bat WinRelWithDebInfo --build-only

REM Configure only — inspect cache, do not build
generate_migraphx.bat WinRelWithDebInfo --no-build

REM Full build + run all DxGML tests
generate_migraphx.bat WinRelWithDebInfo --run-tests

REM Debug build
generate_migraphx.bat WinDebug
```

Available presets: `WinRelWithDebInfo` (default), `WinDebug`, `WinRelease`, `WinMinSizeRel`.

Output directory: `build\<preset>\`  (e.g. `build\WinRelWithDebInfo\bin\amdxgml.dll`)

### Visual Studio

**Prerequisite — install two ClangCL components:**

1. Open **Visual Studio Installer** → click **Modify** on VS 2022
2. Go to the **Individual components** tab
3. Search for and check **both** of the following:
   - **"C++ Clang Compiler for Windows"** — installs `clang-cl.exe`
   - **"C++ Clang-cl for v143 build tools (x64/x86)"** — registers the ClangCL MSBuild toolset
4. Click **Modify** to install

Both components are required. Without `clang-cl.exe`, the compiler is missing. Without the `v143` toolset integration, CMake configure fails with `MSB8020: The build tools for ClangCL cannot be found`.

```bat
REM Generate VS solution for IDE navigation (no build)
generate_migraphx.bat --vs --no-build
REM  → open build_vs\migraphx.sln in Visual Studio

REM Configure VS solution + build (RelWithDebInfo)
generate_migraphx.bat --vs

REM Rebuild without re-configure
generate_migraphx.bat --vs --build-only

REM Debug config
generate_migraphx.bat --vs Debug

REM VS build + run all DxGML tests
generate_migraphx.bat --vs --run-tests
```

Available configs: `RelWithDebInfo` (default), `Debug`, `Release`, `MinSizeRel`.
Also accepts Ninja preset names as aliases (`WinRelWithDebInfo` → `RelWithDebInfo`).

Output directory: `build_vs\bin\<config>\`  (e.g. `build_vs\bin\RelWithDebInfo\amdxgml.dll`)

### CMake

```cmake
# Ninja (via preset):
cmake --preset WinRelWithDebInfo -DBUILD_TESTING=ON
cmake --build build/WinRelWithDebInfo --parallel --target amdxgml driver

# Visual Studio:
cmake -G "Visual Studio 17 2022" -A x64 -T ClangCL \
      -DMIGRAPHX_ENABLE_DXGML=ON -DDXGML_DROP_DIR=<path> ...
cmake --build build_vs --config RelWithDebInfo --parallel --target amdxgml driver
```

## Running tests

### Parse + driver tests (CPU only)

```bat
REM ctest parse unit tests (5 tests):
ctest --test-dir build\WinRelWithDebInfo -R test_dxgml -V

REM Full driver + parse suite (35 driver tests + 4 parse unit tests):
test\dxgml\run_dxgml_tests.bat

REM Driver tests only:
test\dxgml\run_dxgml_tests.bat mlir

REM Single model:
test\dxgml\run_dxgml_tests.bat simple_gemm
```

### GPU execution tests

`test\dxgml\run_dxgml_gpu_tests.bat` runs models on the local GPU using
`migraphx-driver`.  Five modes are available:

| Mode | Driver command | What it does |
|------|---------------|--------------|
| `parse` | `read --dxgml` | Parse only — no GPU involved |
| `compile` | `compile --dxgml --gpu` | Compile to GPU kernels, no execution |
| `run` | `run --dxgml --gpu` | Compile + execute on GPU, no validation |
| `verify` | `verify --dxgml --gpu --atol --rtol` | GPU execution validated against CPU reference (**default**) |
| `profile` | `rocprofv2 --kernel-trace -- … run --dxgml --gpu` | GPU kernel trace via rocprofv2 (falls back to rocprof or plain run) |

```bat
REM Verify all models (default):
test\dxgml\run_dxgml_gpu_tests.bat

REM Explicit mode:
test\dxgml\run_dxgml_gpu_tests.bat verify
test\dxgml\run_dxgml_gpu_tests.bat parse
test\dxgml\run_dxgml_gpu_tests.bat compile
test\dxgml\run_dxgml_gpu_tests.bat run
test\dxgml\run_dxgml_gpu_tests.bat profile

REM Single model (any mode):
test\dxgml\run_dxgml_gpu_tests.bat verify conv_act_add
test\dxgml\run_dxgml_gpu_tests.bat compile conv_example
test\dxgml\run_dxgml_gpu_tests.bat profile simple_gemm

REM Verify all + save full driver output:
test\dxgml\run_dxgml_gpu_tests.bat verify dump
```

**Loading real weights during GPU tests:**

Models that ship a companion `resources.mlir` file (e.g. `phi_silica_qdq`) have
weight data automatically injected via `--dxgml-resources`.  The script checks for
the file at `$MlirDir\<model>\resources.mlir` and passes the flag only when found,
so tests still run (with random weights) when the resources file is absent.

To inject weights manually for a single model:

```bat
migraphx-driver.exe verify ^
    --dxgml test\dxgml\mlir\phi_silica_qdq\model.mlir ^
    --dxgml-resources test\dxgml\mlir\phi_silica_qdq\resources.mlir ^
    --gpu --atol 1e-2 --rtol 1e-2
```

**Tolerances** (verify mode):
- fp16 models: `--atol 1e-2 --rtol 1e-2`
- fp32 models (`conv_example`): `--atol 1e-4 --rtol 1e-4`

**Profiling output** is written to `test\dxgml\dump\profile_<model>\` — one
subdirectory per model containing the rocprofv2 CSV kernel trace and a
`driver_output.txt`.  `rocprofv2` is preferred; the script falls back to
legacy `rocprof` if only that is on PATH, and to an un-instrumented `run`
if neither profiler is found (still useful for wall-clock timing).

**Log file**: every run appends results to `test\dxgml\dump\gpu_run_results.log`.

---

## Test Layout

```
test/dxgml/
  include/
    dxgml_test.hpp              helper: read_dxgml(), embedded file access
  mlir/                         embedded MLIR fixtures (CompilationInput files)
  parse/
    conv_relu_test.cpp
    gelu_test.cpp
    relu_erf_test.cpp
    standalone_cluster_test.cpp
    debug_parse_test.cpp        diagnostic: try-catch all fixtures, print results
  run_dxgml_tests.bat           CPU parse + driver test runner
  run_dxgml_tests.ps1           PowerShell implementation
  run_dxgml_gpu_tests.bat       GPU test runner (parse/compile/run/verify/profile)
  run_dxgml_gpu_tests.ps1       PowerShell implementation
  dump/                         Log and dump output directory (git-ignored)
```
