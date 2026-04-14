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
weight tensor.  These become `@param` instructions (callers supply the weight data
at runtime).  Constants are inserted with `insert_parameter(module::end(), ...)` so
they appear in source order, after the entry-point argument parameters.

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
| `constant` | `@param` | Named weight parameter |

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

### Ninja (default — fastest incremental rebuilds)

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

### Visual Studio 2022 (ClangCL toolset)

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

### CMake directly (advanced)

```cmake
# Ninja (via preset):
cmake --preset WinRelWithDebInfo -DBUILD_TESTING=ON
cmake --build build/WinRelWithDebInfo --parallel --target amdxgml driver

# Visual Studio:
cmake -G "Visual Studio 17 2022" -A x64 -T ClangCL \
      -DMIGRAPHX_ENABLE_DXGML=ON -DDXGML_DROP_DIR=<path> ...
cmake --build build_vs --config RelWithDebInfo --parallel --target amdxgml driver
```

### Running tests

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

---

## Test Layout

```
test/dxgml/
  include/
    dxgml_test.hpp          helper: read_dxgml(), embedded file access
  mlir/                     embedded MLIR fixtures (CompilationInput files)
  parse/
    conv_relu_test.cpp
    gelu_test.cpp
    relu_erf_test.cpp
    standalone_cluster_test.cpp
    debug_parse_test.cpp    diagnostic: try-catch all fixtures, print results
```
