# DxGML Frontend — Architecture

## Overview

The DxGML frontend converts `.mlir` files written in the **DxGML MLIR dialect** into
MIGraphX `program` objects, which then follow the standard MIGraphX GPU compilation
pipeline.  No DxGML ops ever reach the rocMLIR / GPU layer; all dialect-specific
nodes are translated to MIGraphX built-in ops during frontend parsing.

```
.mlir file / text
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
| `src/dxgml/CMakeLists.txt` | Build rules for `migraphx_dxgml` shared library |
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

The frontend is a standalone shared library (`migraphx_dxgml`).  It has no
dependency on MLIR C API libraries or `dxgml_ir.dll`.

```cmake
# Enable / disable:
cmake -DMIGRAPHX_ENABLE_DXGML=ON ...

# Build:
ninja migraphx_dxgml

# Run tests:
ninja test/dxgml/test
ctest -R test_dxgml -V
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
