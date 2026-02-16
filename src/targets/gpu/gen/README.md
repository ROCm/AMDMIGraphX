# GPU Gen IR

The GPU Gen IR is an extension to the MIGraphX IR for generating optimized GPU kernels. It lowers high-level tensor operations through four stages -- gridwise, blockwise, lanewise, and final -- producing C++ HIP kernel code that is JIT-compiled to GPU code objects.

## Pipeline Overview

The gen IR sits in the GPU target compilation pipeline between `fuse_mlir` and `lowering`. The `fuse_gen` pass identifies eligible operations, wraps them in `gpu::gen::op`, and the `gen_compiler` lowers and compiles them during `compile_ops`.

```
MIGraphX high-level IR
    |
    | fuse_pointwise, fuse_pointwise_reduce (existing passes)
    | fuse_mlir (existing pass)
    v
    | fuse_gen pass  <-- identifies pointwise, reduce, gather/pad/reverse inputs
    v
gpu::gen::op[submodule]  (wrapped in gpu::precompile_op)
    |
    | compile_ops pass invokes gen_compiler
    v
gen_compiler::compile()
    |  1. gen_gridwise  -- add output buffer, handle reductions, set up grid
    |  2. gen_blockwise -- allocate LDS for block reductions, tiling
    |  3. gen_lanewise  -- lower to per-element load/store/reduce ops
    |  4. gen_final     -- hardware-specific optimizations (placeholder)
    |  5. DCE           -- remove dead instructions
    |  6. codegen       -- generate C++ source from gen IR
    |  7. compile_hip   -- JIT compile to GPU code object
    v
gpu::code_object_op (executable GPU kernel)
```

## Fusion (`fuse_gen.cpp`)

The `fuse_gen` pass runs after `fuse_mlir` and matches:

**Pointwise operations** (`find_gen_pointwise`): Matches `pointwise` instructions not handled by MLIR. When a pointwise op's inputs include `pad`, `gather`, or `reverse` (1D only), those index transform ops are inlined into the gen module.

**Standalone reductions** (`find_gen_reduce`): Matches bare `reduce_sum`, `reduce_max`, etc. that weren't fused by `fuse_pointwise_reduce`. Limited to 1D inputs.

For each matched operation, the pass creates a gen module containing the inlined operations and replaces the original instruction with `gpu::gen::op`.

### Input Fusion Example

For `add(pad(x), y)`, the fusion traces back through the `pad` input and inlines it:

```
Before fuse_gen:
  x = @param:x -> {8}
  pad = pad[pads={2,2}](x) -> {12}
  y = @param:y -> {12}
  add = pointwise(pad, y) [add submodule]

After fuse_gen:
  x = @param:x -> {8}
  y = @param:y -> {12}
  gen_op = gpu::gen::op(x, y) [gen module containing: pad + add]
```

## Gridwise Pass (`gridwise.cpp`)

The first lowering stage. Transforms MIGraphX IR to grid-level gen IR.

**For all modules:**
- Adds `z_output` parameter (output buffer -- MIGraphX IR doesn't have explicit output buffers)
- Inserts `gpu::gen::copy(result, z_output)` to write computation result to output

**For pointwise:**
- Inserts `gpu::gen::workgroup_id` for multi-tile distribution (when tile config has ntiles > 0)

**For reductions:**
- Detects `reduce_*` instructions
- Analyzes input/output shapes to compute `reduce_elements`
- Selects reduction algorithm based on shape analysis:
  - **lane**: Strided reductions (min stride > 2)
  - **wave**: Small reductions (reduce_elements <= wavefront size, typically 64)
  - **block**: Large reductions (reduce_elements > wavefront size)
- Replaces `reduce_*` with `gpu::gen::gridwise_reduce[op, algo, reduce_elements, block_size]`

### Example: reduce_sum lowering through gridwise

```
Before gridwise:
  x = @param:x -> {256}
  reduce_sum[axes={0}](x) -> {1}
  @return(reduce_sum)

After gridwise:
  x = @param:x -> {256}
  z_output = @param:z_output -> {1}
  gridwise_reduce[op=sum, algo=block, reduce_elements=256, block_size=256](x) -> {1}
  copy(gridwise_reduce, z_output) -> {1}
  @return(copy)
```

## Blockwise Pass (`blockwise.cpp`)

The second stage. Manages workgroup-level resources.

**For block reductions:**
- Detects `gridwise_reduce[algo="block"]`
- Inserts `gpu::gen::lds_allocate` (one element per wave for cross-wave communication)
- Adds the LDS buffer as an additional input to the `gridwise_reduce`

**For multi-dimensional pointwise (tiling):**
- Computes tile configuration from parameter shapes
- Inserts `gpu::gen::tile_region` for each multi-dimensional parameter
- Replaces parameter references with tiled regions

### Example: block reduce through blockwise

```
Before blockwise:
  gridwise_reduce[algo=block, block_size=256](x) -> {1}

After blockwise:
  lds = gpu::gen::lds_allocate[shape={float, {4}}]  // 4 waves in 256 threads
  gridwise_reduce[algo=block, block_size=256](x, lds) -> {1}
```

## Lanewise Pass (`lanewise.cpp`)

The third stage and the main lowering pass. Converts tensor operations to per-element scalar operations.

### Thread ID Insertion

Inserts `gpu::gen::local_id` (for tiled ops) or `gpu::gen::global_id` (for non-tiled) as the thread index.

### Copy Lowering

Converts `gpu::gen::copy(src, dst)` to `gpu::gen::store(dst, tid, src)`.

### Index Transform Lowering

Detects `pad`, `gather`, `reverse` instructions in the gen module and lowers them to gen IR index transforms + loads:

**Pad** becomes `pad_index` + `conditional_load`:
```
pad_idx = gpu::gen::pad_index[input_shape, pads](tid)     // -1 if out of bounds
val = gpu::gen::conditional_load(input, pad_idx, 0.0)      // load or fill value
```

**Gather** becomes `gather_index` + `load`:
```
g_idx = gpu::gen::gather_index[input_shape, axis](indices, tid)
val = gpu::gen::load(data, g_idx)
```

**Reverse** becomes `reverse_index` + `load`:
```
r_idx = gpu::gen::reverse_index[input_shape, axes](tid)
val = gpu::gen::load(input, r_idx)
```

### Pointwise Lowering

For tensor parameters feeding into pointwise ops:
1. Inserts `gpu::gen::load(param, tid)` to produce scalar values
2. Rebuilds pointwise ops as new scalar instructions (creates NEW instructions to avoid intermediate shape mismatches)
3. Rewires `store` to use the scalar result

### Reduction Lowering

Lowers `gridwise_reduce` based on the `algo` field:

**Lane** (strided): `strided_load` + `lane_reduce`
```
val = gpu::gen::strided_load[size=N, stride=1](input, tid)
result = gpu::gen::lane_reduce[op=sum](val)
```

**Wave** (small, <=64 elements): `load` + `dpp_reduce`
```
val = gpu::gen::load(input, tid)
result = gpu::gen::dpp_reduce[op=sum](val)
```

**Block** (large): `[strided_load + lane_reduce] + dpp_reduce + reduce_waves`
```
// When reduce_elements > block_size: multiple elements per thread
val = gpu::gen::strided_load[size=N, stride=block_size](input, tid)
partial = gpu::gen::lane_reduce[op=sum](val)
// When reduce_elements <= block_size: one element per thread
partial = gpu::gen::load(input, tid)

wave_result = gpu::gen::dpp_reduce[op=sum](partial)
result = gpu::gen::reduce_waves[op=sum](wave_result, lds_buffer)
```

### Full Pointwise Example

```
Before lanewise:                        After lanewise:
  x = @param:x -> {16}                   x = @param:x -> {16}
  y = @param:y -> {16}                   y = @param:y -> {16}
  z_output = @param:z_output -> {16}     z_output = @param:z_output -> {16}
  add(x, y) -> {16}                      gid = gpu::gen::global_id
  copy(add, z_output)                    lx = gpu::gen::load(x, gid) -> scalar
  @return(copy)                          ly = gpu::gen::load(y, gid) -> scalar
                                         add = add(lx, ly) -> scalar
                                         store = gpu::gen::store(z_output, gid, add)
                                         @return(store)
```

## Code Generation (`codegen.cpp`)

The `compile_gen` function orchestrates the full pipeline:

1. Runs the 4-level lowering passes directly (not through `run_passes` to avoid intermediate shape validation)
2. Runs `dead_code_elimination` to clean up
3. Generates the C++ kernel source using `cpp_generator`
4. Compiles to a HIP code object via `compile_hip_code_object`

### How C++ Code is Generated

Each gen IR op specifies its generated code via the `gpu_gen` attribute. The `cpp_generator::generate_module` callback:

1. For ops with `gpu_gen` attribute: substitutes `${0}`, `${1}`, etc. with argument variable names
2. For ops with `point_op` attribute (standard pointwise): uses `generate_point_op`
3. Special cases: `multibroadcast` (pass-through), `tile_region` (pass-through), `lds_allocate` (GCC statement expression for `__shared__` memory)

### Generated Kernel Structure

```cpp
namespace migraphx {

template<class Tx0, class Tx1, class Tz_output, class Tidx>
__device__ auto gen_func(Tx0 x0, Tx1 x1, Tz_output z_output, Tidx idx) {
    auto zz0 = idx.global;                    // gpu::gen::global_id
    auto zz1 = x0[zz0];                       // gpu::gen::load
    auto zz2 = x1[zz0];                       // gpu::gen::load
    auto zz3 = zz1 + zz2;                     // add (pointwise)
    auto zz4 = z_output[zz0] = zz3;           // gpu::gen::store
    return zz4;
}

extern "C" {
MIGRAPHX_GLOBAL void add_gen_kernel(void* p0, void* p1, void* p2) {
    auto idx = make_index();
    make_tensors()(p0, p1, p2)([&](auto... xs) {
        gen_func(xs..., idx);
    });
}
}
} // namespace migraphx
```

### Launch Parameters

The `compile_gen` function selects launch parameters based on the lowered module:

- **Pointwise (no tiling)**: `global = output_elements`, `local = 256`
- **Pointwise (tiled)**: `global = grid_size * block_size`, `local = block_size`
- **Wave/lane reduce**: `global = input_elements (rounded up)`, `local = 256`
- **Block reduce**: `global = output_elements * block_size`, `local = block_size`

## Operations Reference

### ID Operations

| Operation | Generated Code | Description |
|-----------|---------------|-------------|
| `gpu::gen::global_id` | `idx.global` | Global thread index |
| `gpu::gen::local_id` | `idx.local` | Local thread index within workgroup |
| `gpu::gen::workgroup_id` | `idx.group` | Workgroup index |
| `gpu::gen::workgroup_size` | `idx.nlocal()` | Threads per workgroup |
| `gpu::gen::lane_id` | `idx.local_wave()` | Lane index within wave |

### Memory Operations

| Operation | Generated Code | Description |
|-----------|---------------|-------------|
| `gpu::gen::load` | `${0}[${1}]` | Load element from tensor at index |
| `gpu::gen::store` | `${0}[${1}] = ${2}` | Store element to tensor at index |
| `gpu::gen::vector_load[size=N]` | `gen::vec_load<N>(${0}.data(), ${1})` | Vectorized load |
| `gpu::gen::vector_store[size=N]` | `gen::vec_store<N>(${0}.data(), ${1}, ${2})` | Vectorized store |
| `gpu::gen::strided_load[size,stride]` | `gen::strided_load<size,stride>(${0}.data(), ${1})` | Strided load for reductions |
| `gpu::gen::copy[schedule]` | (lowered by lanewise to store) | High-level copy |
| `gpu::gen::lds_allocate[shape]` | `({static __shared__ T buf[N]; buf;})` | LDS allocation |
| `gpu::gen::conditional_load` | `gen::conditional_load(${0}, ${1}, ${2})` | Bounds-checked load |

### Reduction Operations

| Operation | Generated Code | Description |
|-----------|---------------|-------------|
| `gpu::gen::gridwise_reduce[op,algo,...]` | (lowered by lanewise) | High-level reduce with algorithm choice |
| `gpu::gen::lane_reduce[op]` | `gen::lane_reduce_<op>(${0})` | Per-thread array reduction |
| `gpu::gen::dpp_reduce[op]` | `gen::dpp_reduce_<op>(${0})` | Wave-level DPP reduction |
| `gpu::gen::reduce_waves[op]` | `gen::block_reduce_<op>(${0}, ${1}, ...)` | Cross-wave LDS reduction |

### Index Transform Operations

| Operation | Generated Code | Description |
|-----------|---------------|-------------|
| `gpu::gen::pad_index[input_shape,pads]` | `gen::pad_index_1d(len, pad, ${0})` | Pad index (-1 if out of bounds) |
| `gpu::gen::gather_index[input_shape,axis]` | `gen::gather_index_1d(${0}, ${1})` | Gather index lookup |
| `gpu::gen::reverse_index[input_shape,axes]` | `gen::reverse_index_1d(len, ${0})` | Reverse index |

## Kernel-Side Helpers (`migraphx/kernels/gen.hpp`)

Device functions called by generated kernel code:

**Memory access:**
- `gen::vec_load<N>(data, offset)` / `gen::vec_store<N>(data, offset, value)` -- Vectorized access
- `gen::strided_load<N, Stride>(data, base)` / `gen::strided_store<N, Stride>(data, base, value)` -- Strided access
- `gen::conditional_load(data, idx, fill)` -- Bounds-checked load with fallback

**Reductions:**
- `gen::lane_reduce_sum/product/max/min(array)` -- Sequential array reduction
- `gen::dpp_reduce_sum/product/max/min(x)` -- Wavefront reduction via `__shfl_xor` butterfly pattern
- `gen::block_reduce_sum/product/max/min(x, lds, nwaves, wave_id, lane_id)` -- Cross-wave LDS reduction

**Index transforms:**
- `gen::pad_index_1d(input_len, pad_before, out_idx)` -- Returns input index or -1
- `gen::gather_index_1d(indices, out_idx)` -- Looks up `indices[out_idx]`
- `gen::reverse_index_1d(len, out_idx)` -- Returns `(len-1) - out_idx`

## Testing

### Unit Tests (`test/gpu/gen/`)

Run: `cd build && ./bin/test_gpu_gen_<name>`

| File | Tests | Description |
|------|-------|-------------|
| `tiling_test.cpp` | 6 | Tile config computation, vectorization |
| `gridwise_test.cpp` | 5 | Output buffer insertion, reduce algo selection |
| `blockwise_test.cpp` | 2 | Tiling, LDS allocation |
| `lanewise_test.cpp` | 5 | Load/store insertion, copy lowering, reduce lowering |
| `codegen_test.cpp` | 4 | C++ code generation from gen IR |
| `ops_test.cpp` | 9 | Shape computation, codegen for gen ops |
| `fuse_test.cpp` | 3 | Fusion pass with fuse_pointwise |

### Verify Tests (`test/verify/`)

Run: `cd build && ./bin/test_verify '*_gen_*'`

| File | Description |
|------|-------------|
| `test_gen_pointwise.cpp` | add, mul, mul+add |
| `test_gen_pad_add.cpp` | pad + add |
| `test_gen_reverse_add.cpp` | reverse + add |
| `test_gen_gather_pointwise.cpp` | gather + add |
| `test_gen_concat_pointwise.cpp` | concat + add |
| `test_gen_reduce_sum.cpp` | reduce_sum, reduce_max (standalone) |
| `test_gen_reduce_mul_sum.cpp` | reduce_sum (standalone) |
| `test_gen_pad_reduce.cpp` | pad + reduce_sum |
| `test_gen_gather_reduce.cpp` | gather + reduce_sum |
| `test_gen_concat_reduce.cpp` | concat + reduce_sum |
| `test_gen_reverse_reduce.cpp` | reverse + reduce_sum |
| `test_gen_pad_mul_reduce.cpp` | pad + mul + reduce_sum |
| `test_gen_gather_add_reduce.cpp` | gather + add + reduce_sum |

## Current Limitations

- **1D only**: Index transform fusion (pad, gather, reverse) is limited to 1D tensors. Multi-dimensional versions fall back to existing JIT compilers.
- **No concat fusion**: Concat is not fused into gen modules (complex multi-input selection).
- **No fused reduce**: Fused reductions (pointwise + reduce in one kernel) are not supported. The `fuse_pointwise_reduce` + gen pipeline handles them separately.
- **Final pass placeholder**: The final pass doesn't yet implement hardware-specific optimizations.

## File Structure

```
src/targets/gpu/gen/
  ops.cpp                   # All gen IR operations (30+ ops)
  fuse_gen.cpp              # Fusion pass (pointwise, reduce, index input fusion)
  tiling.cpp                # Tile config computation
  gridwise.cpp              # Stage 1: output buffer, reduce algo selection
  blockwise.cpp             # Stage 2: LDS allocation, tiling
  lanewise.cpp              # Stage 3: load/store/reduce/index lowering
  final.cpp                 # Stage 4: HW optimizations (placeholder)
  codegen.cpp               # C++ code generation + compile_gen function
  README.md                 # This file
  include/migraphx/gpu/gen/
    fuse_gen.hpp, tiling.hpp, gridwise.hpp, blockwise.hpp,
    lanewise.hpp, final.hpp, codegen.hpp

src/targets/gpu/jit/
  gen.cpp                   # gen_compiler registration (CRTP compiler)

src/targets/gpu/kernels/include/migraphx/kernels/
  gen.hpp                   # Device-side helpers (vec_load, dpp_reduce, etc.)

test/gpu/gen/               # Unit tests (34 tests, 7 files)
test/verify/test_gen_*.cpp  # Verify tests (16 tests, 13 files)
```
