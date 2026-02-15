# GPU Gen IR

The GPU Gen IR is an extension to the MIGraphX IR that represents GPU-specific operations for kernel generation. It provides a lower-level representation that enables fine-grained control over tiling, vectorization, and memory access patterns.

## Overview

The gen IR system takes high-level MIGraphX operations (like `pointwise`, `reduce`, `gather`, etc.) and progressively lowers them through four stages to produce optimized GPU kernels. The key idea is to represent GPU concepts like thread IDs, workgroups, and memory operations explicitly in the IR, then generate C++ code from this representation.

```
High-level MIGraphX IR (pointwise, reduce, etc.)
    |
    | fuse_gen pass
    v
gpu::gen::op (wraps submodule)
    |
    | gen_compiler (inside compile_ops)
    v
    +---> gen_gridwise   (grid/block size, workgroup distribution)
    +---> gen_blockwise  (LDS, tile_region, work distribution)
    +---> gen_lanewise   (vector_load/store, per-lane computation)
    +---> gen_final      (HW intrinsics, peephole optimizations)
    +---> cpp_generator  (C++ kernel source)
    +---> compile_hip    (GPU code object binary)
    v
code_object_op (executable kernel)
```

## Fusion

The `fuse_gen` pass runs after `fuse_mlir` in the GPU target pipeline and fuses the following operators:

- pointwise
- reductions (reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_prod)
- gather
- pad
- reverse
- concat

Operations already handled by `fuse_mlir` are skipped. The pass wraps eligible operations in `gpu::gen::op` inside `gpu::precompile_op`, which is later compiled by the gen_compiler during the `compile_ops` pass.

## 4-Level Lowering Pipeline

### Gridwise

Describes computation at the grid level:
- Specifies grid size and block size
- Defines tuning parameters
- Creates copy operations representing data movement
- Inserts `workgroup_id` for multi-tile distribution

### Blockwise

Manages work within a single workgroup:
- Inserts `tile_region` operations to create tiled views of tensors
- Manages shared memory (LDS) via `lds_allocate`
- Handles data loading from global memory to LDS
- Distributes work across all lanes in a block
- Inserts `barrier` for workgroup synchronization

### Lanewise

Operates on registers with per-lane computation:
- Replaces copy ops with explicit `vector_load` / `vector_store`
- Determines vectorization width based on shape alignment
- Inserts `local_id` for tiled copies, `global_id` for non-tiled
- For reductions: inserts `strided_load`, `lane_reduce`, `dpp_reduce`, `reduce_waves`

### Final

Applies hardware-specific optimizations:
- Replace operations with hardware intrinsics (FMA, etc.)
- Expand `dpp_reduce` to hardware DPP instruction sequences
- Insert scheduling barriers for instruction ordering
- Apply target-specific peephole optimizations

## Operations

All gen operations are in the `gpu::gen` namespace.

### ID Operations

| Operation | Description | Generated Code |
|-----------|-------------|----------------|
| `gpu::gen::global_id` | Global thread index | `idx.global` |
| `gpu::gen::local_id` | Local thread index within workgroup | `idx.local` |
| `gpu::gen::workgroup_id` | Workgroup index | `idx.group` |
| `gpu::gen::workgroup_size` | Number of threads per workgroup | `idx.nlocal()` |
| `gpu::gen::lane_id` | Lane index within wave | `idx.local_wave()` |

### Memory Operations

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `gpu::gen::load` | Load single element from tensor | - |
| `gpu::gen::store` | Store a single element to tensor | - |
| `gpu::gen::vector_load` | Vectorized load from tensor | `size`: vector width |
| `gpu::gen::vector_store` | Vectorized store to tensor | `size`: vector width |
| `gpu::gen::strided_load` | Load elements at strided positions | `size`, `stride` |
| `gpu::gen::strided_store` | Store to tensor at strided positions | `size`, `stride` |
| `gpu::gen::copy` | Copy data between tensors | `schedule`: `"per_lane"` or `"per_block"` |
| `gpu::gen::lds_allocate` | Allocate LDS (shared) memory | `shape`: tensor shape |

### Tiling Operations

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `gpu::gen::tile_region` | Create a tiled view of a tensor | `tile_dims`, `axis` |

### Synchronization and Control

| Operation | Description |
|-----------|-------------|
| `gpu::gen::barrier` | Workgroup barrier (`__syncthreads()`) |
| `gpu::gen::check` | Runtime assertion (`MIGRAPHX_CHECK`) |

### Reduction Operations

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `gpu::gen::lane_reduce` | Reduce within a lane (array) | `op`: "sum", "product", "max", "min" |
| `gpu::gen::dpp_reduce` | Reduce within a wavefront using DPP | `op`: "sum", "product", "max", "min" |
| `gpu::gen::reduce_waves` | Reduce across wavefronts using LDS | `op`: "sum", "product", "max", "min" |

### Index Transformation Operations

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `gpu::gen::pad_index` | Transform index for padded tensor | `input_shape`, `pads` |
| `gpu::gen::gather_index` | Transform index for gather | `input_shape`, `axis` |
| `gpu::gen::reverse_index` | Transform index for reversed axes | `input_shape`, `axes` |
| `gpu::gen::shape_index` | Transform index for slice/broadcast/transpose | `input_shape`, `output_shape` |
| `gpu::gen::conditional_load` | Load with bounds check | `size` |

## Code Generation

The code generation uses `cpp_generator` to produce C++ kernel code from the lowered gen IR. Operations specify their generated code via the `gpu_gen` attribute.

### Generated Kernel Structure

```cpp
namespace migraphx {

// Generated device function
template<class Tx, class Tz, class Tidx>
__device__ auto gen_func(Tx x, Tz z, Tidx idx) {
    auto zz0 = idx.global;
    auto zz1 = gen::vec_load<4>(x.data(), zz0);
    gen::vec_store<4>(z.data(), zz0, zz1);
}

extern "C" {
MIGRAPHX_GLOBAL void kernel(void* p0, void* p1) {
    auto idx = make_index();
    make_tensors()(p0, p1)([&](auto... xs) {
        gen_func(xs..., idx);
    });
}
}

} // namespace migraphx
```

## Kernel-Side Helpers

Located in `migraphx/kernels/gen.hpp`:

- `gen::vec_load<N>` / `gen::vec_store<N>` -- Vectorized memory access
- `gen::strided_load<N, Stride>` / `gen::strided_store<N, Stride>` -- Strided access
- `gen::conditional_load` -- Bounds-checked load
- `gen::lane_reduce_sum/product/max/min` -- Per-lane array reduction
- `gen::dpp_reduce_sum/product/max/min` -- Wave-level DPP reductions
- `gen::block_reduce_sum/product/max/min` -- Cross-wavefront LDS reductions

## Testing

### Structural Tests (`test/gpu/gen/`)

| File | Executable | Description |
|------|-----------|-------------|
| `ops_test.cpp` | `test_gpu_gen_ops_test` | Shape computation for all gen ops |
| `fuse_test.cpp` | `test_gpu_gen_fuse_test` | Fusion pass wraps operations correctly |
| `tiling_test.cpp` | `test_gpu_gen_tiling_test` | Tile config computation |
| `gridwise_test.cpp` | `test_gpu_gen_gridwise_test` | Gridwise pass behavior |
| `blockwise_test.cpp` | `test_gpu_gen_blockwise_test` | Blockwise pass behavior |
| `lanewise_test.cpp` | `test_gpu_gen_lanewise_test` | Lanewise pass behavior |
| `codegen_test.cpp` | `test_gpu_gen_codegen_test` | C++ code generation |

### Verify Tests (`test/verify/`)

| File | Description |
|------|-------------|
| `test_gen_pointwise.cpp` | Pointwise add/mul via gen IR |
| `test_gen_pad_add.cpp` | Pad + add fusion |
| `test_gen_reverse_add.cpp` | Reverse + add fusion |
| `test_gen_gather_add.cpp` | Gather + add fusion |

Run structural tests:
```bash
ctest -R '^test_gpu_gen_'
```

Run verify tests:
```bash
build/bin/test_verify general
```

## File Structure

```
src/targets/gpu/gen/
  ops.cpp                          # All gen IR operation definitions
  fuse_gen.cpp                     # Fusion pass
  tiling.cpp                       # Tiling utility (tile_config, computation)
  gridwise.cpp                     # Gridwise lowering pass
  blockwise.cpp                    # Blockwise lowering pass
  lanewise.cpp                     # Lanewise lowering pass
  final.cpp                        # Final optimization pass
  codegen.cpp                      # Code generation + gen_compiler
  README.md                        # This file
  include/migraphx/gpu/gen/
    fuse_gen.hpp
    tiling.hpp
    gridwise.hpp
    blockwise.hpp
    lanewise.hpp
    final.hpp
    codegen.hpp

src/targets/gpu/kernels/include/migraphx/kernels/
  gen.hpp                          # Device-side kernel helpers
```
