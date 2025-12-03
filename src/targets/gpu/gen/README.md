# GPU Gen IR

The GPU Gen IR is an extension to the MIGraphX IR that represents GPU-specific operations for kernel generation. It provides a lower-level representation that enables fine-grained control over tiling, vectorization, and memory access patterns.

## Overview

The gen IR system takes high-level MIGraphX operations (like `pointwise`) and progressively lowers them through several passes to produce optimized GPU kernels. The key idea is to represent GPU concepts like thread IDs, workgroups, and memory operations explicitly in the IR, then generate C++ code from this representation.

```
┌─────────────────┐
│   pointwise     │  High-level fused operation
└────────┬────────┘
         │ fuse_gen pass
         ▼
┌─────────────────┐
│ gpu::gen::copy  │  Gen IR with copy operations
└────────┬────────┘
         │ gen_tiling pass
         ▼
┌─────────────────┐
│  tile_region    │  Tiled operations with workgroup_id
│  workgroup_id   │
└────────┬────────┘
         │ gen_lower pass
         ▼
┌─────────────────┐
│  vector_load    │  Low-level memory operations
│  vector_store   │
│  local_id       │
└────────┬────────┘
         │ codegen
         ▼
┌─────────────────┐
│   C++ kernel    │  Generated HIP code
└─────────────────┘
```

## Operations

All gen operations are in the `gpu::gen` namespace and use `attributes()` with a `"point_op"` key to specify how they generate C++ code.

### ID Operations

These operations provide access to thread and workgroup indices:

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
| `gpu::gen::vector_load` | Vectorized load from tensor | `size`: vector width |
| `gpu::gen::vector_store` | Vectorized store to tensor | `size`: vector width |
| `gpu::gen::strided_load` | Load single element at strided position | - |
| `gpu::gen::strided_store` | Store to tensor at index | - |
| `gpu::gen::copy` | Copy data between tensors | `schedule`: `"per_thread"` or `"per_block"` |
| `gpu::gen::lds_allocate` | Allocate LDS (shared) memory | `shape`: tensor shape |

**`strided_load`** loads a single element at position `base + iter * stride`:
- Inputs: tensor, base_index, iteration, stride
- Used in reduction loops where each thread loads from: base, base+stride, base+2*stride, ...

**`accumulate`** accumulates a value into an accumulator:
- Inputs: accumulator, new_value
- Attribute: `op` - "sum", "product", "max", "min"
- Used for per-lane accumulation in reductions

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
| `gpu::gen::dpp_reduce` | Reduce within a wavefront using DPP | `op`: "sum", "product", "max", "min" |
| `gpu::gen::reduce_waves` | Reduce across wavefronts using LDS | `op`: "sum", "product", "max", "min" |

**`dpp_reduce`** performs an intra-wave reduction using AMD DPP (Data Parallel Primitives) instructions. The result is broadcast to all lanes in the wavefront.

**`reduce_waves`** reduces values across all wavefronts in a workgroup:
1. Each wave writes its partial result to LDS at `lds[wave_id]`
2. Synchronization barrier
3. First wave reads and reduces all partial results
4. Result is available to all threads

Example usage:
```
// Per-thread accumulation
@1 = ...compute partial sum...

// Reduce within wave
@2 = gpu::gen::dpp_reduce[op="sum"](@1)

// Allocate LDS for cross-wave reduction (one element per wave)
@3 = gpu::gen::lds_allocate[shape={float, {8}}]

// Reduce across waves  
@4 = gpu::gen::reduce_waves[op="sum"](@2, @3)
```

### Index Transformation Operations

These operations transform indices for operations like padding, gather, and reverse:

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `gpu::gen::offset` | Compute linear offset from index | `shape` |
| `gpu::gen::pad_index` | Transform index for padded tensor | `input_shape`, `pads` |
| `gpu::gen::gather_index` | Transform index for gather | `input_shape`, `axis` |
| `gpu::gen::reverse_index` | Transform index for reversed axes | `input_shape`, `axes` |
| `gpu::gen::shape_index` | Transform index for slice/broadcast/transpose | `input_shape`, `output_shape` |
| `gpu::gen::conditional_load` | Load with bounds check | `size` |

## Passes

### 1. `fuse_gen` Pass

**Location:** `fuse_gen.cpp`  
**Runs after:** `fuse_mlir` in the GPU target pipeline

This pass identifies operations that can be handled by the gen IR system and wraps them in `gpu::precompile_op` for JIT compilation.

**What it does:**
- Finds `pointwise` operations not handled by other fusion passes
- Skips multi-output, broadcasted, and mixed-type operations
- Wraps eligible operations with `gpu::gen::op` inside `gpu::precompile_op`
- The original submodule is passed directly to `gpu::gen::op`

**`gpu::gen::op`**: A wrapper operation similar to `mlir_op` that contains a submodule with high-level tensor operations (pointwise, reduce, pad, gather, etc.). The tiling and lower passes generate the appropriate tiles and loads/stores based on the operations in the submodule.

### 2. `gen_tiling` Pass

**Location:** `tiling.cpp`

This pass inserts tiling operations for multi-dimensional tensors.

**What it does:**
- Finds `gpu::gen::copy` operations
- Computes optimal tile configuration based on tensor shapes
- Inserts `workgroup_id` and `tile_region` operations
- Updates copy inputs to use tiled regions

**Before:**
```
x = @param:x -> float_type, {64, 32, 128}
z = @param:z -> float_type, {64, 32, 128}
copy = gpu::gen::copy(x, z)
```

**After:**
```
x = @param:x -> float_type, {64, 32, 128}
z = @param:z -> float_type, {64, 32, 128}
wg_id = gpu::gen::workgroup_id()
x_tile = gpu::gen::tile_region[tile_dims=[32, 64], axis=1](x, wg_id)
z_tile = gpu::gen::tile_region[tile_dims=[32, 64], axis=1](z, wg_id)
copy = gpu::gen::copy(x_tile, z_tile)
```

### 3. `gen_lower` Pass

**Location:** `lower.cpp`

This pass lowers copy operations to explicit vector load/store instructions.

**What it does:**
- Finds `gpu::gen::copy` operations
- Determines vectorization width based on shape alignment
- For tiled copies (inputs are `tile_region`): uses `local_id`
- For non-tiled copies: uses `global_id`
- Inserts `vector_load` and `vector_store` operations

**Before:**
```
copy = gpu::gen::copy(x_tile, z_tile)
```

**After:**
```
lid = gpu::gen::local_id()
load = gpu::gen::vector_load[size=4](x_tile, lid)
store = gpu::gen::vector_store[size=4](z_tile, lid, load)
```

## Code Generation

**Location:** `codegen.cpp`

The code generation uses `cpp_generator` to produce C++ kernel code from the lowered gen IR.

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

### Key Components

1. **`generate_gen_function`**: Uses `cpp_generator::generate_module()` to create a `__device__` function from the IR
2. **`compile_gen`**: Compiles a gen IR program to a GPU code object
3. **`gen_compiler`**: Registered compiler for `gpu::gen::op` operations

## Kernel-Side Helpers

**Location:** `migraphx/kernels/gen.hpp`

Helper functions used by generated code:

```cpp
namespace migraphx::gen {
    // Compute linear offset from multi-dimensional index
    template<class Shape>
    __device__ auto compute_offset(Shape shape, index_int i);

    // Index transformations
    template<class Shape, class Pads>
    __device__ auto pad_index(Shape shape, Pads pads, index_int i);

    template<class Shape, class Axes>
    __device__ auto reverse_index(Shape shape, Axes axes, index_int i);

    template<class Shape, class Indices, index_int Axis>
    __device__ auto gather_index(Shape shape, Indices indices, index_int i);

    // Conditional load with bounds check
    template<class Tensor, class Fill>
    __device__ auto conditional_load(Tensor tensor, int64_t offset, Fill fill);

    // Vectorized memory access
    template<index_int N, class T>
    __device__ auto vec_load(T* data, index_int offset);

    template<index_int N, class T, class V>
    __device__ void vec_store(T* data, index_int offset, V value);

    // Strided load (single element at base + iter * stride)
    template<class T>
    __device__ auto strided_load(T* data, index_int base, index_int iter, index_int stride);

    template<class T, class V>
    __device__ void strided_store(T* data, index_int idx, V value);

    // Per-lane accumulation
    template<class T>
    __device__ auto accumulate_sum(T acc, T val);
    template<class T>
    __device__ auto accumulate_product(T acc, T val);
    template<class T>
    __device__ auto accumulate_max(T acc, T val);
    template<class T>
    __device__ auto accumulate_min(T acc, T val);

    // Reduction operations
    template<class T>
    __device__ auto dpp_reduce_sum(T x);      // Wave-level sum
    
    template<class T>
    __device__ auto dpp_reduce_product(T x);  // Wave-level product
    
    template<class T>
    __device__ auto dpp_reduce_max(T x);      // Wave-level max
    
    template<class T>
    __device__ auto dpp_reduce_min(T x);      // Wave-level min

    // Block-level reductions (across wavefronts)
    template<class T>
    __device__ auto block_reduce_sum(T x, T* lds, index_int nwaves, 
                                     index_int wave_id, index_int lane_id);
    
    template<class T>
    __device__ auto block_reduce_max(T x, T* lds, index_int nwaves,
                                     index_int wave_id, index_int lane_id);
    // ... similar for product, min
}
```

## Adding New Operations

To add a new gen operation:

1. **Define the operation** in `ops.cpp`:
```cpp
struct my_new_op
{
    std::string name() const { return "gpu::gen::my_new_op"; }
    
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        // Return output shape
    }
    
    value attributes() const
    {
        return {{"point_op", "my_generated_code(${0}, ${1})"}};
    }
};
MIGRAPHX_REGISTER_OP(my_new_op)
```

2. **Handle in codegen** if needed in `codegen.cpp`:
```cpp
if(ins->name() == "gpu::gen::my_new_op")
{
    return "gen::my_function(" + args[0] + ", " + args[1] + ")";
}
```

3. **Add kernel-side helper** if needed in `gen.hpp`:
```cpp
template<class T>
__device__ auto my_function(T a, T b) { ... }
```

## Testing

Tests are located in `test/gpu/gen/`:

- `ops_test.cpp`: Unit tests for individual operations
- `tiling_test.cpp`: Tests for the tiling pass
- `lower_test.cpp`: Tests for the lower pass
- `codegen_test.cpp`: Tests for code generation
- `fuse_gen_test.cpp`: Tests for the fusion pass

Verify tests in `test/verify/`:
- `test_gen_pad_add.cpp`: Pad + add fusion
- `test_gen_reverse_add.cpp`: Reverse + add fusion
- `test_gen_gather_add.cpp`: Gather + add fusion

## File Structure

```
src/targets/gpu/gen/
├── CMakeLists.txt
├── README.md                 # This file
├── include/migraphx/gpu/gen/
│   ├── export.h              # Export macros
│   ├── codegen.hpp           # Code generation API
│   ├── fuse_gen.hpp          # Fusion pass
│   ├── tiling.hpp            # Tiling pass and tile_config
│   └── lower.hpp             # Lowering pass
├── ops.cpp                   # All gen operation definitions
├── fuse_gen.cpp              # Fusion pass implementation
├── tiling.cpp                # Tiling pass implementation
├── lower.cpp                 # Lowering pass implementation
└── codegen.cpp               # Code generation and compiler
```

