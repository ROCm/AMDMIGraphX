---
applyTo: "src/targets/gpu/kernels/**/*.hpp"
---

# GPU Kernel Review Instructions

- Use `migraphx::array` instead of raw C arrays
- Use tensor_view or iterators instead of raw pointers
- Use the index calculation from tensor_view or shape instead of raw divisions and mod. Try to make slices of tensor_view/shape for tiling instead of manually calculating offset
- Use `uninitialized_buffer` for shared LDS memory
- Use the `vec<T, N>` type to declare vector types
- Use `repeat_c` for unrolling loops
- Use the `index` class to read the local and global index instead of `threadIdx.x` or `blockIdx.x`.
- Use the `local_stride`, `global_stride` or `local_wave_stride` to do strided loops instead of manually writing them.
- Don't use `::value` to read integral constant values, they already implicitly convert to an integer
  - Try to pass integral_constants as much as possible. Don't assign them to int variables, declare the variable as `auto` so it can capture integral constant.
  - `foo(ic)` or `array<T, ic>` already work

