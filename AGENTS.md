# Repository Guidelines

This file provides guidance for agents and contributors working with code in this repository.

## Project Overview

MIGraphX is AMD's graph inference engine that accelerates machine learning model inference on ROCm. It uses a three-level IR hierarchy:
- **Programs** (`migraphx::program`): Top-level containers for entire neural networks
- **Modules** (`migraphx::module`): DAGs of instructions representing computational subgraphs
- **Instructions** (`migraphx::instruction`): Atomic operations wrapping an `operation` with inputs

The compilation flow transforms high-level models (ONNX, TensorFlow) through optimization passes into target-specific code (GPU kernels via HIP/MIOpen/rocBLAS, CPU ops, or reference implementations).

## Dependencies

**ROCm ecosystem:** ROCm, MIOpen, rocBLAS, hipBLASLt, HIP (paths under `/opt/rocm`)
**Core libraries:** Protobuf (ONNX), Half (IEEE 754 fp16), pybind11 (Python), nlohmann/json, MessagePack, SQLite3
**Build tools:** CMake 3.15+, rbuild (optional but recommended), ccache (via rbuild develop)
**Language level:** C++17 (set in CMake)

## Project Structure & Module Organization

- `src/include/migraphx/*.hpp` - Public API headers
- `src/op/` - Operation implementations
- `src/targets/gpu/` - GPU backend (HIP, MIOpen, rocBLAS lowering)
- `src/targets/cpu/` - CPU backend
- `src/targets/ref/` - Reference implementation
- `src/driver/` - CLI tools (main.cpp, perf.cpp, verify.cpp, passes.cpp)
- `src/py/` - Python bindings (pybind11)
- `test/` - Unit and integration tests (mirrors `src/` structure)
- `test/gpu/` - GPU-specific tests (build as `test_gpu_<topic>`)
- `test/verify/` - End-to-end verification tests
- `test/onnx/` - ONNX model parsing/verification
- `tools/` and `cmake/` - Build helpers, scripts, format scripts, CMake modules
- `examples/` - Usage examples (C++ API, Python, diffusion, transformers, vision)
- `docs/` - Sphinx/Doxygen documentation; build artifacts in `docs/_build/`
- `build/` - Default generated build output

## Build Commands

**Development build (recommended):**
```bash
rbuild develop -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
```

**Manual CMake build:**
```bash
cmake -S . -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
make -C build -j$(nproc)
```

**Build with dependencies (rbuild):**
```bash
rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
```

**Run all tests:**
```bash
make -C build check
```

**Install:**
```bash
make -C build install
```

**Build documentation:**
```bash
make -C build doc
# Or manually in docs/:
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

**Format code:**
```bash
python3 tools/format.py -i
```

**Install git hooks:**
```bash
./.githooks/install
```

## Coding Standards

### Naming Conventions

- **Files and symbols**: `snake_case` (e.g., `compute_shape`, `eliminate_concat.cpp`)
- **Template parameters**: `CamelCase`
- **Macros**: `UPPER_CASE` with `MIGRAPHX_` prefix
- Public headers go under `src/include/migraphx/*.hpp`
- Prefer clear names over abbreviations
- Use `using type_alias = int` not `typedef int type_alias`
- Use `nullptr` not `NULL` or `0`
- Use `or` and `and` instead of `&&` and `||`

### Formatting (.clang-format)

- 4-space indent, no tabs
- 100 column limit
- Braces on new line for classes/functions/control structures
- Align consecutive assignments

### Linting

- `.clang-tidy` runs via CMake; treat warnings seriously (they are enforced in CI)
- `cppcheck` is enabled with custom rules

### Style Guidelines

- **Avoid raw loops** - Prefer algorithms from `<migraphx/algorithm.hpp>` and STL `<algorithm>`:
  ```cpp
  // Good: Declarative, expresses intent clearly
  std::transform(in.begin(), in.end(), std::back_inserter(out),
                 [](int i) { return i * 2; });

  // Avoid: Raw loops are error-prone and less optimizable
  for (int i : in) { out.push_back(i * 2); }

  // Note: std::for_each should NOT be used as it doesn't encapsulate a raw loop
  // It's just a wrapper around a for loop - use std::transform or other algorithms instead
  ```

  **MIGraphX-specific algorithms** (`#include <migraphx/algorithm.hpp>`):
  - `transform_if(start, last, out, pred, f)` - Transform with filtering
  - `transform_accumulate(first, last, init, binop, unaryop)` - Accumulate with projection
  - `transform_partial_sum(first, last, out, binop, unaryop)` - Partial sum with projection
  - `group_by(start, last, out, pred)` - Group elements by predicate
  - `group_unique(start, last, out, pred)` - Group consecutive unique elements
  - `group_find(start, last, pred, out)` - Find and group matching ranges
  - `adjacent_remove_if(first, last, pred)` - Remove based on adjacent pairs
  - `adjacent_for_each(first, last, f)` - Iterate over adjacent pairs
  - `for_each(first1, last1, first2, f)` - Two-range iteration (like std::transform)

  **If no suitable algorithm exists:** Add a new algorithm to `migraphx/algorithm.hpp` rather than using raw loops

- **Memory management** - Use `std::make_unique/shared`, avoid raw `new`/`delete`
- **Non-memory resources** - Use `MIGRAPHX_MANAGE_PTR` for C-style acquire/release APIs:
  ```cpp
  using file_ptr = MIGRAPHX_MANAGE_PTR(FILE*, fclose);
  file_ptr f{fopen(filename, "r")};  // Automatically closed on scope exit
  ```

- **Type erasure over inheritance** - MIGraphX uses type erasure extensively for `pass`, `operation`, `target`
  - No need to inherit from base class - just implement required interface methods
  - Generate boilerplate with `make generate` when interfaces change
  - Enables value semantics with polymorphism, no virtual inheritance overhead

- **Pass design principles:**
  - Keep passes idempotent (running twice = running once)
  - Keep passes deterministic (same input → same output)
  - Always handle dynamic shapes in `compute_shape` (unknown dimensions)

- **Generic programming** - Write reusable, type-independent code using templates and STL algorithms

- **Avoid Casts** - Don't use a cast unless absolutely necessary. Declare correct types. Casts indicate a type mismatch that should be resolved at the source.

- **Encapsulate Bit Manipulation** - Put bit-twiddling and low-level operations behind well-named utility functions:
    - Prefer `std::bitset` for bit manipulation

- **Use std::tie for Lexicographical Comparisons** - Use `std::tie` or `std::lexicographical_compare` instead of manually writing lexicographical comparisons.

- **Use shape class to compute offsets and indexing** - The `migraphx::shape` class provides methods for computing offsets, strides, and indexing. Use these instead of manual calculations with mod and division.

## Code Quality

- Prefer correct, complete implementations over minimal ones.
- Use appropriate data structures and algorithms — don't brute-force what has a known better solution.
- When fixing a bug, fix the root cause, not the symptom.
- If something I asked for requires error handling or validation to work reliably, include it without asking.

## Code Architecture

### Core IR Components

**Program (`src/include/migraphx/program.hpp`):**
- Contains one or more modules
- Manages compilation via `program::compile(target)`
- Entry point: `get_main_module()` returns main execution module

**Module (`src/include/migraphx/module.hpp`):**
- DAG of instructions
- Add instructions via `module::add_instruction(op, inputs...)`
- Add parameters via `module::add_parameter(name, shape)`

**Instruction (`src/include/migraphx/instruction.hpp`):**
- References an operation and input instructions
- Mutable IR node/reference; passes may rewrite instructions in place via APIs such as `replace`, `set_normalized`, and `set_target_id`

**Operation (`src/include/migraphx/operation.hpp`):**
- Defines computation semantics
- Must implement: `name()`, `compute_shape(inputs)`
- Optionally: `compute(ctx, output, inputs)`, `finalize(ctx, shape, inputs)`

### Backends

A `target` provides `get_passes`, `get_context`, and memory ops; see GPU under `src/targets/gpu/` and CPU under `src/targets/cpu/`.

### Compilation Pipeline

1. Parse model (ONNX/TF) → Generic IR
2. Run optimization passes (`pass_manager.cpp`)
3. Target lowering (`targets/{gpu,cpu}/`) applies backend-specific transformations
4. Code generation (GPU: kernel fusion, HIP code objects; CPU: direct ops)
5. Execution via `program::eval(params)` or `program::run(params)`

## Core Data Structures

### Shape, Literal, Argument, and Tensor View

**`migraphx::shape`** - Describes tensor properties (one of the most important classes):
- Data type (e.g., `float_type`, `int32_type`)
- Dimensions/lengths (e.g., `{1, 3, 224, 224}`)
- Memory layout/strides
- States: `.packed()`, `.transposed()`, `.broadcasted()`, `.standard()`

**`migraphx::literal`** - Immutable data buffer with a shape (compile-time constants like weights)

**`migraphx::argument`** - Mutable data buffer with a shape (runtime results and inputs)

**`migraphx::tensor_view<T>`** - Type-safe multi-dimensional view combining `T*` pointer with a shape

**Visitor pattern for type-safe data access:**
```cpp
migraphx::literal l{migraphx::shape{migraphx::shape::float_type, {2, 2}}, {1.0f, 2.0f, 3.0f, 4.0f}};
l.visit([](auto view) {
    // 'view' is a tensor_view<float>
    std::cout << view(1, 1) << std::endl; // Prints 4.0
});
```

### Parameters and Evaluation

Add parameters to modules for runtime inputs:
```cpp
migraphx::shape s{migraphx::shape::int32_type, {1}};
auto x = mm->add_parameter("x", s);  // Creates placeholder instruction

// Later, evaluate with actual data:
migraphx::parameter_map params;
std::vector<int> data = {4};
params["x"] = migraphx::argument(s, data.data());
auto result = p.eval(params).back();
```

## Extension Patterns

### Adding an Operation

1. **Create operation struct** in `src/include/migraphx/op/<name>.hpp`:
```cpp
struct my_op {
    std::string name() const { return "my_op"; }

    shape compute_shape(std::vector<shape> inputs) const {
        // Compute output shape from inputs
        return inputs[0];
    }

    // Optional: for reference target
    argument compute(context& ctx, const shape& output,
                     std::vector<argument> args) const {
        // Compute implementation
    }
};
```

2. **Register operation** (automatic via include pattern or explicit):
```cpp
MIGRAPHX_REGISTER_OP(my_op)
```

3. **Add tests** in `test/op_shape_test.cpp` or `test/my_op_test.cpp`

Place headers in `src/include/migraphx/op/` and implementation in `src/` if not header-only. Reflect attributes if needed.

### Adding an Optimization Pass

**Simple implementation:**
```cpp
#include <migraphx/pass.hpp>

struct my_pass {
    std::string name() const { return "my_pass"; }

    void apply(module& m) const {
        // Transform module instructions
    }
    // Or use module_pass_manager:
    // void apply(module_pass_manager& mpm) const;
};
```

**Using matchers (preferred for pattern matching):**

Create a matcher struct with `matcher()` and `apply()` methods:
```cpp
namespace {
struct find_mul_by_one
{
    auto matcher() const
    {
        // Match: mul(1, x) or mul(x, 1)
        return match::name("mul")(match::either_arg(0, 1)(
            match::is_constant(1.0f),
            match::any().bind("x")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto x = r.instructions["x"];
        m.replace_instruction(ins, x);
    }
};
} // namespace

void simplify_mul::apply(module& m) const
{
    match::find_matches(m, find_mul_by_one{});
}
```

**Built-in matchers:**
- `match::name("op_name")` - Match operation by name
- `match::any()` - Match any instruction
- `match::is_constant(value)` - Match literal with specific value
- `match::args(m1, m2, ...)` - Match positional arguments
- `match::arg(i)(m)` - Match i-th argument
- `match::either_arg(0, 1)(m1, m2)` - Match either argument order
- `.bind("name")` - Bind matched instruction to retrieve later

**Custom matchers:**
```cpp
MIGRAPHX_PRED_MATCHER(broadcasted_shape, instruction_ref ins) {
    return ins->get_shape().broadcasted();
}
// Usage: auto m = broadcasted_shape();
```

**Integration:**
1. Wire into target's `get_passes()` method
2. For driver testing: register in `src/driver/passes.cpp`
3. Add tests mirroring `test/eliminate_concat_test.cpp`

### Adding a Backend Target

1. **Implement target interface** in `src/targets/mytarget/`:
```cpp
struct mytarget {
    std::string name() const { return "mytarget"; }

    std::vector<pass> get_passes(context& ctx,
                                  const compile_options& options) const {
        // Return optimization passes
    }

    context get_context() const {
        // Return execution context
    }

    // Memory operations
    argument allocate(const shape& s) const;
    argument copy_to(const argument& arg) const;
    argument copy_from(const argument& arg) const;
};
```

2. **Register target:**
```cpp
MIGRAPHX_REGISTER_TARGET(mytarget)
```

3. **Add target-specific tests** in `test/mytarget/`

## Testing Guidelines

### Framework

- CTest-driven executables generated from files in `test/` (see `test/CMakeLists.txt`)
- Naming: prefer `*_test.cpp`; CMake auto-creates an executable `test_<topic>` and registers it with CTest
- Colocate helpers under `test/include/` when needed
- Coverage: CI uploads via Codecov; keep unit tests focused and fast

### Test Directory Organization

- `test/` - Core unit tests (no specific backend), target-independent
- `test/gpu/` - GPU-specific tests (requires `MIGRAPHX_ENABLE_GPU=On` and a ROCm device); build as `test_gpu_<topic>`
- `test/ref/` - Reference operator behavior tests
- `test/verify/` - Multi-target result comparison (GPU vs reference)
- `test/onnx/parse/` - ONNX model parsing tests (no execution)
- `test/onnx/verify/` - ONNX model execution + numerical comparison
- `test/tf/` - TensorFlow parser tests
- `test/api/` - C/C++ API validation
- `test/py/` - Python binding tests

### Running Tests

```bash
# All tests
make -C build check

# Single test file
ctest -R '^test_<topic>$'

# Single test case with pattern
./bin/test_<topic> 'pattern*'

# GPU tests only
ctest -R '^test_gpu_'

# List all test cases in a file
./bin/test_<topic> --list
```

### Unit Tests

**Basic test structure:**
```cpp
#include "test.hpp"  // From test/include/

TEST_CASE(my_feature) {
    EXPECT(condition);
    CHECK(other_condition);
}
```

**Module/Program test pattern:**
```cpp
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include "test.hpp"

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::my_pass{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(test_my_pass)
{
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", {migraphx::shape::float_type, {3, 3}});
        auto result = m1.add_instruction(migraphx::make_op("relu"), input);
        m1.add_return({result});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input = m2.add_parameter("input", {migraphx::shape::float_type, {3, 3}});
        auto result = m2.add_instruction(migraphx::make_op("relu"), input);
        m2.add_return({result});
    }

    EXPECT(m1 == m2);
}
```

### Testing Passes

When adding a pass that rewrites structure:
1. Build a small graph with the pattern to optimize
2. Run only the new pass (plus prerequisites and dead_code_elimination for cleanup)
3. Create a new module with the expected final structure and assert that it is the same

Use driver for inspection: `migraphx-driver compile --text --apply-pass your_pass`

### Numerical Verification

- Use `verify_rms_range` for tolerance-based comparisons with floating-point noise
- Provide deterministic inputs (explicit literals or seeded random generators)
- Avoid flaky thresholds by using stable, reproducible test data

### Verify tests

- Tests under `test/verify` will verify the results using the ref target
- Each verify class should go into a separate .cpp file

### Test Best Practices

**Speed principles:**
- Favor shapes < 64 elements for scalar algorithms
- Use single-digit iteration counts
- Reuse shapes across tests to exploit compiler cache hits
- To skip large-buffer tests, configure `-DMIGRAPHX_DISABLE_LARGE_BUFFER_TESTS=On`

**Edge cases to cover:**
- Zero-length dimensions
- Dynamic shapes at extreme min/max
- Mixed type promotion (int + float → float)
- Broadcasting asymmetries
- Reduction axes ordering

**Regression strategy:**
- When fixing subtle bugs (e.g., stride miscalculation), add test reproducing original failure
- Keep golden ONNX/TF assets minimal (large models slow CI and obscure failures)
- For new operator parsing, a model containing just that op suffices

**Test utilities:** `test/include/test.hpp`, `basic_ops.hpp`, `pointwise.hpp`, `reduce.hpp`

## GPU Backend Details

The GPU backend (`src/targets/gpu/`) implements:

- **Lowering** (`lowering.cpp`): Transform generic ops to GPU-specific ops
- **Fusion** (`fuse_ops.cpp`, `fuse_ck.cpp`; shared pointwise fusion in `src/fuse_pointwise.cpp`): Kernel fusion optimizations
- **Code generation** (`compile_hip*.cpp`, `compile_miopen.cpp`): HIP/MIOpen/rocBLAS dispatch
- **Scheduling** (`schedule_model.cpp`): Select optimal implementation
- **Performance DB** (`perfdb.cpp`, `problem_cache.cpp`): Cache tuning results

**Key subdirectories:**
- `device/` - Device-side utilities and kernels
- `kernels/` - Custom kernel implementations
- `include/migraphx/gpu/` - GPU-specific headers

### Fusion Strategies

MIGraphX employs multiple fusion stages to reduce kernel overhead and improve memory locality:

**1. Prefusion (GPU-specific):**
- Replace recognizable operator clusters with custom placeholders before lowering
- Passes: `prefuse_ops`, `fuse_attention`
- Detects high-level patterns like transformer attention (Q*K^T scaling, softmax, V multiplication)

**2. Pointwise/Reduce Fusion:**
- `fuse_pointwise`: Groups pure elementwise chains into submodules for single-kernel code generation
- `fuse_reduce` / `fuse_pointwise_reduce`: Combines reductions sharing compatible axes
- Flags: `enable_rewrite_reshapes`, `enable_rewrite_broadcasts`, `enable_multi_output`
- Critical for layernorm and attention patterns

**3. Backend Post-Lowering Fusion:**
- `fuse_ops`: Combine lowered ops with adjacent pointwise or layout adjustments
- `fuse_ck`: Leverage Composable Kernel library for GEMM + epilogues (bias, activation)
- `fuse_mlir`: Groups pointwise + reshape sequences around GEMM/Conv for MLIR pipeline

**Troubleshooting missing fusion:**
- Check if earlier passes canonicalized shapes (`simplify_reshapes`)
- Verify broadcasts are explicit/implicit as expected
- Check if dynamic dims prevent grouping
- Ensure op type lists support new data types (fp8/bf16)
- Use `MIGRAPHX_TRACE_COMPILE=1` to confirm fusion pass execution

### Target Compilation Flow

**Target object provides:**
- `get_passes(context&, compile_options&)` - Ordered backend-specific pass list
- Allocation model and context types
- Lowering rules mapping high-level ops to target intrinsics/library calls

**Compile options control:**
- `fast_math` - Relaxed math transformations
- `exhaustive_tune` - Full search for fastest GEMM/convolution implementation
- Quantization toggles (fp16/int8/fp8) for op substitution

**Memory planning:**
- `memory_coloring` - Assign buffers to reuse memory where lifetimes don't overlap
- `adjust_allocation` / `replace_allocate` - Coordinate with target allocation models
- `schedule` - Order instructions for optimal overlap/locality (disable: `MIGRAPHX_DISABLE_SCHEDULE_PASS=1`)

**Library integration:**
- GEMM/DOT: HIPBLASLt or Composable Kernel (`MIGRAPHX_SET_GEMM_PROVIDER`)
- Convolution: MIOpen if available (`MIGRAPHX_USE_MIOPEN`), else custom kernels
- Precision handling: `propagate_precision`, `rewrite_low_precision` for mixed precision (fp16/fp8/bf16)

## Type Erasure System

MIGraphX uses type erasure extensively to create extensible systems without traditional inheritance. This pattern is used for `pass`, `operation`, `target`, `instruction`, and other core abstractions. It enables value semantics with polymorphic behavior, no virtual inheritance overhead.

### Creating New Type-Erased Interfaces

**1. Define interface template** in `tools/include/<name>.hpp`:
```python
<%
interface('pass',
    virtual('name', returns='std::string', const=True),
    virtual('apply', returns='void', mpm='module_pass_manager &', const=True,
            default='migraphx::detail::module_pass_manager_apply'),
    virtual('apply', returns='void', p='program &', const=True,
            default='migraphx::nop')
)
%>
```

**Syntax explanation:**
- `interface('pass', ...)` - Defines type-erased interface named `pass`
- `virtual(...)` - Defines one method in the interface
  - First arg: method name (e.g., `'name'`)
  - `returns='...'` - C++ return type
  - `const=True` - Method is const
  - Method arguments as keyword args: `p='program &'`
  - `default='...'` - Optional default implementation fallback

**2. Generate boilerplate:**
```bash
make generate  # Runs tools/te.py on templates in tools/include/migraphx/
```

**3. Generated header** placed in `src/include/migraphx/<name>.hpp` contains wrapper class with concept/model pattern for virtual dispatch.

### Using Type-Erased Objects

No inheritance required - just implement the interface:
```cpp
struct my_custom_pass {
    std::string name() const { return "my_custom_pass"; }
    void apply(module& m) const { /* transformation logic */ }
};

// Use directly:
migraphx::pass p = my_custom_pass{};  // Value semantics!
```

## Frontend Parsers

### ONNX Parser (`src/onnx/`)

- Parses ONNX models into MIGraphX IR
- Located in `src/onnx/parse_*.cpp` files
- Each ONNX operator has a corresponding parse function

**Adding a new ONNX operator:**
1. Add parse function in `src/onnx/parse_<op>.cpp`
2. Register in ONNX parser operator map
3. Map ONNX attributes to MIGraphX operation parameters
4. Handle shape inference and broadcasting
5. Add test in `test/onnx/parse/` (parsing only) and `test/onnx/verify/` (execution)

**TensorFlow Parser** follows similar pattern in `src/tf/`.

## Python Bindings

Located in `src/py/migraphx_py.cpp` using pybind11.

**Basic pattern:**
```python
import migraphx

# Parse model
model = migraphx.parse_onnx("model.onnx")

# Compile for target
model.compile(migraphx.get_target("gpu"))

# Run inference
results = model.run({"input": input_data})
```

When adding C++ APIs, mirror in Python bindings following existing patterns.

## migraphx-driver CLI Tool

The driver (`./bin/migraphx-driver`) is essential for development, debugging, and testing without writing C++ code.

### Commands

| Command | Description |
|---------|-------------|
| `op -l/--list` | Print all available MIGraphX operators |
| `params` | Print input/output parameter shapes |
| `read` | Load and print input graph |
| `compile` | Compile and print input graph |
| `run` | Compile, allocate parameters, evaluate, and print graph |
| `verify` | Run on reference and GPU, check output consistency |
| `perf` | Compile and run, then print performance report |

### Key Options

| Option | Description |
|--------|-------------|
| `--onnx` | Load file as ONNX graph |
| `--tf` | Load file as TensorFlow graph |
| `--migraphx` | Load file as MIGraphX graph |
| `--gpu` / `--cpu` / `--ref` | Target backend for compilation |
| `--fp16` / `--int8` | Quantization mode |
| `--text` / `--json` / `--binary` | Output format |
| `--batch N` | Set batch size for model |
| `--input-dim @input 1 3 224 224` | Set static dimensions |
| `-n N` / `--iterations N` | Number of iterations for perf |
| `--fill0` / `--fill1` | Fill parameters with 0s or 1s |
| `-o FILE` / `--output FILE` | Write output to file |

### Usage Examples

**Compile and inspect IR:**
```bash
migraphx-driver compile model.onnx --gpu --text
```

**Apply a driver pass:**
```bash
migraphx-driver compile model.onnx --apply-pass dead_code_elimination --text
```

**Performance benchmarking:**
```bash
migraphx-driver perf model.onnx --gpu -n 100
```

**Verify numerical correctness:**
```bash
migraphx-driver verify model.onnx --atol 1e-5 --rtol 1e-5
```

**Register custom passes:** Add to `src/driver/passes.cpp` for driver access.

## Debugging & Tracing

**Environment variables:**
- `MIGRAPHX_TRACE_COMPILE=1` - Trace compilation passes
- `MIGRAPHX_TRACE_EVAL=1` - Trace evaluation
- `MIGRAPHX_DISABLE_SCHEDULE_PASS=1` - Disable scheduling for debugging

**Performance reporting:**
- `program.perf_report(std::ostream&, iterations, parameter_map)` - Get timing stats
- Driver: `migraphx-driver perf --onnx model.onnx --gpu -n 50`

**IR inspection:**
- `program.debug_print()` - Print IR to stdout
- GraphViz output via `graphviz.cpp` helpers
- ROCm profiling markers via `marker_roctx.*`

## Key CMake Options

- `MIGRAPHX_ENABLE_GPU` - Enable GPU backend (default: ON if HIP available)
- `MIGRAPHX_ENABLE_CPU` - Enable CPU backend (default: OFF)
- `MIGRAPHX_ENABLE_PYTHON` - Build Python bindings (default: ON)
- `MIGRAPHX_USE_MIOPEN` - Use MIOpen library (default: ON)
- `MIGRAPHX_USE_ROCBLAS` - Use rocBLAS library (default: ON)
- `MIGRAPHX_USE_HIPBLASLT` - Use hipBLASLt (default: ON, requires rocBLAS)
- `MIGRAPHX_USE_COMPOSABLEKERNEL` - Use composable kernel JIT (default: ON)
- `MIGRAPHX_DISABLE_LARGE_BUFFER_TESTS` - Skip large-buffer tests
- `BUILD_DEV` - Development mode with extra checks
- `GPU_TARGETS` - Target GPU architectures (use detection snippet)

**Example:** `cmake -S . -B build -DMIGRAPHX_ENABLE_GPU=On -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')`

## Common Issues

- **Missing GPU_TARGETS**: Always specify via `$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')`
- **Dynamic shape errors**: Ensure `compute_shape()` handles unknown dimensions
- **Pass ordering**: Passes have dependencies - check existing target pipelines
- **Kernel packaging**: New GPU kernels must be included in CMake install targets
- **Build failures**: Check clang-tidy warnings - they're enforced in CI

## Reference Materials

**When implementing new features, search for analogous implementations:**
- Operations: `grep -r "similar_op" src/op/`
- Passes: `grep -r "similar_pass" src/`
- GPU lowering: Check `src/targets/gpu/lowering.cpp`
- Tests: Mirror structure from `test/<similar_feature>_test.cpp`
- Matchers: See existing patterns in `src/matcher.cpp`
- ONNX ops: Look at similar operators in `src/onnx/parse_*.cpp`

**Key header files to understand:**
- `src/include/migraphx/program.hpp` - Top-level container
- `src/include/migraphx/module.hpp` - DAG of instructions
- `src/include/migraphx/instruction.hpp` - Graph nodes
- `src/include/migraphx/operation.hpp` - Operation abstraction
- `src/include/migraphx/shape.hpp` - Tensor shape/layout
- `src/include/migraphx/pass.hpp` - Pass interface
- `src/include/migraphx/matcher.hpp` - Pattern matching DSL

**Documentation:**
- Official docs: `docs/` directory (Sphinx + Doxygen)
- Developer guides: `docs/` directory (detailed contributor documentation)
- Build documentation: `make -C build doc`

## Commit & Pull Request Guidelines

- **Commits:** Use imperative, concise subject lines (optionally add a scope), e.g., `fix: correct hip stream handling`. Reference issues/PRs (`Fixes #123`).
- **PRs:** Include problem statement, approach, and risk notes; link issues. Show test output for affected areas and add tests for new behavior. Ensure formatting/linters pass and CI is green.

## Running Commands

- Do not use `$(nproc)`, run `nproc` first, then use the numeric value directly.
- When running `make -j$(nproc)`, first resolve `$(nproc)` to a literal number, e.g. run `nproc` first, then use the numeric value directly in the make command (i.e., `make -j32` if `nproc` returns `32`).

## Workspace Edit Rules

- When creating *any* temporary/scratch/generated file, write it under `tmp/` at the workspace root.
- Use `tmp/<task-slug>/<YYYYMMDD-HHMMSS>/` for groups of files.
- Never place temp files outside `tmp/` or inside source directories.
