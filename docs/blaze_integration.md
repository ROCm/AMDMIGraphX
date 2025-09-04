# Blaze Integration with MIGraphX

This document describes how to integrate and use the [Blaze](https://blaze-lib.org/) high-performance math library with MIGraphX.

## Overview

Blaze is a high-performance C++ math library for dense and sparse arithmetic. The integration allows MIGraphX users to leverage Blaze's optimized linear algebra operations for enhanced performance in specific scenarios.

## Building MIGraphX with Blaze Support

### Prerequisites

1. Install the Blaze library following the instructions at https://blaze-lib.org/
2. Ensure Blaze is in your `CMAKE_PREFIX_PATH` or provide the path during configuration

### Configuration

To enable Blaze support when building MIGraphX:

```bash
cmake -DMIGRAPHX_USE_BLAZE=ON -DCMAKE_PREFIX_PATH=/path/to/blaze/installation ..
```

Or if using UAI build system:
```bash
uai configure -DMIGRAPHX_USE_BLAZE=ON -DCMAKE_PREFIX_PATH=/path/to/blaze/installation
```

### Building

```bash
uai build
```

## Usage

### Basic Usage

```cpp
#include <migraphx/blaze_utils.hpp>
#include <migraphx/operators.hpp>

// Check if Blaze support is available
if (migraphx::blaze_utils::is_blaze_available()) {
    // Create MIGraphX tensors
    auto shape_a = migraphx::shape{migraphx::shape::float_type, {3, 4}};
    auto shape_b = migraphx::shape{migraphx::shape::float_type, {4, 2}};
    auto shape_c = migraphx::shape{migraphx::shape::float_type, {3, 2}};
    
    // ... initialize tensors ...
    
    // Convert to tensor views
    auto view_a = arg_a.get<float>();
    auto view_b = arg_b.get<float>();
    auto view_c = arg_c.get<float>();
    
    // Perform optimized matrix multiplication using Blaze
    migraphx::blaze_utils::blaze_gemm(view_c, view_a, view_b);
    
    // Convert MIGraphX tensors to Blaze matrices for advanced operations
    auto blaze_matrix = migraphx::blaze_utils::to_blaze_matrix(view_a);
    auto norm = blaze::norm(blaze_matrix);
}
```

### Advanced Usage

The integration provides several utility functions:

- `to_blaze_matrix()`: Convert 2D MIGraphX tensor_view to Blaze matrix
- `to_blaze_vector()`: Convert 1D MIGraphX tensor_view to Blaze vector  
- `blaze_gemm()`: Optimized matrix multiplication using Blaze
- `is_blaze_available()`: Runtime check for Blaze support

### Example

See `examples/blaze_integration_example/` for a complete working example.

## Performance Considerations

- Blaze operations create views of MIGraphX tensor data (no copying)
- Best performance is achieved with properly aligned, contiguous tensor data
- Consider using Blaze for compute-intensive linear algebra operations
- The integration is most beneficial for large matrix operations

## Configuration Macros

- `MIGRAPHX_USE_BLAZE`: Defined when Blaze support is enabled (1) or disabled (0)
- Available in C++ code via `#include <migraphx/config.h>`

## Troubleshooting

### Blaze not found during build

Ensure Blaze is properly installed and its installation directory is in `CMAKE_PREFIX_PATH`:

```bash
export CMAKE_PREFIX_PATH=/path/to/blaze/installation:$CMAKE_PREFIX_PATH
```

### Runtime issues

Verify Blaze support is enabled:
```cpp
assert(migraphx::blaze_utils::is_blaze_available());
```

### Performance issues

- Ensure tensors are contiguous in memory
- Use appropriate Blaze matrix storage order (row-major vs column-major)
- Profile to determine if Blaze operations provide benefit for your workload
