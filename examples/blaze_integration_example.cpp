/*
 * Example demonstrating MIGraphX integration with Blaze library
 * 
 * This example shows how to:
 * 1. Enable Blaze support in MIGraphX
 * 2. Convert MIGraphX tensors to Blaze matrices
 * 3. Perform optimized matrix operations using Blaze
 * 4. Use Blaze's high-performance linear algebra routines
 */

#include <migraphx/blaze_utils.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/generate.hpp>
#include <iostream>

int main() {
    // Check if Blaze support is available
    if (!migraphx::blaze_utils::is_blaze_available()) {
        std::cout << "Blaze support is not enabled. Please build MIGraphX with -DMIGRAPHX_USE_BLAZE=ON" << std::endl;
        return 1;
    }

    std::cout << "Blaze integration example for MIGraphX" << std::endl;

#if MIGRAPHX_USE_BLAZE
    // Create MIGraphX tensors
    auto shape_a = migraphx::shape{migraphx::shape::float_type, {3, 4}};
    auto shape_b = migraphx::shape{migraphx::shape::float_type, {4, 2}};
    auto shape_c = migraphx::shape{migraphx::shape::float_type, {3, 2}};

    auto arg_a = migraphx::generate_argument(shape_a, 1);  // Fill with 1.0f
    auto arg_b = migraphx::generate_argument(shape_b, 2);  // Fill with 2.0f
    auto arg_c = migraphx::generate_argument(shape_c, 0);  // Fill with 0.0f

    // Get tensor views
    auto view_a = arg_a.get<float>();
    auto view_b = arg_b.get<float>();
    auto view_c = arg_c.get<float>();

    std::cout << "Matrix A (3x4): " << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            std::cout << view_a(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix B (4x2): " << std::endl;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            std::cout << view_b(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // Perform matrix multiplication using Blaze
    std::cout << "Performing C = A * B using Blaze..." << std::endl;
    migraphx::blaze_utils::blaze_gemm(view_c, view_a, view_b);

    std::cout << "Result matrix C (3x2): " << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            std::cout << view_c(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // Expected result: each element should be 8.0 (4 * 1 * 2)
    std::cout << "Expected: Each element should be 8.0" << std::endl;

    // Demonstrate Blaze matrix conversion
    auto blaze_a = migraphx::blaze_utils::to_blaze_matrix(view_a);
    std::cout << "Blaze matrix A dimensions: " << blaze_a.rows() << "x" << blaze_a.columns() << std::endl;
    
    // Show some Blaze-specific operations
    auto blaze_norm = blaze::norm(blaze_a);
    std::cout << "Frobenius norm of matrix A: " << blaze_norm << std::endl;

#endif

    return 0;
}
