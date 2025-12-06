/*
 * Test program for gather optimization selector
 * 
 * This demonstrates how the gather optimization selector chooses
 * between different kernel implementations based on operation characteristics.
 */

#include <migraphx/shape.hpp>
#include <migraphx/gpu/gather_optimizer.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace migraphx;
using namespace migraphx::gpu;

struct test_case
{
    std::string name;
    std::vector<std::size_t> data_shape;
    std::vector<std::size_t> indices_shape;
    int axis;
    std::string expected_kernel;
};

void print_analysis(const std::string& name, 
                   const gather_analysis& analysis, 
                   const std::string& selected_kernel)
{
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Test Case: " << name << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    std::cout << "Analysis:\n";
    std::cout << "  Output elements:      " << analysis.num_elements << "\n";
    std::cout << "  Axis:                 " << analysis.axis << "\n";
    std::cout << "  Axis size:            " << analysis.axis_size << "\n";
    std::cout << "  Num indices:          " << analysis.num_indices << "\n";
    std::cout << "  Is innermost axis:    " << (analysis.is_innermost_axis ? "YES" : "NO") << "\n";
    std::cout << "  Contiguous input:     " << (analysis.is_contiguous_input ? "YES" : "NO") << "\n";
    std::cout << "  Large gather:         " << (analysis.is_large_gather ? "YES" : "NO") << "\n";
    
    std::cout << "\nSelected Kernel: " << selected_kernel << "\n";
}

int main()
{
    std::cout << "MIGraphX Gather Optimization Selector Test\n";
    std::cout << std::string(60, '=') << "\n";
    
    std::vector<test_case> test_cases = {
        // Small gather - should use basic
        {
            "Small Gather (Basic Expected)",
            {100, 50},      // data shape
            {10},           // indices shape
            0,              // axis
            "gather"        // expected
        },
        
        // Medium gather, not innermost - should use optimized
        {
            "Medium Gather on Outer Axis (Optimized Expected)",
            {1000, 500},    // data shape
            {100},          // indices shape
            0,              // axis
            "gather_opt"    // expected
        },
        
        // Large gather on innermost axis - should use vectorized
        {
            "Large Innermost Axis Gather (Vectorized Expected)",
            {100, 1000},    // data shape
            {200},          // indices shape
            1,              // axis (innermost)
            "gather_vectorized" // expected
        },
        
        // Large gather on outer axis - should use optimized
        {
            "Large Outer Axis Gather (Optimized Expected)",
            {500, 1000},    // data shape
            {200},          // indices shape
            0,              // axis (outer)
            "gather_opt"    // expected
        },
        
        // Very large innermost - should use vectorized
        {
            "Very Large Innermost (Vectorized Expected)",
            {256, 2048},    // data shape
            {512},          // indices shape
            1,              // axis (innermost)
            "gather_vectorized" // expected
        },
        
        // 3D tensor, middle axis
        {
            "3D Tensor Middle Axis (Optimized Expected)",
            {64, 128, 256}, // data shape
            {100},          // indices shape
            1,              // axis (middle)
            "gather_opt"    // expected
        },
        
        // 3D tensor, innermost axis, large
        {
            "3D Tensor Innermost Axis (Vectorized Expected)",
            {32, 64, 512},  // data shape
            {200},          // indices shape
            2,              // axis (innermost)
            "gather_vectorized" // expected
        },
    };
    
    int passed = 0;
    int failed = 0;
    
    for(const auto& tc : test_cases)
    {
        // Create shapes
        shape data_shape{shape::float_type, tc.data_shape};
        shape indices_shape{shape::int32_type, tc.indices_shape};
        
        // Calculate output shape
        auto output_lens = tc.data_shape;
        output_lens[tc.axis] = indices_shape.elements();
        shape output_shape{shape::float_type, output_lens};
        
        std::vector<shape> inputs = {data_shape, indices_shape, output_shape};
        
        // Analyze and select kernel
        auto analysis = analyze_gather(inputs, tc.axis);
        auto selected_kernel = select_gather_kernel(inputs, tc.axis);
        
        // Print results
        print_analysis(tc.name, analysis, selected_kernel);
        
        // Check if selection matches expected
        bool matches = (selected_kernel == tc.expected_kernel);
        std::cout << "Expected: " << tc.expected_kernel << "\n";
        std::cout << "Result:   " << (matches ? "✓ PASS" : "✗ FAIL") << "\n";
        
        if(matches)
            passed++;
        else
            failed++;
    }
    
    // Summary
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Summary\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "Total tests: " << (passed + failed) << "\n";
    std::cout << "Passed:      " << passed << " ✓\n";
    std::cout << "Failed:      " << failed << (failed > 0 ? " ✗" : "") << "\n";
    std::cout << std::string(60, '=') << "\n";
    
    return (failed == 0) ? 0 : 1;
}

