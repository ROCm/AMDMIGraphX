/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/op/pooling.hpp>

// Reusable Function to create a pooling program based on input parameters
migraphx::program create_pooling_program(const std::vector<int>& lengths,
                                         const std::vector<int>& stride,
                                         const std::vector<int>& padding)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::float_type, {1, 3, 32, 32}};
    auto l0 = mm->add_parameter("x", s);

    // Create a pooling operation with the specified parameters
    migraphx::op::pooling pooling_op{migraphx::op::pooling_mode::max};
    pooling_op.lengths = lengths;   
    pooling_op.stride  = stride;    
    pooling_op.padding = padding;   

    mm->add_instruction(pooling_op, l0);

    return p;
}

// Test Case 1: Valid pooling parameters (should work with OneDNN)
struct test_valid_pooling : verify_program<test_valid_pooling>
{
    migraphx::program create_program() const
    {
        // Valid case: Kernel size {2, 2}, Stride {2, 2}, No padding
        return create_pooling_program({2, 2}, {2, 2}, {0, 0});
    }
};

// Test Case 2: Large kernel size (should trigger fallback)
struct test_large_kernel_pooling : verify_program<test_large_kernel_pooling>
{
    migraphx::program create_program() const
    {
        // Invalid case: Kernel size {15, 15}, Stride {1, 1}, No padding
        return create_pooling_program({15, 15}, {1, 1}, {0, 0});
    }
};

// Test Case 3: Invalid padding (should trigger fallback)
struct test_invalid_padding_pooling : verify_program<test_invalid_padding_pooling>
{
    migraphx::program create_program() const
    {
        // Invalid case: Kernel size {2, 2}, Stride {2, 2}, Large padding {5, 5}
        return create_pooling_program({2, 2}, {2, 2}, {5, 5});
    }
};
