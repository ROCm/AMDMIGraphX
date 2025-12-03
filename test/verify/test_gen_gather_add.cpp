/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/make_op.hpp>

// Test gather followed by pointwise add - verifies gather index transformation fusion
template <migraphx::shape::type_t DType>
struct test_gen_gather_add : verify_program<test_gen_gather_add<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s_data{DType, {8, 16, 32}};
        auto data = mm->add_parameter("data", s_data);
        // Use literal indices that are within bounds [0, 8)
        std::vector<int> indices_data = {0, 2, 4, 6};
        migraphx::shape s_indices{migraphx::shape::int32_type, {4}};
        auto indices = mm->add_literal(migraphx::literal{s_indices, indices_data});
        auto gathered =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        // Output shape: {4, 16, 32}
        migraphx::shape s_y{DType, {4, 16, 32}};
        auto y = mm->add_parameter("y", s_y);
        mm->add_instruction(migraphx::make_op("add"), gathered, y);
        return p;
    }
};

template struct test_gen_gather_add<migraphx::shape::float_type>;
template struct test_gen_gather_add<migraphx::shape::half_type>;

// Test gather on different axis followed by multiply
template <migraphx::shape::type_t DType>
struct test_gen_gather_axis1_mul : verify_program<test_gen_gather_axis1_mul<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s_data{DType, {4, 16, 8}};
        auto data = mm->add_parameter("data", s_data);
        // Use literal indices that are within bounds [0, 16)
        std::vector<int> indices_data = {0, 3, 7, 11, 15};
        migraphx::shape s_indices{migraphx::shape::int32_type, {5}};
        auto indices = mm->add_literal(migraphx::literal{s_indices, indices_data});
        auto gathered =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), data, indices);
        // Output shape: {4, 5, 8}
        migraphx::shape s_y{DType, {4, 5, 8}};
        auto y = mm->add_parameter("y", s_y);
        mm->add_instruction(migraphx::make_op("mul"), gathered, y);
        return p;
    }
};

template struct test_gen_gather_axis1_mul<migraphx::shape::float_type>;
template struct test_gen_gather_axis1_mul<migraphx::shape::half_type>;

// Test gather followed by relu
template <migraphx::shape::type_t DType>
struct test_gen_gather_relu : verify_program<test_gen_gather_relu<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s_data{DType, {16, 32}};
        auto data = mm->add_parameter("data", s_data);
        // Use literal indices that are within bounds [0, 16)
        std::vector<int> indices_data = {0, 2, 4, 6, 8, 10, 12, 14};
        migraphx::shape s_indices{migraphx::shape::int32_type, {8}};
        auto indices = mm->add_literal(migraphx::literal{s_indices, indices_data});
        auto gathered =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        mm->add_instruction(migraphx::make_op("relu"), gathered);
        return p;
    }
};

template struct test_gen_gather_relu<migraphx::shape::float_type>;
template struct test_gen_gather_relu<migraphx::shape::half_type>;
