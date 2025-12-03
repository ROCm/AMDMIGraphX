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

// Test gather followed by reduce_sum - tests index transformation with reduction
template <migraphx::shape::type_t DType>
struct test_gen_gather_reduce_sum : verify_program<test_gen_gather_reduce_sum<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s_data{DType, {16, 32}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {8}};
        auto data    = mm->add_parameter("data", s_data);
        auto indices = mm->add_literal(
            migraphx::literal{s_indices, std::vector<int32_t>{0, 2, 4, 6, 8, 10, 12, 14}});
        auto gathered =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        // Gathered shape: {8, 32}, reduce along last axis
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), gathered);
        return p;
    }
    std::string section() const { return "reduce"; }
};

template struct test_gen_gather_reduce_sum<migraphx::shape::float_type>;
template struct test_gen_gather_reduce_sum<migraphx::shape::half_type>;

// Test gather followed by reduce_max
template <migraphx::shape::type_t DType>
struct test_gen_gather_reduce_max : verify_program<test_gen_gather_reduce_max<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s_data{DType, {4, 8, 16}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {3}};
        auto data    = mm->add_parameter("data", s_data);
        auto indices = mm->add_literal(migraphx::literal{s_indices, std::vector<int32_t>{1, 3, 5}});
        auto gathered =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), data, indices);
        // Gathered shape: {4, 3, 16}, reduce along axis 2
        mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), gathered);
        return p;
    }
    std::string section() const { return "reduce"; }
};

template struct test_gen_gather_reduce_max<migraphx::shape::float_type>;
template struct test_gen_gather_reduce_max<migraphx::shape::half_type>;

// Test gather + pointwise + reduce fusion
template <migraphx::shape::type_t DType>
struct test_gen_gather_add_reduce : verify_program<test_gen_gather_add_reduce<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s_data{DType, {16, 8}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {4}};
        auto data    = mm->add_parameter("data", s_data);
        auto indices = mm->add_literal(migraphx::literal{s_indices, std::vector<int32_t>{0, 4, 8, 12}});
        auto gathered =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        // Gathered shape: {4, 8}
        migraphx::shape s_y{DType, {4, 8}};
        auto y   = mm->add_parameter("y", s_y);
        auto add = mm->add_instruction(migraphx::make_op("add"), gathered, y);
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), add);
        return p;
    }
    std::string section() const { return "reduce"; }
};

template struct test_gen_gather_add_reduce<migraphx::shape::float_type>;
template struct test_gen_gather_add_reduce<migraphx::shape::half_type>;


