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
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <layernorm.hpp>

// Test layernorm with pad fusion - pad input before layernorm
// This tests if gen IR can fuse pad with the layernorm operations
template <migraphx::shape::type_t DType>
struct test_gen_pad_layernorm : verify_program<test_gen_pad_layernorm<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 2, 4};
        auto x                   = mm->add_parameter("x", migraphx::shape{DType, dims});
        // Pad the input
        std::vector<int64_t> pads = {0, 0, 1, 0, 0, 1};
        auto padded = mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}}), x);
        // Apply layernorm to padded tensor {1, 2, 6}
        std::vector<size_t> padded_dims = {1, 2, 6};
        add_layernorm(*mm, padded, padded_dims);
        return p;
    }
    std::string section() const { return "reduce"; }
};

template struct test_gen_pad_layernorm<migraphx::shape::float_type>;
template struct test_gen_pad_layernorm<migraphx::shape::half_type>;

// Test layernorm with gather fusion - gather then layernorm
template <migraphx::shape::type_t DType>
struct test_gen_gather_layernorm : verify_program<test_gen_gather_layernorm<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s_data{DType, {8, 4, 16}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {4}};
        auto data = mm->add_parameter("data", s_data);
        auto indices =
            mm->add_literal(migraphx::literal{s_indices, std::vector<int32_t>{0, 2, 4, 6}});
        auto gathered =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        // Gathered shape: {4, 4, 16}
        std::vector<size_t> gathered_dims = {4, 4, 16};
        add_layernorm(*mm, gathered, gathered_dims);
        return p;
    }
    std::string section() const { return "reduce"; }
};

template struct test_gen_gather_layernorm<migraphx::shape::float_type>;
template struct test_gen_gather_layernorm<migraphx::shape::half_type>;

// Test layernorm with add fusion - add before layernorm
// This is a common pattern in transformers (residual connection + layernorm)
template <migraphx::shape::type_t DType>
struct test_gen_add_layernorm : verify_program<test_gen_add_layernorm<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {2, 4, 8};
        auto x                   = mm->add_parameter("x", migraphx::shape{DType, dims});
        auto y                   = mm->add_parameter("y", migraphx::shape{DType, dims});
        auto add                 = mm->add_instruction(migraphx::make_op("add"), x, y);
        add_layernorm(*mm, add, dims);
        return p;
    }
    std::string section() const { return "reduce"; }
};

template struct test_gen_add_layernorm<migraphx::shape::float_type>;
template struct test_gen_add_layernorm<migraphx::shape::half_type>;
