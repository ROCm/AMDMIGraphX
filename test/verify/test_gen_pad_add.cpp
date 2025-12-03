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

// Test pad followed by pointwise add - verifies pad index transformation fusion
// Pad format: [before_d0, before_d1, ..., after_d0, after_d1, ...]
template <migraphx::shape::type_t DType>
struct test_gen_pad_add : verify_program<test_gen_pad_add<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {4, 8, 16}};
        // Pad format: [before_d0, before_d1, before_d2, after_d0, after_d1, after_d2]
        std::vector<int64_t> pads = {0, 1, 1, 0, 1, 1};
        auto x                    = mm->add_parameter("x", s);
        auto padded = mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}}), x);
        // Padded shape: {4, 10, 18}
        migraphx::shape s_padded{DType, {4, 10, 18}};
        auto y = mm->add_parameter("y", s_padded);
        mm->add_instruction(migraphx::make_op("add"), padded, y);
        return p;
    }
};

template struct test_gen_pad_add<migraphx::shape::float_type>;
template struct test_gen_pad_add<migraphx::shape::half_type>;

// Test pad with constant value followed by multiply
template <migraphx::shape::type_t DType>
struct test_gen_pad_mul : verify_program<test_gen_pad_mul<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {2, 4, 8}};
        // Pad format: [before_d0, before_d1, before_d2, after_d0, after_d1, after_d2]
        std::vector<int64_t> pads = {1, 1, 0, 1, 1, 0};
        auto x                    = mm->add_parameter("x", s);
        auto padded =
            mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}, {"value", 1.0f}}), x);
        // Padded shape: {4, 6, 8}
        migraphx::shape s_padded{DType, {4, 6, 8}};
        auto y = mm->add_parameter("y", s_padded);
        mm->add_instruction(migraphx::make_op("mul"), padded, y);
        return p;
    }
};

template struct test_gen_pad_mul<migraphx::shape::float_type>;
template struct test_gen_pad_mul<migraphx::shape::half_type>;
