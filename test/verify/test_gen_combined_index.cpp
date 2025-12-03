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

// Test pad followed by reverse followed by add
// This tests chained index transformations
template <migraphx::shape::type_t DType>
struct test_gen_pad_reverse_add : verify_program<test_gen_pad_reverse_add<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {4, 8}};
        // Pad format: [before_d0, before_d1, after_d0, after_d1]
        std::vector<int64_t> pads = {1, 0, 1, 0};
        auto x                    = mm->add_parameter("x", s);
        auto padded = mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}}), x);
        // Padded shape: {6, 8}
        std::vector<int64_t> axes = {0};
        auto reversed = mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), padded);
        migraphx::shape s_y{DType, {6, 8}};
        auto y = mm->add_parameter("y", s_y);
        mm->add_instruction(migraphx::make_op("add"), reversed, y);
        return p;
    }
};

template struct test_gen_pad_reverse_add<migraphx::shape::float_type>;
template struct test_gen_pad_reverse_add<migraphx::shape::half_type>;

// Test slice followed by pointwise
template <migraphx::shape::type_t DType>
struct test_gen_slice_add : verify_program<test_gen_slice_add<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {8, 16, 32}};
        auto x      = mm->add_parameter("x", s);
        auto sliced = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 2}}, {"starts", {2, 4}}, {"ends", {6, 28}}}),
            x);
        // Sliced shape: {4, 16, 24}
        migraphx::shape s_y{DType, {4, 16, 24}};
        auto y = mm->add_parameter("y", s_y);
        mm->add_instruction(migraphx::make_op("add"), sliced, y);
        return p;
    }
};

template struct test_gen_slice_add<migraphx::shape::float_type>;
template struct test_gen_slice_add<migraphx::shape::half_type>;

// Test broadcast followed by pointwise
template <migraphx::shape::type_t DType>
struct test_gen_broadcast_add : verify_program<test_gen_broadcast_add<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s_x{DType, {1, 8, 1}};
        migraphx::shape s_y{DType, {4, 8, 16}};
        auto x = mm->add_parameter("x", s_x);
        auto broadcasted =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 8, 16}}}), x);
        auto y = mm->add_parameter("y", s_y);
        mm->add_instruction(migraphx::make_op("add"), broadcasted, y);
        return p;
    }
};

template struct test_gen_broadcast_add<migraphx::shape::float_type>;
template struct test_gen_broadcast_add<migraphx::shape::half_type>;

// Test transpose followed by pointwise
template <migraphx::shape::type_t DType>
struct test_gen_transpose_add : verify_program<test_gen_transpose_add<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s_x{DType, {4, 8, 16}};
        auto x = mm->add_parameter("x", s_x);
        auto transposed =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), x);
        // Transposed shape: {4, 16, 8}
        migraphx::shape s_y{DType, {4, 16, 8}};
        auto y = mm->add_parameter("y", s_y);
        mm->add_instruction(migraphx::make_op("add"), transposed, y);
        return p;
    }
};

template struct test_gen_transpose_add<migraphx::shape::float_type>;
template struct test_gen_transpose_add<migraphx::shape::half_type>;
