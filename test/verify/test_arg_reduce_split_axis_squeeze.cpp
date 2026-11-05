/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/make_op.hpp>

template <bool IsMax, bool Last, bool Tied>
struct test_arg_reduce_split_axis_squeeze
    : verify_program<test_arg_reduce_split_axis_squeeze<IsMax, Last, Tied>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 12, 3}});
        // Keep the CPU int64-add fallback exact while preserving strict integer verification.
        auto y = mm->add_parameter("y", {migraphx::shape::int8_type, {1, 4, 3, 3}});
        if constexpr(Tied)
        {
            // Repeat a runtime row to force ties without a foldable constant reduction.
            x = mm->add_instruction(
                migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), x);
            x = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 3}}}),
                                    x);
        }
        auto arg =
            mm->add_instruction(migraphx::make_op(IsMax ? "argmax" : "argmin",
                                                  {{"axis", 1}, {"select_last_index", Last}}),
                                x);
        auto squeezed = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), arg);
        auto restored =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), squeezed);
        auto broadcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 3}}}), restored);
        auto reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 3, 3}}}), broadcast);
        auto y_int64 = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), y);
        auto add = mm->add_instruction(migraphx::make_op("add"), reshape, y_int64);
        mm->add_return({add});
        return p;
    }

    std::string section() const { return "reduce"; }
};

template struct test_arg_reduce_split_axis_squeeze<true, false, false>;
template struct test_arg_reduce_split_axis_squeeze<true, false, true>;
template struct test_arg_reduce_split_axis_squeeze<true, true, false>;
template struct test_arg_reduce_split_axis_squeeze<true, true, true>;
template struct test_arg_reduce_split_axis_squeeze<false, false, false>;
template struct test_arg_reduce_split_axis_squeeze<false, false, true>;
template struct test_arg_reduce_split_axis_squeeze<false, true, false>;
template struct test_arg_reduce_split_axis_squeeze<false, true, true>;
