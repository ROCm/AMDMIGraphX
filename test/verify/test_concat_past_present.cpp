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
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/sym.hpp>

// Decode-mode kv-cache append fed straight from a parameter, so nothing fuses
// and the copy kernel runs; fp8 covers its unvectorized path
template <migraphx::shape::type_t DType>
struct test_concat_past_present : verify_program<test_concat_past_present<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {1, 2, 1, 4}};
        migraphx::shape cs{DType, {1, 2, 8, 4}};
        auto present = mm->add_parameter("present", s);
        auto cache   = mm->add_parameter("cache", cs);
        auto slk     = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1, 1}}, {3}});
        mm->add_instruction(
            migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), present, slk, cache);
        return p;
    }
};

template struct test_concat_past_present<migraphx::shape::half_type>;
template struct test_concat_past_present<migraphx::shape::fp8e4m3fn_type>;

struct test_concat_past_present_runtime_sequence
    : verify_program<test_concat_past_present_runtime_sequence>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto present =
            mm->add_parameter("present", migraphx::shape{migraphx::shape::half_type, {1, 2, 4, 4}});
        auto cache =
            mm->add_parameter("cache", migraphx::shape{migraphx::shape::half_type, {1, 2, 8, 4}});
        auto slk = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1, 1}}, {0}});
        auto sequence_length = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {2}});
        mm->add_instruction(migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}),
                            present,
                            slk,
                            sequence_length,
                            cache);
        return p;
    }
};

struct test_concat_past_present_dynamic_runtime_sequence
    : verify_program<test_concat_past_present_dynamic_runtime_sequence>
{
    migraphx::program create_program() const
    {
        using dd = migraphx::shape::dynamic_dimension;
        using migraphx::sym::lit;

        auto sequence = migraphx::sym::var("sequence", {1, 4}, {2});
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto present =
            mm->add_parameter("present",
                              migraphx::shape{migraphx::shape::half_type,
                                              {dd{lit(1)}, dd{lit(2)}, dd{sequence}, dd{lit(4)}}});
        auto cache =
            mm->add_parameter("cache", migraphx::shape{migraphx::shape::half_type, {1, 2, 8, 4}});
        auto slk = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1, 1}}, {0}});
        auto sequence_length = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {2}});
        mm->add_instruction(migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}),
                            present,
                            slk,
                            sequence_length,
                            cache);
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"present", migraphx::shape{migraphx::shape::half_type, {1, 2, 2, 4}}}};
    }
};
