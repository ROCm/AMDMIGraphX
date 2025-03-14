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
#include <migraphx/float8.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

template <typename CType>
struct test_rsqrt : verify_program<test_rsqrt<CType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                      = p.get_main_module();
        migraphx::shape::type_t dtype = migraphx::shape::get_type<CType>();
        std::vector<size_t> input_lens{1, 3, 16, 16};
        migraphx::shape s{dtype, input_lens};
        auto x       = mm->add_parameter("x", s);
        auto min_val = mm->add_literal(migraphx::literal{migraphx::shape{dtype}, {1.0}});
        auto max_val = mm->add_literal(
            migraphx::literal{migraphx::shape{dtype}, {std::numeric_limits<CType>::max()}});
        min_val = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), min_val);
        max_val = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), max_val);
        auto l0 = mm->add_instruction(migraphx::make_op("clip"), x, min_val, max_val);
        mm->add_instruction(migraphx::make_op("rsqrt"), l0);
        return p;
    };
};

template struct test_rsqrt<float>;
template struct test_rsqrt<migraphx::half>;
template struct test_rsqrt<migraphx::bf16>;
template struct test_rsqrt<migraphx::fp8::fp8e4m3fnuz>;
template struct test_rsqrt<migraphx::fp8::fp8e5m2fnuz>;
template struct test_rsqrt<migraphx::fp8::fp8e4m3fn>;
template struct test_rsqrt<migraphx::fp8::fp8e5m2>;
