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
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/half.hpp>
#include <migraphx/bf16.hpp>
#include <cmath>
#include <limits>
#include "test.hpp"

static migraphx::argument cast_inf(float x, migraphx::shape::type_t target)
{
    migraphx::module m;
    auto lit = m.add_literal(x);
    auto cvt = m.add_instruction(migraphx::make_op("convert", {{"target_type", target}}), lit);
    return cvt->eval();
}

template <class T>
static bool is_neg_inf(T x)
{
    return std::isinf(x) and x < 0;
}

template <class T>
static bool is_pos_inf(T x)
{
    return std::isinf(x) and x > 0;
}

TEST_CASE(float_inf_to_half)
{
    EXPECT(is_pos_inf(cast_inf(std::numeric_limits<float>::infinity(), migraphx::shape::half_type)
                          .at<migraphx::half>()));
}

TEST_CASE(float_ninf_to_half)
{
    EXPECT(is_neg_inf(cast_inf(-std::numeric_limits<float>::infinity(), migraphx::shape::half_type)
                          .at<migraphx::half>()));
}

TEST_CASE(double_inf_to_float)
{
    EXPECT(is_pos_inf(cast_inf(std::numeric_limits<double>::infinity(), migraphx::shape::float_type)
                          .at<float>()));
}

TEST_CASE(double_ninf_to_float)
{
    EXPECT(
        is_neg_inf(cast_inf(-std::numeric_limits<double>::infinity(), migraphx::shape::float_type)
                       .at<float>()));
}

TEST_CASE(double_inf_to_half)
{
    EXPECT(is_pos_inf(cast_inf(std::numeric_limits<double>::infinity(), migraphx::shape::half_type)
                          .at<migraphx::half>()));
}

TEST_CASE(double_ninf_to_half)
{
    EXPECT(is_neg_inf(cast_inf(-std::numeric_limits<double>::infinity(), migraphx::shape::half_type)
                          .at<migraphx::half>()));
}

TEST_CASE(float_inf_to_int32)
{
    EXPECT(cast_inf(std::numeric_limits<float>::infinity(), migraphx::shape::int32_type)
               .at<int32_t>() == std::numeric_limits<int32_t>::max());
}

TEST_CASE(float_ninf_to_int32)
{
    EXPECT(cast_inf(-std::numeric_limits<float>::infinity(), migraphx::shape::int32_type)
               .at<int32_t>() == std::numeric_limits<int32_t>::min());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
