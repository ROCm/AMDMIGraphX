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
 *
 */
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/float_equal.hpp>
#include <migraphx/kernels/test.hpp>

TEST_CASE(vec_dot_float4)
{
    migraphx::vec<float, 4> x = {1.0f, 2.0f, 3.0f, 4.0f};
    migraphx::vec<float, 4> y = {1.0f, 1.0f, 1.0f, 1.0f};
    EXPECT(migraphx::float_equal(migraphx::vec_dot(x, y), 10.0f));
}

TEST_CASE(vec_dot_int)
{
    migraphx::vec<int, 4> x = {1, 2, 3, 4};
    migraphx::vec<int, 4> y = {5, 6, 7, 8};
    EXPECT(migraphx::vec_dot(x, y) == 70);
}

TEST_CASE(vec_dot_scalar_accumulate)
{
    EXPECT(migraphx::float_equal(migraphx::vec_dot(2.0f, 3.0f, 1.0f), 7.0f));
}

TEST_CASE(vec_dot_float_accumulate)
{
    migraphx::vec<float, 2> x = {1.0f, 2.0f};
    migraphx::vec<float, 2> y = {3.0f, 4.0f};
    EXPECT(migraphx::float_equal(migraphx::vec_dot(x, y, 0.5f), 11.5f));
}

TEST_CASE(vec_dot_half2)
{
    migraphx::vec<migraphx::half, 2> x = {1, 2};
    migraphx::vec<migraphx::half, 2> y = {3, 4};
    auto result                        = migraphx::vec_dot(x, y);
    EXPECT(migraphx::is_same<decltype(result), float>{});
    EXPECT(migraphx::float_equal(result, 11.0f));
}

TEST_CASE(vec_dot_half8)
{
    migraphx::vec<migraphx::half, 8> x = {1, 2, 3, 4, 5, 6, 7, 8};
    migraphx::vec<migraphx::half, 8> y = {1, 1, 1, 1, 1, 1, 1, 1};
    EXPECT(migraphx::float_equal(migraphx::vec_dot(x, y), 36.0f));
}

TEST_CASE(vec_dot_half_accumulate)
{
    migraphx::vec<migraphx::half, 4> x = {1, 2, 3, 4};
    migraphx::vec<migraphx::half, 4> y = {4, 3, 2, 1};
    EXPECT(migraphx::float_equal(migraphx::vec_dot(x, y, 0.5f), 20.5f));
}

// An odd size has no pairs for the dot instruction and multiplies one element at a time
TEST_CASE(vec_dot_half3_accumulate)
{
    migraphx::vec<migraphx::half, 3> x = {1, 2, 3};
    migraphx::vec<migraphx::half, 3> y = {2, 2, 2};
    EXPECT(migraphx::float_equal(migraphx::vec_dot(x, y, 1.0f), 13.0f));
}

// 2049 has no half representation, so the sum must be kept in float
TEST_CASE(vec_dot_half_float_sum)
{
    migraphx::vec<migraphx::half, 2> x = {2048, 1};
    migraphx::vec<migraphx::half, 2> y = {1, 1};
    EXPECT(migraphx::float_equal(migraphx::vec_dot(x, y), 2049.0f));
    EXPECT(migraphx::float_equal(migraphx::vec_dot(x, y, 2.0f), 2051.0f));
}

TEST_CASE(vec_dot_half3_float_sum)
{
    migraphx::vec<migraphx::half, 3> x = {2048, 1, 1};
    migraphx::vec<migraphx::half, 3> y = {1, 1, 1};
    auto result                        = migraphx::vec_dot(x, y);
    EXPECT(migraphx::is_same<decltype(result), float>{});
    EXPECT(migraphx::float_equal(result, 2050.0f));
}

TEST_CASE(vec_dot_half_scalar)
{
    auto result = migraphx::vec_dot(migraphx::half{2048}, migraphx::half{3});
    EXPECT(migraphx::is_same<decltype(result), float>{});
    EXPECT(migraphx::float_equal(result, 6144.0f));
}

// 257 has no bf16 representation
TEST_CASE(vec_dot_bf16_float_sum)
{
    migraphx::vec<migraphx::bf16, 2> x = {256, 1};
    migraphx::vec<migraphx::bf16, 2> y = {1, 1};
    auto result                        = migraphx::vec_dot(x, y);
    EXPECT(migraphx::is_same<decltype(result), float>{});
    EXPECT(migraphx::float_equal(result, 257.0f));
}

TEST_CASE(vec_dot_int8_int32_sum)
{
    migraphx::vec<int8_t, 4> x = {100, 100, 100, -100};
    migraphx::vec<int8_t, 4> y = {100, 100, 100, 100};
    auto result                = migraphx::vec_dot(x, y);
    EXPECT(migraphx::is_same<decltype(result), int32_t>{});
    EXPECT(result == 20000);
}

TEST_CASE(vec_dot_double)
{
    migraphx::vec<double, 2> x = {1.0, 2.0};
    migraphx::vec<double, 2> y = {3.0, 4.0};
    auto result                = migraphx::vec_dot(x, y);
    EXPECT(migraphx::is_same<decltype(result), double>{});
    EXPECT(migraphx::float_equal(result, 11.0));
}
