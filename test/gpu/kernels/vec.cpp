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

// vec_size

TEST_CASE(vec_size_scalar)
{
    EXPECT(migraphx::vec_size<int>() == 0);
    EXPECT(migraphx::vec_size<float>() == 0);
}

TEST_CASE(vec_size_vec2) { EXPECT(migraphx::vec_size<migraphx::vec<float, 2>>() == 2); }

TEST_CASE(vec_size_vec4) { EXPECT(migraphx::vec_size<migraphx::vec<int, 4>>() == 4); }

// is_any_vec

TEST_CASE(is_any_vec_empty) { EXPECT(not migraphx::is_any_vec<>()); }

TEST_CASE(is_any_vec_scalar) { EXPECT(not migraphx::is_any_vec<int>()); }

TEST_CASE(is_any_vec_vec) { EXPECT(migraphx::is_any_vec<migraphx::vec<float, 2>>()); }

TEST_CASE(is_any_vec_mixed) { EXPECT(migraphx::is_any_vec<int, migraphx::vec<float, 2>>()); }

// vec_at

TEST_CASE(vec_at_scalar) { EXPECT(migraphx::vec_at(42, 0) == 42); }

TEST_CASE(vec_at_vec)
{
    migraphx::vec<int, 4> v = {10, 20, 30, 40};
    EXPECT(migraphx::vec_at(v, 0) == 10);
    EXPECT(migraphx::vec_at(v, 1) == 20);
    EXPECT(migraphx::vec_at(v, 2) == 30);
    EXPECT(migraphx::vec_at(v, 3) == 40);
}

// common_vec_size

TEST_CASE(common_vec_size_scalars) { EXPECT(migraphx::common_vec_size<int, float>() == 0); }

TEST_CASE(common_vec_size_mixed)
{
    EXPECT(migraphx::common_vec_size<int, migraphx::vec<float, 4>>() == 4);
}

TEST_CASE(common_vec_size_vecs)
{
    EXPECT(migraphx::common_vec_size<migraphx::vec<int, 2>, migraphx::vec<float, 4>>() == 4);
}

// vec_transform

TEST_CASE(vec_transform_scalar)
{
    auto result = migraphx::vec_transform(3, 4)([](auto a, auto b) { return a + b; });
    EXPECT(result == 7);
}

TEST_CASE(vec_transform_vec)
{
    migraphx::vec<int, 4> a = {1, 2, 3, 4};
    migraphx::vec<int, 4> b = {10, 20, 30, 40};
    auto result             = migraphx::vec_transform(a, b)([](auto x, auto y) { return x + y; });
    EXPECT(migraphx::vec_at(result, 0) == 11);
    EXPECT(migraphx::vec_at(result, 1) == 22);
    EXPECT(migraphx::vec_at(result, 2) == 33);
    EXPECT(migraphx::vec_at(result, 3) == 44);
}

// vec_packed_at

TEST_CASE(vec_packed_at_scalar)
{
    auto result = migraphx::vec_packed_at<2>(5, 0);
    EXPECT(migraphx::vec_at(result, 0) == 5);
}

TEST_CASE(vec_packed_at_vec)
{
    migraphx::vec<int, 4> v = {10, 20, 30, 40};
    auto first              = migraphx::vec_packed_at<2>(v, 0);
    EXPECT(migraphx::vec_at(first, 0) == 10);
    EXPECT(migraphx::vec_at(first, 1) == 20);
    auto second = migraphx::vec_packed_at<2>(v, 2);
    EXPECT(migraphx::vec_at(second, 0) == 30);
    EXPECT(migraphx::vec_at(second, 1) == 40);
}

// vec_reduce

TEST_CASE(vec_reduce_sum)
{
    migraphx::vec<int, 4> v = {1, 2, 3, 4};
    auto result             = migraphx::vec_reduce(v, [](auto a, auto b) { return a + b; });
    EXPECT(result == 10);
}

TEST_CASE(vec_reduce_max)
{
    migraphx::vec<int, 4> v = {3, 1, 4, 2};
    auto result             = migraphx::vec_reduce(v, [](auto a, auto b) { return a > b ? a : b; });
    EXPECT(result == 4);
}

TEST_CASE(vec_reduce_scalar)
{
    auto result = migraphx::vec_reduce(42, [](auto a, auto b) { return a + b; });
    EXPECT(result == 42);
}

// vec_generate

TEST_CASE(vec_generate_basic)
{
    auto v = migraphx::vec_generate<4>([](auto i) { return int(i * 10); });
    EXPECT(migraphx::vec_at(v, 0) == 0);
    EXPECT(migraphx::vec_at(v, 1) == 10);
    EXPECT(migraphx::vec_at(v, 2) == 20);
    EXPECT(migraphx::vec_at(v, 3) == 30);
}

// vec_dot generic (float)

TEST_CASE(vec_dot_float2)
{
    migraphx::vec<float, 2> x = {1.0f, 2.0f};
    migraphx::vec<float, 2> y = {3.0f, 4.0f};
    auto result               = migraphx::vec_dot(x, y);
    EXPECT(migraphx::float_equal(result, 11.0f));
}

TEST_CASE(vec_dot_float4)
{
    migraphx::vec<float, 4> x = {1.0f, 2.0f, 3.0f, 4.0f};
    migraphx::vec<float, 4> y = {1.0f, 1.0f, 1.0f, 1.0f};
    auto result               = migraphx::vec_dot(x, y);
    EXPECT(migraphx::float_equal(result, 10.0f));
}

TEST_CASE(vec_dot_int)
{
    migraphx::vec<int, 4> x = {1, 2, 3, 4};
    migraphx::vec<int, 4> y = {5, 6, 7, 8};
    auto result             = migraphx::vec_dot(x, y);
    EXPECT(result == 70);
}

// vec_dot half (hardware fdot2)

TEST_CASE(vec_dot_half2)
{
    migraphx::vec<migraphx::half, 2> x = {1, 2};
    migraphx::vec<migraphx::half, 2> y = {3, 4};
    float result                       = migraphx::vec_dot(x, y);
    EXPECT(migraphx::float_equal(result, 11.0f));
}

TEST_CASE(vec_dot_half4)
{
    migraphx::vec<migraphx::half, 4> x = {1, 2, 3, 4};
    migraphx::vec<migraphx::half, 4> y = {1, 1, 1, 1};
    float result                       = migraphx::vec_dot(x, y);
    EXPECT(migraphx::float_equal(result, 10.0f));
}

TEST_CASE(vec_dot_half8)
{
    migraphx::vec<migraphx::half, 8> x = {1, 2, 3, 4, 5, 6, 7, 8};
    migraphx::vec<migraphx::half, 8> y = {1, 1, 1, 1, 1, 1, 1, 1};
    float result                       = migraphx::vec_dot(x, y);
    EXPECT(migraphx::float_equal(result, 36.0f));
}

TEST_CASE(vec_dot_half2_zeros)
{
    migraphx::vec<migraphx::half, 2> x = {0, 0};
    migraphx::vec<migraphx::half, 2> y = {1, 2};
    float result                       = migraphx::vec_dot(x, y);
    EXPECT(migraphx::float_equal(result, 0.0f));
}

// vec_dot bf16

TEST_CASE(vec_dot_bf16_2)
{
    migraphx::vec<migraphx::bf16, 2> x = {1.0f, 2.0f};
    migraphx::vec<migraphx::bf16, 2> y = {3.0f, 4.0f};
    float result                       = migraphx::vec_dot(x, y);
    EXPECT(migraphx::float_equal(result, 11.0f));
}

TEST_CASE(vec_dot_bf16_4)
{
    migraphx::vec<migraphx::bf16, 4> x = {1.0f, 2.0f, 3.0f, 4.0f};
    migraphx::vec<migraphx::bf16, 4> y = {1.0f, 1.0f, 1.0f, 1.0f};
    float result                       = migraphx::vec_dot(x, y);
    EXPECT(migraphx::float_equal(result, 10.0f));
}

TEST_CASE(vec_dot_bf16_8)
{
    migraphx::vec<migraphx::bf16, 8> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    migraphx::vec<migraphx::bf16, 8> y = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float result                       = migraphx::vec_dot(x, y);
    EXPECT(migraphx::float_equal(result, 36.0f));
}

TEST_CASE(vec_dot_bf16_2_zeros)
{
    migraphx::vec<migraphx::bf16, 2> x = {0.0f, 0.0f};
    migraphx::vec<migraphx::bf16, 2> y = {1.0f, 2.0f};
    float result                       = migraphx::vec_dot(x, y);
    EXPECT(migraphx::float_equal(result, 0.0f));
}

// vec_dot int8

TEST_CASE(vec_dot_int8_4)
{
    migraphx::vec<migraphx::int8_t, 4> x = {1, 2, 3, 4};
    migraphx::vec<migraphx::int8_t, 4> y = {5, 6, 7, 8};
    migraphx::int32_t result             = migraphx::vec_dot(x, y);
    EXPECT(result == 70);
}

TEST_CASE(vec_dot_int8_8)
{
    migraphx::vec<migraphx::int8_t, 8> x = {1, 2, 3, 4, 5, 6, 7, 8};
    migraphx::vec<migraphx::int8_t, 8> y = {1, 1, 1, 1, 1, 1, 1, 1};
    migraphx::int32_t result             = migraphx::vec_dot(x, y);
    EXPECT(result == 36);
}

TEST_CASE(vec_dot_int8_4_negative)
{
    migraphx::vec<migraphx::int8_t, 4> x = {-1, 2, -3, 4};
    migraphx::vec<migraphx::int8_t, 4> y = {1, 1, 1, 1};
    migraphx::int32_t result             = migraphx::vec_dot(x, y);
    EXPECT(result == 2);
}

TEST_CASE(vec_dot_int8_4_zeros)
{
    migraphx::vec<migraphx::int8_t, 4> x = {0, 0, 0, 0};
    migraphx::vec<migraphx::int8_t, 4> y = {1, 2, 3, 4};
    migraphx::int32_t result             = migraphx::vec_dot(x, y);
    EXPECT(result == 0);
}

// implicit_conversion

TEST_CASE(implicit_conversion_scalar)
{
    auto conv  = migraphx::implicit_conversion(42);
    int result = conv;
    EXPECT(result == 42);
}

TEST_CASE(implicit_conversion_float_to_int)
{
    auto conv  = migraphx::implicit_conversion(3.0f);
    int result = conv;
    EXPECT(result == 3);
}
