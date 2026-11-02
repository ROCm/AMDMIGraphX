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
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/test.hpp>
#include <migraphx/kernels/float_equal.hpp>

// Regression: a scalar mixed with a vector must be splatted across all lanes,
// not placed in lane 0 with the remaining lanes zeroed
TEST_CASE(max_vec_scalar)
{
    migraphx::vec<float, 4> x{-1.0f, 0.5f, 2.0f, 3.0f};
    auto result = migraphx::max(x, 1.1f);
    EXPECT(migraphx::float_equal(result[0], 1.1f));
    EXPECT(migraphx::float_equal(result[1], 1.1f));
    EXPECT(migraphx::float_equal(result[2], 2.0f));
    EXPECT(migraphx::float_equal(result[3], 3.0f));
}

TEST_CASE(max_scalar_vec)
{
    migraphx::vec<float, 4> x{-1.0f, 0.5f, 2.0f, 3.0f};
    auto result = migraphx::max(1.1f, x);
    EXPECT(migraphx::float_equal(result[0], 1.1f));
    EXPECT(migraphx::float_equal(result[1], 1.1f));
    EXPECT(migraphx::float_equal(result[2], 2.0f));
    EXPECT(migraphx::float_equal(result[3], 3.0f));
}

TEST_CASE(min_vec_scalar)
{
    migraphx::vec<float, 4> x{-1.0f, 0.5f, 2.0f, 3.0f};
    auto result = migraphx::min(x, 1.1f);
    EXPECT(migraphx::float_equal(result[0], -1.0f));
    EXPECT(migraphx::float_equal(result[1], 0.5f));
    EXPECT(migraphx::float_equal(result[2], 1.1f));
    EXPECT(migraphx::float_equal(result[3], 1.1f));
}

TEST_CASE(min_scalar_vec)
{
    migraphx::vec<float, 4> x{-1.0f, 0.5f, 2.0f, 3.0f};
    auto result = migraphx::min(1.1f, x);
    EXPECT(migraphx::float_equal(result[0], -1.0f));
    EXPECT(migraphx::float_equal(result[1], 0.5f));
    EXPECT(migraphx::float_equal(result[2], 1.1f));
    EXPECT(migraphx::float_equal(result[3], 1.1f));
}

// Clip pattern: both bounds are scalars mixed with a vector
TEST_CASE(clip_vec_scalar_bounds)
{
    migraphx::vec<float, 4> x{-1.0f, 0.5f, 2.0f, 10.0f};
    auto result = migraphx::min(migraphx::max(x, 1.1f), 4.0f);
    EXPECT(migraphx::float_equal(result[0], 1.1f));
    EXPECT(migraphx::float_equal(result[1], 1.1f));
    EXPECT(migraphx::float_equal(result[2], 2.0f));
    EXPECT(migraphx::float_equal(result[3], 4.0f));
}

TEST_CASE(max_vec_scalar_int)
{
    migraphx::vec<int, 4> x{-2, 0, 3, 5};
    auto result = migraphx::max(x, 1);
    EXPECT(migraphx::vec_at(result, 0) == 1);
    EXPECT(migraphx::vec_at(result, 1) == 1);
    EXPECT(migraphx::vec_at(result, 2) == 3);
    EXPECT(migraphx::vec_at(result, 3) == 5);
}

TEST_CASE(min_vec_scalar_int)
{
    migraphx::vec<int, 4> x{-2, 0, 3, 5};
    auto result = migraphx::min(x, 1);
    EXPECT(migraphx::vec_at(result, 0) == -2);
    EXPECT(migraphx::vec_at(result, 1) == 0);
    EXPECT(migraphx::vec_at(result, 2) == 1);
    EXPECT(migraphx::vec_at(result, 3) == 1);
}

TEST_CASE(max_mixed_scalars)
{
    EXPECT(migraphx::float_equal(migraphx::max(1, 2.5f), 2.5f));
    EXPECT(migraphx::float_equal(migraphx::min(1, 2.5f), 1.0f));
}
