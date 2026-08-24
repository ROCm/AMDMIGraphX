/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <test.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/gpu/compile_gen.hpp>

static const auto find_fast_axis = test::make_function("find_fast_axis", [](auto&&... xs) {
    return migraphx::gpu::gen::find_fast_axis(static_cast<decltype(xs)>(xs)...);
});

TEST_CASE(test_find_fast_axis)
{
    EXPECT(find_fast_axis(migraphx::shape{migraphx::shape::float_type, {2, 2, 2, 6, 3}}) == 4);
    EXPECT(find_fast_axis(migraphx::shape{
               migraphx::shape::float_type, {2, 2, 2, 6, 3}, {72, 6, 1, 12, 2}}) == 2);
    EXPECT(find_fast_axis(
               migraphx::shape{migraphx::shape::float_type, {64, 512, 32, 32}, {0, 1, 0, 0}}) == 1);
    EXPECT(find_fast_axis(
               migraphx::shape{migraphx::shape::float_type, {64, 512, 32, 32}, {0, 0, 0, 0}}) == 3);
}

static const auto compute_factor = test::make_function(
    "tile::compute_factor", MIGRAPHX_LIFT(migraphx::gpu::gen::tile::compute_factor));

TEST_CASE(test_compute_factor_one) { EXPECT(compute_factor(1) == 1); }

TEST_CASE(test_compute_factor_small_primes)
{
    EXPECT(compute_factor(2) == 2);
    EXPECT(compute_factor(3) == 3);
    EXPECT(compute_factor(5) == 5);
    EXPECT(compute_factor(7) == 7);
    EXPECT(compute_factor(11) == 11);
}

TEST_CASE(test_compute_factor_large_primes)
{
    EXPECT(compute_factor(13) == 1);
    EXPECT(compute_factor(101) == 1);
}

TEST_CASE(test_compute_factor_composite)
{
    EXPECT(compute_factor(12) == 12);
    EXPECT(compute_factor(30) == 30);
    EXPECT(compute_factor(35) == 35);
}

TEST_CASE(test_compute_factor_skips_large_prime_factors)
{
    EXPECT(compute_factor(26) == 2);
    EXPECT(compute_factor(39) == 3);
}

TEST_CASE(test_compute_factor_default_max_size)
{
    EXPECT(compute_factor(64) == 64);
    EXPECT(compute_factor(128) == 64);
    EXPECT(compute_factor(1024) == 64);
}

TEST_CASE(test_compute_factor_custom_max_size)
{
    EXPECT(compute_factor(64, 8) == 8);
    EXPECT(compute_factor(64, 1) == 1);
    EXPECT(compute_factor(12, 2) == 2);
}

TEST_CASE(test_compute_factor_can_exceed_max_size)
{
    // max_size is checked before multiplying, so the last factor can overshoot it
    EXPECT(compute_factor(25, 4) == 5);
    EXPECT(compute_factor(12, 6) == 12);
    EXPECT(compute_factor(6, 4) == 6);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
