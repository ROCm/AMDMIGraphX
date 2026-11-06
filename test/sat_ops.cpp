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
#include <migraphx/sat_ops.hpp>
#include <cstdint>
#include <limits>
#include <test.hpp>

using migraphx::add_sat;
using migraphx::mul_sat;
using migraphx::sub_sat;

template <class T>
static constexpr T tmin()
{
    return std::numeric_limits<T>::min();
}

template <class T>
static constexpr T tmax()
{
    return std::numeric_limits<T>::max();
}

// ---- add_sat ----

TEST_CASE(add_sat_no_overflow)
{
    EXPECT(add_sat(int32_t{2}, int32_t{3}) == 5);
    EXPECT(add_sat(int32_t{-4}, int32_t{1}) == -3);
    EXPECT(add_sat(int64_t{100}, int64_t{-250}) == -150);
    EXPECT(add_sat(uint32_t{10}, uint32_t{20}) == 30u);
}

TEST_CASE(add_sat_signed_positive_overflow)
{
    EXPECT(add_sat(tmax<int32_t>(), int32_t{1}) == tmax<int32_t>());
    EXPECT(add_sat(tmax<int32_t>(), tmax<int32_t>()) == tmax<int32_t>());
    EXPECT(add_sat(tmax<int64_t>(), int64_t{1}) == tmax<int64_t>());
}

TEST_CASE(add_sat_signed_negative_overflow)
{
    EXPECT(add_sat(tmin<int32_t>(), int32_t{-1}) == tmin<int32_t>());
    EXPECT(add_sat(tmin<int32_t>(), tmin<int32_t>()) == tmin<int32_t>());
    EXPECT(add_sat(tmin<int64_t>(), int64_t{-5}) == tmin<int64_t>());
}

TEST_CASE(add_sat_unsigned_overflow)
{
    EXPECT(add_sat(tmax<uint32_t>(), uint32_t{1}) == tmax<uint32_t>());
    EXPECT(add_sat(tmax<uint64_t>(), uint64_t{100}) == tmax<uint64_t>());
}

TEST_CASE(add_sat_boundary_exact)
{
    // Reaching the boundary exactly must not saturate.
    EXPECT(add_sat(int32_t(tmax<int32_t>() - 1), int32_t{1}) == tmax<int32_t>());
    EXPECT(add_sat(int32_t(tmin<int32_t>() + 1), int32_t{-1}) == tmin<int32_t>());
}

// ---- sub_sat ----

TEST_CASE(sub_sat_no_overflow)
{
    EXPECT(sub_sat(int32_t{5}, int32_t{3}) == 2);
    EXPECT(sub_sat(int32_t{-4}, int32_t{1}) == -5);
    EXPECT(sub_sat(int64_t{100}, int64_t{-250}) == 350);
    EXPECT(sub_sat(uint32_t{30}, uint32_t{20}) == 10u);
}

TEST_CASE(sub_sat_signed_positive_overflow)
{
    // x - (negative) can overflow past max.
    EXPECT(sub_sat(tmax<int32_t>(), int32_t{-1}) == tmax<int32_t>());
    EXPECT(sub_sat(tmax<int64_t>(), tmin<int64_t>()) == tmax<int64_t>());
}

TEST_CASE(sub_sat_signed_negative_overflow)
{
    EXPECT(sub_sat(tmin<int32_t>(), int32_t{1}) == tmin<int32_t>());
    EXPECT(sub_sat(tmin<int32_t>(), tmax<int32_t>()) == tmin<int32_t>());
}

TEST_CASE(sub_sat_unsigned_underflow)
{
    // Unsigned underflow clamps to zero (min), not wraparound.
    EXPECT(sub_sat(uint32_t{0}, uint32_t{1}) == 0u);
    EXPECT(sub_sat(uint32_t{5}, uint32_t{10}) == 0u);
    EXPECT(sub_sat(tmin<uint64_t>(), tmax<uint64_t>()) == 0u);
}

TEST_CASE(sub_sat_unsigned_no_underflow)
{
    EXPECT(sub_sat(tmax<uint32_t>(), tmax<uint32_t>()) == 0u);
    EXPECT(sub_sat(uint32_t{10}, uint32_t{10}) == 0u);
}

// ---- mul_sat ----

TEST_CASE(mul_sat_no_overflow)
{
    EXPECT(mul_sat(int32_t{6}, int32_t{7}) == 42);
    EXPECT(mul_sat(int32_t{-6}, int32_t{7}) == -42);
    EXPECT(mul_sat(int32_t{-6}, int32_t{-7}) == 42);
    EXPECT(mul_sat(int64_t{0}, tmax<int64_t>()) == 0);
    EXPECT(mul_sat(uint32_t{6}, uint32_t{7}) == 42u);
}

TEST_CASE(mul_sat_signed_positive_overflow)
{
    // (+ * +) and (- * -) overflow toward max.
    EXPECT(mul_sat(tmax<int32_t>(), int32_t{2}) == tmax<int32_t>());
    EXPECT(mul_sat(tmax<int64_t>(), int64_t{2}) == tmax<int64_t>());
    EXPECT(mul_sat(tmin<int32_t>(), tmin<int32_t>()) == tmax<int32_t>());
}

TEST_CASE(mul_sat_signed_negative_overflow)
{
    // Mixed signs overflow toward min.
    EXPECT(mul_sat(tmax<int32_t>(), int32_t{-2}) == tmin<int32_t>());
    EXPECT(mul_sat(tmin<int32_t>(), int32_t{2}) == tmin<int32_t>());
    EXPECT(mul_sat(tmin<int64_t>(), int64_t{2}) == tmin<int64_t>());
}

TEST_CASE(mul_sat_unsigned_overflow)
{
    EXPECT(mul_sat(tmax<uint32_t>(), uint32_t{2}) == tmax<uint32_t>());
    EXPECT(mul_sat(tmax<uint64_t>(), tmax<uint64_t>()) == tmax<uint64_t>());
}

TEST_CASE(mul_sat_by_zero)
{
    EXPECT(mul_sat(tmax<int32_t>(), int32_t{0}) == 0);
    EXPECT(mul_sat(tmin<int32_t>(), int32_t{0}) == 0);
    EXPECT(mul_sat(tmax<uint32_t>(), uint32_t{0}) == 0u);
}

// ---- constexpr usability ----

TEST_CASE(sat_ops_are_constexpr)
{
    static_assert(add_sat(tmax<int32_t>(), int32_t{1}) == tmax<int32_t>(), "add_sat constexpr");
    static_assert(sub_sat(uint32_t{0}, uint32_t{1}) == 0u, "sub_sat constexpr");
    static_assert(mul_sat(tmax<int32_t>(), int32_t{2}) == tmax<int32_t>(), "mul_sat constexpr");
    EXPECT(true);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
