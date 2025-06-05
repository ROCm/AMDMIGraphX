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
#include <cmath>
#include <migraphx/float_equal.hpp>
#include <migraphx/float4.hpp>
#include <migraphx/ranges.hpp>
#include "test.hpp"

#include <limits>
#include <bitset>

template <class T, class U>
static bool bit_equal(const T& x, const U& y)
{
    static_assert(sizeof(T) == sizeof(U));
    using type = std::array<char, sizeof(T)>;
    return migraphx::bit_cast<type>(x) == migraphx::bit_cast<type>(y);
}

TEST_CASE(check_numeric_limits)
{
    migraphx::float4 a = std::numeric_limits<migraphx::float4>::min();
    std::bitset<8> x(a);
    std::cout << x << std::endl;
    auto b = uint8_t{0b01000000};
    std::bitset<8> y(b);
    std::cout << y << std::endl;
    migraphx::float4 c = std::numeric_limits<migraphx::float4>::min();
    std::bitset<8> z(c);
    std::cout << z << std::endl;
    migraphx::float4 d = std::numeric_limits<migraphx::float4>::min();
    std::bitset<8> g(d);
    std::cout << g << std::endl;
    CHECK(not std::numeric_limits<migraphx::float4>::has_infinity);
    CHECK(not std::numeric_limits<migraphx::float4>::has_quiet_NaN);
    CHECK(not std::numeric_limits<migraphx::float4>::has_signaling_NaN);
    CHECK(bit_equal(std::numeric_limits<migraphx::float4>::min(), uint8_t{0b01000000}));
    CHECK(bit_equal(std::numeric_limits<migraphx::float4>::lowest(), uint8_t{0b11110000}));
    CHECK(bit_equal(std::numeric_limits<migraphx::float4>::max(), uint8_t{0b01110000}));
    CHECK(bit_equal(std::numeric_limits<migraphx::float4>::epsilon(), uint8_t{0b01000000}));
    CHECK(bit_equal(std::numeric_limits<migraphx::float4>::denorm_min(), uint8_t{0b01000000}));
    CHECK(bit_equal(std::numeric_limits<migraphx::float4>::infinity(), uint8_t{0b00000000}));
    CHECK(bit_equal(std::numeric_limits<migraphx::float4>::quiet_NaN(), uint8_t{0b00000000}));
    CHECK(bit_equal(std::numeric_limits<migraphx::float4>::signaling_NaN(), uint8_t{0b00000000}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
