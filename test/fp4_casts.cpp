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
#include <migraphx/float_equal.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/fp4_casts.hpp>
#include "test.hpp"

#include <cmath>
#include <limits>

static float fp4e2m1_to_fp32_value(uint8_t input)
{
    constexpr std::array<float, 16> e2m1_lut = {
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
    return e2m1_lut[input];
}

TEST_CASE(test_fp4_to_float)
{
    std::vector<uint8_t> bit_vals(16);
    std::iota(bit_vals.begin(), bit_vals.end(), 0);
    EXPECT(bool{std::all_of(bit_vals.begin(), bit_vals.end(), [](uint8_t bit_val) {
        float float_val = migraphx::fp4_to_float(bit_val);
        return migraphx::float_equal(float_val, fp4e2m1_to_fp32_value(bit_val));
    })});
}

TEST_CASE(test_float_to_fp4)
{
    std::vector<std::pair<float, uint8_t>> test_vals = {
        {10.f, 0x7},
        {-20.f, 0xF},
        {0.11, 0x0},
        {-0.11, 0x8},
        {2.5, 0x4},
        {-2.5, 0xC},
        {2.6, 0x5},
        {-2.6, 0xD},
        {0.f, 0x0},
        {-0.f, 0x8},
        {4.212, 0x6},
        {0.387, 0x1},
        {0.00128, 0x0},
        {-0.5, 0x9},
        {0.5, 0x1},
        {std::numeric_limits<float>::infinity(), 0x7},
        {-std::numeric_limits<float>::infinity(), 0xF},
        {std::numeric_limits<float>::signaling_NaN(), 0x7},
        {std::numeric_limits<float>::quiet_NaN(), 0x7}};
    EXPECT(bool{std::all_of(test_vals.begin(), test_vals.end(), [](const auto sample) {
        return migraphx::float_to_fp4(sample.first) == sample.second;
    })});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
