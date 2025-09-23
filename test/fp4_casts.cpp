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
#include <migraphx/float8.hpp>
#include "test.hpp"

#include <cmath>
#include <limits>

namespace test_fp4_casts {
static constexpr std::array<float, 16> e2m1_lut = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
} // namespace test_fp4_casts

static float fp4e2m1_to_fp32_value(uint8_t input) { return test_fp4_casts::e2m1_lut[input]; }

TEST_CASE(test_fp4_to_float)
{
    std::vector<uint8_t> bit_vals(16);
    std::iota(bit_vals.begin(), bit_vals.end(), 0);
    EXPECT(std::all_of(bit_vals.begin(), bit_vals.end(), [](uint8_t bit_val) {
        float float_val = migraphx::fp4_to_float(bit_val);
        return migraphx::float_equal(float_val, fp4e2m1_to_fp32_value(bit_val));
    }));
}

TEST_CASE(test_fp4_to_fp8)
{
    std::vector<uint8_t> bit_vals(16);
    std::iota(bit_vals.begin(), bit_vals.end(), 0);
    EXPECT(std::all_of(bit_vals.begin(), bit_vals.end(), [](uint8_t bit_val) {
        auto float_val = migraphx::fp4_to_fp8(bit_val);
        return migraphx::float_equal(float_val, fp4e2m1_to_fp32_value(bit_val));
    }));
}

TEST_CASE(test_constexpr_fp4_to_float)
{
    constexpr std::array<float, 16> res_array = {migraphx::fp4_to_float(0x0),
                                                 migraphx::fp4_to_float(0x1),
                                                 migraphx::fp4_to_float(0x2),
                                                 migraphx::fp4_to_float(0x3),
                                                 migraphx::fp4_to_float(0x4),
                                                 migraphx::fp4_to_float(0x5),
                                                 migraphx::fp4_to_float(0x6),
                                                 migraphx::fp4_to_float(0x7),
                                                 migraphx::fp4_to_float(0x8),
                                                 migraphx::fp4_to_float(0x9),
                                                 migraphx::fp4_to_float(0xA),
                                                 migraphx::fp4_to_float(0xB),
                                                 migraphx::fp4_to_float(0xC),
                                                 migraphx::fp4_to_float(0xD),
                                                 migraphx::fp4_to_float(0xE),
                                                 migraphx::fp4_to_float(0xF)};
    EXPECT(std::equal(res_array.begin(), res_array.end(), test_fp4_casts::e2m1_lut.begin()));
}

TEST_CASE(test_constexpr_fp4_to_fp8)
{
    constexpr std::array<migraphx::fp8::fp8e4m3fn, 16> res_array = {migraphx::fp4_to_fp8(0x0),
                                                                    migraphx::fp4_to_fp8(0x1),
                                                                    migraphx::fp4_to_fp8(0x2),
                                                                    migraphx::fp4_to_fp8(0x3),
                                                                    migraphx::fp4_to_fp8(0x4),
                                                                    migraphx::fp4_to_fp8(0x5),
                                                                    migraphx::fp4_to_fp8(0x6),
                                                                    migraphx::fp4_to_fp8(0x7),
                                                                    migraphx::fp4_to_fp8(0x8),
                                                                    migraphx::fp4_to_fp8(0x9),
                                                                    migraphx::fp4_to_fp8(0xA),
                                                                    migraphx::fp4_to_fp8(0xB),
                                                                    migraphx::fp4_to_fp8(0xC),
                                                                    migraphx::fp4_to_fp8(0xD),
                                                                    migraphx::fp4_to_fp8(0xE),
                                                                    migraphx::fp4_to_fp8(0xF)};
    EXPECT(std::equal(res_array.begin(), res_array.end(), test_fp4_casts::e2m1_lut.begin()));
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
        {-5.25, 0xF},
        {0.25, 0x0},
        {0.5, 0x1},
        {0.75, 0x2},
        {1.25, 0x2},
        {1.75, 0x4},
        {2.5, 0x4},
        {3.5, 0x6},
        {5.0, 0x6},
        {-0.25, 0x8},
        {-0.5, 0x9},
        {-0.75, 0xA},
        {-1.25, 0xA},
        {-1.75, 0xC},
        {-2.5, 0xC},
        {-3.5, 0xE},
        {-5.0, 0xE},
        {std::numeric_limits<float>::infinity(), 0x7},
        {-std::numeric_limits<float>::infinity(), 0xF},
        {std::numeric_limits<float>::signaling_NaN(), 0x0},
        {std::numeric_limits<float>::quiet_NaN(), 0x0}};
    EXPECT(bool{std::all_of(test_vals.begin(), test_vals.end(), [](const auto sample) {
        uint8_t conv = migraphx::cast_to_fp4(sample.first);
        uint8_t gold = sample.second;
        if(conv != gold)
            std::cout << "conv: " << int(conv) << ", gold: " << int(gold) << "\n";
        return conv == gold;
    })});
}

// Disabled test for constexpr version of float_to_fp4
// TEST_CASE(test_constexpr_float_to_fp4)
//{
//    constexpr std::array<uint8_t, 12> res_array = {
//        migraphx::cast_to_fp4(10.f),
//        migraphx::cast_to_fp4(-20.f),
//        migraphx::cast_to_fp4(0.11f),
//        migraphx::cast_to_fp4(-0.11f),
//        migraphx::cast_to_fp4(0.25f),
//        migraphx::cast_to_fp4(1.75f),
//        migraphx::cast_to_fp4(-0.25f),
//        migraphx::cast_to_fp4(-1.75f),
//        migraphx::cast_to_fp4(3.5f),
//        migraphx::cast_to_fp4(-3.5f),
//        migraphx::cast_to_fp4(0.00128f),
//        migraphx::cast_to_fp4(82910.0f),
//    };
//
//    std::array<uint8_t, 12> gold_array = {
//        0x7, 0xF, 0x0, 0x8, 0x0, 0x4, 0x8, 0xC, 0x6, 0xE, 0x0, 0x7};
//    EXPECT(std::equal(res_array.begin(), res_array.end(), gold_array.begin()));
//}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
