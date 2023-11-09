/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/migraphx_float8.hpp>
#include <migraphx/half.hpp>
#include <migraphx/ranges.hpp>
#include "test.hpp"

#include <limits>

float fp8e4m3fnuz_to_fp32_value(uint8_t input)
{
    constexpr std::array<float, 256> e4m3fnuz_lut = {
        0.0f,           0.0009765625f,  0.001953125f,
        0.0029296875f,  0.00390625f,    0.0048828125f,
        0.005859375f,   0.0068359375f,  0.0078125f,
        0.0087890625f,  0.009765625f,   0.0107421875f,
        0.01171875f,    0.0126953125f,  0.013671875f,
        0.0146484375f,  0.015625f,      0.017578125f,
        0.01953125f,    0.021484375f,   0.0234375f,
        0.025390625f,   0.02734375f,    0.029296875f,
        0.03125f,       0.03515625f,    0.0390625f,
        0.04296875f,    0.046875f,      0.05078125f,
        0.0546875f,     0.05859375f,    0.0625f,
        0.0703125f,     0.078125f,      0.0859375f,
        0.09375f,       0.1015625f,     0.109375f,
        0.1171875f,     0.125f,         0.140625f,
        0.15625f,       0.171875f,      0.1875f,
        0.203125f,      0.21875f,       0.234375f,
        0.25f,          0.28125f,       0.3125f,
        0.34375f,       0.375f,         0.40625f,
        0.4375f,        0.46875f,       0.5f,
        0.5625f,        0.625f,         0.6875f,
        0.75f,          0.8125f,        0.875f,
        0.9375f,        1.0f,           1.125f,
        1.25f,          1.375f,         1.5f,
        1.625f,         1.75f,          1.875f,
        2.0f,           2.25f,          2.5f,
        2.75f,          3.0f,           3.25f,
        3.5f,           3.75f,          4.0f,
        4.5f,           5.0f,           5.5f,
        6.0f,           6.5f,           7.0f,
        7.5f,           8.0f,           9.0f,
        10.0f,          11.0f,          12.0f,
        13.0f,          14.0f,          15.0f,
        16.0f,          18.0f,          20.0f,
        22.0f,          24.0f,          26.0f,
        28.0f,          30.0f,          32.0f,
        36.0f,          40.0f,          44.0f,
        48.0f,          52.0f,          56.0f,
        60.0f,          64.0f,          72.0f,
        80.0f,          88.0f,          96.0f,
        104.0f,         112.0f,         120.0f,
        128.0f,         144.0f,         160.0f,
        176.0f,         192.0f,         208.0f,
        224.0f,         240.0f,         std::numeric_limits<float>::quiet_NaN(),
        -0.0009765625f, -0.001953125f,  -0.0029296875f,
        -0.00390625f,   -0.0048828125f, -0.005859375f,
        -0.0068359375f, -0.0078125f,    -0.0087890625f,
        -0.009765625f,  -0.0107421875f, -0.01171875f,
        -0.0126953125f, -0.013671875f,  -0.0146484375f,
        -0.015625f,     -0.017578125f,  -0.01953125f,
        -0.021484375f,  -0.0234375f,    -0.025390625f,
        -0.02734375f,   -0.029296875f,  -0.03125f,
        -0.03515625f,   -0.0390625f,    -0.04296875f,
        -0.046875f,     -0.05078125f,   -0.0546875f,
        -0.05859375f,   -0.0625f,       -0.0703125f,
        -0.078125f,     -0.0859375f,    -0.09375f,
        -0.1015625f,    -0.109375f,     -0.1171875f,
        -0.125f,        -0.140625f,     -0.15625f,
        -0.171875f,     -0.1875f,       -0.203125f,
        -0.21875f,      -0.234375f,     -0.25f,
        -0.28125f,      -0.3125f,       -0.34375f,
        -0.375f,        -0.40625f,      -0.4375f,
        -0.46875f,      -0.5f,          -0.5625f,
        -0.625f,        -0.6875f,       -0.75f,
        -0.8125f,       -0.875f,        -0.9375f,
        -1.0f,          -1.125f,        -1.25f,
        -1.375f,        -1.5f,          -1.625f,
        -1.75f,         -1.875f,        -2.0f,
        -2.25f,         -2.5f,          -2.75f,
        -3.0f,          -3.25f,         -3.5f,
        -3.75f,         -4.0f,          -4.5f,
        -5.0f,          -5.5f,          -6.0f,
        -6.5f,          -7.0f,          -7.5f,
        -8.0f,          -9.0f,          -10.0f,
        -11.0f,         -12.0f,         -13.0f,
        -14.0f,         -15.0f,         -16.0f,
        -18.0f,         -20.0f,         -22.0f,
        -24.0f,         -26.0f,         -28.0f,
        -30.0f,         -32.0f,         -36.0f,
        -40.0f,         -44.0f,         -48.0f,
        -52.0f,         -56.0f,         -60.0f,
        -64.0f,         -72.0f,         -80.0f,
        -88.0f,         -96.0f,         -104.0f,
        -112.0f,        -120.0f,        -128.0f,
        -144.0f,        -160.0f,        -176.0f,
        -192.0f,        -208.0f,        -224.0f,
        -240.0f,
    };

    return e4m3fnuz_lut[input];
}

TEST_CASE(test_fp8_cast_to_float)
{
    std::vector<uint8_t> bit_vals(256);
    std::iota(bit_vals.begin(), bit_vals.end(), 0);
    EXPECT(bool{std::all_of(bit_vals.begin(), bit_vals.end(), [](uint8_t bit_val) {
        migraphx_fp8::fp8e4m3fnuz fp8_val(bit_val, migraphx_fp8::fp8e4m3fnuz::from_bits());
        if(std::isnan(float(fp8_val)) and std::isnan(fp8e4m3fnuz_to_fp32_value(bit_val)))
        {
            return true;
        }
        return migraphx::float_equal(float(fp8_val), fp8e4m3fnuz_to_fp32_value(bit_val));
    })});
}

TEST_CASE(test_positive_zero)
{
    float zero = 0.0;
    migraphx_fp8::fp8e4m3fnuz fp8_zero(zero);
    EXPECT(fp8_zero.is_zero());
    EXPECT(migraphx::float_equal(zero, float(fp8_zero)));
}

TEST_CASE(test_negative_zero)
{
    float nzero = -0.0;
    float pzero = 0.0;
    migraphx_fp8::fp8e4m3fnuz fp8_nzero(nzero);
    EXPECT(fp8_nzero.is_zero());
    //  negative zero gets converted to positive zero
    EXPECT(migraphx::float_equal(pzero, float(fp8_nzero)));
}

TEST_CASE(test_nan_1)
{
    float fnan = std::numeric_limits<float>::quiet_NaN();
    migraphx_fp8::fp8e4m3fnuz fp8_nan(fnan);
    EXPECT(fp8_nan.is_nan());
    EXPECT(std::isnan(fp8_nan));
}

TEST_CASE(test_nan_2)
{
    migraphx_fp8::fp8e4m3fnuz fp8_nan(std::numeric_limits<migraphx_fp8::fp8e4m3fnuz>::quiet_NaN());
    EXPECT(fp8_nan.is_nan());
    EXPECT(std::isnan(fp8_nan));
    EXPECT(std::isnan(float(fp8_nan)));
}

TEST_CASE(test_infinity_1)
{
    float finf = std::numeric_limits<float>::infinity();
    // no inf in fp8e4m3fnuz
    migraphx_fp8::fp8e4m3fnuz fp8_nan(finf);
    EXPECT(fp8_nan.is_nan());
    EXPECT(std::isnan(float(fp8_nan)));
}

TEST_CASE(test_infinity_2)
{
    // no inf in fp8e4m3fnuz, it gets converted to NaNs
    migraphx_fp8::fp8e4m3fnuz fp8_nan(std::numeric_limits<migraphx_fp8::fp8e4m3fnuz>::infinity());
    EXPECT(fp8_nan.is_nan());
    EXPECT(std::isnan(float(fp8_nan)));
}

TEST_CASE(test_infinity_3)
{
    // neg inf
    float finf = -1.0 * std::numeric_limits<float>::infinity();
    // no inf in fp8e4m3fnuz
    migraphx_fp8::fp8e4m3fnuz fp8_nan(finf);
    EXPECT(fp8_nan.is_nan());
    EXPECT(std::isnan(float(fp8_nan)));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
