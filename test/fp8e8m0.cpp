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
#include <migraphx/ranges.hpp>
#include <migraphx/fp8e8m0.hpp>
#include "test.hpp"

#include <limits>
#include <map>
#include <iomanip>
#include <bitset>
#include <set>
#include <random>

template <class T, class U>
static bool bit_equal(const T& x, const U& y)
{
    static_assert(sizeof(T) == sizeof(U));
    using type = std::array<char, sizeof(T)>;
    return migraphx::bit_cast<type>(x) == migraphx::bit_cast<type>(y);
}

TEST_CASE(check_numeric_limits)
{
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::min(), uint8_t{0x00}));
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::lowest(), uint8_t{0x00}));
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::max(), uint8_t{0xfe}));
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::epsilon(), uint8_t{0x7f}));
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::denorm_min(), uint8_t{0x00}));
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::infinity(), uint8_t{0x7f}));
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::quiet_NaN(), uint8_t{0xff}));
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::signaling_NaN(), uint8_t{0xff}));
}

const static std::map<float, uint8_t>& fp8e8m0_lut()
{
    static const std::map<float, uint8_t> result = {
        {5.877471754111437539843683E-39, 0x00},
        {-5.877471754111437539843683E-39, 0x00},
        {1.175494350822287507968737E-38, 0x01},
        {-1.175494350822287507968737E-38, 0x01},
        {7.888609052210118054117286E-31, 0x1b},
        {-7.888609052210118054117286E-31, 0x1b},
        {1.355252715606880542509316E-20, 0x3d},
        {-1.355252715606880542509316E-20, 0x3d},
        {0.0625, 0x7b},
        {-0.0625, 0x7b},
        {0.125, 0x7c},
        {-0.125, 0x7c},
        {0.25, 0x7d},
        {-0.25, 0x7d},
        {0.5, 0x7e},
        {-0.5, 0x7e},
        {1., 0x7f},
        {-1., 0x7f},
        {2., 0x80},
        {-2., 0x80},
        {4., 0x81},
        {-4., 0x81},
        {8., 0x82},
        {-8., 0x82},
        {16., 0x83},
        {-16., 0x83},
        {268435456, 0x9b},
        {-268435456, 0x9b},
        {4611686018427387904, 0xbd},
        {-4611686018427387904, 0xbd},
        {1.701411834604692317316873E+38, 0xfe},
        {-1.701411834604692317316873E+38, 0xfe},
        {std::numeric_limits<float>::quiet_NaN(), 0xff}};
    return result;
}
TEST_CASE(check_fp8e8m0_values)
{
    for(auto [f, x] : fp8e8m0_lut())
    {

        auto h = migraphx::bit_cast<migraphx::fp8e8m0>(x);
        if(std::isnan(f))
        {
            CHECK(std::isnan(h));
        }
        else
        {
            CHECK(bit_equal(x, migraphx::fp8e8m0(f)));
            // absolute value because fp8e8m0 only takes exponent bits
            // and has no negative numbers
            CHECK(migraphx::float_equal(float(h), std::abs(f)));
        }
    }
}

TEST_CASE(check_flows)
{
    // underflow, lowest possible value is 2^(-127)
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::min() *
                        std::numeric_limits<migraphx::fp8e8m0>::min(),
                    std::numeric_limits<migraphx::fp8e8m0>::min()));
    // overflow
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::max() +
                        std::numeric_limits<migraphx::fp8e8m0>::max(),
                    std::numeric_limits<migraphx::fp8e8m0>::signaling_NaN()));
    CHECK(bit_equal(std::numeric_limits<migraphx::fp8e8m0>::max() +
                        std::numeric_limits<migraphx::fp8e8m0>::min(),
                    std::numeric_limits<migraphx::fp8e8m0>::max()));
}

TEST_CASE(test_nan)
{
    float f_qnan = std::numeric_limits<float>::quiet_NaN();
    migraphx::fp8e8m0 fp8e8m0_qnan(f_qnan);
    EXPECT(fp8e8m0_qnan.is_nan());
    EXPECT(std::isnan(fp8e8m0_qnan));

    float f_snan = std::numeric_limits<float>::signaling_NaN();
    migraphx::fp8e8m0 fp8e8m0_snan(f_snan);
    EXPECT(fp8e8m0_snan.is_nan());
    EXPECT(std::isnan(fp8e8m0_snan));
}

TEST_CASE(test_bool)
{
    // there is no zero value in E8M0
    float two   = 2.0;
    float other = 0.125;
    migraphx::fp8e8m0 fp8e8m0_two(two);
    migraphx::fp8e8m0 fp8e8m0_other(other);
    EXPECT(static_cast<bool>(std::numeric_limits<migraphx::fp8e8m0>::min()));
    EXPECT(static_cast<bool>(fp8e8m0_two));
    EXPECT(static_cast<bool>(fp8e8m0_other));
}

TEST_CASE(test_pos_infinity)
{
    float finf = std::numeric_limits<float>::infinity();
    CHECK(bit_equal(migraphx::fp8e8m0(finf), std::numeric_limits<migraphx::fp8e8m0>::quiet_NaN()));
}

TEST_CASE(test_neg_infinity)
{
    float finf = -1.0 * std::numeric_limits<float>::infinity();
    CHECK(bit_equal(migraphx::fp8e8m0(finf), std::numeric_limits<migraphx::fp8e8m0>::quiet_NaN()));
}

TEST_CASE(test_f32_max)
{
    float fmax = std::numeric_limits<float>::max();
    CHECK(bit_equal(migraphx::fp8e8m0(fmax), std::numeric_limits<migraphx::fp8e8m0>::max()));
}

TEST_CASE(test_f32_denorm_min)
{
    float fmin = std::numeric_limits<float>::denorm_min();
    CHECK(bit_equal(migraphx::fp8e8m0(fmin), std::numeric_limits<migraphx::fp8e8m0>::min()));
}

TEST_CASE(test_f32_lowest)
{
    float flowest = std::numeric_limits<float>::lowest();
    CHECK(bit_equal(migraphx::fp8e8m0(flowest), std::numeric_limits<migraphx::fp8e8m0>::max()));
}

TEST_CASE(test_isfinite)
{
    EXPECT(std::isfinite(migraphx::fp8e8m0(0.0)));
    EXPECT(std::isfinite(migraphx::fp8e8m0(-0.0)));
    EXPECT(
        not std::isfinite(migraphx::fp8e8m0(std::numeric_limits<migraphx::fp8e8m0>::quiet_NaN())));
}

TEST_CASE(test_binary_ops)
{
    auto a = migraphx::fp8e8m0(1.0);
    auto b = migraphx::fp8e8m0(2.0);
    auto c = migraphx::fp8e8m0(4.0);
    EXPECT(migraphx::float_equal((a + a), b));
    EXPECT(migraphx::float_equal((b + b), c));
    EXPECT(migraphx::float_equal((a + b), b));

    auto e = migraphx::fp8e8m0(8.0);
    auto f = migraphx::fp8e8m0(0.125);
    EXPECT(e > f);
    EXPECT(f < e);
    EXPECT(f <= e);
    EXPECT(e >= f);
    EXPECT(e <= e);
    EXPECT(f >= f);
    EXPECT(not migraphx::float_equal(f, e));
}

TEST_CASE(test_stream_op)
{
    auto a = migraphx::fp8e8m0(4.0);
    std::stringstream ss;
    ss << a;
    EXPECT(std::string("4") == ss.str());
    ss     = std::stringstream();
    auto b = std::numeric_limits<migraphx::fp8e8m0>::quiet_NaN();
    ss << b;
    EXPECT(std::string("nan") == ss.str());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
