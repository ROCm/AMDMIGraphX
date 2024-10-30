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
#include <cmath>
#include <migraphx/float_equal.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/bf16.hpp>
#include "test.hpp"

#include <limits>
#include <map>
#include <iomanip>

template <class T, class U>
bool bit_equal(const T& x, const U& y)
{
    static_assert(sizeof(T) == sizeof(U));
    using type = std::array<char, sizeof(T)>;
    return migraphx::bit_cast<type>(x) == migraphx::bit_cast<type>(y);
}

TEST_CASE(check_numeric_limits)
{
    std::bitset<8> exponentBits(std::numeric_limits<migraphx::bf16>::quiet_NaN().exponent);
    std::cout << "Exponent bits: " << exponentBits << std::endl;

    std::bitset<7> mantissaBits(std::numeric_limits<migraphx::bf16>::quiet_NaN().mantissa);
    std::cout << "Mantissa bits: " << mantissaBits << std::endl;

    std::cout << std::numeric_limits<migraphx::bf16>::min() << std::endl;
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::min(), uint16_t{0x0080}));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::lowest(), uint16_t{0xff7f}));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::max(), uint16_t{0x7f7f}));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::epsilon(), uint16_t{0x3c00}));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::denorm_min(), uint16_t{0x0001}));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::infinity(), uint16_t{0x7f80}));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::quiet_NaN(), uint16_t{0x7fc0}));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::signaling_NaN(), uint16_t{0x7fa0}));
}


TEST_CASE(check_flows)
{
    // check positive underflow
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::min() *
                        std::numeric_limits<migraphx::bf16>::min(),
                    migraphx::bf16(0)));

    // check overflow
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::infinity() +
                        std::numeric_limits<migraphx::bf16>::infinity(),
                    std::numeric_limits<migraphx::bf16>::infinity()));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::max() +
                        std::numeric_limits<migraphx::bf16>::max(),
                    std::numeric_limits<migraphx::bf16>::infinity()));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::max() /
                        std::numeric_limits<migraphx::bf16>::epsilon(),
                    std::numeric_limits<migraphx::bf16>::infinity()));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::max() +
                        std::numeric_limits<migraphx::bf16>::min(),
                    std::numeric_limits<migraphx::bf16>::max()));

    // check negative underflow
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::lowest() +
                        std::numeric_limits<migraphx::bf16>::lowest(),
                    -std::numeric_limits<migraphx::bf16>::infinity()));
    CHECK(bit_equal(-std::numeric_limits<migraphx::bf16>::infinity() -
                        std::numeric_limits<migraphx::bf16>::infinity(),
                    -std::numeric_limits<migraphx::bf16>::infinity()));
    CHECK(bit_equal(std::numeric_limits<migraphx::bf16>::lowest() -
                        std::numeric_limits<migraphx::bf16>::min(),
                    std::numeric_limits<migraphx::bf16>::lowest()));
}

TEST_CASE(test_nan)
{
    float f_qnan = std::numeric_limits<float>::quiet_NaN();
    migraphx::bf16 bf16_qnan(f_qnan);
    EXPECT(bf16_qnan.is_nan());
    EXPECT(std::isnan(bf16_qnan));

    float f_snan = std::numeric_limits<float>::signaling_NaN();
    migraphx::bf16 bf16_snan(f_snan);
    EXPECT(bf16_snan.is_nan());
    EXPECT(std::isnan(bf16_snan));
}

TEST_CASE(test_bool)
{
    float zero  = 0.0;
    float two   = 2.0;
    float other = -0.375;
    migraphx::bf16 bf16_zero(zero);
    migraphx::bf16 bf16_two(two);
    migraphx::bf16 bf16_other(other);
    EXPECT(not static_cast<bool>(bf16_zero));
    EXPECT(static_cast<bool>(bf16_two));
    EXPECT(static_cast<bool>(bf16_other));
}

TEST_CASE(test_pos_infinity)
{
    float finf = std::numeric_limits<float>::infinity();
    migraphx::bf16 bf16_inf_1(finf);
    CHECK(bit_equal(bf16_inf_1, std::numeric_limits<migraphx::bf16>::infinity()));
}

TEST_CASE(test_neg_infinity)
{
    float finf = -1.0 * std::numeric_limits<float>::infinity();
    migraphx::bf16 bf16_neginf_1(finf);
    CHECK(bit_equal(bf16_neginf_1, -std::numeric_limits<migraphx::bf16>::infinity()));
}

TEST_CASE(test_numeric_max_1)
{
    float fmax = std::numeric_limits<float>::max(); // fp32 max is fp16 inf
    migraphx::bf16 bf16_inf(fmax);
    CHECK(bit_equal(bf16_inf, std::numeric_limits<migraphx::bf16>::max()));
}

TEST_CASE(test_numeric_lowest_1)
{
    float flowest = std::numeric_limits<float>::lowest();
    migraphx::bf16 bf16_neginf(flowest);
    CHECK(bit_equal(bf16_neginf, std::numeric_limits<migraphx::bf16>::lowest()));
}

TEST_CASE(test_max_eq_lowest)
{
    EXPECT(migraphx::float_equal(std::numeric_limits<migraphx::bf16>::lowest(),
                                 -1 * std::numeric_limits<migraphx::bf16>::max()));
}

TEST_CASE(test_isfinite)
{
    EXPECT(std::isfinite(migraphx::bf16(0.0)));
    EXPECT(std::isfinite(migraphx::bf16(-0.0)));
    EXPECT(not std::isfinite(migraphx::bf16(std::numeric_limits<migraphx::bf16>::quiet_NaN())));
}

TEST_CASE(test_binary_ops)
{
    auto a = migraphx::bf16(-1.0);
    auto b = migraphx::bf16(1.0);
    auto c = migraphx::bf16(0.0);
    auto d = migraphx::bf16(-0.0);
    EXPECT(migraphx::float_equal((c + d), c));
    // EXPECT(migraphx::float_equal((c + d), d));
    EXPECT(migraphx::float_equal((a + b), c));
    EXPECT(migraphx::float_equal((a + b), d));

    auto e = migraphx::bf16(10.0);
    auto f = migraphx::bf16(-10.0);
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
    auto a = migraphx::bf16(-1.0);
    std::stringstream ss;
    ss << a;
    EXPECT(std::string("-1") == ss.str());
    ss     = std::stringstream();
    auto b = std::numeric_limits<migraphx::bf16>::quiet_NaN();
    ss << b;
    EXPECT(std::string("nan") == ss.str());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
