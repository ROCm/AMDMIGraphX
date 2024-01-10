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
*
*/

#include <rocm/limits.hpp>
#include <limits>
#include <cmath>
#include <test.hpp>

template <class T>
void test_numeric_limits()
{
    EXPECT(std::numeric_limits<T>::is_specialized == rocm::numeric_limits<T>::is_specialized);
    EXPECT(std::numeric_limits<T>::digits == rocm::numeric_limits<T>::digits);
    EXPECT(std::numeric_limits<T>::digits10 == rocm::numeric_limits<T>::digits10);
    EXPECT(std::numeric_limits<T>::max_digits10 == rocm::numeric_limits<T>::max_digits10);
    EXPECT(std::numeric_limits<T>::is_signed == rocm::numeric_limits<T>::is_signed);
    EXPECT(std::numeric_limits<T>::is_integer == rocm::numeric_limits<T>::is_integer);
    EXPECT(std::numeric_limits<T>::is_exact == rocm::numeric_limits<T>::is_exact);
    EXPECT(std::numeric_limits<T>::radix == rocm::numeric_limits<T>::radix);
    EXPECT(std::numeric_limits<T>::min_exponent == rocm::numeric_limits<T>::min_exponent);
    EXPECT(std::numeric_limits<T>::min_exponent10 == rocm::numeric_limits<T>::min_exponent10);
    EXPECT(std::numeric_limits<T>::max_exponent == rocm::numeric_limits<T>::max_exponent);
    EXPECT(std::numeric_limits<T>::max_exponent10 == rocm::numeric_limits<T>::max_exponent10);
    EXPECT(std::numeric_limits<T>::has_infinity == rocm::numeric_limits<T>::has_infinity);
    EXPECT(std::numeric_limits<T>::has_quiet_NaN == rocm::numeric_limits<T>::has_quiet_NaN);
    EXPECT(std::numeric_limits<T>::has_signaling_NaN == rocm::numeric_limits<T>::has_signaling_NaN);
    EXPECT(std::numeric_limits<T>::is_iec559 == rocm::numeric_limits<T>::is_iec559);
    EXPECT(std::numeric_limits<T>::is_bounded == rocm::numeric_limits<T>::is_bounded);
    EXPECT(std::numeric_limits<T>::is_modulo == rocm::numeric_limits<T>::is_modulo);
    EXPECT(std::numeric_limits<T>::tinyness_before == rocm::numeric_limits<T>::tinyness_before);

    EXPECT(std::numeric_limits<T>::min() == rocm::numeric_limits<T>::min());
    EXPECT(std::numeric_limits<T>::lowest() == rocm::numeric_limits<T>::lowest());
    EXPECT(std::numeric_limits<T>::max() == rocm::numeric_limits<T>::max());
    EXPECT(std::numeric_limits<T>::epsilon() == rocm::numeric_limits<T>::epsilon());
    EXPECT(std::numeric_limits<T>::round_error() == rocm::numeric_limits<T>::round_error());
    EXPECT(std::numeric_limits<T>::infinity() == rocm::numeric_limits<T>::infinity());
    EXPECT(std::numeric_limits<T>::denorm_min() == rocm::numeric_limits<T>::denorm_min());
    if constexpr(std::is_floating_point<T>{})
    {
        EXPECT(std::isnan(rocm::numeric_limits<T>::quiet_NaN()));
        EXPECT(std::isnan(rocm::numeric_limits<T>::signaling_NaN()));
    }
    else
    {
        EXPECT(std::numeric_limits<T>::quiet_NaN() == rocm::numeric_limits<T>::quiet_NaN());
        EXPECT(std::numeric_limits<T>::signaling_NaN() == rocm::numeric_limits<T>::signaling_NaN());
    }
}

TEST_CASE_REGISTER(test_numeric_limits<char>);
TEST_CASE_REGISTER(test_numeric_limits<signed char>);
TEST_CASE_REGISTER(test_numeric_limits<unsigned char>);
TEST_CASE_REGISTER(test_numeric_limits<wchar_t>);
TEST_CASE_REGISTER(test_numeric_limits<char16_t>);
TEST_CASE_REGISTER(test_numeric_limits<char32_t>);
TEST_CASE_REGISTER(test_numeric_limits<short>);
TEST_CASE_REGISTER(test_numeric_limits<unsigned short>);
TEST_CASE_REGISTER(test_numeric_limits<int>);
TEST_CASE_REGISTER(test_numeric_limits<unsigned int>);
TEST_CASE_REGISTER(test_numeric_limits<long>);
TEST_CASE_REGISTER(test_numeric_limits<unsigned long>);
TEST_CASE_REGISTER(test_numeric_limits<long long>);
TEST_CASE_REGISTER(test_numeric_limits<unsigned long long>);
TEST_CASE_REGISTER(test_numeric_limits<double>);
TEST_CASE_REGISTER(test_numeric_limits<float>);

int main(int argc, const char* argv[]) { test::run(argc, argv); }
