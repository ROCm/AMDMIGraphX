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
#include <type_traits_test.hpp>

struct c1
{
};

struct c2
{
};

struct c3 : c2
{
};
struct c1c2
{
    c1c2() {}
    c1c2(c1 const&) {}
    c1c2(c2 const&) {}
    c1c2& operator=(c1c2 const&) { return *this; }
};

#define ROCM_CHECK_COMMON_TYPE(expected, ...)                        \
    ROCM_CHECK_TYPE(rocm::common_type<__VA_ARGS__>::type, expected); \
    ROCM_CHECK_TYPE(rocm::common_type_t<__VA_ARGS__>, expected);

#define ROCM_CHECK_COMMON_TYP_E2(expected, a, b) \
    ROCM_CHECK_COMMON_TYPE(expected, a, b);      \
    ROCM_CHECK_COMMON_TYPE(expected, b, a);

ROCM_DUAL_TEST_CASE()
{
    ROCM_CHECK_COMMON_TYPE(int, int);
    ROCM_CHECK_COMMON_TYPE(int, int, int);
    ROCM_CHECK_COMMON_TYPE(unsigned int, unsigned int, unsigned int);
    ROCM_CHECK_COMMON_TYPE(double, double, double);

    ROCM_CHECK_COMMON_TYP_E2(c1c2, c1c2&, c1&);
    ROCM_CHECK_COMMON_TYP_E2(c2*, c3*, c2*);
    ROCM_CHECK_COMMON_TYP_E2(const int*, int*, const int*);
    ROCM_CHECK_COMMON_TYP_E2(const volatile int*, volatile int*, const int*);
    ROCM_CHECK_COMMON_TYP_E2(volatile int*, int*, volatile int*);
    ROCM_CHECK_COMMON_TYP_E2(int, char, unsigned char);
    ROCM_CHECK_COMMON_TYP_E2(int, char, short);
    ROCM_CHECK_COMMON_TYP_E2(int, char, unsigned short);
    ROCM_CHECK_COMMON_TYP_E2(int, char, int);
    ROCM_CHECK_COMMON_TYP_E2(unsigned int, char, unsigned int);
    ROCM_CHECK_COMMON_TYP_E2(double, double, unsigned int);

    ROCM_CHECK_COMMON_TYPE(double, double, char, int);

    ROCM_CHECK_COMMON_TYPE(int, int&);
    ROCM_CHECK_COMMON_TYPE(int, const int);
    ROCM_CHECK_COMMON_TYPE(int, const int, const int);
    ROCM_CHECK_COMMON_TYP_E2(long, const int, const long);
}
