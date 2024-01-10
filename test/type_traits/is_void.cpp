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

ROCM_DUAL_TEST_CASE()
{
    static_assert(rocm::is_void<void>{});
    static_assert(rocm::is_void<void const>{});
    static_assert(rocm::is_void<void volatile>{});
    static_assert(rocm::is_void<void const volatile>{});

    static_assert(not rocm::is_void<void*>{});
    static_assert(not rocm::is_void<int>{});
    static_assert(not rocm::is_void<test_tt::f1>{});
    static_assert(not rocm::is_void<test_tt::foo0_t>{});
    static_assert(not rocm::is_void<test_tt::foo1_t>{});
    static_assert(not rocm::is_void<test_tt::foo2_t>{});
    static_assert(not rocm::is_void<test_tt::foo3_t>{});
    static_assert(not rocm::is_void<test_tt::foo4_t>{});
    static_assert(not rocm::is_void<test_tt::incomplete_type>{});
    static_assert(not rocm::is_void<int&>{});
    static_assert(not rocm::is_void<int&&>{});
}
