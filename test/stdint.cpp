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
#include <dual_test.hpp>
#include <rocm/stdint.hpp>

ROCM_DUAL_TEST_CASE()
{
    static_assert(sizeof(rocm::int8_t) == 1, "int8_t must be 1 bytes");
    static_assert(sizeof(rocm::uint8_t) == 1, "uint8_t must be 1 bytes");
    static_assert(sizeof(rocm::int16_t) == 2, "int16_t must be 2 bytes");
    static_assert(sizeof(rocm::uint16_t) == 2, "uint16_t must be 2 bytes");
    static_assert(sizeof(rocm::int32_t) == 4, "int32_t must be 4 bytes");
    static_assert(sizeof(rocm::uint32_t) == 4, "uint32_t must be 4 bytes");
    static_assert(sizeof(rocm::int64_t) == 8, "int64_t must be 8 bytes");
    static_assert(sizeof(rocm::uint64_t) == 8, "uint64_t must be 8 bytes");
}
