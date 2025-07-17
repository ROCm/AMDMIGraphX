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
#ifndef MIGRAPHX_GUARD_RTGLIB_FLOAT4_CASTS_HPP
#define MIGRAPHX_GUARD_RTGLIB_FLOAT4_CASTS_HPP

#include <cstdint>
#include <algorithm>
#include <array>
#include <cmath>
#include <migraphx/bit_cast.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/bit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace fp4_detail {
static constexpr std::array<float, 16> fp4_lut = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

static constexpr uint8_t fp4_6_0 = 0x7;
static constexpr uint8_t fp4_4_0 = 0x6;
static constexpr uint8_t fp4_3_0 = 0x5;
static constexpr uint8_t fp4_2_0 = 0x4;
static constexpr uint8_t fp4_1_5 = 0x3;
static constexpr uint8_t fp4_1_0 = 0x2;
static constexpr uint8_t fp4_0_5 = 0x1;
} // namespace fp4_detail

constexpr float fp4_to_float(uint8_t x) { return fp4_detail::fp4_lut[x & 0xFu]; }

// roundTiesToEven
// NOTE: Not straightfoward to make constexpr because std::signbit is not constexpr till
// C++23. Doing f_x < 0 does the wrong thing for negative zero since IEEE754 has +0 == -0.
uint8_t float_to_fp4(float f_x)
{
    bool sign        = std::signbit(f_x);
    uint8_t sign_add = 0x8 * static_cast<uint8_t>(sign);
    float abs_f      = std::abs(f_x);
    if(abs_f >= 1.75)
    {
        if(abs_f >= 3.5)
        {
            if(abs_f > 5)
            {
                return fp4_detail::fp4_6_0 + sign_add;
            }
            return fp4_detail::fp4_4_0 + sign_add;
        }
        if(abs_f > 2.5)
        {
            return fp4_detail::fp4_3_0 + sign_add;
        }
        return fp4_detail::fp4_2_0 + sign_add;
    }
    if(abs_f >= 0.75)
    {
        if(abs_f > 1.25)
        {
            return fp4_detail::fp4_1_5 + sign_add;
        }
        return fp4_detail::fp4_1_0 + sign_add;
    }
    if(abs_f > 0.25)
    {
        return fp4_detail::fp4_0_5 + sign_add;
    }
    // zeros, Nan, and Inf
    return 0x0 + sign_add;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
