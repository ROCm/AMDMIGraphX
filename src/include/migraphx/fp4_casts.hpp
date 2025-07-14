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
#include <migraphx/bit_cast.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/bit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace fp4_detail {
static constexpr std::array<float, 16> fp4_lut = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
} // namespace fp4_detail

constexpr float fp4_to_float(uint8_t x) { return fp4_detail::fp4_lut[x & 0xF]; }

// roundTiesToEven
// based on code in float8_impl
constexpr uint8_t float_to_fp4(float f_x)
{
    const uint32_t f32_mantissa_width = 23;
    const uint32_t f4_mantissa_width  = 1;

    uint32_t x            = migraphx::bit_cast<uint32_t>(f_x);
    uint32_t head         = 0;
    uint32_t f32_mantissa = 0;
    uint32_t f32_exponent = 0;
    uint32_t f32_bias     = 0;
    uint32_t sign         = 0;
    head                  = x & 0xFF800000;
    f32_mantissa          = x & 0x7FFFFF;
    f32_exponent          = (head >> 23) & 0xFF;
    sign                  = head >> 31;
    f32_bias              = 127;

    // input is inf or NaN. No inf or NaN in fp4
    if((x & 0x7F800000) == 0x7F800000)
    {
        // inf
        if(f32_mantissa == 0)
        {
            if(sign == 0)
                return 0x7;
            else
                return 0xF;
        }
        else
        {
            return 0x7;
        }
    }
    // positive zero
    if(x == 0)
        return 0x0;
    // negative zero
    else if(x == 0x80000000)
        return 0x8;

    const int f4_bias                  = 1;
    const int f4_denormal_act_exponent = 0; // actual exponent of f4 denormal
    int act_exponent                   = 0;
    int f4_exponent                    = 0;
    int exponent_diff                  = 0;

    if(f32_exponent == 0 and f32_mantissa != 0)
    {
        // fp32/fp16 is in denormal.
        act_exponent = 1 - f32_bias;
        // actual exponent is exponent - f32_bias + 1 as it is denormal
        exponent_diff = f4_denormal_act_exponent - act_exponent;
    }
    else
    {
        // fp32/fp16 is normal with implicit 1
        act_exponent = f32_exponent - f32_bias;
        if(act_exponent <= f4_denormal_act_exponent)
        {
            exponent_diff = f4_denormal_act_exponent - act_exponent;
        }
        else
        {
            // both fp32/fp16 and f4 are in normal range
            exponent_diff = 0;
        }
        // Add the implicit 1 into mantissa
        f32_mantissa += (1u << f32_mantissa_width);
    }

    // need to know whether the number is right in the middle of two adjacent fp4 numbers. Use max
    // value of 31 to avoid undefined behavior
    bool midpoint =
        (f32_mantissa &
         ((1u << std::min(31u, f32_mantissa_width - f4_mantissa_width + exponent_diff)) - 1)) ==
        (1u << std::min(31u, f32_mantissa_width - f4_mantissa_width + exponent_diff - 1));
    if(exponent_diff > 0)
        f32_mantissa >>= std::min(31u, uint32_t(exponent_diff));
    else if(exponent_diff == -1)
        f32_mantissa <<= -exponent_diff;
    bool implicit_one = f32_mantissa & (1 << f32_mantissa_width);
    // if there is no implicit 1, it  means the f4 is denormal and need to adjust to denorm exponent
    f4_exponent = (act_exponent + exponent_diff) + f4_bias - (implicit_one ? 0 : 1);

    // Adjust exponent and mantissa
    uint32_t drop_mask = (1u << (f32_mantissa_width - f4_mantissa_width)) - 1;
    // if the least significant bit that is not truncated is 1
    bool odd = f32_mantissa & (1u << (f32_mantissa_width - f4_mantissa_width));

    f32_mantissa += (midpoint ? (odd ? f32_mantissa : f32_mantissa - 1) : f32_mantissa) & drop_mask;

    // Deal with overflow
    if(f4_exponent == 0 and ((1 << f32_mantissa_width) & f32_mantissa))
    {
        f4_exponent = 1; // denormal overflow to become normal, promote exponent
    }
    else if((1 << (f32_mantissa_width + 1)) & f32_mantissa)
    {
        f32_mantissa >>= 1;
        f4_exponent++;
    }

    f32_mantissa >>= (f32_mantissa_width - f4_mantissa_width);

    uint32_t signed_all_ones = (sign << 3) + 0x7;
    // above range: quantize to maximum possible float of the same sign
    const int max_exp = 3;
    if(f4_exponent > max_exp)
        return signed_all_ones;

    if(f4_exponent == 0 and f32_mantissa == 0)
        return sign << 3;
    f32_mantissa &= all_ones<f4_mantissa_width>();
    return (sign << 3) | (f4_exponent << f4_mantissa_width) | f32_mantissa;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
