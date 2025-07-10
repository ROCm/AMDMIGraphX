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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {
static std::array<float, 16> _fp4_lut = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
}

constexpr float fp4_to_float(uint8_t x) { return _fp4_lut[x & 0xF]; }

// roundTiesToEven
// based on code in float8_impl
constexpr uint8_t float_to_fp4(float f_x)
{
    // float32 mantissa_width
    const uint32_t mfmt = 23;
    // float4 mantissa_width
    const uint32_t mantissa_size = 1;

    uint32_t x        = migraphx::bit_cast<uint32_t>(f_x);
    uint32_t head     = 0;
    uint32_t mantissa = 0;
    int exponent      = 0;
    uint32_t bias     = 0;
    uint32_t sign     = 0;
    head              = x & 0xFF800000;
    mantissa          = x & 0x7FFFFF;
    exponent          = (head >> 23) & 0xFF;
    sign              = head >> 31;
    bias              = 127;

    // input is inf or NaN. No inf or NaN in fp4
    if((x & 0x7F800000) == 0x7F800000)
    {
        // inf
        if(mantissa == 0)
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

    if(exponent == 0 and mantissa != 0)
    {
        // fp32/fp16 is in denormal.
        act_exponent = 1 - bias;
        // actual exponent is exponent-bias+1 as it is denormal
        exponent_diff = f4_denormal_act_exponent - act_exponent;
    }
    else
    {
        // fp32/fp16 is normal with implicit 1
        act_exponent = exponent - bias;
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
        mantissa += (1u << mfmt);
    }

    // need to know whether the number is right in the middle of two adjacent fp4 numbers. Use max
    // value of 31 to avoid undefined behavior
    bool midpoint =
        (mantissa & ((1u << std::min(31u, mfmt - mantissa_size + exponent_diff)) - 1)) ==
        (1u << std::min(31u, mfmt - mantissa_size + exponent_diff - 1));
    if(exponent_diff > 0)
        mantissa >>= std::min(31u, uint32_t(exponent_diff));
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1 << mfmt);
    // if there is no implicit 1, it  means the f4 is denormal and need to adjust to denorm exponent
    f4_exponent = (act_exponent + exponent_diff) + f4_bias - (implicit_one ? 0 : 1);

    // Adjust exponent and mantissa
    uint32_t drop_mask = (1u << (mfmt - mantissa_size)) - 1;
    // if the least significant bit that is not truncated is 1
    bool odd = mantissa & (1u << (mfmt - mantissa_size));

    mantissa += (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa) & drop_mask;

    // Deal with overflow
    if(f4_exponent == 0 and ((1 << mfmt) & mantissa))
    {
        f4_exponent = 1; // denormal overflow to become normal, promote exponent
    }
    else if((1 << (mfmt + 1)) & mantissa)
    {
        mantissa >>= 1;
        f4_exponent++;
    }

    mantissa >>= (mfmt - mantissa_size);

    uint32_t signed_all_ones = (sign << 3) + 0x7;
    // above range: quantize to maximum possible float of the same sign
    const int max_exp = 3;
    if(f4_exponent > max_exp)
        return signed_all_ones;

    if(f4_exponent == 0 and mantissa == 0)
        return sign << 3;
    mantissa &= (1 << mantissa_size) - 1;
    return (sign << 3) | (f4_exponent << mantissa_size) | mantissa;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
