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

#include <migraphx/errors.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/float8.hpp>
#include <cstdint>
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace fp4_detail {

static constexpr std::array<float, 16> fp4_to_float_lut = {0.0f,
                                                           0.5f,
                                                           1.0f,
                                                           1.5f,
                                                           2.0f,
                                                           3.0f,
                                                           4.0f,
                                                           6.0f,
                                                           -0.0f,
                                                           -0.5f,
                                                           -1.0f,
                                                           -1.5f,
                                                           -2.0f,
                                                           -3.0f,
                                                           -4.0f,
                                                           -6.0f};

// pair is {fp4_tie_value, round_to_zero}, index is the positive fp4 value
// if round_to_zero round tie towards zero, else round tie away from zero
static constexpr std::array<std::pair<float, uint8_t>, 7> float_to_fp4_lut = {
    {{0.25f, 1}, {0.75f, 0}, {1.25f, 1}, {1.75f, 0}, {2.5f, 1}, {3.5f, 0}, {5.0f, 1}}};

} // namespace fp4_detail

// converts 4 LSB to float
constexpr float fp4_to_float(uint8_t x)
{
    return fp4_detail::fp4_to_float_lut[x % fp4_detail::fp4_to_float_lut.size()];
}

// converts 4 LSB to fp8e4m3fn_type
constexpr auto fp4_to_fp8(uint8_t x)
{
    return migraphx::fp8::fp8e4m3fn(
        fp4_detail::fp4_to_float_lut[x % fp4_detail::fp4_to_float_lut.size()]);
}

// rounding mode = roundToNearestRoundTiesToEven
// Reference quantization code from Microsoft:
// https://github.com/microsoft/microxcaling/blob/main/mx/elemwise_ops.py#L82
// Not constexpr because of std::signbit and std::upper_bound not constexpr in C++17
template <class T>
inline uint8_t cast_to_fp4(T x)
{
    float f_x(x);
    using fp4_detail::float_to_fp4_lut;
    using fp4_detail::fp4_to_float_lut;
    if(std::isnan(f_x))
    {
        return 0;
    }
    bool sign        = std::signbit(f_x);
    uint8_t sign_add = sign ? fp4_to_float_lut.size() / 2 : 0u;
    float abs_f      = std::abs(f_x);
    // index value is the positive fp4 value
    uint8_t i = std::upper_bound(float_to_fp4_lut.begin(),
                                 float_to_fp4_lut.end(),
                                 std::make_pair(abs_f, uint8_t{0})) -
                float_to_fp4_lut.begin();

    return i + sign_add;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
