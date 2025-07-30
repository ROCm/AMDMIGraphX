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
#include <iterator>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace fp4_detail {
static constexpr std::array<float, 16> fp4_lut = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

// pair is {fp4_tie_value, round_to_zero}
// if round_to_zero round tie towards zero, else round tie away from zero
static constexpr std::array<std::pair<float, uint8_t>, 7> fp4_even_round = {
    {{0.25, 1}, {0.75, 0}, {1.25, 1}, {1.75, 0}, {2.5, 1}, {3.5, 0}, {5, 1}}};
} // namespace fp4_detail

// converts 4 LSB to float
constexpr float fp4_to_float(uint8_t x)
{
    return fp4_detail::fp4_lut[x % fp4_detail::fp4_lut.size()];
}

// rounding mode = roundToNearestRoundTiesToEven
// Reference quantization code from Microsoft:
// https://github.com/microsoft/microxcaling/blob/main/mx/elemwise_ops.py#L82
// Not constexpr because std::signbit is not constexpr until C++23
inline uint8_t float_to_fp4(float f_x)
{
    using fp4_detail::fp4_even_round;
    using fp4_detail::fp4_lut;
    if(std::isnan(f_x))
    {
        return 0;
    }
    bool sign        = std::signbit(f_x);
    uint8_t sign_add = sign ? fp4_lut.size() / 2 : 0u;
    float abs_f      = std::abs(f_x);
    // index value is the positive fp4 value
    uint8_t i = std::upper_bound(fp4_even_round.begin(),
                                 fp4_even_round.end(),
                                 std::make_pair(abs_f, uint8_t{0})) -
                fp4_even_round.begin();

    return i + sign_add;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
