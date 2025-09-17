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
#ifndef MIGRAPHX_GUARD_KERNELS_FP4_CASTS_HPP
#define MIGRAPHX_GUARD_KERNELS_FP4_CASTS_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/tuple.hpp>
#include <migraphx/kernels/hip.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

namespace fp4_detail {

static constexpr array<float, 16> fp4_lut = {0.0f,
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

// pair is {fp4_tie_value, round_to_zero}
// if round_to_zero round tie towards zero, else round tie away from zero
static constexpr array<migraphx::tuple<float, uint8_t>, 7> fp4_even_round = {make_tuple(0.25, 1),
                                                                             make_tuple(0.75, 0),
                                                                             make_tuple(1.25, 1),
                                                                             make_tuple(1.75, 0),
                                                                             make_tuple(2.5, 1),
                                                                             make_tuple(3.5, 0),
                                                                             make_tuple(5, 1)};
} // namespace fp4_detail

// NOTE: possible to remove float/T casts by making LUTs for each type
// converts 4 LSB to float
template <class T>
__device__ constexpr T cast_from_fp4(uint8_t x)
{
    return T(fp4_detail::fp4_lut[x % fp4_detail::fp4_lut.size()]);
}

// rounding mode = roundToNearestRoundTiesToEven
template <class T>
__device__ inline uint8_t cast_to_fp4(T x)
{
    float f_x = float(x);
    using fp4_detail::fp4_even_round;
    using fp4_detail::fp4_lut;
    if(isnan(f_x))
    {
        return 0;
    }
    bool sign        = signbit(f_x);
    uint8_t sign_add = sign ? fp4_lut.size() / 2 : 0u;
    float abs_f      = abs(f_x);
    // index value is the positive fp4 value
    uint8_t i = migraphx::upper_bound(fp4_even_round.begin(),
                                      fp4_even_round.end(),
                                      migraphx::make_tuple(abs_f, uint8_t{0}),
                                      [&](const auto& a, const auto& b) { return a < b; }) -
                fp4_even_round.begin();

    return i + sign_add;
}

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_FP4_CASTS_HPP
