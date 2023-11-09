/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef MIGRAPHX_HIP_FP8_IMPL_HPP
#define MIGRAPHX_HIP_FP8_IMPL_HPP
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#endif

#define CONST_FOLD(x) (__builtin_constant_p(x) ? (x) : (x))
namespace migraphx_hip_f8_impl {
namespace detail {
template <bool B, class T, class F>
struct conditional
{
    using type = T;
};

template <class T, class F>
struct conditional<false, T, F>
{
    using type = F;
};

template <typename To, typename From>
inline constexpr To bit_cast(From fr) noexcept
{
    static_assert(sizeof(To) == sizeof(From));
#if defined(__GNUC__) and !defined(__clang__)
    To x = CONST_FOLD(*reinterpret_cast<To*>(&fr));
#else
    To x = __builtin_bit_cast(To, fr);
#endif
    return x;
}
} // namespace detail

// #ifdef __HIP_PLATFORM_HCC__
// __device__ inline int clz(uint32_t x) { return __clz(x); }
// #else
// __host__ inline int clz(uint32_t x) { return __builtin_clz(x); }
// #endif

template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
MIGRAPHX_HIP_HOST_DEVICE constexpr uint8_t cast_to_f8(T _x, bool stoch, uint32_t rng)
{

    static_assert(wm + we == 7, "wm+we==7");

    const int mfmt = (sizeof(T) == 4) ? 23 : 10;
    typename detail::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type x;

    if constexpr(sizeof(T) == 4)
        x = detail::bit_cast<uint32_t>(_x);
    else
        x = detail::bit_cast<uint16_t>(_x);

    uint32_t head, mantissa;
    int exponent, bias;
    uint32_t sign;

    if constexpr(sizeof(T) == 4)
    {
        head     = x & 0xFF800000;
        mantissa = x & 0x7FFFFF;
        exponent = (head >> 23) & 0xFF;
        sign     = head >> 31;
        bias     = 127;
    }
    else
    {
        head     = x & 0xFC00;
        mantissa = x & 0x3FF;
        exponent = (head >> 10) & 0x1F;
        sign     = head >> 15;
        bias     = 15;
    }

    uint32_t signed_inf = (sign << 7) + (((1 << we) - 1) << wm);

    // Deal with inf and NaNs
    if(negative_zero_nan)
    {
        if(sizeof(T) == 4)
        {
            if((x & 0x7F800000) == 0x7F800000)
                return 0x80;
        }
        else
        {
            // if(__hisinf(x) || __hisnan(x))
            if((x & 0x7C00) == 0x7C00)
                return 0x80;
        }
    }
    else
    {
        if(sizeof(T) == 4)
        {
            if((x & 0x7F800000) == 0x7F800000)
                return signed_inf + (mantissa != 0 ? 1 : 0);
        }
        else
        {
            if((x & 0x7C00) == 0x7C00)
                return signed_inf + (mantissa != 0 ? 1 : 0);
        }
    }
    // handle positive zero
    if(x == 0)
        return 0;
    // handle negative zero
    if((sizeof(T) == 4 and x == 0x80000000) or (sizeof(T) == 2 and x == 0x8000))
    {
        if(negative_zero_nan)
        {
            return 0;
        }
        else
        {
            return 0x80;
        }
    }

    // First need to check if it is normal or denorm as there is a difference of implict 1
    // Then need to adjust the exponent to align with the F8 exponent, in the meanwhile, shift
    // The mantissa. Then for stochastic rounding, add rng to mantissa and truncate. And for
    // RNE, no need to add rng. Then probably need to check whether there is carry and adjust
    // exponent and mantissa again

    // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent bits
    const int f8_bias                  = (1 << (we - 1)) - 1 + (negative_zero_nan ? 1 : 0);
    const int f8_denormal_act_exponent = 1 - f8_bias; // actual exponent of f8 denormal
    // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
    // f8_exponent is the converted f8 exponent with bias encoding
    // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
    // the difference needs to be adjusted and mantissa shifted
    int act_exponent, f8_exponent, exponent_diff;

    if(exponent == 0)
    { // fp32/fp16 is in denormal.
        /* fp32 denormal is below 2^-127 so it is usually not a concern here, we mostly concern fp16
here. In this case, f8 is usually in denormal. But there could be exceptions. fp16 denormal has
exponent bias 15 while bf8 with NANOO has exponent bias 16. It means that there are some numbers in
fp16 denormal but they are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15. fp16 numbers
where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8 (NANOO) normal. In
this case, the fp16 mantissa should be shift left by 1  */
        act_exponent  = exponent - bias + 1;
        exponent_diff = f8_denormal_act_exponent -
                        act_exponent; // actual exponent is exponent-bias+1 as it is denormal
    }
    else
    { // fp32/fp16 is normal with implicit 1
        act_exponent = exponent - bias;
        if(act_exponent <= f8_denormal_act_exponent)
        {
            /* This is the case where fp32/fp16 is normal but it is in f8 denormal range.
   For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
   actual exponent is -7, it is actually larger due to the implict 1,
   Therefore it needs to be adjust to -6 and mantissa shift right by 1.
   So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
            exponent_diff = f8_denormal_act_exponent - act_exponent;
        }
        else
        {          // both fp32/fp16 and f8 are in normal range
            exponent_diff =
                0; // exponent_diff=0 does not mean there is no difference for this case,
            // act_exponent could be larger. Just that it does not need shift mantissa
        }
        mantissa += (1 << mfmt); // Add the implicit 1 into mantissa
    }

    bool midpoint = (mantissa & ((1 << (mfmt - wm + exponent_diff)) - 1)) ==
                    (1 << (mfmt - wm + exponent_diff - 1));
    /* This part is a bit tricky. The judgment of whether it is a tie needs to be done before we
 shift right as shift right could rip off some residual part and make something not midpoint look
 like midpoint. For example, the fp16 number 0x1002 (0 00100 0000000010), it is larger than
 midpoint, but after shift right by 4 bits, it would look like midpoint.
*/

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1 << mfmt);
    // if there is no implict 1, it  means the f8 is denormal and need to adjust to denorm exponent
    f8_exponent =
        (act_exponent + exponent_diff) /*actual f8 exponent*/ + f8_bias - (implicit_one ? 0 : 1);

    // Now we have the exponent and mantissa adjusted
    uint32_t drop_mask = (1 << (mfmt - wm)) - 1;
    bool odd =
        mantissa & (1 << (mfmt - wm)); // if the least significant bit that is not truncated is 1
    mantissa += (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa)) & drop_mask;

    // Now we deal with overflow
    if(f8_exponent == 0)
    {
        if((1 << mfmt) & mantissa)
        {
            f8_exponent = 1; // denormal overflow to become normal, promote exponent
        }
    }
    else
    {
        if((1 << (mfmt + 1)) & mantissa)
        {
            mantissa >>= 1;
            f8_exponent++;
        }
    }

    mantissa >>= (mfmt - wm);

    // above range: quantize to maximum possible float of the same sign
    const int max_exp = (1 << we) - (negative_zero_nan ? 1 : 2);
    if(f8_exponent > max_exp)
    {
        if(clip)
        {
            mantissa    = (1 << wm) - 1;
            f8_exponent = max_exp;
        }
        else
        {
            return signed_inf;
        }
    }

    if(f8_exponent == 0 && mantissa == 0)
        return negative_zero_nan ? 0 : (sign << 7);
    mantissa &= (1 << wm) - 1;
    return (sign << 7) | (f8_exponent << wm) | mantissa;
}

template <int wm, int we, typename T, bool negative_zero_nan>
MIGRAPHX_HIP_HOST_DEVICE constexpr T cast_from_f8(uint8_t x)
{
    constexpr int weo = 8;
    constexpr int wmo = 23;

    T fInf, fNegInf, fNaN, fNeg0;
    uint32_t ifInf    = 0x7F800000;
    uint32_t ifNegInf = 0xFF800000;
    uint32_t ifNaN    = 0x7F800001;
    uint32_t ifNeg0   = 0x80000000;
    // TODO: need to change T for half but right now it would never  called with half
    fInf    = detail::bit_cast<float>(ifInf);
    fNegInf = detail::bit_cast<float>(ifNegInf);
    fNaN    = detail::bit_cast<float>(ifNaN);
    fNeg0   = detail::bit_cast<float>(ifNeg0);

    if(x == 0)
        return 0;

    uint32_t sign     = x >> 7;
    uint32_t mantissa = x & ((1 << wm) - 1);
    int exponent      = (x & 0x7F) >> wm;
    if(negative_zero_nan)
    {
        if(x == 0x80)
            return fNaN;
    }
    else
    {
        if(x == 0x80)
            return fNeg0;
        if(exponent == ((1 << we) - 1))
            return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
    }
    typename detail::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type retval;

    const int exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

    // subnormal input
    if(exponent == 0)
    {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        int sh = 1 + __builtin_clz(mantissa) - (32 - wm);
        mantissa <<= sh;
        exponent += 1 - sh;
        mantissa &= ((1 << wm) - 1);
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= wmo - wm;

    // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << wmo;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    if(sizeof(T) == 2)
        retval = (sign << 15) | (exponent << 10) | mantissa;
    else
        retval = (sign << 31) | (exponent << 23) | mantissa;
    return detail::bit_cast<T>(retval);
}

} // namespace migraphx_hip_f8_impl
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif // MIGRAPHX_HIP_FP8_IMPL_HPP
