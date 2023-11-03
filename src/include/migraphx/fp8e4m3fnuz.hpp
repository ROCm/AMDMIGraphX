/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_RTGLIB_FP8E4M3FNUZ_HPP
#define MIGRAPHX_GUARD_RTGLIB_FP8E4M3FNUZ_HPP

/// Defines the Float8_e4m3fnuz type (8-bit floating-point) including
/// conversions to standard C types and basic arithmetic operations. Note that
/// arithmetic operations are implemented by converting to floating point and
/// performing the operation in float32.
///
/// Binary configuration remains the same as Float8_e4m3fn:
/// s eeee mmm
/// 1 sign bit
/// 4 exponent bits
/// 3 mantissa bits
///
/// The key differences versus Float8_e4m3fn are:
/// bias = 8
/// no infinities or negative zero
/// NaN only when sign bit is 1, rest all 0s
///
/// Implementation based on the paper https://arxiv.org/pdf/2206.02915.pdf and
/// the existing Float8_e4m3fn implementation.

#include <type_traits>
#include <cmath>
#include <cstdint>
#include <climits>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <sstream>
#include <iostream>
#include <migraphx/config.hpp>
#include <string>
#include <utility>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
// MIGraphX by default does not have device code in the regular compilation paths,
// therefore, when this file is used from the host side, compilation takes much
// longer. By guarding the __device__ directive we can control that such compilation
// only happens for kernels which include this file.
// need to include hip_runtime.h otherwise it complains about __host__ and __device__
#include <hip/hip_runtime.h>
#define MIGRAPHX_HIP_HOST_DEVICE __host__ __device__
#else
#define MIGRAPHX_HIP_HOST_DEVICE
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wreserved-identifier"
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace detail {

inline MIGRAPHX_HIP_HOST_DEVICE float fp32_from_bits(uint32_t w)
{
    union
    {
        uint32_t as_bits;
        float as_value;
    } fp32 = {w};
    return fp32.as_value;
}

inline MIGRAPHX_HIP_HOST_DEVICE uint32_t fp32_to_bits(float f)
{
    union
    {
        float as_value;
        uint32_t as_bits;
    } fp32 = {f};
    return fp32.as_bits;
}

/*
 * Convert a 8-bit floating-point number in fp8 E4M3FNUZ format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
inline MIGRAPHX_HIP_HOST_DEVICE constexpr float fp8e4m3fnuz_to_fp32_value(uint8_t input)
{
    constexpr float e4m3fnuz_lut[256] = {
        0.0f,           0.0009765625f,  0.001953125f,
        0.0029296875f,  0.00390625f,    0.0048828125f,
        0.005859375f,   0.0068359375f,  0.0078125f,
        0.0087890625f,  0.009765625f,   0.0107421875f,
        0.01171875f,    0.0126953125f,  0.013671875f,
        0.0146484375f,  0.015625f,      0.017578125f,
        0.01953125f,    0.021484375f,   0.0234375f,
        0.025390625f,   0.02734375f,    0.029296875f,
        0.03125f,       0.03515625f,    0.0390625f,
        0.04296875f,    0.046875f,      0.05078125f,
        0.0546875f,     0.05859375f,    0.0625f,
        0.0703125f,     0.078125f,      0.0859375f,
        0.09375f,       0.1015625f,     0.109375f,
        0.1171875f,     0.125f,         0.140625f,
        0.15625f,       0.171875f,      0.1875f,
        0.203125f,      0.21875f,       0.234375f,
        0.25f,          0.28125f,       0.3125f,
        0.34375f,       0.375f,         0.40625f,
        0.4375f,        0.46875f,       0.5f,
        0.5625f,        0.625f,         0.6875f,
        0.75f,          0.8125f,        0.875f,
        0.9375f,        1.0f,           1.125f,
        1.25f,          1.375f,         1.5f,
        1.625f,         1.75f,          1.875f,
        2.0f,           2.25f,          2.5f,
        2.75f,          3.0f,           3.25f,
        3.5f,           3.75f,          4.0f,
        4.5f,           5.0f,           5.5f,
        6.0f,           6.5f,           7.0f,
        7.5f,           8.0f,           9.0f,
        10.0f,          11.0f,          12.0f,
        13.0f,          14.0f,          15.0f,
        16.0f,          18.0f,          20.0f,
        22.0f,          24.0f,          26.0f,
        28.0f,          30.0f,          32.0f,
        36.0f,          40.0f,          44.0f,
        48.0f,          52.0f,          56.0f,
        60.0f,          64.0f,          72.0f,
        80.0f,          88.0f,          96.0f,
        104.0f,         112.0f,         120.0f,
        128.0f,         144.0f,         160.0f,
        176.0f,         192.0f,         208.0f,
        224.0f,         240.0f,         std::numeric_limits<float>::quiet_NaN(),
        -0.0009765625f, -0.001953125f,  -0.0029296875f,
        -0.00390625f,   -0.0048828125f, -0.005859375f,
        -0.0068359375f, -0.0078125f,    -0.0087890625f,
        -0.009765625f,  -0.0107421875f, -0.01171875f,
        -0.0126953125f, -0.013671875f,  -0.0146484375f,
        -0.015625f,     -0.017578125f,  -0.01953125f,
        -0.021484375f,  -0.0234375f,    -0.025390625f,
        -0.02734375f,   -0.029296875f,  -0.03125f,
        -0.03515625f,   -0.0390625f,    -0.04296875f,
        -0.046875f,     -0.05078125f,   -0.0546875f,
        -0.05859375f,   -0.0625f,       -0.0703125f,
        -0.078125f,     -0.0859375f,    -0.09375f,
        -0.1015625f,    -0.109375f,     -0.1171875f,
        -0.125f,        -0.140625f,     -0.15625f,
        -0.171875f,     -0.1875f,       -0.203125f,
        -0.21875f,      -0.234375f,     -0.25f,
        -0.28125f,      -0.3125f,       -0.34375f,
        -0.375f,        -0.40625f,      -0.4375f,
        -0.46875f,      -0.5f,          -0.5625f,
        -0.625f,        -0.6875f,       -0.75f,
        -0.8125f,       -0.875f,        -0.9375f,
        -1.0f,          -1.125f,        -1.25f,
        -1.375f,        -1.5f,          -1.625f,
        -1.75f,         -1.875f,        -2.0f,
        -2.25f,         -2.5f,          -2.75f,
        -3.0f,          -3.25f,         -3.5f,
        -3.75f,         -4.0f,          -4.5f,
        -5.0f,          -5.5f,          -6.0f,
        -6.5f,          -7.0f,          -7.5f,
        -8.0f,          -9.0f,          -10.0f,
        -11.0f,         -12.0f,         -13.0f,
        -14.0f,         -15.0f,         -16.0f,
        -18.0f,         -20.0f,         -22.0f,
        -24.0f,         -26.0f,         -28.0f,
        -30.0f,         -32.0f,         -36.0f,
        -40.0f,         -44.0f,         -48.0f,
        -52.0f,         -56.0f,         -60.0f,
        -64.0f,         -72.0f,         -80.0f,
        -88.0f,         -96.0f,         -104.0f,
        -112.0f,        -120.0f,        -128.0f,
        -144.0f,        -160.0f,        -176.0f,
        -192.0f,        -208.0f,        -224.0f,
        -240.0f,
    };

    return e4m3fnuz_lut[input];
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E4M3FNUZ format, in bit representation.
 */
inline MIGRAPHX_HIP_HOST_DEVICE uint8_t fp8e4m3fnuz_from_fp32_value(float f)
{
    /*
     * Binary representation of 256.0f, which is the first value not representable
     * (i.e. the first value which would overflow in to the sign bit, resulting in
     * a NaN) in fp8e4m3fnuz range:
     * 1 0000 000 - fp8e4m3fnuz
     * 0 10000110 00000000000000000000000 - fp32
     */
    constexpr uint32_t fnuz_max = UINT32_C(0x87) << 23;

    /*
     * A mask for converting fp32 numbers lower than fp8e4m3fnuz normal range
     * into denormalized representation.
     * magic number: ((127 - 8) + (23 - 3) + 1)
     */
    constexpr uint32_t denorm_mask = UINT32_C(0x8C) << 23;

    uint32_t f_bits = fp32_to_bits(f);

    uint32_t result = 0u;

    /*
     * Extract the sign of the input number into the high bit of the 32-bit word:
     *
     *      +---+----------------------------------+
     *      | S |0000000 00000000 00000000 00000000|
     *      +---+----------------------------------+
     * Bits  31                 0-31
     */
    const uint32_t sign = f_bits & UINT32_C(0x80000000);

    /*
     * Set sign bit to 0
     */
    f_bits ^= sign;

    if(f_bits >= fnuz_max)
    {
        // NaN -- sign bit set to 1, rest 0s
        return 0x80;
    }

    if(f_bits < (UINT32_C(121) << 23))
    {
        // Input number is smaller than 2^(-7), which is the smallest
        // fp8e4m3fnuz normal number
        f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
        result = static_cast<uint8_t>(f_bits - denorm_mask);
        if(result == 0)
        {
            // fnuz types don't have negative zero.
            return 0;
        }
    }
    else
    {
        // resulting mantissa is odd
        uint8_t mant_odd = (f_bits >> 20) & 1;

        // update exponent, rounding bias part 1
        f_bits += ((uint32_t)(8 - 127) << 23) + 0x7FFFF;

        // rounding bias part 2
        f_bits += mant_odd;

        // take the bits!
        result = static_cast<uint8_t>(f_bits >> 20);
    }

    result |= sign >> 24;

    return result;
}

/// Temporary half-precision expression.
/// This class represents a half-precision expression which just stores a single-precision value
/// internally.
struct expr
{
    /// Conversion constructor.
    /// \param f single-precision value to convert
    explicit expr(float f) : value_(f) {}

    /// Conversion to single-precision.
    /// \return single precision value representing expression value
    operator float() const { return value_; }

    private:
    /// Internal expression value stored in single-precision.
    float value_;
};

} // namespace detail

struct alignas(1) fp8e4m3fnuz
{
    uint8_t x;

    struct from_bits_t
    {
    };
    static constexpr MIGRAPHX_HIP_HOST_DEVICE from_bits_t from_bits() { return from_bits_t(); }

    MIGRAPHX_HIP_HOST_DEVICE fp8e4m3fnuz() : x(0) {}

    MIGRAPHX_HIP_HOST_DEVICE constexpr fp8e4m3fnuz(uint8_t bits, from_bits_t) : x(bits) {}

    MIGRAPHX_HIP_HOST_DEVICE fp8e4m3fnuz(const fp8e4m3fnuz& y) = default;

    inline explicit MIGRAPHX_HIP_HOST_DEVICE fp8e4m3fnuz(float value)
        : x(detail::fp8e4m3fnuz_from_fp32_value(value))
    {
    }

#if !defined(__HIP_NO_F8_CONVERSIONS__)
    // for the device kernels, this needs to be disabled since implicit_conversion op can type cast
    // any type to any other type and that results in conflicts in candidate overload resolutions.
    fp8e4m3fnuz& MIGRAPHX_HIP_HOST_DEVICE operator=(float rhs)
    {
        x = detail::fp8e4m3fnuz_from_fp32_value(rhs);
        return *this;
    }
#endif
    inline constexpr MIGRAPHX_HIP_HOST_DEVICE operator float() const
    {
        return detail::fp8e4m3fnuz_to_fp32_value(x);
    }
    fp8e4m3fnuz& MIGRAPHX_HIP_HOST_DEVICE operator=(const fp8e4m3fnuz& rhs) = default;

    fp8e4m3fnuz& MIGRAPHX_HIP_HOST_DEVICE operator=(fp8e4m3fnuz&& rhs) = default;

    inline bool MIGRAPHX_HIP_HOST_DEVICE isnan() const { return x == 0b10000000; }

    fp8e4m3fnuz& MIGRAPHX_HIP_HOST_DEVICE operator+=(float rhs)
    {
        x = detail::fp8e4m3fnuz_from_fp32_value(rhs + float(x));
        return *this;
    }
    fp8e4m3fnuz& MIGRAPHX_HIP_HOST_DEVICE operator-=(float rhs)
    {
        x = detail::fp8e4m3fnuz_from_fp32_value(rhs - float(x));
        return *this;
    }
    fp8e4m3fnuz& MIGRAPHX_HIP_HOST_DEVICE operator*=(float rhs)
    {
        x = detail::fp8e4m3fnuz_from_fp32_value(rhs * float(x));
        return *this;
    }
    fp8e4m3fnuz& MIGRAPHX_HIP_HOST_DEVICE operator/=(float rhs)
    {
        x = detail::fp8e4m3fnuz_from_fp32_value(rhs / float(x));
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& out, const fp8e4m3fnuz& value)
{
    out << (float)(value);
    return out;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace std {

template <>
class numeric_limits<migraphx::fp8e4m3fnuz>
{
    public:
    static constexpr bool is_specialized    = true;
    static constexpr bool is_signed         = true;
    static constexpr bool is_integer        = false;
    static constexpr bool is_exact          = false;
    static constexpr bool has_infinity      = false;
    static constexpr bool has_quiet_NaN     = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr auto has_denorm        = true;
    static constexpr auto has_denorm_loss   = true;
    static constexpr auto round_style       = numeric_limits<float>::round_style;
    static constexpr bool is_iec559         = false;
    static constexpr bool is_bounded        = true;
    static constexpr bool is_modulo         = false;
    static constexpr int digits             = 4;
    static constexpr int digits10           = 0;
    static constexpr int max_digits10       = 3;
    static constexpr int radix              = 2;
    static constexpr int min_exponent       = -5;
    static constexpr int min_exponent10     = -1;
    static constexpr int max_exponent       = 8;
    static constexpr int max_exponent10     = 2;
    static constexpr auto traps             = numeric_limits<float>::traps;
    static constexpr auto tinyness_before   = false;

    static constexpr migraphx::fp8e4m3fnuz min()
    {
        return migraphx::fp8e4m3fnuz(0x08, migraphx::fp8e4m3fnuz::from_bits());
    }
    static constexpr migraphx::fp8e4m3fnuz lowest()
    {
        return migraphx::fp8e4m3fnuz(0xFF, migraphx::fp8e4m3fnuz::from_bits());
    }
    static constexpr migraphx::fp8e4m3fnuz max()
    {
        return migraphx::fp8e4m3fnuz(0x7F, migraphx::fp8e4m3fnuz::from_bits());
    }
    static constexpr migraphx::fp8e4m3fnuz epsilon()
    {
        return migraphx::fp8e4m3fnuz(0x28, migraphx::fp8e4m3fnuz::from_bits());
    }
    static constexpr migraphx::fp8e4m3fnuz round_error()
    {
        return migraphx::fp8e4m3fnuz(0x38, migraphx::fp8e4m3fnuz::from_bits());
    }
    static constexpr migraphx::fp8e4m3fnuz infinity()
    {
        // NaN (no infinities)
        return migraphx::fp8e4m3fnuz(0x80, migraphx::fp8e4m3fnuz::from_bits());
    }
    static constexpr migraphx::fp8e4m3fnuz quiet_NaN()
    {
        return migraphx::fp8e4m3fnuz(0x80, migraphx::fp8e4m3fnuz::from_bits());
    }
    static constexpr migraphx::fp8e4m3fnuz denorm_min()
    {
        return migraphx::fp8e4m3fnuz(0x01, migraphx::fp8e4m3fnuz::from_bits());
    }
};

template <class T>
struct common_type<migraphx::fp8e4m3fnuz, T> : std::common_type<float, T> // NOLINT
{
};

template <class T>
struct common_type<T, migraphx::fp8e4m3fnuz> : std::common_type<float, T> // NOLINT
{
};

template <>
struct common_type<migraphx::fp8e4m3fnuz, migraphx::fp8e4m3fnuz>
{
    using type = float;
};

// template <>
// struct common_type<migraphx::fp8e4m3fnuz, migraphx::half>
// {
//     using type = float;
// };

// template <>
// struct common_type<migraphx::half, migraphx::fp8e4m3fnuz>
// {
//     using type = float;
// };

} // namespace std
#pragma clang diagnostic pop
#endif
