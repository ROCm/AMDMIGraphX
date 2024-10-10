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
 */

#ifndef MIGRAPHX_GUARD_RTGLIB_HALF_HPP
#define MIGRAPHX_GUARD_RTGLIB_HALF_HPP

#include <half/half.hpp>
#include <migraphx/config.hpp>
#include <migraphx/float8.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace half {

template<unsigned int N>
constexpr unsigned int all_ones() noexcept
{
    return (1 << N) - 1;
}

struct float16_parts 
{
    unsigned int mantissa : 10;
    unsigned int exponent : 5;
    unsigned int sign : 1;

    static constexpr unsigned int mantissa_width()
    {
        return 23;
    }

    static constexpr unsigned int max_exponent()
    {
        return all_ones<5>();
    }

    static constexpr int exponent_bias()
    {
        return all_ones<4>();
    }

    constexpr float to_float() const noexcept
    {
        return migraphx::bit_cast<float>(*this);
    }
};

constexpr float16_parts get_parts(float f)
{
    return migraphx::bit_cast<float16_parts>(f);
}

template<unsigned int MantissaSize, unsigned int ExponentSize, unsigned int Flags = 0>
struct generic_float
{
    unsigned int mantissa : MantissaSize;
    unsigned int exponent : ExponentSize;
    unsigned int sign : 1;

    static constexpr int exponent_bias()
    {
        return all_ones<ExponentSize - 1>();
    }

    explicit generic_float(float f = 0.0) noexcept
    {
        from_float(get_parts(f));
    }

    constexpr float to_float() const noexcept
    {
        float16_parts f{};
        f.sign = sign;
        f.mantissa = mantissa << (float16_parts::mantissa_width() - MantissaSize);
        if(exponent == all_ones<ExponentSize>())
        {
            f.exponent = float16_parts::max_exponent();
        }
        else
        {
            constexpr const auto diff = float16_parts::exponent_bias() - exponent_bias();
            f.exponent = exponent + diff;
        }
        return f.to_float();
    }

    constexpr void from_float(float16_parts f) noexcept
    {
        sign  = f.sign;
        mantissa = f.mantissa >> (float16_parts::mantissa_width() - MantissaSize);

        if(f.exponent == 0)
        {
            exponent = 0;
        }
        else if(f.exponent == float16_parts::max_exponent())
        {
            exponent = all_ones<ExponentSize>();
        }
        else
        {
            constexpr const int diff = float16_parts::exponent_bias() - exponent_bias();
            auto e = int(f.exponent) - diff;
            if(e >= all_ones<ExponentSize>())
            {
                exponent = all_ones<ExponentSize>();
                mantissa = 0;
            }
            else if(e < 0)
            {
                exponent = 0;
                mantissa = 0;
            }
            else
            {
                exponent = f.exponent - diff;
            }
        }

        exponent = std::min(f.exponent, all_ones<ExponentSize>());
    }

    constexpr bool is_normal() const noexcept
    {
        return exponent != all_ones<ExponentSize>() and exponent != 0;
    }

    constexpr bool is_inf() const noexcept
    {
        return exponent == all_ones<ExponentSize>() and mantissa == 0;
    }

    constexpr bool is_nan() const noexcept
    {
        return exponent == all_ones<ExponentSize>() and mantissa != 0;
    }

    constexpr bool is_finite() const noexcept
    {
        return exponent != all_ones<ExponentSize>();
    }

    constexpr operator float() const noexcept
    {
        return this->to_float();
    }

    static constexpr generic_float infinity()
    {
        generic_float x{};
        x.exponent = all_ones<ExponentSize>();
        return x;
    }

    static constexpr generic_float snan()
    {
        generic_float x{};
        x.exponent = all_ones<ExponentSize>();
        x.mantissa = 1 << (MantissaSize - 2);
        return x;
    }

    static constexpr generic_float qnan()
    {
        generic_float x{};
        x.exponent = all_ones<ExponentSize>();
        x.mantissa = 1 << (MantissaSize - 1);
        return x;
    }

    static constexpr generic_float min()
    {
        generic_float x{};
        x.exponent = 1;
        x.mantissa = 0;
        return x;
    }

    static constexpr generic_float denorm_min()
    {
        generic_float x{};
        x.exponent = 0;
        x.mantissa = 1;
        x.sign = 0;
        return x;
    }

    static constexpr generic_float lowest()
    {
        generic_float x{};
        x.exponent = all_ones<ExponentSize>() - 1;
        x.mantissa = all_ones<MantissaSize>();
        x.sign = 1;
        return x;
    }

    static constexpr generic_float max()
    {
        generic_float x{};
        x.exponent = all_ones<ExponentSize>() - 1;
        x.mantissa = all_ones<MantissaSize>();
        x.sign = 0;
        return x;
    }

    static constexpr generic_float epsilon()
    {
        generic_float x{1.0};
        x.mantissa++;
        return generic_float{x.to_float() - 1.0f};
    }
// NOLINTNEXTLINE
#define MIGRAPHX_GENERIC_FLOAT_ASSIGN_OP(op) \
    constexpr generic_float& operator op(const generic_float& rhs) \
    { \
        float self = *this; \
        float frhs = rhs; \
        self op frhs; \
        *this = generic_float(self); \
        return *this; \
    }
    MIGRAPHX_GENERIC_FLOAT_ASSIGN_OP(*=)
    MIGRAPHX_GENERIC_FLOAT_ASSIGN_OP(-=)
    MIGRAPHX_GENERIC_FLOAT_ASSIGN_OP(+=)
    MIGRAPHX_GENERIC_FLOAT_ASSIGN_OP(/=)
// NOLINTNEXTLINE
#define MIGRAPHX_GENERIC_FLOAT_BINARY_OP(op) \
    friend constexpr generic_float operator op(const generic_float& x, const generic_float& y) \
    { \
        return generic_float(float(x) op float(y)); \
    }
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(*)
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(-)
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(+)
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(/)
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(<)
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(<=)
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(>)
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(>=)

    friend constexpr generic_float operator==(const generic_float& x, const generic_float& y)
    {
        if (not x.is_finite() or not y.is_finite())
            return false;
        return std::tie(x.mantissa, x.exponent, x.sign) == std::tie(y.mantissa, y.exponent, y.sign);
    }

    friend constexpr generic_float operator!=(const generic_float& x, const generic_float& y)
    {
        return not(x == y);
    }
};

using half = migraphx::half::generic_float<10, 5>;

namespace detail {
template <class T>
struct deduce
{
    using type = T;
};

#ifdef HAS_HALF_V1
template <>
struct deduce<half_float::detail::expr>
{
    using type = half;
};
#endif
} // namespace detail

template <class T>
using deduce = typename detail::deduce<T>::type;

} // namespace half
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace std {

template <class T>
struct common_type<migraphx::half::half, T> : std::common_type<float, T> // NOLINT
{
};

template <class T>
struct common_type<T, migraphx::half::half> : std::common_type<float, T> // NOLINT
{
};

template <>
struct common_type<migraphx::fp8::fp8e4m3fnuz, migraphx::half::half>
{
    using type = float;
};

template <>
struct common_type<migraphx::half::half, migraphx::fp8::fp8e4m3fnuz>
{
    using type = float;
};

template <>
struct common_type<migraphx::fp8::fp8e4m3fn, migraphx::half::half>
{
    using type = float;
};

template <>
struct common_type<migraphx::half::half, migraphx::fp8::fp8e4m3fn>
{
    using type = float;
};

template <>
struct common_type<migraphx::fp8::fp8e5m2, migraphx::half::half>
{
    using type = float;
};

template <>
struct common_type<migraphx::half::half, migraphx::fp8::fp8e5m2>
{
    using type = float;
};

template <>
struct common_type<migraphx::half::half, migraphx::half::half>
{
    using type = migraphx::half::half;
};

} // namespace std

#endif
