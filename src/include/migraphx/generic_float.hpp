/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef MIGRAPHX_GUARD_MIGRAPHX_GENERIC_FLOAT_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_GENERIC_FLOAT_HPP

#include <migraphx/config.hpp>
#include <migraphx/bit_cast.hpp>
#include <migraphx/bit.hpp>
#include <algorithm>
#include <limits>
#include <iostream>
#include <tuple>
#include <cstdint>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

constexpr std::size_t integer_divide_ceil(std::size_t x, std::size_t y)
{
    return (x + y - std::size_t{1}) / y;
}

// compute the smallest multiple of y that is greater than or equal to x
// this is equivalent to y * ceil(x / y)
constexpr std::size_t ceil_mul_of(std::size_t x, std::size_t y)
{
    return y * integer_divide_ceil(x, y);
}

template <unsigned int Bytes>
struct unsigned_type
{
};

template <>
struct unsigned_type<1>
{
    using type = std::uint8_t;
};

template <>
struct unsigned_type<2>
{
    using type = std::uint16_t;
};

template <>
struct unsigned_type<4>
{
    using type = std::uint32_t;
};

template <>
struct unsigned_type<8>
{
    using type = std::uint64_t;
};

// CRTP base for operators
template <class Derived>
struct generic_float_operators
{

// NOLINTNEXTLINE
#define MIGRAPHX_GENERIC_FLOAT_ASSIGN_OP(op)                                  \
    friend constexpr Derived& operator op(Derived & lhs, const Derived & rhs) \
    {                                                                         \
        float self = lhs;                                                     \
        float frhs = rhs;                                                     \
        self op frhs;                                                         \
        lhs = self;                                                           \
        return lhs;                                                           \
    }
    MIGRAPHX_GENERIC_FLOAT_ASSIGN_OP(*=)
    MIGRAPHX_GENERIC_FLOAT_ASSIGN_OP(-=)
    MIGRAPHX_GENERIC_FLOAT_ASSIGN_OP(+=)
    MIGRAPHX_GENERIC_FLOAT_ASSIGN_OP(/=)

// NOLINTNEXTLINE
#define MIGRAPHX_GENERIC_FLOAT_BINARY_OP(op)                                 \
    friend constexpr Derived operator op(const Derived& x, const Derived& y) \
    {                                                                        \
        return Derived(float(x) op float(y));                                \
    }
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(*)
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(-)
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(+)
    MIGRAPHX_GENERIC_FLOAT_BINARY_OP(/)

// NOLINTNEXTLINE
#define MIGRAPHX_GENERIC_FLOAT_COMPARE_OP(op)                             \
    friend constexpr bool operator op(const Derived& x, const Derived& y) \
    {                                                                     \
        return float(x) op float(y);                                      \
    }
    MIGRAPHX_GENERIC_FLOAT_COMPARE_OP(<)
    MIGRAPHX_GENERIC_FLOAT_COMPARE_OP(<=)
    MIGRAPHX_GENERIC_FLOAT_COMPARE_OP(>)
    MIGRAPHX_GENERIC_FLOAT_COMPARE_OP(>=)

    protected:
    // prohibit creation of this base object
    generic_float_operators() = default;
};

struct float32_parts
{
    unsigned int mantissa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;

    static constexpr unsigned int exponent_width() { return 8; }

    static constexpr unsigned int mantissa_width() { return 23; }

    static constexpr unsigned int max_exponent() { return all_ones<8>(); }

    static constexpr int exponent_bias() { return all_ones<7>(); }

    constexpr float to_float() const noexcept { return migraphx::bit_cast<float>(*this); }
};

constexpr float32_parts get_parts(float f) { return migraphx::bit_cast<float32_parts>(f); }

template <unsigned int MantissaSize, unsigned int ExponentSize, unsigned int Flags = 0>
struct __attribute__((packed, may_alias)) generic_float
    : generic_float_operators<generic_float<MantissaSize, ExponentSize, Flags>>
{
    using type = typename unsigned_type<bit_ceil(
        integer_divide_ceil(MantissaSize + ExponentSize + 1, 8))>::type;

    type mantissa : MantissaSize;
    type exponent : ExponentSize;
    type sign : 1;

    static constexpr int exponent_bias() { return all_ones<ExponentSize - 1>(); }

    explicit constexpr generic_float(float f = 0.0) noexcept { from_float(get_parts(f)); }

    constexpr generic_float& operator=(float f) noexcept
    {
        from_float(get_parts(f));
        return *this;
    }

    constexpr generic_float operator-() const noexcept
    {
        generic_float result = *this;
        result.sign          = not this->sign;
        return result;
    }

    constexpr generic_float operator+() const noexcept { return *this; }

    constexpr float to_float() const noexcept
    {
        float32_parts f{};
        f.sign = sign;

        if(exponent == 0 and ExponentSize != float32_parts::exponent_width()) // subnormal fps
        {

            if(mantissa == 0)
            {
                f.exponent = 0;
                f.mantissa = 0;
            }
            else
            {
                type shift = 0;
                f.mantissa = mantissa;

                if(MantissaSize < float32_parts::mantissa_width())
                {
                    shift = MantissaSize - ((sizeof(type) * 8) - countl_zero(mantissa));
                    f.mantissa <<= (shift + 1u);
                }

                f.exponent = float32_parts::exponent_bias() - exponent_bias() - shift;
                f.mantissa = f.mantissa << (float32_parts::mantissa_width() - MantissaSize);
            }
        }
        else if(exponent == all_ones<ExponentSize>())
        {
            f.mantissa = mantissa << (float32_parts::mantissa_width() - MantissaSize);
            f.exponent = float32_parts::max_exponent();
        }
        else
        {
            f.mantissa               = mantissa << (float32_parts::mantissa_width() - MantissaSize);
            constexpr const int diff = float32_parts::exponent_bias() - exponent_bias();
            f.exponent               = int(exponent) + diff;
        }

        return f.to_float();
    }

    constexpr void from_float(float32_parts f) noexcept
    {
        sign = f.sign;

        if(f.exponent == 0)
        {
            exponent = 0;
            mantissa = f.mantissa >> (float32_parts::mantissa_width() - MantissaSize);
        }
        else if(f.exponent == float32_parts::max_exponent())
        {
            exponent = all_ones<ExponentSize>();
            mantissa = f.mantissa >> (float32_parts::mantissa_width() - MantissaSize);
        }
        else
        {
            constexpr const int diff = float32_parts::exponent_bias() - exponent_bias();
            auto e                   = int(f.exponent) - diff;

            if(e >= static_cast<int>(all_ones<ExponentSize>()))
            {
                exponent = all_ones<ExponentSize>();
                mantissa = 0;
            }
            else if(e < 1)
            {
                exponent = 0;

                auto shift        = diff - int(f.exponent);
                auto shift_amount = shift + (float32_parts::mantissa_width() - MantissaSize) + 1;

                if(shift_amount < (sizeof(unsigned int) * 8))
                {
                    mantissa = (f.mantissa | (1u << float32_parts::mantissa_width())) >>
                               (shift + (float32_parts::mantissa_width() - MantissaSize) + 1);
                }
                else
                {
                    mantissa = 0;
                }
            }
            else
            {
                exponent = int(f.exponent) - diff;
                mantissa = f.mantissa >> (float32_parts::mantissa_width() - MantissaSize);
            }
        }

        exponent = std::min<type>(exponent, all_ones<ExponentSize>());
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

    constexpr bool has_infinity() const noexcept { return true; }

    constexpr bool is_finite() const noexcept { return exponent != all_ones<ExponentSize>(); }

    constexpr operator float() const noexcept { return this->to_float(); }

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
        x.mantissa = 1u << (MantissaSize - 2u);
        return x;
    }

    static constexpr generic_float qnan()
    {
        generic_float x{};
        x.exponent = all_ones<ExponentSize>();
        x.mantissa = 1u << (MantissaSize - 1u);
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
        x.sign     = 0;
        return x;
    }

    static constexpr generic_float lowest()
    {
        generic_float x{};
        x.exponent = all_ones<ExponentSize>() - 1;
        x.mantissa = all_ones<MantissaSize>();
        x.sign     = 1;
        return x;
    }

    static constexpr generic_float max()
    {
        generic_float x{};
        x.exponent = all_ones<ExponentSize>() - 1;
        x.mantissa = all_ones<MantissaSize>();
        x.sign     = 0;
        return x;
    }

    static constexpr generic_float epsilon()
    {
        generic_float x{1.0};
        x.mantissa++;
        return generic_float{x.to_float() - 1.0f};
    }

    friend constexpr bool operator==(const generic_float& x, const generic_float& y)
    {
        if(not x.is_finite() or not y.is_finite())
            return false;

        if((x.mantissa == 0 and x.exponent == 0) and (y.mantissa == 0 and y.exponent == 0))
        {
            return true;
        }

        return std::tie(x.mantissa, x.exponent, x.sign) == std::tie(y.mantissa, y.exponent, y.sign);
    }

    friend constexpr bool operator!=(const generic_float& x, const generic_float& y)
    {
        return not(x == y);
    }

    constexpr generic_float& operator++() noexcept
    {
        *this += generic_float(1.0f);
        return *this;
    }

    const generic_float operator++(int) noexcept // NOLINT(readability-const-return-type)
    {
        generic_float temp = *this;
        *this += generic_float(1.0f);
        return temp;
    }
};

template <unsigned int Flags>
struct __attribute__((packed, may_alias)) generic_float<0, 8, Flags>
    : generic_float_operators<generic_float<0, 8, Flags>>
{
    uint8_t exponent;

    static constexpr int exponent_bias() { return all_ones<7>(); }

    explicit constexpr generic_float(float f = 1.0) noexcept { from_float(get_parts(f)); }

    constexpr generic_float& operator=(float f) noexcept
    {
        from_float(get_parts(f));
        return *this;
    }

    // No sign for this type
    constexpr generic_float operator-() const noexcept { return snan(); }

    constexpr generic_float operator+() const noexcept { return *this; }

    constexpr float to_float() const noexcept
    {
        float32_parts f{};
        f.sign     = 0;
        if(exponent == 0)
        {
            // 2^(-127) is a fp32 denormal number
            f.mantissa = 1;
            f.mantissa = f.mantissa << (float32_parts::mantissa_width() - 1);
        }
        else if(exponent == all_ones<8>())
        {
            // setting to fp32 qNaN
            f.mantissa = (1 << (float32_parts::mantissa_width() - 1)) + 1;
        }
        else
        {
            f.mantissa = 0;
        }
        f.exponent = exponent;
        return f.to_float();
    }

    /**
     * Extracts only exponent bits from float.
     * All fp32 denorm numbers will go to fp8e8m0{2^(-127)}.
     * All fp32 NaN and infinity go to fp8e8m0{NaN}.
     */
    constexpr void from_float(float32_parts f) noexcept { exponent = f.exponent; }

    // No denorm numbers in fp8e8m0.
    constexpr bool is_normal() const noexcept { return not is_nan(); }

    // No infinity numbers in fp8e8m0.
    constexpr bool is_inf() const noexcept { return false; }

    constexpr bool is_nan() const noexcept { return exponent == all_ones<8>(); }

    constexpr bool is_finite() const noexcept { return not is_nan(); }

    constexpr bool has_infinity() const noexcept
    {
        return false;
        ;
    }

    constexpr operator float() const noexcept { return this->to_float(); }

    // doesn't have infinity, returning 2**0
    static constexpr generic_float infinity()
    {
        generic_float x{};
        x.exponent = all_ones<8>() >> 1u;
        return x;
    }

    // only one NaN value
    static constexpr generic_float snan()
    {
        generic_float x{};
        x.exponent = all_ones<8>();
        return x;
    }

    // only one NaN value
    static constexpr generic_float qnan() { return snan(); }

    // min value = 2**(-127)
    static constexpr generic_float min()
    {
        generic_float x{};
        x.exponent = 0;
        return x;
    }

    // No subnormal numbers in FP8E8M0
    static constexpr generic_float denorm_min() { return min(); }

    static constexpr generic_float lowest() { return min(); }

    // max value = 2**(127)
    static constexpr generic_float max()
    {
        generic_float x{};
        x.exponent = all_ones<8>() - 1;
        return x;
    }

    // next number from 2**0 is 2**1 so epsilon is 2**0
    static constexpr generic_float epsilon()
    {
        generic_float x{};
        x.exponent = all_ones<8>() >> 1u;
        return x;
    }

    friend constexpr bool operator==(const generic_float& x, const generic_float& y)
    {

        return x.exponent == y.exponent;
    }

    friend constexpr bool operator!=(const generic_float& x, const generic_float& y)
    {
        return not(x == y);
    }

    constexpr generic_float& operator++() noexcept
    {
        ++exponent;
        return *this;
    }

    const generic_float operator++(int) noexcept // NOLINT(readability-const-return-type)
    {
        generic_float temp = *this;
        operator++(this->exponent);
        return temp;
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

// NOLINTBEGIN(cert-dcl58-cpp)
namespace std {

template <unsigned int M, unsigned int E, unsigned int F>
class numeric_limits<migraphx::generic_float<M, E, F>>
{
    public:
    static constexpr bool has_infinity = not(M == 0 and E == 0);
    static constexpr migraphx::generic_float<M, E, F> epsilon()
    {
        return migraphx::generic_float<M, E, F>::epsilon();
    }

    static constexpr migraphx::generic_float<M, E, F> quiet_NaN()
    {
        return migraphx::generic_float<M, E, F>::qnan();
    }

    static constexpr migraphx::generic_float<M, E, F> signaling_NaN()
    {
        return migraphx::generic_float<M, E, F>::snan();
    }

    static constexpr migraphx::generic_float<M, E, F> max()
    {
        return migraphx::generic_float<M, E, F>::max();
    }

    static constexpr migraphx::generic_float<M, E, F> min()
    {
        return migraphx::generic_float<M, E, F>::min();
    }

    static constexpr migraphx::generic_float<M, E, F> lowest()
    {
        return migraphx::generic_float<M, E, F>::lowest();
    }

    static constexpr migraphx::generic_float<M, E, F> infinity()
    {
        return migraphx::generic_float<M, E, F>::infinity();
    }

    static constexpr migraphx::generic_float<M, E, F> denorm_min()
    {
        return migraphx::generic_float<M, E, F>::denorm_min();
    }
};

template <unsigned int M, unsigned int E, unsigned int F, class T>
struct common_type<migraphx::generic_float<M, E, F>, T> : std::common_type<float, T>
{
};

template <unsigned int M, unsigned int E, unsigned int F, class T>
struct common_type<T, migraphx::generic_float<M, E, F>> : std::common_type<float, T>
{
};

template <unsigned int M, unsigned int E, unsigned int F>
struct common_type<migraphx::generic_float<M, E, F>, migraphx::generic_float<M, E, F>>
{
    using type = migraphx::generic_float<M, E, F>;
};

template <unsigned int M1,
          unsigned int E1,
          unsigned int F1,
          unsigned int M2,
          unsigned int E2,
          unsigned int F2>
struct common_type<migraphx::generic_float<M1, E1, F1>, migraphx::generic_float<M2, E2, F2>>
{
    using type = float;
};

} // namespace std
// NOLINTEND(cert-dcl58-cpp)

#endif // MIGRAPHX_GUARD_MIGRAPHX_GENERIC_FLOAT_HPP
