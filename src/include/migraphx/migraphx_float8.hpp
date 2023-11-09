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

#ifndef MIGRAPHX_GUARD_RTGLIB_FLOAT8_HPP
#define MIGRAPHX_GUARD_RTGLIB_FLOAT8_HPP
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wfloat-equal"
#pragma clang diagnostic ignored "-Wc++20-extensions"
#endif // __clang__

#ifndef MIGRAPHX_FP8_FNUZ
#define MIGRAPHX_FP8_FNUZ true
#endif // MIGRAPHX_FP8_FNUZ

// We are clipping in down conversion by default
#define MIGRAPHX_F8_DOWNCAST_CLIPPING 1

#include <cmath>
#include <cstdint>
#include <climits>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <sstream>
#include <iostream>
#include <string>
#include <utility>

namespace migraphx_f8_impl {

template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
constexpr uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

template <int wm, int we, typename T, bool negative_zero_nan>
constexpr T cast_from_f8(uint8_t x);

} // namespace migraphx_f8_impl

#include <migraphx/migraphx_f8_impl.hpp>

namespace migraphx_fp8 {

enum class migraphx_f8_rounding_mode
{
    standard, // standard rounding is doing RNE -- round to nearest even
    stochastic
};

enum class f8_type
{
    bf8 = 0, // s1e5m2
    fp8 = 1  // s1e4m3
};

template <typename T>
class numeric_limits;

template <migraphx_fp8::f8_type T = migraphx_fp8::f8_type::fp8>
struct float8
{
    uint8_t data = 0x00;
    // default constructor
    constexpr float8() = default;
    // default copy constructor
    constexpr float8(const float8<T>& y) = default;
    struct from_bits_t
    {
    };
    static constexpr from_bits_t from_bits() { return from_bits_t(); }

    explicit constexpr float8(uint8_t bits, from_bits_t) : data(bits) {}

    explicit constexpr float8(float v,
                              migraphx_fp8::migraphx_f8_rounding_mode rm =
                                  migraphx_fp8::migraphx_f8_rounding_mode::standard,
                              uint32_t rng = 0)
    {
        if constexpr(T == migraphx_fp8::f8_type::fp8)
        {
#ifdef MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_f8_impl::
                cast_to_f8<3, 4, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/, true /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_f8_rounding_mode::stochastic), rng);
#else  // MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_f8_impl::
                cast_to_f8<3, 4, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/, false /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_f8_rounding_mode::stochastic), rng);
#endif // MIGRAPHX_F8_DOWNCAST_CLIPPING
        }
        else
        {
#ifdef MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_f8_impl::
                cast_to_f8<2, 5, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/, true /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_f8_rounding_mode::stochastic), rng);
#else  // MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_f8_impl::
                cast_to_f8<2, 5, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/, false /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_f8_rounding_mode::stochastic), rng);
#endif // rocblas_F8_downcast_clipping}
        }
    }

    inline constexpr operator float() const
    {
        if constexpr(T == migraphx_fp8::f8_type::fp8)
        {
            return migraphx_f8_impl::
                cast_from_f8<3, 4, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/>(data);
        } // else
        return migraphx_f8_impl::cast_from_f8<2, 5, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/>(
            data);
    }

    inline constexpr bool is_zero() const
    {
        if constexpr(MIGRAPHX_FP8_FNUZ)
        {
            return data == 0x00;
        }
        else
        {
            return (data == 0x00) or (data == 0x80);
        }
    }

    inline constexpr bool is_nan() const
    {
        if constexpr(MIGRAPHX_FP8_FNUZ)
        {
            return data == 0x80;
        }
        else
        {
            if(T == migraphx_fp8::f8_type::bf8)
            {
                return (data == 0x7d) or (data == 0x7e) or (data == 0x7f) or (data == 0xfd) or
                       (data == 0xfe) or (data == 0xff);
            }
            else
            {
                return (data == 0x79) or (data == 0x7a) or (data == 0x7b) or (data == 0x7c) or
                       (data == 0x7d) or (data == 0x7e) or (data == 0x7f) or (data == 0xf9) or
                       (data == 0xfa) or (data == 0xfb) or (data == 0xfc) or (data == 0xfd) or
                       (data == 0xfe) or (data == 0xff);
            }
        }
    }

    inline constexpr bool is_inf() const
    {
        if constexpr(MIGRAPHX_FP8_FNUZ)
        {
            return data == 0x80;
        }
        else
        {
            if(T == migraphx_fp8::f8_type::bf8)
            {
                return (data == 0x7c) or (data == 0xfc);
            }
            else
            {
                return (data == 0x78) or (data == 0xf8);
            }
        }
    }

#define MIGRAPHX_FP8_UNARY_OP(unary_op, binary_op)                                    \
    constexpr float8& operator unary_op(const float8& rhs)                            \
    {                                                                                 \
        const auto tmp = static_cast<float>(*this) binary_op static_cast<float>(rhs); \
        *this          = static_cast<float8>(tmp);                                    \
        return *this;                                                                 \
    }                                                                                 \
    constexpr float8& operator unary_op(const float& rhs)                             \
    {                                                                                 \
        const auto tmp = static_cast<float>(*this) binary_op static_cast<float>(rhs); \
        *this          = static_cast<float8>(tmp);                                    \
        return *this;                                                                 \
    }

    MIGRAPHX_FP8_UNARY_OP(*=, *)
    MIGRAPHX_FP8_UNARY_OP(-=, -)
    MIGRAPHX_FP8_UNARY_OP(+=, +)
    MIGRAPHX_FP8_UNARY_OP(/=, /)

    inline constexpr float8& operator=(const float8& rhs) = default;
    inline constexpr float8& operator=(float8&& rhs)      = default;

    inline constexpr float8& operator=(float rhs)
    {
        *this = static_cast<float8>(rhs);
        return *this;
    }

    inline constexpr bool operator==(const float8& rhs) const
    {
        if((rhs.is_zero() and this->is_zero()) or
           (fabs(rhs - *this) < migraphx_fp8::numeric_limits<float8<T>>::epsilon()))
            return true;
        else if(rhs.is_nan() or rhs.is_inf() or this->is_nan() or this->is_inf())
            return false;

        return false;
    }

    inline constexpr bool operator<(const float8& rhs) const
    {
        const auto we   = static_cast<float>(*this);
        const auto them = static_cast<float>(rhs);
        return we < them;
    }

    inline constexpr bool operator>(const float8& rhs) const
    {
        const auto we   = static_cast<float>(*this);
        const auto them = static_cast<float>(rhs);
        return we > them;
    }
};

// Special operator overloading
template <migraphx_fp8::f8_type T>
inline std::ostream& operator<<(std::ostream& os, const migraphx_fp8::float8<T>& rhs)
{
    return os << static_cast<float>(rhs);
}

// NOLINTNEXTLINE
#define MIGRAPHX_FP8_BINARY_OP(binary_op, U)                                  \
    template <migraphx_fp8::f8_type T>                                        \
    inline constexpr U operator binary_op(const migraphx_fp8::float8<T>& lhs, \
                                          const migraphx_fp8::float8<T>& rhs) \
    {                                                                         \
        return U(static_cast<float>(lhs) binary_op static_cast<float>(rhs));  \
    }

// TODO: these should return floats
MIGRAPHX_FP8_BINARY_OP(*, migraphx_fp8::float8<T>)
MIGRAPHX_FP8_BINARY_OP(-, migraphx_fp8::float8<T>)
MIGRAPHX_FP8_BINARY_OP(/, migraphx_fp8::float8<T>)
MIGRAPHX_FP8_BINARY_OP(+, migraphx_fp8::float8<T>)
// TODO: Comparison ops shouldn't convert to float, maybe need to take care of rounding effects.
MIGRAPHX_FP8_BINARY_OP(==, bool)
MIGRAPHX_FP8_BINARY_OP(>=, bool)
MIGRAPHX_FP8_BINARY_OP(<=, bool)
MIGRAPHX_FP8_BINARY_OP(>, bool)
MIGRAPHX_FP8_BINARY_OP(<, bool)
MIGRAPHX_FP8_BINARY_OP(!=, bool)

template <migraphx_fp8::f8_type T>
inline migraphx_fp8::float8<T> fabs(migraphx_fp8::float8<T> v)
{
    v.data = v.data & 0x7f;
    return v;
}

template <class T>
constexpr T F8_Max()
{
    return T{0x7F, T::from_bits()};
}

template <class T>
constexpr T F8_Lowest()
{
    return T{0xFF, T::from_bits()};
}

using fp8e4m3fnuz = float8<migraphx_fp8::f8_type::fp8>;

template <>
class numeric_limits<migraphx_fp8::float8<migraphx_fp8::f8_type::fp8>>
{
    public:
    // TODO :figure out epsilon in Hex to make it constexpr
    static constexpr migraphx_fp8::float8<migraphx_fp8::f8_type::fp8> epsilon()
    {
        return migraphx_fp8::float8<migraphx_fp8::f8_type::fp8>(
            0x28, migraphx_fp8::float8<>::from_bits());
    }

    static constexpr migraphx_fp8::float8<migraphx_fp8::f8_type::fp8> quiet_NaN()
    {
        return migraphx_fp8::float8<migraphx_fp8::f8_type::fp8>(
            MIGRAPHX_FP8_FNUZ ? 0x80 : 0x7F, migraphx_fp8::float8<>::from_bits());
    }

    static constexpr migraphx_fp8::float8<migraphx_fp8::f8_type::fp8> max()
    {
        return migraphx_fp8::F8_Max<migraphx_fp8::float8<migraphx_fp8::f8_type::fp8>>();
    }

    // TODO figure out Hex value
    static migraphx_fp8::float8<migraphx_fp8::f8_type::fp8> min()
    {
        return static_cast<migraphx_fp8::float8<migraphx_fp8::f8_type::fp8>>(-1.0f) *
               migraphx_fp8::F8_Max<migraphx_fp8::float8<migraphx_fp8::f8_type::fp8>>();
    }

    static constexpr migraphx_fp8::float8<migraphx_fp8::f8_type::fp8> lowest()
    {
        return migraphx_fp8::F8_Lowest<migraphx_fp8::float8<migraphx_fp8::f8_type::fp8>>();
    }

    static constexpr migraphx_fp8::float8<migraphx_fp8::f8_type::fp8> infinity()
    {
        return migraphx_fp8::float8<migraphx_fp8::f8_type::fp8>(
            MIGRAPHX_FP8_FNUZ ? 0x80 : 0x7F, migraphx_fp8::float8<>::from_bits());
    }
};

template <>
class numeric_limits<migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>>
{
    public:
    static constexpr migraphx_fp8::float8<migraphx_fp8::f8_type::bf8> epsilon()
    {
        return migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>(
            0x34, migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>::from_bits());
    }

    static constexpr migraphx_fp8::float8<migraphx_fp8::f8_type::bf8> quiet_NaN()
    {
        return migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>(
            MIGRAPHX_FP8_FNUZ ? 0x80 : 0x7d,
            migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>::from_bits());
    }

    static constexpr migraphx_fp8::float8<migraphx_fp8::f8_type::bf8> max()
    {
        return static_cast<migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>>(
            migraphx_fp8::F8_Max<migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>>());
    }
    // TODO figure  out constexpr value
    static migraphx_fp8::float8<migraphx_fp8::f8_type::bf8> min()
    {
        return static_cast<migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>>(float(-1.0f)) *
               migraphx_fp8::F8_Max<migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>>();
    }
    static constexpr migraphx_fp8::float8<migraphx_fp8::f8_type::bf8> lowest()
    {
        return migraphx_fp8::F8_Lowest<migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>>();
    }

    static constexpr migraphx_fp8::float8<migraphx_fp8::f8_type::bf8> infinity()
    {
        return migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>(
            MIGRAPHX_FP8_FNUZ ? 0x80 : 0x7c,
            migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>::from_bits());
    }
};
} // namespace migraphx_fp8

// =================================================================================================
// define numeric limits for the new data type
namespace std {
inline bool isfinite(migraphx_fp8::float8<migraphx_fp8::f8_type::fp8> x) // NOLINT
{
    return x.is_inf();
}

inline bool isfinite(migraphx_fp8::float8<migraphx_fp8::f8_type::bf8> x) // NOLINT
{
    return x.is_inf();
}

inline bool isnan(migraphx_fp8::float8<migraphx_fp8::f8_type::fp8> x) // NOLINT
{
    return x.is_nan();
}

inline bool isnan(migraphx_fp8::float8<migraphx_fp8::f8_type::bf8> x) // NOLINT
{
    return x.is_nan();
}

template <>
class numeric_limits<migraphx_fp8::float8<migraphx_fp8::f8_type::fp8>>
    : public migraphx_fp8::numeric_limits<migraphx_fp8::float8<migraphx_fp8::f8_type::fp8>>
{
};

template <>
class numeric_limits<migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>>
    : public migraphx_fp8::numeric_limits<migraphx_fp8::float8<migraphx_fp8::f8_type::bf8>>
{
};

template <class T>
struct common_type<migraphx_fp8::fp8e4m3fnuz, T> : std::common_type<float, T> // NOLINT
{
};

template <class T>
struct common_type<T, migraphx_fp8::fp8e4m3fnuz> : std::common_type<float, T> // NOLINT
{
};

template <>
struct common_type<migraphx_fp8::fp8e4m3fnuz, migraphx_fp8::fp8e4m3fnuz>
{
    using type = float;
};

} // namespace std
// =================================================================================================
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif // MIGRAPHX_GUARD_RTGLIB_FLOAT8_HPP
