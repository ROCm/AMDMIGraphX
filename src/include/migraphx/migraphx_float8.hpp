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
#pragma clang diagnostic ignored "-Wmacro-redefined"
#pragma clang diagnostic ignored "-Wc++20-extensions"
#endif // __clang__

#if(defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__))
// need to include hip_runtime.h otherwise it complains about __host__ and __device__
#if defined(MIGRAPHX_JIT_USE_HIPRTC)
#include <migraphx/kernels/hip.hpp>
#else
#include <hip/hip_runtime.h>
#endif
#define MIGRAPHX_HIP_HOST_DEVICE __host__ __device__
#define MIGRAPHX_HIP_HOST __host__
#else
#define MIGRAPHX_HIP_HOST_DEVICE
#define MIGRAPHX_HIP_HOST
#endif // HIP_PLATFORM_AMD

#define MIGRAPHX_HIP_DEVICE __device__

#ifndef MIGRAPHX_FP8_FNUZ
#define MIGRAPHX_FP8_FNUZ true
#endif // MIGRAPHX_FP8_FNUZ

// We are clipping in down conversion by default
#define MIGRAPHX_F8_DOWNCAST_CLIPPING 1
#if defined(MIGRAPHX_JIT_USE_HIPRTC)
#include <migraphx/kernels/types.hpp>
using uint8_t  = migraphx::uint8_t;
using uint16_t = migraphx::uint16_t;
using uint32_t = migraphx::uint32_t;
#else
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
#endif

namespace migraphx_hip_f8_impl {

template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
MIGRAPHX_HIP_HOST_DEVICE constexpr uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

template <int wm, int we, typename T, bool negative_zero_nan>
MIGRAPHX_HIP_HOST_DEVICE constexpr T cast_from_f8(uint8_t x);

} // namespace migraphx_hip_f8_impl

#include <migraphx/migraphx_hip_f8_impl.hpp>

namespace migraphx_fp8 {

enum class migraphx_hip_f8_rounding_mode
{
    standard, // standard rounding is doing RNE -- round to nearest even
    stochastic
};

enum class hip_f8_type
{
    bf8 = 0, // s1e5m2
    fp8 = 1  // s1e4m3
};

template <typename T>
class NumericLimits;

template <migraphx_fp8::hip_f8_type T = migraphx_fp8::hip_f8_type::fp8>
struct hip_f8
{
    uint8_t data;
    // default constructor
    MIGRAPHX_HIP_HOST_DEVICE constexpr hip_f8() = default;
    // default copy constructor
    MIGRAPHX_HIP_HOST_DEVICE constexpr hip_f8(const hip_f8<T>& y) = default;
    struct from_bits_t
    {
    };
    static constexpr MIGRAPHX_HIP_HOST_DEVICE from_bits_t from_bits() { return from_bits_t(); }

    MIGRAPHX_HIP_HOST_DEVICE explicit constexpr hip_f8(uint8_t bits, from_bits_t) : data(bits) {}

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    // device specific optimized F8 down-conversion code

    template <bool stochastic_rounding = false>
    static MIGRAPHX_HIP_DEVICE uint8_t cast_to_f8_from_f32(float v, uint32_t rng = 0)
    {
        uint8_t i8data;
        union
        {
            float fval;
            uint32_t i32val;
            uint8_t i8val[4]; // NOTE: not endian independent
        } val;

        uint32_t ival = 0;
        val.fval      = v;

#ifdef MIGRAPHX_F8_DOWNCAST_CLIPPING
        if constexpr(T == migraphx_fp8::hip_f8_type::fp8)
        {
            if((val.i32val & 0x7F800000) != 0x7F800000) /// propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
        }
        else
        {
            if((val.i32val & 0x7F800000) != 0x7F800000) // propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);
        }
#endif
        if(stochastic_rounding)
        {
            if constexpr(T == migraphx_fp8::hip_f8_type::fp8)
            {
                ival = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
            }
            else
            {
                ival = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
            }
        }
        else // RNE CVT
        {
            if constexpr(T == migraphx_fp8::hip_f8_type::fp8)
            {
                ival = __builtin_amdgcn_cvt_pk_fp8_f32(
                    val.fval, val.fval, ival, false); // false -> WORD0
            }
            else
            {
                ival = __builtin_amdgcn_cvt_pk_bf8_f32(
                    val.fval, val.fval, ival, false); // false -> WORD0}
            }
        }
        val.i32val = ival;
        i8data     = val.i8val[0]; // little endian

        return i8data;
    }
#endif // __gfx940__

       // constructor from float
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)

    // NOTE: ON-DEVICE... always optimal bias
    explicit MIGRAPHX_HIP_DEVICE hip_f8(float v,
                                        migraphx_fp8::migraphx_hip_f8_rounding_mode rm =
                                            migraphx_fp8::migraphx_hip_f8_rounding_mode::standard,
                                        uint32_t rng = 0)
    {
        // runtime branch, use cast_to_f8_from_f32 if want to avoid it
        if(rm == migraphx_fp8::migraphx_hip_f8_rounding_mode::stochastic)
            data = cast_to_f8_from_f32<true>(v, rng);
        else
            data = cast_to_f8_from_f32<false>(v);
    }

    // Host only implementation using s/w simulation
    explicit MIGRAPHX_HIP_HOST
#else
    // both Host and DEVICE for non-gfx940 using s/w simulation
    explicit constexpr MIGRAPHX_HIP_HOST_DEVICE
#endif
    hip_f8(float v,
           migraphx_fp8::migraphx_hip_f8_rounding_mode rm =
               migraphx_fp8::migraphx_hip_f8_rounding_mode::standard,
           uint32_t rng = 0)
    {
        if constexpr(T == migraphx_fp8::hip_f8_type::fp8)
        {
#ifdef MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_hip_f8_impl::
                cast_to_f8<3, 4, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/, true /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_hip_f8_rounding_mode::stochastic), rng);
#else  // MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_hip_f8_impl::
                cast_to_f8<3, 4, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/, false /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_hip_f8_rounding_mode::stochastic), rng);
#endif // MIGRAPHX_F8_DOWNCAST_CLIPPING
        }
        else
        {
#ifdef MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_hip_f8_impl::
                cast_to_f8<2, 5, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/, true /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_hip_f8_rounding_mode::stochastic), rng);
#else  // MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_hip_f8_impl::
                cast_to_f8<2, 5, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/, false /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_hip_f8_rounding_mode::stochastic), rng);
#endif // rocblas_F8_downcast_clipping}
        }
    }

    /*
        // Constructor from half
        explicit constexpr MIGRAPHX_HIP_HOST_DEVICE
        hip_f8(migraphx::half v,
               migraphx_fp8::migraphx_hip_f8_rounding_mode rm =
                   migraphx_fp8::migraphx_hip_f8_rounding_mode::standard,
               uint32_t rng = 0)
            : hip_f8((float)v, rm, rng)
        {
        }

    // constructor from int
    explicit constexpr MIGRAPHX_HIP_HOST_DEVICE
    hip_f8(int v,
           migraphx_fp8::migraphx_hip_f8_rounding_mode rm =
               migraphx_fp8::migraphx_hip_f8_rounding_mode::standard,
           uint32_t rng = 0)
        : hip_f8((float)v, rm, rng)
    {
    }

    // constructor from double
    explicit constexpr MIGRAPHX_HIP_HOST_DEVICE
    hip_f8(double v,
           migraphx_fp8::migraphx_hip_f8_rounding_mode rm =
               migraphx_fp8::migraphx_hip_f8_rounding_mode::standard,
           uint32_t rng = 0)
        : hip_f8((float)v, rm, rng)
    {
    }
    */
    /**/
    // convert to float
// #if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#if 0 // need constexpr operator(). This version can't be constexpr
    // upcast using device specific intrinsic
    inline MIGRAPHX_HIP_DEVICE operator float() const
    {
        float fval;
        uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
        if constexpr(T == migraphx_fp8::hip_f8_type::fp8)
        {
            asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
        }
        else
        {
            asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
        }

        return fval;
    }

    inline constexpr MIGRAPHX_HIP_HOST operator float() const
#else // non gfx940
    inline constexpr MIGRAPHX_HIP_HOST_DEVICE operator float() const
#endif
    {
        if constexpr(T == migraphx_fp8::hip_f8_type::fp8)
        {
            return migraphx_hip_f8_impl::
                cast_from_f8<3, 4, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/>(data);
        } // else
        return migraphx_hip_f8_impl::
            cast_from_f8<2, 5, float, MIGRAPHX_FP8_FNUZ /*negative_zero_nan*/>(data);
    }

    /*
        // convert to half
        explicit inline MIGRAPHX_HIP_HOST_DEVICE operator migraphx::half() const
        {
            return migraphx::half(float(*this)); // convert to float, then convert to f16
        }
    */

    // check for zero
    inline MIGRAPHX_HIP_HOST_DEVICE constexpr bool is_zero() const
    {
        if constexpr(MIGRAPHX_FP8_FNUZ)
        {
            return data == 0x00;
        }
        else
        {
            return (data == 0x00) || (data == 0x80);
        }
    }

    // check for nan
    inline MIGRAPHX_HIP_HOST_DEVICE constexpr bool is_nan() const
    {
        if constexpr(MIGRAPHX_FP8_FNUZ)
        {
            return data == 0x80;
        }
        else
        {
            if(T == migraphx_fp8::hip_f8_type::bf8)
            {
                return (data == 0x7d) || (data == 0x7e) || (data == 0x7f) || (data == 0xfd) ||
                       (data == 0xfe) || (data == 0xff);
            }
            else
            {
                return (data == 0x79) || (data == 0x7a) || (data == 0x7b) || (data == 0x7c) ||
                       (data == 0x7d) || (data == 0x7e) || (data == 0x7f) || (data == 0xf9) ||
                       (data == 0xfa) || (data == 0xfb) || (data == 0xfc) || (data == 0xfd) ||
                       (data == 0xfe) || (data == 0xff);
            }
        }
    }

    // check for inf
    inline MIGRAPHX_HIP_HOST_DEVICE constexpr bool is_inf() const
    {
        if constexpr(MIGRAPHX_FP8_FNUZ)
        {
            return data == 0x80;
        }
        else
        {
            if(T == migraphx_fp8::hip_f8_type::bf8)
            {
                return (data == 0x7c) || (data == 0xfc);
            }
            else
            {
                return (data == 0x78) || (data == 0xf8);
            }
        }
    }

#define MIGRAPHX_FP8_UNARY_OP(unary_op, binary_op)                                    \
    constexpr hip_f8& MIGRAPHX_HIP_HOST_DEVICE operator unary_op(const hip_f8& rhs)   \
    {                                                                                 \
        const auto tmp = static_cast<float>(*this) binary_op static_cast<float>(rhs); \
        *this          = static_cast<hip_f8>(tmp);                                    \
        return *this;                                                                 \
    }                                                                                 \
    constexpr hip_f8& MIGRAPHX_HIP_HOST_DEVICE operator unary_op(const float& rhs)    \
    {                                                                                 \
        const auto tmp = static_cast<float>(*this) binary_op static_cast<float>(rhs); \
        *this          = static_cast<hip_f8>(tmp);                                    \
        return *this;                                                                 \
    }

    MIGRAPHX_FP8_UNARY_OP(*=, *)
    MIGRAPHX_FP8_UNARY_OP(-=, -)
    MIGRAPHX_FP8_UNARY_OP(+=, +)
    MIGRAPHX_FP8_UNARY_OP(/=, /)

    inline MIGRAPHX_HIP_HOST_DEVICE constexpr hip_f8& operator=(const hip_f8& rhs) = default;
    inline MIGRAPHX_HIP_HOST_DEVICE constexpr hip_f8& operator=(hip_f8&& rhs)      = default;

#if !defined(__HIP_NO_F8_CONVERSIONS__)
    // for the device kernels, this needs to be disabled since implicit_conversion op can type cast
    // any type to any other type and that results in conflicts in candidate overload resolutions.
    inline constexpr hip_f8& MIGRAPHX_HIP_HOST_DEVICE operator=(float rhs)
    {
        *this = static_cast<hip_f8>(rhs);
        return *this;
    }
#endif

    inline MIGRAPHX_HIP_HOST_DEVICE constexpr bool operator==(const hip_f8& rhs) const
    {
        if((rhs.is_zero() && this->is_zero()) ||
           (fabs(rhs - *this) < migraphx_fp8::NumericLimits<hip_f8<T>>::epsilon()))
            return true;
        else if(rhs.is_nan() || rhs.is_inf() || this->is_nan() || this->is_inf())
            return false;

        return false;
    }

    inline MIGRAPHX_HIP_HOST_DEVICE constexpr bool operator<(const hip_f8& rhs) const
    {
        const auto we   = static_cast<float>(*this);
        const auto them = static_cast<float>(rhs);
        return we < them;
    }

    inline MIGRAPHX_HIP_HOST_DEVICE constexpr bool operator>(const hip_f8& rhs) const
    {
        const auto we   = static_cast<float>(*this);
        const auto them = static_cast<float>(rhs);
        return we > them;
    }
};

#ifndef MIGRAPHX_JIT_USE_HIPRTC
// Special operator overloading
template <migraphx_fp8::hip_f8_type T>
inline std::ostream& operator<<(std::ostream& os, const migraphx_fp8::hip_f8<T>& rhs)
{
    return os << static_cast<float>(rhs);
}
#endif

// NOLINTNEXTLINE
#define MIGRAPHX_FP8_BINARY_OP(binary_op, U)                                    \
    template <migraphx_fp8::hip_f8_type T>                                      \
    inline constexpr U MIGRAPHX_HIP_HOST_DEVICE operator binary_op(             \
        const migraphx_fp8::hip_f8<T>& lhs, const migraphx_fp8::hip_f8<T>& rhs) \
    {                                                                           \
        return U(static_cast<float>(lhs) binary_op static_cast<float>(rhs));    \
    }

// TODO: these should return floats
MIGRAPHX_FP8_BINARY_OP(*, migraphx_fp8::hip_f8<T>)
MIGRAPHX_FP8_BINARY_OP(-, migraphx_fp8::hip_f8<T>)
MIGRAPHX_FP8_BINARY_OP(/, migraphx_fp8::hip_f8<T>)
MIGRAPHX_FP8_BINARY_OP(+, migraphx_fp8::hip_f8<T>)
// TODO: Comparison ops shouldn't convert to float, maybe need to take care of rounding effects.
MIGRAPHX_FP8_BINARY_OP(==, bool)
MIGRAPHX_FP8_BINARY_OP(>=, bool)
MIGRAPHX_FP8_BINARY_OP(<=, bool)
MIGRAPHX_FP8_BINARY_OP(>, bool)
MIGRAPHX_FP8_BINARY_OP(<, bool)
MIGRAPHX_FP8_BINARY_OP(!=, bool)

template <migraphx_fp8::hip_f8_type T>
inline MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<T> fabs(migraphx_fp8::hip_f8<T> v)
{
    v.data = v.data & 0x7f;
    return v;
}

template <class T>
MIGRAPHX_HIP_HOST_DEVICE constexpr T F8_Max()
{
    return T{0x7F, T::from_bits()};
}

template <class T>
MIGRAPHX_HIP_HOST_DEVICE constexpr T F8_Lowest()
{
    return T{0xFF, T::from_bits()};
}

using fp8e4m3fnuz = hip_f8<migraphx_fp8::hip_f8_type::fp8>;

template <>
class NumericLimits<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>>
{
    public:
    // TODO :figure out epsilon in Hex to make it constexpr
    static constexpr MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>
    epsilon()
    {
        return migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>(
            0x28, migraphx_fp8::hip_f8<>::from_bits());
    }

    static constexpr MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>
    quiet_NaN()
    {
        return migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>(
            MIGRAPHX_FP8_FNUZ ? 0x80 : 0x7F, migraphx_fp8::hip_f8<>::from_bits());
    }

    static constexpr MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>
    max()
    {
        return migraphx_fp8::F8_Max<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>>();
    }

    // TODO figure out Hex value
    static MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8> min()
    {
        return static_cast<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>>(-1.0f) *
               migraphx_fp8::F8_Max<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>>();
    }

    static constexpr MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>
    lowest()
    {
        return migraphx_fp8::F8_Lowest<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>>();
    }

    static constexpr MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>
    infinity()
    {
        return migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>(
            MIGRAPHX_FP8_FNUZ ? 0x80 : 0x7F, migraphx_fp8::hip_f8<>::from_bits());
    }
};

template <>
class NumericLimits<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>>
{
    public:
    static constexpr MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>
    epsilon()
    {
        return migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>(
            0x34, migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>::from_bits());
    }

    static constexpr MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>
    quiet_NaN()
    {
        return migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>(
            MIGRAPHX_FP8_FNUZ ? 0x80 : 0x7d,
            migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>::from_bits());
    }

    static constexpr MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>
    max()
    {
        return static_cast<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>>(
            migraphx_fp8::F8_Max<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>>());
    }
    // TODO figure  out constexpr value
    static MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8> min()
    {
        return static_cast<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>>(float(-1.0f)) *
               migraphx_fp8::F8_Max<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>>();
    }
    static constexpr MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>
    lowest()
    {
        return migraphx_fp8::F8_Lowest<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>>();
    }

    static constexpr MIGRAPHX_HIP_HOST_DEVICE migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>
    infinity()
    {
        return migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>(
            MIGRAPHX_FP8_FNUZ ? 0x80 : 0x7c,
            migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>::from_bits());
    }
};
/*
// Use h/w intrinsic and optimized version when __gfx940__
template <typename T,
          typename Ta,
          bool stochastic_rounding,
          typename std::enable_if<(!(migraphx::is_same<T, Ta>{}) &&
                                   (migraphx::is_same<T, migraphx_f8>{} ||
                                    migraphx::is_same<T, migraphx_bf8>{})),
                                  int>::type = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    // NOTE: we are directly calling cast_to_f8_from_f32 instead of constructor to optimize
    // away one runtime branch
    T val;
    if(migraphx::is_same<T, migraphx_f8>::value)
        val.data = migraphx_f8::cast_to_f8_from_f32<stochastic_rounding>(float(a), rng);
    else
        val.data = migraphx_bf8::cast_to_bf8_from_f32<stochastic_rounding>(float(a), rng);
    return val;
#else  // non gfx940
    return T(float(a),
             stochastic_rounding ? migraphx_fp8::migraphx_hip_f8_rounding_mode::stochastic
                                 : migraphx_fp8::migraphx_hip_f8_rounding_mode::standard,
             rng);
#endif // __gfx940__
}

// NOTE NOTE: The above code is good if we don't consider HIP-GEMM code and only consider
// the quantization However, if we need HIP-GEMM for fall-back, we would need explicit_cast
// handles Tacc=f32 to To=f16/bf16 conversion
template <typename T,
          typename Ta,
          bool stochastic_rounding,
          typename std::enable_if<(!(migraphx::is_same<T, Ta>{}) &&
                                   !(migraphx::is_same<T, migraphx_f8>{} ||
                                     migraphx::is_same<T, migraphx_bf8>{})),
                                  int>::type = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng)
{
    // the return type is not a F8 types, no SR for those types
    // not sure if we have direct conversion, so converting to float first
    // no effect if the input type is float
    return T(float(a));
}
*/
} // namespace migraphx_fp8
// define numeric limits for the new data type
#ifndef MIGRAPHX_JIT_USE_HIPRTC
namespace std {
inline bool isfinite(migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8> x) // NOLINT
{
    return x.is_inf();
}

inline bool isfinite(migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8> x) // NOLINT
{
    return x.is_inf();
}

inline bool isnan(migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8> x) // NOLINT
{
    return x.is_nan();
}

inline bool isnan(migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8> x) // NOLINT
{
    return x.is_nan();
}

template <>
class numeric_limits<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>>
    : public migraphx_fp8::NumericLimits<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::fp8>>
{
};

template <>
class numeric_limits<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>>
    : public migraphx_fp8::NumericLimits<migraphx_fp8::hip_f8<migraphx_fp8::hip_f8_type::bf8>>
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
#endif
// =================================================================================================
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif // MIGRAPHX_GUARD_RTGLIB_FLOAT8_HPP
