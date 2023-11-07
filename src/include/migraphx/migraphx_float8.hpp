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

#ifndef MIGRAPHX_FLOAT8_HPP
#define MIGRAPHX_FLOAT8_HPP

#ifdef __HIP_PLATFORM_HCC__
#define MIGRAPHX_HIP_HOST_DEVICE __host__ __device__
#else
#define MIGRAPHX_HIP_HOST_DEVICE
#endif
#define MIGRAPHX_HIP_HOST __host__
#define MIGRAPHX_HIP_DEVICE __device__

// We are clipping in down conversion by default
#define MIGRAPHX_F8_DOWNCAST_CLIPPING 1
#ifndef __HIPCC_RTC__
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
#include <migraphx/type_traits.hpp>
#else
#include <migraphx/kernels/type_traits.hpp>
#endif

namespace migraphx_hip_f8_impl {

template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
MIGRAPHX_HIP_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

template <int wm, int we, typename T, bool negative_zero_nan>
MIGRAPHX_HIP_HOST_DEVICE T cast_from_f8(uint8_t x);

} // namespace migraphx_hip_f8_impl

#include "migraphx_hip_f8_impl.hpp"

namespace migraphx_fp8 {

enum class migraphx_hip_f8_rounding_mode
{
    standard,
    stochastic
};

enum class hip_f8_type
{
    bf8 = 0, // s1e5m2
    fp8 = 1  // s1e4m3
};

template <migraphx_fp8::hip_f8_type T = migraphx_fp8::hip_f8_type::fp8>
struct MIGRAPHX_EXPORT migraphx_f8
{
    uint8_t data;
    // default constructor
    MIGRAPHX_HIP_HOST_DEVICE migraphx_f8() = default;

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
            val.i32val = ival;
            i8data     = val.i8val[0]; // little endian
        }
        else                           // RNE CVT
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
                val.i32val = ival;
                i8data     = val.i8val[0];
            }
            return i8data;
        }
    }
#endif // __gfx940__

       // constructor from float
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)

    // NOTE: ON-DEVICE... always optimal bias
    explicit MIGRAPHX_HIP_DEVICE
    migraphx_f8(float v,
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
    explicit MIGRAPHX_HIP_HOST_DEVICE
#endif
    migraphx_f8(float v,
                migraphx_fp8::migraphx_hip_f8_rounding_mode rm =
                    migraphx_fp8::migraphx_hip_f8_rounding_mode::standard,
                uint32_t rng = 0)
    {
        if constexpr(T == migraphx_fp8::hip_f8_type::fp8)
        {
#ifdef MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_hip_f8_impl::
                cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_hip_f8_rounding_mode::stochastic), rng);
#else  // MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_hip_f8_impl::
                cast_to_f8<3, 4, float, true /*negative_zero_nan*/, false /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_hip_f8_rounding_mode::stochastic), rng);
#endif // MIGRAPHX_F8_DOWNCAST_CLIPPING
        }
        else
        {
#ifdef MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_hip_f8_impl::
                cast_to_f8<2, 5, float, true /*negative_zero_nan*/, true /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_hip_f8_rounding_mode::stochastic), rng);
#else  // MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx_hip_f8_impl::
                cast_to_f8<2, 5, float, true /*negative_zero_nan*/, false /*clip*/>(
                    v, (rm == migraphx_fp8::migraphx_hip_f8_rounding_mode::stochastic), rng);
#endif // rocblas_F8_downcast_clipping}
        }
    }

    // Constructor from half
    explicit MIGRAPHX_HIP_HOST_DEVICE
    migraphx_f8(migraphx::half v,
                migraphx_fp8::migraphx_hip_f8_rounding_mode rm =
                    migraphx_fp8::migraphx_hip_f8_rounding_mode::standard,
                uint32_t rng = 0)
        : migraphx_f8((float)v, rm, rng)
    {
    }

    // constructor from int
    explicit MIGRAPHX_HIP_HOST_DEVICE
    migraphx_f8(int v,
                migraphx_fp8::migraphx_hip_f8_rounding_mode rm =
                    migraphx_fp8::migraphx_hip_f8_rounding_mode::standard,
                uint32_t rng = 0)
        : migraphx_f8((float)v, rm, rng)
    {
    }
    // constructor from double
    explicit MIGRAPHX_HIP_HOST_DEVICE
    migraphx_f8(double v,
                migraphx_fp8::migraphx_hip_f8_rounding_mode rm =
                    migraphx_fp8::migraphx_hip_f8_rounding_mode::standard,
                uint32_t rng = 0)
        : migraphx_f8((float)v, rm, rng)
    {
    }

    // convert to float
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    // upcast using device specific intrinsic
    explicit inline MIGRAPHX_HIP_DEVICE operator float() const
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

    explicit inline MIGRAPHX_HIP_HOST operator float() const
#else // non gfx940
    explicit inline MIGRAPHX_HIP_HOST_DEVICE operator float() const
#endif
    {
        if constexpr(T == migraphx_fp8::hip_f8_type::fp8)
        {
            return migraphx_hip_f8_impl::cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(
                data);
        } // else
        return migraphx_hip_f8_impl::cast_from_f8<2, 5, float, true /*negative_zero_nan*/>(data);
    }

    // convert to half
    explicit inline MIGRAPHX_HIP_HOST_DEVICE operator migraphx::half() const
    {
        return migraphx::half(float(*this)); // convert to float, then convert to f16
    }

    // check for zero
    inline MIGRAPHX_HIP_HOST_DEVICE bool is_zero() const { return data == 0x00; }

    // check for nan
    inline MIGRAPHX_HIP_HOST_DEVICE bool is_nan() const { return data == 0x80; }

    // check for inf
    inline MIGRAPHX_HIP_HOST_DEVICE bool is_inf() const { return data == 0x80; }

    // assignment overloading only from the same F8 types
    inline __host__ __device__ migraphx_f8& operator=(const migraphx_f8& a)
    {
        data = a.data;
        return *this;
    }
};

/*
// Special operator overloading
inline std::ostream& operator<<(std::ostream& os, const migraphx_f8& f8) { return os << float(f8); }

inline std::ostream& operator<<(std::ostream& os, const migraphx_bf8& bf8)
{
    return os << float(bf8);
}

// all + operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline __host__ __device__ float operator+(const float fa, migraphx_f8 b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(const float fa, migraphx_bf8 b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(migraphx_f8 a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(migraphx_bf8 a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(migraphx_f8 a, migraphx_bf8 b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ float operator+(migraphx_bf8 a, migraphx_f8 b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ migraphx_f8 operator+(migraphx_f8 a, migraphx_f8 b)
{
    return migraphx_f8(float(a) + float(b));
}

inline __host__ __device__ migraphx_bf8 operator+(migraphx_bf8 a, migraphx_bf8 b)
{
    return migraphx_bf8(float(a) + float(b));
}

inline __host__ __device__ migraphx_f8& operator+=(migraphx_f8& a, migraphx_f8 b)
{
    return a = migraphx_f8(float(a) + float(b));
}

inline __host__ __device__ migraphx_bf8& operator+=(migraphx_bf8& a, migraphx_bf8 b)
{
    return a = migraphx_bf8(float(a) + float(b));
}

// overloading multiplication, always returns float,
inline __host__ __device__ float operator*(migraphx_f8 a, migraphx_f8 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, migraphx_f8 b) { return (a * float(b)); }

inline __host__ __device__ float operator*(migraphx_f8 a, float b) { return (float(a) * b); }

inline __host__ __device__ float operator*(int32_t a, migraphx_f8 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(double a, migraphx_f8 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(migraphx_bf8 a, migraphx_bf8 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, migraphx_bf8 b) { return (a * float(b)); }

inline __host__ __device__ float operator*(migraphx_bf8 a, float b) { return (float(a) * b); }

inline __host__ __device__ float operator*(int32_t a, migraphx_bf8 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(double a, migraphx_bf8 b)
{
    return ((float)a * float(b));
}

// overloading for mixed f8 and bf8 types
inline __host__ __device__ float operator*(migraphx_f8 a, migraphx_bf8 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(migraphx_bf8 a, migraphx_f8 b)
{
    return float(a) * float(b);
}

// all - operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline __host__ __device__ float operator-(const float fa, migraphx_f8 b)
{
    return (fa - float(b));
}

inline __host__ __device__ float operator-(const float fa, migraphx_bf8 b)
{
    return (fa - float(b));
}

inline __host__ __device__ float operator-(migraphx_f8 a, const float fb)
{
    return (float(a) - fb);
}

inline __host__ __device__ float operator-(migraphx_bf8 a, const float fb)
{
    return (float(a) - fb);
}

inline __host__ __device__ float operator-(migraphx_f8 a, migraphx_bf8 b)
{
    return (float(a) - float(b));
}

inline __host__ __device__ float operator-(migraphx_bf8 a, migraphx_f8 b)
{
    return (float(a) - float(b));
}

inline __host__ __device__ migraphx_f8 operator-(migraphx_f8 a, migraphx_f8 b)
{
    return migraphx_f8(float(a) - float(b));
}

inline __host__ __device__ migraphx_bf8 operator-(migraphx_bf8 a, migraphx_bf8 b)
{
    return migraphx_bf8(float(a) - float(b));
}

inline __host__ __device__ migraphx_f8& operator-=(migraphx_f8& a, migraphx_f8 b)
{
    return a = migraphx_f8(float(a) - float(b));
}

inline __host__ __device__ migraphx_bf8& operator-=(migraphx_bf8& a, migraphx_bf8 b)
{
    return a = migraphx_bf8(float(a) - float(b));
}

// overloading division, always returns float,
inline __host__ __device__ float operator/(migraphx_f8 a, migraphx_f8 b)
{
    return float(a) / float(b);
}

inline __host__ __device__ float operator/(float a, migraphx_f8 b) { return (a / float(b)); }

inline __host__ __device__ float operator/(migraphx_f8 a, float b) { return (float(a) / b); }

inline __host__ __device__ float operator/(int32_t a, migraphx_f8 b)
{
    return ((float)a / float(b));
}

inline __host__ __device__ float operator/(double a, migraphx_f8 b)
{
    return ((float)a / float(b));
}

inline __host__ __device__ float operator/(migraphx_bf8 a, migraphx_bf8 b)
{
    return float(a) / float(b);
}

inline __host__ __device__ float operator/(float a, migraphx_bf8 b) { return (a / float(b)); }

inline __host__ __device__ float operator/(migraphx_bf8 a, float b) { return (float(a) / b); }

inline __host__ __device__ float operator/(int32_t a, migraphx_bf8 b)
{
    return ((float)a / float(b));
}

inline __host__ __device__ float operator/(double a, migraphx_bf8 b)
{
    return ((float)a / float(b));
}

// overloading for mixed f8 and bf8 types
inline __host__ __device__ float operator/(migraphx_f8 a, migraphx_bf8 b)
{
    return float(a) / float(b);
}

inline __host__ __device__ float operator/(migraphx_bf8 a, migraphx_f8 b)
{
    return float(a) / float(b);
}

// overloading for compare
inline __host__ __device__ bool operator==(migraphx_f8 a, migraphx_f8 b)
{
    return (a.data == b.data);
}

inline __host__ __device__ bool operator==(migraphx_bf8 a, migraphx_bf8 b)
{
    return (a.data == b.data);
}

inline __host__ __device__ bool operator!=(migraphx_f8 a, migraphx_f8 b)
{
    return (a.data != b.data);
}

inline __host__ __device__ bool operator!=(migraphx_bf8 a, migraphx_bf8 b)
{
    return (a.data != b.data);
}

// ================ Explicit downcasting to support different rounding (RNE, SR)
// =============== NOTE: we going to remove all assignment operator overloading from other
// types and enforce this explicit_downcast function to make any roudning behavior default
// We have to explicitly call this function with SR flag

template <typename T,
          typename Ta,
          bool stochastic_rounding,
          typename std::enable_if<migraphx::is_same<T, Ta>{}, int>::type = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng = 0)
{
    // same type, no conversion
    return a;
}

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
/*
namespace std {
inline migraphx_f8 sin(migraphx_f8 a) { return migraphx_f8(sinf(float(a))); }
inline migraphx_f8 cos(migraphx_f8 a) { return migraphx_f8(cosf(float(a))); }
inline migraphx_bf8 sin(migraphx_bf8 a) { return migraphx_bf8(sinf(float(a))); }
inline migraphx_bf8 cos(migraphx_bf8 a) { return migraphx_bf8(cosf(float(a))); }
__device__ __host__ constexpr migraphx_f8 real(const migraphx_f8& a) { return a; }
__device__ __host__ constexpr migraphx_bf8 real(const migraphx_bf8& a) { return a; }
} // namespace std
*/
// =================================================================================================
#endif // MIGRAPHX_FLOAT8_HPP
