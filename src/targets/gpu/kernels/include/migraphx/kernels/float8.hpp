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

#ifndef MIGRAPHX_GUARD_KERNELS_FLOAT8_HPP
#define MIGRAPHX_GUARD_KERNELS_FLOAT8_HPP
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wfloat-equal"
#pragma clang diagnostic ignored "-Wmacro-redefined"
#pragma clang diagnostic ignored "-Wc++20-extensions"
#endif // __clang__

#define MIGRAPHX_HIP_DEVICE __device__

// We are clipping in down conversion by default
#define MIGRAPHX_F8_DOWNCAST_CLIPPING 1

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/float8_impl.hpp>

namespace migraphx {
namespace fp8 {

enum class rounding_mode
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

template <migraphx::fp8::f8_type T = migraphx::fp8::f8_type::fp8, bool FNUZ = true>
struct float8
{
    uint8_t data;
    // default constructor
    MIGRAPHX_HIP_DEVICE constexpr float8() = default;
    // default copy constructor
    MIGRAPHX_HIP_DEVICE constexpr float8(const float8& y) = default;
    struct from_bits_t
    {
    };
    static constexpr MIGRAPHX_HIP_DEVICE from_bits_t from_bits() { return from_bits_t(); }

    MIGRAPHX_HIP_DEVICE explicit constexpr float8(uint8_t bits, from_bits_t) : data(bits) {}

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
        if constexpr(T == migraphx::fp8::f8_type::fp8)
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
            if constexpr(T == migraphx::fp8::f8_type::fp8)
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
            if constexpr(T == migraphx::fp8::f8_type::fp8)
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
    explicit MIGRAPHX_HIP_DEVICE
    float8(float v,
           migraphx::fp8::rounding_mode rm = migraphx::fp8::rounding_mode::standard,
           uint32_t rng                    = 0)
    {
        // runtime branch, use cast_to_f8_from_f32 if want to avoid it
        if(rm == migraphx::fp8::rounding_mode::stochastic)
            data = cast_to_f8_from_f32<true>(v, rng);
        else
            data = cast_to_f8_from_f32<false>(v);
    }
#else
    // DEVICE for non-gfx940 using s/w simulation
    explicit constexpr MIGRAPHX_HIP_DEVICE
#endif
    float8(float v,
           migraphx::fp8::rounding_mode rm = migraphx::fp8::rounding_mode::standard,
           uint32_t rng                    = 0)
    {
        if constexpr(T == migraphx::fp8::f8_type::fp8)
        {
#ifdef MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx::fp8::impl::
                cast_to_f8<3, 4, float, FNUZ /*negative_zero_nan*/, true /*clip*/>(
                    v, (rm == migraphx::fp8::rounding_mode::stochastic), rng);
#else  // MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx::fp8::impl::
                cast_to_f8<3, 4, float, FNUZ /*negative_zero_nan*/, false /*clip*/>(
                    v, (rm == migraphx::fp8::rounding_mode::stochastic), rng);
#endif // MIGRAPHX_F8_DOWNCAST_CLIPPING
        }
        else
        {
#ifdef MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx::fp8::impl::
                cast_to_f8<2, 5, float, FNUZ /*negative_zero_nan*/, true /*clip*/>(
                    v, (rm == migraphx::fp8::rounding_mode::stochastic), rng);
#else  // MIGRAPHX_F8_DOWNCAST_CLIPPING
            data = migraphx::fp8::impl::
                cast_to_f8<2, 5, float, FNUZ /*negative_zero_nan*/, false /*clip*/>(
                    v, (rm == migraphx::fp8::rounding_mode::stochastic), rng);
#endif // rocblas_F8_downcast_clipping}
        }
    }

    // convert to float
// #if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#if 0 // need constexpr operator(). This version can't be constexpr
    // upcast using device specific intrinsic
    inline MIGRAPHX_HIP_DEVICE operator float() const
    {
        float fval;
        uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
        if constexpr(T == migraphx::fp8::f8_type::fp8)
        {
            asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
        }
        else
        {
            asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
        }

        return fval;
    }

#else // non gfx940
    inline constexpr MIGRAPHX_HIP_DEVICE operator float() const
#endif
    {
        if constexpr(T == migraphx::fp8::f8_type::fp8)
        {
            return migraphx::fp8::impl::cast_from_f8<3, 4, float, FNUZ /*negative_zero_nan*/>(data);
        } // else
        return migraphx::fp8::impl::cast_from_f8<2, 5, float, FNUZ /*negative_zero_nan*/>(data);
    }

    // check for zero
    inline MIGRAPHX_HIP_DEVICE constexpr bool is_zero() const
    {
        if constexpr(FNUZ)
        {
            return data == 0x00;
        }
        else
        {
            return (data == 0x00) || (data == 0x80);
        }
    }

    // check for nan
    inline MIGRAPHX_HIP_DEVICE constexpr bool is_nan() const
    {
        if constexpr(FNUZ)
        {
            return data == 0x80;
        }
        else
        {
            if(T == migraphx::fp8::f8_type::bf8)
            {
                return (data == 0x7D) or (data == 0x7E) or (data == 0x7F) or (data == 0xFD) or
                       (data == 0xFE) or (data == 0xFF);
            }
            else
            {
                return (data == 0x7F) or (data == 0xFF);
            }
        }
    }

    // check for inf
    inline MIGRAPHX_HIP_DEVICE constexpr bool is_inf() const
    {
        if constexpr(FNUZ)
        {
            return data == 0x80;
        }
        else
        {
            if(T == migraphx::fp8::f8_type::bf8)
            {
                return (data == 0x7C) or (data == 0xFC);
            }
            else
            {
                // no infinities in e4m3fn, represent them as NaNs
                return (data == 0x7F) or (data == 0xFF);
            }
        }
    }

#define MIGRAPHX_FP8_UNARY_OP(unary_op, binary_op)                                    \
    constexpr float8& MIGRAPHX_HIP_DEVICE operator unary_op(const float8& rhs)        \
    {                                                                                 \
        const auto tmp = static_cast<float>(*this) binary_op static_cast<float>(rhs); \
        *this          = static_cast<float8>(tmp);                                    \
        return *this;                                                                 \
    }                                                                                 \
    constexpr float8& MIGRAPHX_HIP_DEVICE operator unary_op(const float& rhs)         \
    {                                                                                 \
        const auto tmp = static_cast<float>(*this) binary_op static_cast<float>(rhs); \
        *this          = static_cast<float8>(tmp);                                    \
        return *this;                                                                 \
    }

    MIGRAPHX_FP8_UNARY_OP(*=, *)
    MIGRAPHX_FP8_UNARY_OP(-=, -)
    MIGRAPHX_FP8_UNARY_OP(+=, +)
    MIGRAPHX_FP8_UNARY_OP(/=, /)

    inline MIGRAPHX_HIP_DEVICE constexpr float8& operator=(const float8& rhs) = default;
    inline MIGRAPHX_HIP_DEVICE constexpr float8& operator=(float8&& rhs)      = default;

    inline MIGRAPHX_HIP_DEVICE constexpr bool operator==(const float8& rhs) const
    {
        if(rhs.is_nan() or rhs.is_inf() or this->is_nan() or this->is_inf())
            return false;
        else if((rhs.is_zero() and this->is_zero()) or (this->data == rhs.data))
            return true;
        return false;
    }

    inline MIGRAPHX_HIP_DEVICE constexpr bool operator<(const float8& rhs) const
    {
        const auto we   = static_cast<float>(*this);
        const auto them = static_cast<float>(rhs);
        return we < them;
    }

    inline MIGRAPHX_HIP_DEVICE constexpr bool operator>(const float8& rhs) const
    {
        const auto we   = static_cast<float>(*this);
        const auto them = static_cast<float>(rhs);
        return we > them;
    }
};

// NOLINTNEXTLINE
#define MIGRAPHX_FP8_BINARY_OP(binary_op, U)                                                       \
    template <migraphx::fp8::f8_type T>                                                            \
    inline constexpr U MIGRAPHX_HIP_DEVICE operator binary_op(const migraphx::fp8::float8<T>& lhs, \
                                                              const migraphx::fp8::float8<T>& rhs) \
    {                                                                                              \
        return U(static_cast<float>(lhs) binary_op static_cast<float>(rhs));                       \
    }

// TODO: these should return floats
MIGRAPHX_FP8_BINARY_OP(*, migraphx::fp8::float8<T>)
MIGRAPHX_FP8_BINARY_OP(-, migraphx::fp8::float8<T>)
MIGRAPHX_FP8_BINARY_OP(/, migraphx::fp8::float8<T>)
MIGRAPHX_FP8_BINARY_OP(+, migraphx::fp8::float8<T>)
// TODO: Comparison ops shouldn't convert to float, maybe need to take care of rounding effects.
MIGRAPHX_FP8_BINARY_OP(==, bool)
MIGRAPHX_FP8_BINARY_OP(>=, bool)
MIGRAPHX_FP8_BINARY_OP(<=, bool)
MIGRAPHX_FP8_BINARY_OP(>, bool)
MIGRAPHX_FP8_BINARY_OP(<, bool)
MIGRAPHX_FP8_BINARY_OP(!=, bool)

// https://onnx.ai/onnx/technical/float8.html
using fp8e4m3fn   = float8<migraphx::fp8::f8_type::fp8, false>;
using fp8e5m2     = float8<migraphx::fp8::f8_type::bf8, false>;
using fp8e4m3fnuz = float8<migraphx::fp8::f8_type::fp8, true>;
using fp8e5m2fnuz = float8<migraphx::fp8::f8_type::bf8, true>;
;

inline MIGRAPHX_HIP_DEVICE fp8e4m3fnuz fabs(fp8e4m3fnuz v)
{
    v.data = v.data & 0x7f;
    return v;
}

inline MIGRAPHX_HIP_DEVICE fp8e4m3fn fabs(fp8e4m3fn v)
{
    v.data = v.data & 0x7f;
    return v;
}

inline MIGRAPHX_HIP_DEVICE fp8e5m2fnuz fabs(fp8e5m2fnuz v)
{
    v.data = v.data & 0x7f;
    return v;
}

inline MIGRAPHX_HIP_DEVICE fp8e5m2 fabs(fp8e5m2 v)
{
    v.data = v.data & 0x7f;
    return v;
}

template <>
class numeric_limits<fp8e4m3fnuz>
{
    public:
    static constexpr bool has_infinity = false;
    static constexpr MIGRAPHX_HIP_DEVICE fp8e4m3fnuz epsilon()
    {
        return fp8e4m3fnuz(0x28, fp8e4m3fnuz::from_bits());
    }
    // NOLINTNEXTLINE
    static constexpr MIGRAPHX_HIP_DEVICE fp8e4m3fnuz quiet_NaN()
    {
        return fp8e4m3fnuz(0x80, fp8e4m3fnuz::from_bits());
    }

    static constexpr MIGRAPHX_HIP_DEVICE fp8e4m3fnuz max()
    {
        return fp8e4m3fnuz(0x7F, fp8e4m3fnuz::from_bits());
    }
    // this is min value that is not DeNorm. DeNorm min is 0x01
    static constexpr MIGRAPHX_HIP_DEVICE fp8e4m3fnuz min()
    {
        return fp8e4m3fnuz(0x08, fp8e4m3fnuz::from_bits());
    }

    static constexpr MIGRAPHX_HIP_DEVICE fp8e4m3fnuz lowest()
    {
        return fp8e4m3fnuz(0xFF, fp8e4m3fnuz::from_bits());
    }
};

template <>
class numeric_limits<fp8e4m3fn>
{
    public:
    static constexpr bool has_infinity = false;
    static constexpr MIGRAPHX_HIP_DEVICE fp8e4m3fn epsilon()
    {
        return fp8e4m3fn(0x20, fp8e4m3fn::from_bits());
    }
    // NOLINTNEXTLINE
    static constexpr MIGRAPHX_HIP_DEVICE fp8e4m3fn quiet_NaN()
    {
        return fp8e4m3fn(0x7F, fp8e4m3fn::from_bits());
    }

    static constexpr MIGRAPHX_HIP_DEVICE fp8e4m3fn max()
    {
        return fp8e4m3fn(0x7E, fp8e4m3fn::from_bits());
    }
    // this is min value that is not DeNorm. DeNorm min is 0x01
    static constexpr MIGRAPHX_HIP_DEVICE fp8e4m3fn min()
    {
        return fp8e4m3fn(0x08, fp8e4m3fn::from_bits());
    }

    static constexpr MIGRAPHX_HIP_DEVICE fp8e4m3fn lowest()
    {
        return fp8e4m3fn(0xFE, fp8e4m3fn::from_bits());
    }
};

template <>
class numeric_limits<fp8e5m2fnuz>
{
    public:
    static constexpr bool has_infinity = false;
    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2fnuz epsilon()
    {
        return fp8e5m2fnuz(0x34, fp8e5m2fnuz::from_bits());
    }

    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2fnuz quiet_NaN() // NOLINT
    {
        return fp8e5m2fnuz(0x80, fp8e5m2fnuz::from_bits());
    }

    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2fnuz max()
    {
        return fp8e5m2fnuz(0x7F, fp8e5m2fnuz::from_bits());
    }
    // this is min value that is not DeNorm. DeNorm min is 0x01. I am not sure if we want to make
    // this distinction. For the floating points we would end up using lowest most of the times.
    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2fnuz min()
    {
        return fp8e5m2fnuz(0x4, fp8e5m2fnuz::from_bits());
    }

    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2fnuz lowest()
    {
        return fp8e5m2fnuz(0xFF, fp8e5m2fnuz::from_bits());
    }
};

template <>
class numeric_limits<fp8e5m2>
{
    public:
    static constexpr bool has_infinity = true;
    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2 epsilon()
    {
        return fp8e5m2(0x34, fp8e5m2::from_bits());
    }
    // 7D, 7E, 7F are positive NaNs and FD, FE, FF are negative NaNs
    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2 quiet_NaN()
    {
        return fp8e5m2(0xFF, fp8e5m2::from_bits());
    } // NOLINT

    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2 max()
    {
        return fp8e5m2(0x7B, fp8e5m2::from_bits());
    }
    // this is min value that is not DeNorm. DeNorm min is 0x01. I am not sure if we want to make
    // this distinction. For the floating points we would end up using lowest most of the times.
    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2 min()
    {
        return fp8e5m2(0x4, fp8e5m2::from_bits());
    }

    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2 lowest()
    {
        return fp8e5m2(0xFB, fp8e5m2::from_bits());
    }
    // 7C and FC both are infinity
    static constexpr MIGRAPHX_HIP_DEVICE fp8e5m2 infinity()
    {
        return fp8e5m2(0x7C, fp8e5m2::from_bits());
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
             stochastic_rounding ? migraphx::fp8::rounding_mode::stochastic
                                 : migraphx::fp8::rounding_mode::standard,
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
} // namespace fp8
} // namespace migraphx
// =================================================================================================
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif // MIGRAPHX_GUARD_KERNELS_FLOAT8_HPP
