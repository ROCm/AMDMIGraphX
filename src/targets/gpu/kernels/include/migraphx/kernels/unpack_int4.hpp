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
#ifndef MIGRAPHX_GUARD_KERNELS_UNPACK_INT4_HPP
#define MIGRAPHX_GUARD_KERNELS_UNPACK_INT4_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/type_traits.hpp>
#include <migraphx/kernels/bit_cast.hpp>
#include <migraphx/kernels/debug.hpp>

namespace migraphx {

// Value-level unpack used by fused kernels: each packed byte expands into
// two adjacent lanes, low nibble first
template <class T, index_int N>
constexpr vec<T, N * 2> unpack_int4(vec<T, N> x)
{
    vec<T, N * 2> result{};
    for(index_int i = 0; i < N; i++)
    {
        if constexpr(is_unsigned<T>{})
        {
            result[2 * i]     = x[i] & 0xfu;
            result[2 * i + 1] = x[i] >> 4u;
        }
        else
        {
            // NOLINTNEXTLINE(hicpp-signed-bitwise)
            result[2 * i] = static_cast<int8_t>(static_cast<uint8_t>(x[i]) << 4u) >> 4;
            // NOLINTNEXTLINE(hicpp-signed-bitwise)
            result[2 * i + 1] = x[i] >> 4;
        }
    }
    return result;
}

template <class T, MIGRAPHX_REQUIRES(is_integral<T>{})>
constexpr vec<T, 2> unpack_int4(T x)
{
    return unpack_int4(vec<T, 1>{x});
}

namespace detail {

// The packed bytes as 32-bit words of eight nibbles. Signed nibbles get
// their sign bit flipped so they read as unsigned offsets: value = nibble - 8
template <class T, index_int N>
__device__ vec<uint32_t, N / 4> unpack_int4_words(vec<T, N> x)
{
    auto words = __builtin_bit_cast(vec<uint32_t, N / 4>, x);
    if constexpr(is_unsigned<T>{})
        return words;
    else
        return words ^ 0x88888888u;
}

// Byte j of x in bytes 0 and 2 of the result: [b, 0, b, 0]. The perm
// selector picks byte j of x (j < 4) and 0x0c selects a zero byte
__device__ inline uint32_t unpack_int4_byte_pair(uint32_t x, uint32_t j)
{
    MIGRAPHX_ASSERT(j < 4);
    return __builtin_amdgcn_perm(x, x, 0x0c000c00u | (j << 16u) | j);
}

// Both nibbles of byte j as fp16 without a convert: or-ed into the mantissa
// of 1024.0 (0x6400) the low nibble reads as 1024 + lo and the high nibble
// as 1024 + 16 * hi, which one packed fma turns into lo + bias and hi + bias
__device__ inline vec<half, 2> unpack_int4_half_pair(uint32_t word, uint32_t j, vec<half, 2> offset)
{
    auto p                   = unpack_int4_byte_pair(word, j);
    auto m                   = bit_cast<vec<half, 2>>((p & 0x00f0000fu) | 0x64006400u);
    const vec<half, 2> scale = {half(1), half(1.0 / 16)};
    return __builtin_elementwise_fma(m, scale, offset);
}

template <class U, index_int N>
__device__ vec<half, N * 2> unpack_int4_as_half(vec<U, N> x, half bias)
{
    static_assert(N % 4 == 0, "Packed bytes must form whole words");
    auto words = unpack_int4_words(x);
    if constexpr(not is_unsigned<U>{})
        bias = bias - half(8);
    const vec<half, 2> offset = {half(-1024) + bias, half(-64) + bias};
    vec<half, N * 2> result{};
    for(index_int i = 0; i < N / 4; i++)
    {
        for(index_int j = 0; j < 4; j++)
        {
            auto v                    = unpack_int4_half_pair(words[i], j, offset);
            result[8 * i + 2 * j]     = v[0];
            result[8 * i + 2 * j + 1] = v[1];
        }
    }
    return result;
}

// The nibble at bit S or-ed into the mantissa of 2^23 (0x4b000000) reads as
// 2^23 + nibble * 2^S; the fma scales it back and removes the offset
template <index_int S>
__device__ float unpack_int4_float_nibble(uint32_t word, float bias)
{
    static_assert(S + 4 <= 23, "Nibble must fit in the float mantissa");
    auto m                = bit_cast<float>((word & (0xfu << S)) | 0x4b000000u);
    constexpr float scale = 1.0f / (1u << S);
    return __builtin_elementwise_fma(m, scale, bias - float(1u << (23 - S)));
}

template <class U, index_int N>
__device__ vec<float, N * 2> unpack_int4_as_float(vec<U, N> x, float bias)
{
    static_assert(N % 4 == 0, "Packed bytes must form whole words");
    auto words = unpack_int4_words(x);
    if constexpr(not is_unsigned<U>{})
        bias = bias - 8;
    vec<float, N * 2> result{};
    for(index_int i = 0; i < N / 4; i++)
    {
        auto q = words[i];
        // Nibbles 5-7 reach past the 23-bit mantissa, so shift them down first
        auto q12          = q >> 12u;
        result[8 * i]     = unpack_int4_float_nibble<0>(q, bias);
        result[8 * i + 1] = unpack_int4_float_nibble<4>(q, bias);
        result[8 * i + 2] = unpack_int4_float_nibble<8>(q, bias);
        result[8 * i + 3] = unpack_int4_float_nibble<12>(q, bias);
        result[8 * i + 4] = unpack_int4_float_nibble<16>(q, bias);
        result[8 * i + 5] = unpack_int4_float_nibble<8>(q12, bias);
        result[8 * i + 6] = unpack_int4_float_nibble<12>(q12, bias);
        result[8 * i + 7] = unpack_int4_float_nibble<16>(q12, bias);
    }
    return result;
}

} // namespace detail

// Unpack straight to a float type with an integer bias (the negated zero
// point) folded in: value = nibble + bias. Whole words of fp16 or fp32 skip
// the integer unpack and convert by building the floats from the nibbles
template <class T, class U, index_int N>
__device__ vec<T, N * 2> unpack_int4_as(vec<U, N> x, T bias)
{
    if constexpr(N % 4 == 0 and is_same<T, half>{})
        return detail::unpack_int4_as_half(x, bias);
    else if constexpr(N % 4 == 0 and is_same<T, float>{})
        return detail::unpack_int4_as_float(x, bias);
    else
    {
        vec<T, N * 2> result = implicit_conversion(unpack_int4(x));
        return result + bias;
    }
}

template <class T, class U, MIGRAPHX_REQUIRES(is_integral<U>{})>
__device__ vec<T, 2> unpack_int4_as(U x, T bias)
{
    return unpack_int4_as<T>(vec<U, 1>{x}, bias);
}

template <int Axis, class Output, class Input>
__device__ void unpack_int4(Output output, Input input)
{
    const auto input_shape = input.get_shape();

    make_index().global_stride(input_shape.elements(), [&](auto i) {
        auto idx = input_shape.multi(i);
        idx[Axis] *= 2;
        auto values = unpack_int4(input[i]);
        output[idx] = values[0];
        idx[Axis] += 1;
        output[idx] = values[1];
    });
}

} // namespace migraphx
#endif
