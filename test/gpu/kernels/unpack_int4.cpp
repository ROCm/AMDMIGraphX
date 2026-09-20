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
#include <migraphx/kernels/unpack_int4.hpp>
#include <migraphx/kernels/test.hpp>
#include <migraphx/kernels/float_equal.hpp>

// The int4 value of a nibble, sign-extended for signed packed types
template <class U>
__device__ int nibble_value(unsigned nibble)
{
    if constexpr(migraphx::is_unsigned<U>{})
        return static_cast<int>(nibble);
    else
        return static_cast<int>(nibble) - (nibble < 8 ? 0 : 16);
}

// The unpack must give nibble + bias, low nibble first, over every byte
// value, for both biases the reduce codegen emits
template <class T, class U, migraphx::index_int N>
__device__ bool unpack_int4_as_matches(T bias)
{
    for(unsigned start = 0; start < 256; start += N)
    {
        migraphx::vec<U, N> x{};
        for(migraphx::index_int i = 0; i < N; i++)
            x[i] = static_cast<U>(start + i);
        auto result = migraphx::unpack_int4_as<T>(x, bias);
        for(migraphx::index_int i = 0; i < N; i++)
        {
            unsigned byte = start + i;
            T lo          = T(nibble_value<U>(byte & 0xfu)) + bias;
            T hi          = T(nibble_value<U>(byte >> 4u)) + bias;
            if(not migraphx::float_equal(result[2 * i], lo))
                return false;
            if(not migraphx::float_equal(result[2 * i + 1], hi))
                return false;
        }
    }
    return true;
}

TEST_CASE(unpack_int4_as_half_uint8)
{
    EXPECT(unpack_int4_as_matches<migraphx::half, uint8_t, 16>(migraphx::half(0)));
    EXPECT(unpack_int4_as_matches<migraphx::half, uint8_t, 16>(migraphx::half(-8)));
}

TEST_CASE(unpack_int4_as_half_int8)
{
    EXPECT(unpack_int4_as_matches<migraphx::half, int8_t, 16>(migraphx::half(0)));
    EXPECT(unpack_int4_as_matches<migraphx::half, int8_t, 16>(migraphx::half(-8)));
}

TEST_CASE(unpack_int4_as_float_uint8)
{
    EXPECT(unpack_int4_as_matches<float, uint8_t, 16>(0.0f));
    EXPECT(unpack_int4_as_matches<float, uint8_t, 16>(-8.0f));
}

TEST_CASE(unpack_int4_as_float_int8)
{
    EXPECT(unpack_int4_as_matches<float, int8_t, 16>(0.0f));
    EXPECT(unpack_int4_as_matches<float, int8_t, 16>(-8.0f));
}

// A single word and the generic fallback for fewer bytes than a word
TEST_CASE(unpack_int4_as_word)
{
    EXPECT(unpack_int4_as_matches<migraphx::half, uint8_t, 4>(migraphx::half(-8)));
    EXPECT(unpack_int4_as_matches<float, int8_t, 4>(-8.0f));
}

TEST_CASE(unpack_int4_as_generic)
{
    EXPECT(unpack_int4_as_matches<migraphx::half, uint8_t, 2>(migraphx::half(-8)));
    EXPECT(unpack_int4_as_matches<float, int8_t, 2>(-8.0f));
    EXPECT(unpack_int4_as_matches<migraphx::bf16, uint8_t, 16>(migraphx::bf16(-8)));
}

// Spot-check the values: 0x7a holds low nibble 10 and high nibble 7
TEST_CASE(unpack_int4_as_values)
{
    auto u = migraphx::unpack_int4_as<migraphx::half>(uint8_t{0x7a}, migraphx::half(-8));
    EXPECT(migraphx::float_equal(u[0], migraphx::half(2)));
    EXPECT(migraphx::float_equal(u[1], migraphx::half(-1)));
    // The signed low nibble 10 is -6
    auto s = migraphx::unpack_int4_as<float>(int8_t{0x7a}, 0.0f);
    EXPECT(migraphx::float_equal(s[0], -6.0f));
    EXPECT(migraphx::float_equal(s[1], 7.0f));
    migraphx::vec<uint8_t, 16> w{};
    w[5]   = 0x7a;
    auto v = migraphx::unpack_int4_as<float>(w, -8.0f);
    EXPECT(migraphx::float_equal(v[10], 2.0f));
    EXPECT(migraphx::float_equal(v[11], -1.0f));
    EXPECT(migraphx::float_equal(v[0], -8.0f));
    EXPECT(migraphx::float_equal(v[31], -8.0f));
}
