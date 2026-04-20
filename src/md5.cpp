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
#include <migraphx/md5.hpp>
#include <migraphx/bit_cast.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

constexpr std::size_t block_size = 64;

// Per-round shift amounts (RFC 1321 section 3.4).
constexpr std::array<std::uint32_t, 64> shifts = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                                        5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                                        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                                        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

// Sine-derived constants: floor(2^32 * abs(sin(i + 1))), i = 0..63.
constexpr std::array<std::uint32_t, 64> sine_table = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

constexpr std::uint32_t rotate_left(std::uint32_t x, std::uint32_t n) { return (x << n) | (x >> (32u - n)); }

template <class It>
constexpr std::uint32_t load_le32(It p)
{
    return std::uint32_t{p[0]} | (std::uint32_t{p[1]} << 8u) | (std::uint32_t{p[2]} << 16u) | (std::uint32_t{p[3]} << 24u);
}

std::array<std::uint32_t, 4> process_block(std::array<std::uint32_t, 4> state,
                                 const std::array<std::uint8_t, block_size>& block)
{
    std::array<std::uint32_t, 16> m{};
    const auto word_indices = range(m.size());
    std::transform(word_indices.begin(), word_indices.end(), m.begin(), [&](std::ptrdiff_t i) {
        return load_le32(block.begin() + (i * 4));
    });

    // v holds the round state; after each step v[0] is overwritten with the new
    // 'b' and std::rotate shifts the labels so that (a, b, c, d) tracks the
    // canonical MD5 register carousel (a <- d, b <- new_b, c <- b, d <- c).
    std::array<std::uint32_t, 4> v = state;
    auto& [a, b, c, d]   = v;

    for(std::uint32_t i = 0; i < 64; ++i)
    {
        std::array<std::uint32_t, 2> fg{};
        if(i < 16)
        {
            fg = {(b & c) | ((~b) & d), i};
        }
        else if(i < 32)
        {
            fg = {(d & b) | ((~d) & c), (5u * i + 1u) % 16u};
        }
        else if(i < 48)
        {
            fg = {b ^ c ^ d, (3u * i + 5u) % 16u};
        }
        else
        {
            fg = {c ^ (b | (~d)), (7u * i) % 16u};
        }

        a = b + rotate_left(a + fg[0] + sine_table[i] + m[fg[1]], shifts[i]);
        std::rotate(v.begin(), v.end() - 1, v.end());
    }

    return {state[0] + a, state[1] + b, state[2] + c, state[3] + d};
}

std::uint8_t to_uint8(std::int8_t c) { return bit_cast<std::uint8_t>(c); }

} // namespace

std::string md5(const std::string_view& str)
{
    std::array<std::uint32_t, 4> state = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};

    const std::size_t full_blocks = str.size() / block_size;
    const std::size_t remainder   = str.size() % block_size;

    std::array<std::uint8_t, block_size> block{};
    for(std::size_t i = 0; i < full_blocks; ++i)
    {
        const auto chunk_begin = str.begin() + (i * block_size);
        std::transform(chunk_begin, chunk_begin + block_size, block.begin(), &to_uint8);
        state = process_block(state, block);
    }

    // Final block(s): remaining bytes, a 0x80 terminator, zero fill, and the
    // message bit length in the last 8 bytes (little-endian). Two blocks are
    // needed when the bit-length field no longer fits in the current block.
    std::array<std::array<std::uint8_t, block_size>, 2> tail{};
    const auto tail_src_begin = str.begin() + (full_blocks * block_size);
    std::transform(tail_src_begin, str.end(), tail[0].begin(), &to_uint8);
    tail[0][remainder] = 0x80;

    const bool need_two    = (remainder >= block_size - 8);
    const std::uint64_t bit_length   = std::uint64_t{str.size()} * 8u;
    auto& last             = need_two ? tail[1] : tail[0];
    const auto bit_indices = range(8);
    transform_partial_sum(
        bit_indices.begin(),
        bit_indices.end(),
        last.end() - 8,
        [](std::uint64_t acc, std::uint64_t) { return acc >> 8u; },
        [&](auto) { return bit_length; });

    state = process_block(state, tail[0]);
    if(need_two)
    {
        state = process_block(state, tail[1]);
    }

    return to_hex_string(state, true);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
