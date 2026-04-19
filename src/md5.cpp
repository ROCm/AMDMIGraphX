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
#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <iomanip>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr std::size_t block_size = 64;

// Per-round shift amounts (RFC 1321 section 3.4).
constexpr std::array<u32, 64> shifts = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

// sine-derived constants: floor(2^32 * abs(sin(i + 1))), i = 0..63.
constexpr std::array<u32, 64> sine_table = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
    0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
    0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
    0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
    0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
    0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
    0xeb86d391};

constexpr u32 rotate_left(u32 x, u32 n) { return (x << n) | (x >> (32u - n)); }

void process_block(std::array<u32, 4>& state, const u8* block)
{
    std::array<u32, 16> m{};
    for(std::size_t i = 0; i < 16; ++i)
    {
        const std::size_t base = i * 4;
        m[i]                   = u32{block[base]} | (u32{block[base + 1]} << 8u) |
               (u32{block[base + 2]} << 16u) | (u32{block[base + 3]} << 24u);
    }

    u32 a = state[0];
    u32 b = state[1];
    u32 c = state[2];
    u32 d = state[3];

    for(u32 i = 0; i < 64; ++i)
    {
        u32 f = 0;
        u32 g = 0;
        if(i < 16)
        {
            f = (b & c) | ((~b) & d);
            g = i;
        }
        else if(i < 32)
        {
            f = (d & b) | ((~d) & c);
            g = (5u * i + 1u) % 16u;
        }
        else if(i < 48)
        {
            f = b ^ c ^ d;
            g = (3u * i + 5u) % 16u;
        }
        else
        {
            f = c ^ (b | (~d));
            g = (7u * i) % 16u;
        }

        const u32 temp = d;
        d              = c;
        c              = b;
        b              = b + rotate_left(a + f + sine_table[i] + m[g], shifts[i]);
        a              = temp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

std::string digest_to_hex(const std::array<u32, 4>& state)
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for(u32 word : state)
    {
        for(u32 i = 0; i < 4; ++i)
        {
            oss << std::setw(2) << ((word >> (8u * i)) & 0xffu);
        }
    }
    return oss.str();
}

} // namespace

std::string md5(const std::string_view& str)
{
    std::array<u32, 4> state = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};

    const u64 bit_length            = static_cast<u64>(str.size()) * 8u;
    const std::size_t full_blocks   = str.size() / block_size;
    const std::size_t remainder     = str.size() % block_size;
    const auto* data                = reinterpret_cast<const u8*>(str.data());

    for(std::size_t i = 0; i < full_blocks; ++i)
    {
        process_block(state, data + i * block_size);
    }

    // Padding: one 0x80 byte, zero fill, and an 8-byte little-endian bit length
    // in the last 8 bytes. This consumes one block if remainder < 56, else two.
    std::array<u8, 2 * block_size> tail{};
    std::memcpy(tail.data(), data + full_blocks * block_size, remainder);
    tail[remainder]              = 0x80;
    const std::size_t pad_blocks = (remainder < block_size - 8) ? 1 : 2;
    const std::size_t tail_size  = pad_blocks * block_size;
    for(std::size_t i = 0; i < 8; ++i)
    {
        tail[tail_size - 8 + i] = static_cast<u8>((bit_length >> (8u * i)) & 0xffu);
    }

    for(std::size_t i = 0; i < pad_blocks; ++i)
    {
        process_block(state, tail.data() + i * block_size);
    }

    return digest_to_hex(state);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
