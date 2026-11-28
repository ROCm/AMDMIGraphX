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
#include <migraphx/compression.hpp>
#include <migraphx/errors.hpp>
#include <test.hpp>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

/// Bytes with no redundancy for zstd to remove. Deterministic, so a failure reproduces.
static std::vector<char> random_bytes(std::size_t n)
{
    std::vector<char> result(n);
    std::mt19937 gen{3}; // NOLINT(cert-msc51-cpp)
    std::generate(result.begin(), result.end(), [&] { return static_cast<char>(gen()); });
    return result;
}

TEST_CASE(round_trip_empty)
{
    std::vector<char> data;
    EXPECT(migraphx::zstd_decompress(migraphx::zstd_compress(data)).empty());
}

TEST_CASE(round_trip_one_byte)
{
    std::vector<char> data = {'x'};
    EXPECT(migraphx::zstd_decompress(migraphx::zstd_compress(data)) == data);
}

TEST_CASE(round_trip_shrinks_redundant_data)
{
    // Every byte value, over and over.
    std::vector<char> data(1u << 16u);
    std::iota(data.begin(), data.end(), char{0});
    auto frame = migraphx::zstd_compress(data);
    EXPECT(frame.size() < data.size() / 2);
    EXPECT(migraphx::zstd_decompress(frame) == data);
}

TEST_CASE(round_trip_incompressible_data)
{
    auto data = random_bytes(1u << 20u);
    EXPECT(migraphx::zstd_decompress(migraphx::zstd_compress(data)) == data);
}

TEST_CASE(round_trip_at_each_level)
{
    // Every byte value, over and over.
    std::vector<char> data(1u << 16u);
    std::iota(data.begin(), data.end(), char{0});
    auto fast    = migraphx::zstd_compress(data, -5);
    auto normal  = migraphx::zstd_compress(data, 3);
    auto highest = migraphx::zstd_compress(data, 19);
    EXPECT(migraphx::zstd_decompress(fast) == data);
    EXPECT(migraphx::zstd_decompress(normal) == data);
    EXPECT(migraphx::zstd_decompress(highest) == data);
    // The highest level does not do worse than the fastest
    EXPECT(highest.size() <= fast.size());
    // Level 3 is the level used when none is given
    EXPECT(migraphx::zstd_compress(data) == normal);
}

TEST_CASE(unknown_level_throws)
{
    std::vector<char> data = {'x'};
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::zstd_compress(data, 1000); },
                                             "compression level"));
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::zstd_compress(data, -100000000); },
                                             "compression level"));
}

TEST_CASE(not_a_frame_throws)
{
    const std::vector<char> data(4, '\xc1');
    EXPECT(test::throws([&] { migraphx::zstd_decompress(data); }));
}

TEST_CASE(truncated_frame_throws)
{
    auto frame = migraphx::zstd_compress(random_bytes(4096));
    frame.pop_back();
    EXPECT(test::throws([&] { migraphx::zstd_decompress(frame); }));
    frame.resize(3);
    EXPECT(test::throws([&] { migraphx::zstd_decompress(frame); }));
}

TEST_CASE(trailing_data_throws)
{
    auto frame = migraphx::zstd_compress(random_bytes(4096));
    auto twice = frame;
    twice.insert(twice.end(), frame.begin(), frame.end());
    EXPECT(test::throws([&] { migraphx::zstd_decompress(twice); }));
    frame.push_back(0);
    EXPECT(test::throws([&] { migraphx::zstd_decompress(frame); }));
}

// Incompressible content is stored as is, so a flipped byte still decodes and only the checksum
// can catch it.
TEST_CASE(damaged_content_fails_the_checksum)
{
    auto frame = migraphx::zstd_compress(random_bytes(4096));
    frame[frame.size() / 2] ^= '\x10';
    EXPECT(test::throws([&] { migraphx::zstd_decompress(frame); }));
}

TEST_CASE(content_over_the_limit_throws)
{
    auto data  = random_bytes(4096);
    auto frame = migraphx::zstd_compress(data);
    EXPECT(test::throws([&] { migraphx::zstd_decompress(frame, data.size() - 1); }));
    EXPECT(migraphx::zstd_decompress(frame, data.size()) == data);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
