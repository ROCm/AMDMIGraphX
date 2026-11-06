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
#include <migraphx/bit_cast.hpp>
#include <migraphx/charconv.hpp>
#include <test.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>

TEST_CASE(to_chars_signed_integer)
{
    std::array<char, 32> buffer{};
    auto result =
        migraphx::to_chars(buffer.data(), buffer.data() + buffer.size(), std::int64_t{-42});

    EXPECT(result.ec == std::errc{});
    EXPECT(std::string(buffer.data(), result.ptr) == "-42");
}

TEST_CASE(to_chars_floating_point)
{
    auto convert = [](double value) {
        std::array<char, 32> buffer{};
        auto result = migraphx::to_chars(buffer.data(), buffer.data() + buffer.size(), value);

        EXPECT(result.ec == std::errc{});
        return std::string(buffer.data(), result.ptr);
    };
    auto round_trips = [&](double value) {
        return migraphx::bit_cast<std::uint64_t>(std::stod(convert(value))) ==
               migraphx::bit_cast<std::uint64_t>(value);
    };

    EXPECT(convert(3.14) == "3.14");
    EXPECT(round_trips(0.1));
    EXPECT(round_trips(1.0 / 3.0));
    EXPECT(round_trips(1e-9));
    EXPECT(round_trips(std::numeric_limits<double>::max()));
}

TEST_CASE(to_chars_buffer_too_small)
{
    std::array<char, 2> buffer{};
    auto result = migraphx::to_chars(buffer.data(), buffer.data() + buffer.size(), 123);

    EXPECT(result.ptr == buffer.data() + buffer.size());
    EXPECT(result.ec == std::errc::value_too_large);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
