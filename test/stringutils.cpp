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
#include <migraphx/stringutils.hpp>
#include <test.hpp>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

TEST_CASE(interpolate_string_simple1)
{
    std::string input = "Hello ${w}!";
    auto s            = migraphx::interpolate_string(input, {{"w", "world"}});
    EXPECT(s == "Hello world!");
}

TEST_CASE(interpolate_string_simple2)
{
    std::string input = "${hello}";
    auto s            = migraphx::interpolate_string(input, {{"hello", "bye"}});
    EXPECT(s == "bye");
}

TEST_CASE(interpolate_string_unbalanced)
{
    std::string input = "${hello";
    EXPECT(test::throws([&] { migraphx::interpolate_string(input, {{"hello", "bye"}}); }));
}

TEST_CASE(interpolate_string_extra_space)
{
    std::string input = "${  hello  }";
    auto s            = migraphx::interpolate_string(input, {{"hello", "bye"}});
    EXPECT(s == "bye");
}

TEST_CASE(interpolate_string_multiple)
{
    std::string input = "${h} ${w}!";
    auto s            = migraphx::interpolate_string(input, {{"w", "world"}, {"h", "Hello"}});
    EXPECT(s == "Hello world!");
}

TEST_CASE(interpolate_string_next)
{
    std::string input = "${hh}${ww}!";
    auto s            = migraphx::interpolate_string(input, {{"ww", "world"}, {"hh", "Hello"}});
    EXPECT(s == "Helloworld!");
}

TEST_CASE(interpolate_string_dollar_sign)
{
    std::string input = "$hello";
    auto s            = migraphx::interpolate_string(input, {{"hello", "bye"}});
    EXPECT(s == "$hello");
}

TEST_CASE(interpolate_string_missing)
{
    std::string input = "${hello}";
    EXPECT(test::throws([&] { migraphx::interpolate_string(input, {{"h", "bye"}}); }));
}

TEST_CASE(interpolate_string_custom1)
{
    std::string input = "****{{a}}****";
    auto s            = migraphx::interpolate_string(input, {{"a", "b"}}, "{{", "}}");
    EXPECT(s == "****b****");
}

TEST_CASE(interpolate_string_custom2)
{
    std::string input = "****{{{a}}}****";
    auto s            = migraphx::interpolate_string(input, {{"a", "b"}}, "{{{", "}}}");
    EXPECT(s == "****b****");
}

TEST_CASE(interpolate_string_custom3)
{
    std::string input = "****{{{{a}}}}****";
    auto s            = migraphx::interpolate_string(input, {{"a", "b"}}, "{{{{", "}}}}");
    EXPECT(s == "****b****");
}

TEST_CASE(slit_string_simple1)
{
    std::string input = "one,two,three";
    auto resuts       = migraphx::split_string(input, ',');
    EXPECT(resuts.size() == 3);
    EXPECT(resuts.front() == "one");
    EXPECT(resuts.back() == "three");
}

TEST_CASE(slit_string_simple2)
{
    std::string input = "one";
    auto resuts       = migraphx::split_string(input, ',');
    EXPECT(resuts.size() == 1);
    EXPECT(resuts.front() == "one");
}

TEST_CASE(slit_string_simple3)
{
    std::string input = "one two three";
    auto resuts       = migraphx::split_string(input, ',');
    EXPECT(resuts.size() == 1);
    EXPECT(resuts.front() == "one two three");
}

TEST_CASE(to_hex_string_empty)
{
    const std::vector<std::uint8_t> v{};
    EXPECT(migraphx::to_hex_string(v).empty());
    EXPECT(migraphx::to_hex_string(v, true).empty());
}

TEST_CASE(to_hex_string_uint8_single)
{
    const std::vector<std::uint8_t> v{0xab};
    EXPECT(migraphx::to_hex_string(v) == "ab");
    // Single-byte elements are orientation-invariant.
    EXPECT(migraphx::to_hex_string(v, true) == "ab");
}

TEST_CASE(to_hex_string_uint8_multiple)
{
    const std::vector<std::uint8_t> v{0xca, 0xfe, 0xba, 0xbe};
    EXPECT(migraphx::to_hex_string(v) == "cafebabe");
    EXPECT(migraphx::to_hex_string(v, true) == "cafebabe");
}

TEST_CASE(to_hex_string_uint8_zero_and_max)
{
    const std::vector<std::uint8_t> v{0x00, 0xff, 0x01, 0x10};
    EXPECT(migraphx::to_hex_string(v) == "00ff0110");
}

TEST_CASE(to_hex_string_uint16_msb)
{
    const std::vector<std::uint16_t> v{0xabcd};
    EXPECT(migraphx::to_hex_string(v) == "abcd");
}

TEST_CASE(to_hex_string_uint16_lsb)
{
    const std::vector<std::uint16_t> v{0xabcd};
    EXPECT(migraphx::to_hex_string(v, true) == "cdab");
}

TEST_CASE(to_hex_string_uint32_msb)
{
    const std::vector<std::uint32_t> v{0xdeadbeef};
    EXPECT(migraphx::to_hex_string(v) == "deadbeef");
}

TEST_CASE(to_hex_string_uint32_lsb)
{
    const std::vector<std::uint32_t> v{0xdeadbeef};
    EXPECT(migraphx::to_hex_string(v, true) == "efbeadde");
}

TEST_CASE(to_hex_string_uint64_msb)
{
    const std::vector<std::uint64_t> v{0x0123456789abcdefULL};
    EXPECT(migraphx::to_hex_string(v) == "0123456789abcdef");
}

TEST_CASE(to_hex_string_uint64_lsb)
{
    const std::vector<std::uint64_t> v{0x0123456789abcdefULL};
    EXPECT(migraphx::to_hex_string(v, true) == "efcdab8967452301");
}

TEST_CASE(to_hex_string_uint32_multiple)
{
    const std::vector<std::uint32_t> v{0xdeadbeef, 0xcafebabe};
    EXPECT(migraphx::to_hex_string(v) == "deadbeefcafebabe");
    EXPECT(migraphx::to_hex_string(v, true) == "efbeaddebebafeca");
}

TEST_CASE(to_hex_string_zero_padding)
{
    // Each element produces exactly 2 * sizeof(T) characters, so small values
    // are zero-padded to the full element width.
    const std::vector<std::uint32_t> v{0x00000001};
    EXPECT(migraphx::to_hex_string(v) == "00000001");
    EXPECT(migraphx::to_hex_string(v, true) == "01000000");
}

TEST_CASE(to_hex_string_std_array)
{
    const std::array<std::uint16_t, 2> a = {0x1234, 0x5678};
    EXPECT(migraphx::to_hex_string(a) == "12345678");
    EXPECT(migraphx::to_hex_string(a, true) == "34127856");
}

TEST_CASE(to_hex_string_initializer_list)
{
    EXPECT(migraphx::to_hex_string(std::initializer_list<std::uint8_t>{0xde, 0xad}) == "dead");
}

TEST_CASE(to_hex_string_length)
{
    // Output length is exactly 2 * sizeof(T) * n regardless of element values.
    const std::vector<std::uint32_t> v(5, 0);
    EXPECT(migraphx::to_hex_string(v).size() == 2 * sizeof(std::uint32_t) * v.size());
}

TEST_CASE(to_hex_string_signed_is_unsigned_bitpattern)
{
    // Signed inputs are reinterpreted through std::make_unsigned, so -1 prints
    // as the all-ones byte pattern of its underlying width.
    const std::vector<std::int8_t> v{std::int8_t{-1}};
    EXPECT(migraphx::to_hex_string(v) == "ff");

    const std::vector<std::int32_t> w{std::int32_t{-1}};
    EXPECT(migraphx::to_hex_string(w) == "ffffffff");
    EXPECT(migraphx::to_hex_string(w, true) == "ffffffff");
}

TEST_CASE(to_hex_string_md5_initial_state_lsb)
{
    // LSB ordering matches MD5's canonical byte layout: the initial state
    // serialized LSB-first is the well-known digest of the empty string.
    const std::array<std::uint32_t, 4> state = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    EXPECT(migraphx::to_hex_string(state, true) == "0123456789abcdeffedcba9876543210");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
