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
#include <migraphx/base64.hpp>
#include "test.hpp"

TEST_CASE(base64_encoding)
{
    std::string input_0    = "abc";
    std::string expected_0 = "YWJj";
    EXPECT(migraphx::b64_encode(input_0) == expected_0);

    std::string input_1    = "abcd";
    std::string expected_1 = "YWJjZA==";
    EXPECT(migraphx::b64_encode(input_1) == expected_1);

    std::string input_2    = "convolution";
    std::string expected_2 = "Y29udm9sdXRpb24=";
    EXPECT(migraphx::b64_encode(input_2) == expected_2);

    std::string input_3    = "https://www.amd.com/en/products/software/rocm.html";
    std::string expected_3 = "aHR0cHM6Ly93d3cuYW1kLmNvbS9lbi9wcm9kdWN0cy9zb2Z0d2FyZS9yb2NtLmh0bWw=";
    EXPECT(migraphx::b64_encode(input_3) == expected_3);

    std::string input_4    = "{1, 3, 7, 9}";
    std::string expected_4 = "ezEsIDMsIDcsIDl9";
    EXPECT(migraphx::b64_encode(input_4) == expected_4);
}

TEST_CASE(base64_RFC_test_vectors)
{
    std::string input_0    = "";
    std::string expected_0 = "";
    EXPECT(migraphx::b64_encode(input_0) == expected_0);

    std::string input_1    = "f";
    std::string expected_1 = "Zg==";
    EXPECT(migraphx::b64_encode(input_1) == expected_1);

    std::string input_2    = "fo";
    std::string expected_2 = "Zm8=";
    EXPECT(migraphx::b64_encode(input_2) == expected_2);

    std::string input_3    = "foo";
    std::string expected_3 = "Zm9v";
    EXPECT(migraphx::b64_encode(input_3) == expected_3);

    std::string input_4    = "foob";
    std::string expected_4 = "Zm9vYg==";
    EXPECT(migraphx::b64_encode(input_4) == expected_4);

    std::string input_5    = "fooba";
    std::string expected_5 = "Zm9vYmE=";
    EXPECT(migraphx::b64_encode(input_5) == expected_5);

    std::string input_6    = "foobar";
    std::string expected_6 = "Zm9vYmFy";
    EXPECT(migraphx::b64_encode(input_6) == expected_6);
}

// Following tests altered from
// https://github.com/tobiaslocker/base64/blob/master/test/base64_tests.cpp
TEST_CASE(base64_encodes_empty)
{
    std::string const expected{};
    std::string const actual{migraphx::b64_encode({})};
    EXPECT(expected == actual);
}

TEST_CASE(base64_encodes_three_bytes_zeros)
{
    std::array<std::uint8_t, 3> const input{0x00, 0x00, 0x00};
    auto const expected{"AAAA"};
    auto const actual{migraphx::b64_encode({input.begin(), input.end()})};
    EXPECT(expected == actual);
}

TEST_CASE(base64_encodes_three_bytes_random)
{
    std::array<std::uint8_t, 3> const input{0xFE, 0xE9, 0x72};
    std::string const expected{"/uly"};
    std::string const actual{migraphx::b64_encode({input.begin(), input.end()})};
    EXPECT(expected == actual);
}

TEST_CASE(base64_encodes_two_bytes)
{
    std::array<std::uint8_t, 2> const input{0x00, 0x00};
    auto const expected{"AAA="};
    auto const actual{migraphx::b64_encode({input.begin(), input.end()})};
    EXPECT(expected == actual);
}

TEST_CASE(base64_encodes_one_byte)
{
    std::array<std::uint8_t, 1> const input{0x00};
    auto const expected{"AA=="};
    auto const actual{migraphx::b64_encode({input.begin(), input.end()})};
    EXPECT(expected == actual);
}

TEST_CASE(base64_encodes_four_bytes)
{
    std::array<std::uint8_t, 4> const input{0x74, 0x68, 0x65, 0x20};
    auto const expected{"dGhlIA=="};
    auto const actual{migraphx::b64_encode({input.begin(), input.end()})};
    EXPECT(expected == actual);
}

TEST_CASE(base64_encodes_five_bytes)
{
    std::array<std::uint8_t, 5> const input{0x20, 0x62, 0x72, 0x6f, 0x77};
    auto const expected{"IGJyb3c="};
    auto const actual{migraphx::b64_encode({input.begin(), input.end()})};
    EXPECT(expected == actual);
}

TEST_CASE(base64_encodes_six_bytes)
{
    std::array<std::uint8_t, 6> const input{0x20, 0x6a, 0x75, 0x6d, 0x70, 0x73};
    auto const expected{"IGp1bXBz"};
    auto const actual{migraphx::b64_encode({input.begin(), input.end()})};
    EXPECT(expected == actual);
}

TEST_CASE(base64_encodes_BrownFox)
{
    std::array<std::uint8_t, 43> const input{
        0x74, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b, 0x20, 0x62, 0x72, 0x6f, 0x77, 0x6e,
        0x20, 0x66, 0x6f, 0x78, 0x20, 0x6a, 0x75, 0x6d, 0x70, 0x73, 0x20, 0x6f, 0x76, 0x65, 0x72,
        0x20, 0x74, 0x68, 0x65, 0x20, 0x6c, 0x61, 0x7a, 0x79, 0x20, 0x64, 0x6f, 0x67};

    auto const expected{"dGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw=="};
    auto const actual{migraphx::b64_encode({input.begin(), input.end()})};
    EXPECT(expected == actual);
}

TEST_CASE(base64_encodes_EncodesBrownFastFoxNullInMiddle)
{
    std::array<std::uint8_t, 45> const input{
        0x74, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b, 0x21, 0x20, 0x62, 0x72, 0x6f, 0x77,
        0x6e, 0x20, 0x66, 0x6f, 0x78, 0x20, 0x6a, 0x75, 0x6d, 0x70, 0x73, 0x20, 0x6f, 0x76, 0x65,
        0x72, 0x20, 0x74, 0x68, 0x65, 0x00, 0x20, 0x6c, 0x61, 0x7a, 0x79, 0x20, 0x64, 0x6f, 0x67};

    auto const expected{"dGhlIHF1aWNrISBicm93biBmb3gganVtcHMgb3ZlciB0aGUAIGxhenkgZG9n"};
    auto const actual{migraphx::b64_encode({input.begin(), input.end()})};
    EXPECT(expected == actual);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
