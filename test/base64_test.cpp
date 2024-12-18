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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
