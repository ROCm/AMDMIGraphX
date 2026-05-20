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
#include <migraphx/compile_modes.hpp>
#include <test.hpp>

TEST_CASE(convert_uint8_eager)
{
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(0)) == migraphx::compile_modes::eager);
}

TEST_CASE(convert_uint8_balanced)
{
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(50)) == migraphx::compile_modes::balanced);
}

TEST_CASE(convert_uint8_max)
{
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(100)) == migraphx::compile_modes::max);
}

TEST_CASE(convert_uint8_closest_to_eager)
{
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(1)) == migraphx::compile_modes::eager);
}

TEST_CASE(convert_uint8_closest_to_balanced)
{
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(30)) == migraphx::compile_modes::balanced);
}

TEST_CASE(convert_uint8_closest_to_max)
{
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(99)) == migraphx::compile_modes::max);
}

TEST_CASE(convert_uint8_midpoint)
{
    auto result = migraphx::convert_to_compile_mode(uint8_t(25));
    EXPECT(result == migraphx::compile_modes::eager or result == migraphx::compile_modes::balanced);
}

TEST_CASE(convert_string_eager)
{
    EXPECT(migraphx::convert_to_compile_mode("eager") == migraphx::compile_modes::eager);
}

TEST_CASE(convert_string_balanced)
{
    EXPECT(migraphx::convert_to_compile_mode("balanced") == migraphx::compile_modes::balanced);
}

TEST_CASE(convert_string_max)
{
    EXPECT(migraphx::convert_to_compile_mode("max") == migraphx::compile_modes::max);
}

TEST_CASE(convert_string_case_insensitive)
{
    EXPECT(migraphx::convert_to_compile_mode("EAGER") == migraphx::compile_modes::eager);
    EXPECT(migraphx::convert_to_compile_mode("Balanced") == migraphx::compile_modes::balanced);
    EXPECT(migraphx::convert_to_compile_mode("MAX") == migraphx::compile_modes::max);
}

TEST_CASE(convert_string_integer)
{
    EXPECT(migraphx::convert_to_compile_mode("0") == migraphx::compile_modes::eager);
    EXPECT(migraphx::convert_to_compile_mode("50") == migraphx::compile_modes::balanced);
    EXPECT(migraphx::convert_to_compile_mode("100") == migraphx::compile_modes::max);
}

TEST_CASE(convert_string_integer_closest)
{
    EXPECT(migraphx::convert_to_compile_mode("30") == migraphx::compile_modes::balanced);
}

TEST_CASE(convert_string_invalid)
{
    EXPECT(test::throws([&] { migraphx::convert_to_compile_mode("invalid"); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
