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
#include <stdexcept>
#include <string>
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

TEST_CASE(convert_uint8_out_of_range)
{
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(200)) == migraphx::compile_modes::max);
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(101)) == migraphx::compile_modes::max);
}

TEST_CASE(convert_string_integer_out_of_range)
{
    EXPECT(migraphx::convert_to_compile_mode("-5") == migraphx::compile_modes::eager);
    EXPECT(migraphx::convert_to_compile_mode("200") == migraphx::compile_modes::max);
}

TEST_CASE(convert_string_invalid)
{
    EXPECT(test::throws([&] { migraphx::convert_to_compile_mode("invalid"); }));
}

TEST_CASE(convert_string_empty_throws)
{
    EXPECT(test::throws([&] { migraphx::convert_to_compile_mode(""); }));
}

TEST_CASE(convert_uint8_boundary_25)
{
    auto result = migraphx::convert_to_compile_mode(uint8_t(25));
    EXPECT(result == migraphx::compile_modes::eager or result == migraphx::compile_modes::balanced);
}

TEST_CASE(convert_uint8_boundary_75)
{
    auto result = migraphx::convert_to_compile_mode(uint8_t(75));
    EXPECT(result == migraphx::compile_modes::balanced or result == migraphx::compile_modes::max);
}

TEST_CASE(convert_string_trailing_garbage)
{
    // std::stoi stops at the first non-digit — pin the accepted behaviour
    EXPECT(migraphx::convert_to_compile_mode("50abc") == migraphx::compile_modes::balanced);
}

TEST_CASE(convert_string_leading_whitespace)
{
    EXPECT(migraphx::convert_to_compile_mode(" 100") == migraphx::compile_modes::max);
}

TEST_CASE(convert_string_plus_sign)
{
    EXPECT(migraphx::convert_to_compile_mode("+50") == migraphx::compile_modes::balanced);
}

TEST_CASE(convert_string_overflow_throws)
{
    // std::stoi raises out_of_range here, not invalid_argument
    EXPECT(
        test::throws<std::out_of_range>([&] { migraphx::convert_to_compile_mode("99999999999"); }));
}

TEST_CASE(convert_string_invalid_throws_invalid_argument)
{
    EXPECT(
        test::throws<std::invalid_argument>([&] { migraphx::convert_to_compile_mode("invalid"); }));
}

TEST_CASE(convert_uint8_tie_resolves_to_lower_mode)
{
    // std::min_element keeps the first match, so exact ties pick the lower mode.
    // convert_uint8_boundary_25/_75 accept either answer and so assert nothing;
    // these pin the rule and cover both comparator outcomes.
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(25)) == migraphx::compile_modes::eager);
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(75)) == migraphx::compile_modes::balanced);
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(24)) == migraphx::compile_modes::eager);
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(26)) == migraphx::compile_modes::balanced);
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(74)) == migraphx::compile_modes::balanced);
    EXPECT(migraphx::convert_to_compile_mode(uint8_t(76)) == migraphx::compile_modes::max);
}

TEST_CASE(convert_uint8_in_range_always_yields_known_mode)
{
    for(int i = 0; i <= 100; ++i)
    {
        auto m = migraphx::convert_to_compile_mode(static_cast<uint8_t>(i));
        EXPECT(m == migraphx::compile_modes::eager or m == migraphx::compile_modes::balanced or
               m == migraphx::compile_modes::max);
    }
}

TEST_CASE(convert_string_and_uint8_agree)
{
    for(int i = 0; i <= 100; ++i)
        EXPECT(migraphx::convert_to_compile_mode(std::to_string(i)) ==
               migraphx::convert_to_compile_mode(static_cast<uint8_t>(i)));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
