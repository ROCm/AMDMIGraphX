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
#include <migraphx/pp.hpp>
#include <string>
#include "test.hpp"

// Helper macros for the tests below. They use the `_PP_` infix so clang-tidy recognizes them as
// preprocessor helpers.
#define MIGRAPHX_TEST_PP_IDENT(x) x
#define MIGRAPHX_TEST_PP_QUOTE(x) #x
#define MIGRAPHX_TEST_PP_PLUS(x) +(x)
#define MIGRAPHX_TEST_PP_ADD_ONE(z, ...) +1
#define MIGRAPHX_TEST_PP_ADD_INDEX(z, ...) +(z)

TEST_CASE(cat)
{
    EXPECT(MIGRAPHX_PP_CAT(12, 34) == 1234);
    // Concatenation forms a new identifier.
    int MIGRAPHX_PP_CAT(foo, bar) = 99;
    EXPECT(foobar == 99);
}

TEST_CASE(expand_eat_comma)
{
    EXPECT(MIGRAPHX_PP_EXPAND(1 + 2) == 3);
    // EAT discards its arguments entirely.
    int n = 5 MIGRAPHX_PP_EAT(+100);
    EXPECT(n == 5);
    // COMMA always expands to a single comma, ignoring its arguments.
    int arr[] = {1 MIGRAPHX_PP_COMMA(z) 2};
    EXPECT(arr[0] == 1);
    EXPECT(arr[1] == 2);
}

TEST_CASE(iif)
{
    EXPECT(MIGRAPHX_PP_IIF(1)(10, 20) == 10);
    EXPECT(MIGRAPHX_PP_IIF(0)(10, 20) == 20);
}

TEST_CASE(complement)
{
    EXPECT(MIGRAPHX_PP_COMPL(0) == 1);
    EXPECT(MIGRAPHX_PP_COMPL(1) == 0);
}

TEST_CASE(bit_and)
{
    EXPECT(MIGRAPHX_PP_BITAND(1)(5) == 5);
    EXPECT(MIGRAPHX_PP_BITAND(0)(5) == 0);
}

TEST_CASE(is_paren)
{
    EXPECT(MIGRAPHX_PP_IS_PAREN((7)) == 1);
    EXPECT(MIGRAPHX_PP_IS_PAREN(7) == 0);
}

TEST_CASE(is_empty_arg)
{
    EXPECT(MIGRAPHX_PP_IS_EMPTY_ARG() == 1);
    EXPECT(MIGRAPHX_PP_IS_EMPTY_ARG(7) == 0);
}

TEST_CASE(repeat)
{
    // REPEAT(n, m, ...) invokes m for the indices 0..n inclusive.
    int count = 0 MIGRAPHX_PP_REPEAT(4, MIGRAPHX_TEST_PP_ADD_ONE, ~);
    EXPECT(count == 5);
    int index_sum = 0 MIGRAPHX_PP_REPEAT(3, MIGRAPHX_TEST_PP_ADD_INDEX, ~);
    EXPECT(index_sum == 6); // 0 + 1 + 2 + 3
}

TEST_CASE(each_args)
{
    // EACH_ARGS applies the macro to each argument with no separator.
    int sum = 0 MIGRAPHX_PP_EACH_ARGS(MIGRAPHX_TEST_PP_PLUS, 1, 2, 3, 4);
    EXPECT(sum == 10);
    int one = 0 MIGRAPHX_PP_EACH_ARGS(MIGRAPHX_TEST_PP_PLUS, 9);
    EXPECT(one == 9);
}

TEST_CASE(transform_args_values)
{
    // TRANSFORM_ARGS applies the macro to each argument separated by commas.
    int arr[] = {MIGRAPHX_PP_TRANSFORM_ARGS(MIGRAPHX_TEST_PP_IDENT, 3, 5, 7)};
    EXPECT(sizeof(arr) == 3 * sizeof(int));
    EXPECT(arr[0] == 3);
    EXPECT(arr[1] == 5);
    EXPECT(arr[2] == 7);
}

TEST_CASE(transform_args_single)
{
    // A single argument produces no trailing comma.
    int arr[] = {MIGRAPHX_PP_TRANSFORM_ARGS(MIGRAPHX_TEST_PP_IDENT, 42)};
    EXPECT(sizeof(arr) == sizeof(int));
    EXPECT(arr[0] == 42);
}

TEST_CASE(transform_args_strings)
{
    std::string names[] = {MIGRAPHX_PP_TRANSFORM_ARGS(MIGRAPHX_TEST_PP_QUOTE, red, green, blue)};
    EXPECT(names[0] == "red");
    EXPECT(names[1] == "green");
    EXPECT(names[2] == "blue");
}

TEST_CASE(transform_args_max)
{
    // Sixteen arguments is the maximum supported by the transform.
    int arr[] = {MIGRAPHX_PP_TRANSFORM_ARGS(
        MIGRAPHX_TEST_PP_IDENT, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)};
    EXPECT(sizeof(arr) == 16 * sizeof(int));
    EXPECT(arr[0] == 0);
    EXPECT(arr[15] == 15);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
