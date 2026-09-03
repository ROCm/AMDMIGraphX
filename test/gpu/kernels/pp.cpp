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
 *
 */
#include <migraphx/kernels/pp.hpp>
#include <migraphx/kernels/test.hpp>

// NOLINTBEGIN(*-macro-to-enum)
#define MIGRAPHX_TEST_PP_INDEX 2
// NOLINTEND(*-macro-to-enum)

// NOLINTNEXTLINE
#define MIGRAPHX_TEST_PP_DOUBLE(x) ((x) * 2)
// NOLINTNEXTLINE
#define MIGRAPHX_TEST_PP_MUL(x, d) ((x) * (d))
// NOLINTNEXTLINE
#define MIGRAPHX_TEST_PP_ADD_STMT(x) sum += (x);
// NOLINTNEXTLINE
#define MIGRAPHX_TEST_PP_MUL_ADD_STMT(x, d) sum += (x) * (d);
// NOLINTNEXTLINE
#define MIGRAPHX_TEST_PP_ACCUM(i, x) sum += (i) + (x);

TEST_CASE(pp_primitive_cat)
{
    int x2 = 5;
    // cppcheck-suppress knownConditionTrueFalse
    EXPECT(MIGRAPHX_PP_PRIMITIVE_CAT(x, 2) == 5);
}

TEST_CASE(pp_cat_expands_arguments)
{
    int x2 = 5;
    // cppcheck-suppress knownConditionTrueFalse
    EXPECT(MIGRAPHX_PP_CAT(x, MIGRAPHX_TEST_PP_INDEX) == 5);
}

TEST_CASE(pp_eat)
{
    int x = 1 MIGRAPHX_PP_EAT(+100);
    // cppcheck-suppress knownConditionTrueFalse
    EXPECT(x == 1);
}

TEST_CASE(pp_expand) { EXPECT(MIGRAPHX_PP_EXPAND(1 + 1) == 2); }

TEST_CASE(pp_comma)
{
    int arr[] = {1 MIGRAPHX_PP_COMMA() 2};
    EXPECT(sizeof(arr) == 2 * sizeof(int));
    EXPECT(arr[0] == 1);
    EXPECT(arr[1] == 2);
}

TEST_CASE(pp_iif)
{
    // cppcheck-suppress duplicateExpression
    EXPECT(MIGRAPHX_PP_IIF(1)(10, 20) == 10);
    // cppcheck-suppress duplicateExpression
    EXPECT(MIGRAPHX_PP_IIF(0)(10, 20) == 20);
}

TEST_CASE(pp_compl)
{
    EXPECT(MIGRAPHX_PP_COMPL(0) == 1);
    EXPECT(MIGRAPHX_PP_COMPL(1) == 0);
}

TEST_CASE(pp_bitand)
{
    EXPECT(MIGRAPHX_PP_BITAND(0)(0) == 0);
    EXPECT(MIGRAPHX_PP_BITAND(0)(1) == 0);
    // cppcheck-suppress duplicateExpression
    EXPECT(MIGRAPHX_PP_BITAND(1)(0) == 0);
    // cppcheck-suppress duplicateExpression
    EXPECT(MIGRAPHX_PP_BITAND(1)(1) == 1);
}

TEST_CASE(pp_not)
{
    EXPECT(MIGRAPHX_PP_NOT(0) == 1);
    EXPECT(MIGRAPHX_PP_NOT(1) == 0);
    EXPECT(MIGRAPHX_PP_NOT(5) == 0);
}

TEST_CASE(pp_bool)
{
    EXPECT(MIGRAPHX_PP_BOOL(0) == 0);
    EXPECT(MIGRAPHX_PP_BOOL(1) == 1);
    EXPECT(MIGRAPHX_PP_BOOL(7) == 1);
}

TEST_CASE(pp_is_paren)
{
    EXPECT(MIGRAPHX_PP_IS_PAREN(abc) == 0);
    EXPECT(MIGRAPHX_PP_IS_PAREN(()) == 1);
    EXPECT(MIGRAPHX_PP_IS_PAREN((1, 2)) == 1);
}

TEST_CASE(pp_is_empty_arg)
{
    EXPECT(MIGRAPHX_PP_IS_EMPTY_ARG() == 1);
    EXPECT(MIGRAPHX_PP_IS_EMPTY_ARG(abc) == 0);
    EXPECT(MIGRAPHX_PP_IS_EMPTY_ARG((1, 2)) == 0);
}

TEST_CASE(pp_repeat_fixed)
{
    int sum = 0;
    MIGRAPHX_PP_REPEAT3(MIGRAPHX_TEST_PP_ACCUM, 1)
    EXPECT(sum == 10);
}

TEST_CASE(pp_repeat_selected)
{
    int sum = 0;
    MIGRAPHX_PP_REPEAT(2)(MIGRAPHX_TEST_PP_ACCUM, 5) EXPECT(sum == 18);
}

TEST_CASE(pp_repeat_zero)
{
    int sum = 0;
    MIGRAPHX_PP_REPEAT0(MIGRAPHX_TEST_PP_ACCUM, 3)
    EXPECT(sum == 3);
}

// cppcheck's preprocessor cannot expand the recursive argument-transform macros
#ifndef CPPCHECK
TEST_CASE(pp_generate)
{
    int arr[] = {MIGRAPHX_PP_GENERATE(4)};
    EXPECT(sizeof(arr) == 5 * sizeof(int));
    EXPECT(arr[0] == 0);
    EXPECT(arr[1] == 1);
    EXPECT(arr[2] == 2);
    EXPECT(arr[3] == 3);
    EXPECT(arr[4] == 4);
}

TEST_CASE(pp_each_args)
{
    int sum = 0;
    MIGRAPHX_PP_EACH_ARGS(MIGRAPHX_TEST_PP_ADD_STMT, 1, 2, 3)
    EXPECT(sum == 6);
}

TEST_CASE(pp_each_args_single)
{
    int sum = 0;
    MIGRAPHX_PP_EACH_ARGS(MIGRAPHX_TEST_PP_ADD_STMT, 42)
    EXPECT(sum == 42);
}

TEST_CASE(pp_each_args_max)
{
    int sum = 0;
    MIGRAPHX_PP_EACH_ARGS(
        MIGRAPHX_TEST_PP_ADD_STMT, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    EXPECT(sum == 136);
}

TEST_CASE(pp_each_args_data)
{
    int sum = 0;
    MIGRAPHX_PP_EACH_ARGS_DATA(MIGRAPHX_TEST_PP_MUL_ADD_STMT, 10, 1, 2, 3)
    EXPECT(sum == 60);
}

TEST_CASE(pp_transform_args)
{
    int arr[] = {MIGRAPHX_PP_TRANSFORM_ARGS(MIGRAPHX_TEST_PP_DOUBLE, 1, 2, 3)};
    EXPECT(sizeof(arr) == 3 * sizeof(int));
    EXPECT(arr[0] == 2);
    EXPECT(arr[1] == 4);
    EXPECT(arr[2] == 6);
}

TEST_CASE(pp_transform_args_single)
{
    int arr[] = {MIGRAPHX_PP_TRANSFORM_ARGS(MIGRAPHX_TEST_PP_DOUBLE, 21)};
    EXPECT(sizeof(arr) == sizeof(int));
    EXPECT(arr[0] == 42);
}

TEST_CASE(pp_transform_args_data)
{
    int arr[] = {MIGRAPHX_PP_TRANSFORM_ARGS_DATA(MIGRAPHX_TEST_PP_MUL, 3, 1, 2)};
    EXPECT(sizeof(arr) == 2 * sizeof(int));
    EXPECT(arr[0] == 3);
    EXPECT(arr[1] == 6);
}

TEST_CASE(pp_enum_concat)
{
    int x0    = 1;
    int x1    = 2;
    int x2    = 3;
    int arr[] = {MIGRAPHX_PP_ENUM(2, x)};
    EXPECT(sizeof(arr) == 3 * sizeof(int));
    EXPECT(arr[0] == 1);
    EXPECT(arr[1] == 2);
    EXPECT(arr[2] == 3);
}

TEST_CASE(pp_enum_multi_token)
{
    auto f = [](MIGRAPHX_PP_ENUM(1, int y)) { return y0 * 10 + y1; };
    EXPECT(f(3, 4) == 34);
}

TEST_CASE(pp_enum_pairs)
{
    using type0 = int;
    using type1 = int;
    auto f      = [](MIGRAPHX_PP_ENUM(1, type, y)) { return y0 * 10 + y1; };
    EXPECT(f(3, 4) == 34);
}
#endif
